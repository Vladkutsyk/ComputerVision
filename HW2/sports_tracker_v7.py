"""
SPORTS VIDEO TRACKER V6
=======================
Built on V5 (CSRT + Kalman + Grass-Freeze + Kick Detection).

NEW in V6:
  1. Enhanced Kalman — 6-state [cx,cy,vx,vy,ax,ay] constant-acceleration model
                       + velocity clamp (no runaway predictions).
  2. Appearance Model — HSV 2D histogram (H+S channels) built at init and
                        updated only when the patch is clean enough.
                        Used to detect when CSRT drifted onto a wrong object.
  3. GrabCut Separation — when multi-object overlap is suspected (appearance
                          drop OR bbox area spike), GrabCut segments the bbox
                          into foreground blobs, scores each by appearance +
                          Kalman proximity, and re-inits the tracker on the
                          correct blob.

Controls:
    Draw ROI  — hold LMB + drag
    ENTER     — start / resume
    SPACE     — pause / resume
    R         — re-select target
    P         — toggle PiP
    Q / ESC   — quit

Dependencies:
    pip install opencv-contrib-python numpy
"""

import cv2
import numpy as np
import sys
import os
import time
from collections import deque

# ── CSRT tuning ──────────────────────────────────────────────────────────────
CSRT_PARAMS = {
    "admm_iterations":       4,
    "background_ratio":      2,
    "num_hog_channels_used": 18,
    "padding":               3.0,
    "template_size":         200,
    "gsl_sigma":             1.0,
    "hog_orientations":      9,
    "num_scales":            33,
    "scale_step":            1.02,
    "scale_sigma_factor":    0.25,
    "psr_threshold":         0.035,
    "use_channel_weights":   True,
    "use_color_names":       True,
    "use_gray":              True,
    "use_hog":               True,
}

# ── General tracking ─────────────────────────────────────────────────────────
MAX_LOST_FRAMES     = 45
REDETECT_THRESHOLD  = 0.50
TEMPLATE_UPDATE_INT = 8
MAX_FRAME_WIDTH     = 1280

# ── Trajectory / kick detection (V5) ─────────────────────────────────────────
TRAJECTORY_TOLERANCE = 1.15
MAX_KICK_DISTANCE    = 4.5

# ── NEW V6: Appearance model ──────────────────────────────────────────────────
APPEARANCE_BINS          = 32    # H and S histogram bins
APPEARANCE_THRESHOLD     = 0.42  # min correlation to trust CSRT result
APPEARANCE_UPDATE_THRESH = 0.68  # min correlation to allow model soft-update
APPEARANCE_AREA_SPIKE    = 1.80  # bbox-area ratio above which multi-obj assumed

# ── NEW V6: GrabCut separation ────────────────────────────────────────────────
GRABCUT_ITERS         = 3     # GrabCut EM iterations (3 = fast + good enough)
GRABCUT_BLOB_MIN_RATIO = 0.10  # blob must cover ≥10 % of bbox area to count
GRABCUT_MIN_SIDE      = 24    # skip GrabCut if bbox side < this (px)

# ── Colours ───────────────────────────────────────────────────────────────────
CLR_OK       = (0,   220,  80)
CLR_WARN     = (0,   165, 255)
CLR_COAST    = (255, 105, 180)
CLR_LOST     = (50,   50, 220)
CLR_SELECT   = (0,   220, 255)
CLR_PIP      = (0,   220, 220)
CLR_TRAIL    = (0,   180, 255)
CLR_FROZEN   = (255,  50,  50)
CLR_KICK     = (255, 255,   0)
CLR_MULTIOBJ = (0,   100, 255)   # orange — multi-object / GrabCut event


# ─────────────────────────────────────────────────────────────────────────────
#  1. ENHANCED KALMAN  (6-state: position + velocity + acceleration)
# ─────────────────────────────────────────────────────────────────────────────

class KalmanPredictor:
    """
    Constant-acceleration Kalman filter.
    State  : [cx, cy, vx, vy, ax, ay]
    Measure: [cx, cy]

    Why 6-state?
      - Ball follows a parabolic arc (acceleration ≠ 0).
      - Player decelerates / accelerates when changing direction.
      - 4-state constant-velocity model diverges quickly during such events.
    """

    MAX_SPEED = 130  # px / frame — clamp; prevents runaway after bad measurement

    def __init__(self):
        dt = 1.0
        kf = cv2.KalmanFilter(6, 2)

        # Constant-acceleration transition
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0,  0.5*dt*dt, 0        ],
            [0, 1, 0,  dt, 0,         0.5*dt*dt],
            [0, 0, 1,  0,  dt,        0        ],
            [0, 0, 0,  1,  0,         dt       ],
            [0, 0, 0,  0,  1,         0        ],
            [0, 0, 0,  0,  0,         1        ],
        ], dtype=np.float32)

        # Observe only cx, cy
        kf.measurementMatrix = np.zeros((2, 6), dtype=np.float32)
        kf.measurementMatrix[0, 0] = 1.0
        kf.measurementMatrix[1, 1] = 1.0

        # Process noise: tight on position, medium on velocity, loose on accel
        kf.processNoiseCov = np.diag(
            [1e-3, 1e-3, 5e-2, 5e-2, 5e-1, 5e-1]
        ).astype(np.float32)

        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
        kf.errorCovPost        = np.eye(6, dtype=np.float32) * 0.1

        self.kf          = kf
        self.initialized = False

    def init(self, cx: float, cy: float, vx: float = 0., vy: float = 0.):
        state = np.array([[cx], [cy], [vx], [vy], [0.], [0.]], dtype=np.float32)
        self.kf.statePre  = state.copy()
        self.kf.statePost = state.copy()
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 0.1
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return None
        p = self.kf.predict()
        return float(p[0, 0]), float(p[1, 0])

    def correct(self, cx: float, cy: float):
        if not self.initialized:
            self.init(cx, cy)
            return
        self.kf.correct(np.array([[cx], [cy]], dtype=np.float32))
        # Velocity clamp
        s     = self.kf.statePost
        speed = float(np.hypot(s[2, 0], s[3, 0]))
        if speed > self.MAX_SPEED:
            ratio    = self.MAX_SPEED / speed
            s[2, 0] *= ratio
            s[3, 0] *= ratio
            self.kf.statePost = s

    def get_velocity(self):
        if not self.initialized:
            return 0., 0.
        s = self.kf.statePost
        return float(s[2, 0]), float(s[3, 0])


# ─────────────────────────────────────────────────────────────────────────────
#  2. APPEARANCE MODEL  (2D HSV histogram)
# ─────────────────────────────────────────────────────────────────────────────

class AppearanceModel:
    """
    Maintains a 2D HSV histogram (Hue × Saturation) of the tracked target.

    Why 2D H+S?
      - Value (brightness) is scene-dependent and changes under shadow / floodlights.
      - Hue alone can't distinguish similar-hued kits.
      - H+S together reliably captures jersey colour and skin tones.

    Usage:
      model.build(patch)         — force-build from clean reference
      score = model.score(patch) — Bhattacharyya similarity [0, 1]
      model.soft_update(patch)   — update only if patch looks clean
    """

    def __init__(self, bins: int = APPEARANCE_BINS):
        self.bins = bins
        self.hist = None

    def _make_hist(self, patch: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist(
            [hsv], [0, 1], None,
            [self.bins, self.bins],
            [0, 180, 0, 256]
        )
        cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
        return h

    def build(self, patch: np.ndarray):
        if patch.size == 0:
            return
        self.hist = self._make_hist(patch)

    def score(self, patch: np.ndarray) -> float:
        """Returns correlation similarity in [0, 1]. 1 = perfect match."""
        if self.hist is None or patch.size == 0:
            return 1.0  # no model built yet → always trust CSRT
        h = self._make_hist(patch)
        corr = cv2.compareHist(self.hist, h, cv2.HISTCMP_CORREL)
        return max(0.0, float(corr))

    def soft_update(self, patch: np.ndarray, score: float):
        """Update only if the patch looks like our target (score high enough)."""
        if score >= APPEARANCE_UPDATE_THRESH:
            self.build(patch)


# ─────────────────────────────────────────────────────────────────────────────
#  SPORTS TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class SportsTracker:
    TS_NONE      = "NONE"
    TS_TRACKING  = "TRACKING"
    TS_OCCLUDED  = "OCCLUDED"
    TS_COASTING  = "COASTING"
    TS_OUT_FRAME = "OUT_FRAME"
    TS_LOST      = "LOST"

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        ok, first = self.cap.read()
        if not ok:
            raise IOError("Cannot read first frame")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cur_frame = self._resize(first)

        self.playing   = False
        self.frame_idx = 0

        self.tracker          = None
        self.bbox             = None
        self.track_state      = self.TS_NONE
        self.lost_count       = 0
        self.kalman           = KalmanPredictor()
        self.kalman_bbox_size = None

        self.template         = None
        self.template_fidx    = 0
        self.template_frozen  = False
        self.kick_flash_timer = 0.0

        # ── NEW V6 ──────────────────────────────────────────────────────────
        self.appearance          = AppearanceModel()
        self.app_score           = 1.0   # latest appearance similarity
        self.multi_obj_flash_t   = 0.0  # timestamp of last GrabCut event

        self.trail = deque(maxlen=60)
        self.pip_enabled = True

        self.drawing = False
        self.roi_pt1 = None
        self.roi_pt2 = None

        self.fps_value = 0.0
        self._fps_t0   = time.time()
        self._fps_cnt  = 0

        self.WIN = "Sports Tracker V6"
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        h, w = self.cur_frame.shape[:2]
        cv2.resizeWindow(self.WIN, min(w, 1280), min(h, 720))
        cv2.setMouseCallback(self.WIN, self._mouse_cb)

    # ── Resize ────────────────────────────────────────────────────────────────

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w > MAX_FRAME_WIDTH:
            return cv2.resize(frame, (MAX_FRAME_WIDTH, int(h * MAX_FRAME_WIDTH / w)))
        return frame

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def _win_to_frame(self, wx: int, wy: int):
        try:
            _, _, dw, dh = cv2.getWindowImageRect(self.WIN)
            if dw > 0 and dh > 0 and self.cur_frame is not None:
                fh, fw = self.cur_frame.shape[:2]
                return int(wx * fw / dw), int(wy * fh / dh)
        except Exception:
            pass
        return wx, wy

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def _mouse_cb(self, event, x, y, flags, param):
        fx, fy = self._win_to_frame(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.roi_pt1 = self.roi_pt2 = (fx, fy)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.roi_pt2 = (fx, fy)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.roi_pt2 = (fx, fy)
            if self.cur_frame is not None:
                x1 = min(self.roi_pt1[0], self.roi_pt2[0])
                y1 = min(self.roi_pt1[1], self.roi_pt2[1])
                x2 = max(self.roi_pt1[0], self.roi_pt2[0])
                y2 = max(self.roi_pt1[1], self.roi_pt2[1])
                if (x2 - x1) > 8 and (y2 - y1) > 8:
                    self._init_tracking(self.cur_frame, (x1, y1, x2-x1, y2-y1))

    # ── CSRT factory ──────────────────────────────────────────────────────────

    def _make_csrt(self):
        params = cv2.TrackerCSRT_Params()
        for k, v in CSRT_PARAMS.items():
            if hasattr(params, k):
                try:
                    setattr(params, k, v)
                except Exception:
                    pass
        return cv2.TrackerCSRT_create(params)

    # ── Init tracker ──────────────────────────────────────────────────────────

    def _init_tracking(self, frame: np.ndarray, bbox: tuple):
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x = max(0, min(x, fw - 2));  y = max(0, min(y, fh - 2))
        w = max(10, min(w, fw - x)); h = max(10, min(h, fh - y))
        bbox = (x, y, w, h)

        self.tracker = self._make_csrt()
        self.tracker.init(frame, bbox)
        self.bbox             = bbox
        self.lost_count       = 0
        self.track_state      = self.TS_TRACKING
        self.kalman_bbox_size = (w, h)

        self.kalman = KalmanPredictor()
        self.kalman.init(x + w / 2, y + h / 2)

        self._save_template(frame, bbox, force=True)

        # Build appearance model from the initial selection
        patch = self._get_patch(frame, bbox)
        if patch.size > 0:
            self.appearance.build(patch)
        self.app_score = 1.0

        self.trail.clear()
        self.kick_flash_timer  = 0.0
        self.multi_obj_flash_t = 0.0
        print(f"[TRACKER] Init  bbox={bbox}")

    # ── Grass / environment check (V5) ───────────────────────────────────────

    def _is_open_grass(self, frame: np.ndarray, bbox: tuple) -> bool:
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        mx, my = max(10, w), max(10, h)
        x1, y1 = max(0, x-mx), max(0, y-my)
        x2, y2 = min(fw, x+w+mx), min(fh, y+h+my)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35,40,40]), np.array([85,255,255]))
        mask[y-y1:y-y1+h, x-x1:x-x1+w] = 0
        total = (x2-x1)*(y2-y1) - w*h
        return total > 0 and cv2.countNonZero(mask) / total > 0.40

    def _save_template(self, frame: np.ndarray, bbox: tuple, force: bool = False):
        if not force and not self._is_open_grass(frame, bbox):
            self.template_frozen = True
            return
        self.template_frozen = False
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x1, y1 = max(0,x), max(0,y)
        x2, y2 = min(fw,x+w), min(fh,y+h)
        if x2 > x1 and y2 > y1:
            self.template      = frame[y1:y2, x1:x2].copy()
            self.template_fidx = self.frame_idx

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _center(bbox):
        x, y, w, h = bbox
        return x + w / 2, y + h / 2

    @staticmethod
    def _overlap(bbox, fshape):
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = fshape[:2]
        ix1, iy1 = max(0,x), max(0,y)
        ix2, iy2 = min(fw,x+w), min(fh,y+h)
        if ix2<=ix1 or iy2<=iy1:
            return 0.0
        return (ix2-ix1)*(iy2-iy1) / max(w*h, 1)

    def _get_patch(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """Safe crop of frame at bbox."""
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x1, y1 = max(0,x), max(0,y)
        x2, y2 = min(fw,x+w), min(fh,y+h)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1,1,3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]

    # ── Re-detection (V5) ─────────────────────────────────────────────────────

    def _redetect(self, frame: np.ndarray, pred_pos: tuple):
        if self.template is None or pred_pos is None:
            return None
        g_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g_tmpl  = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        px, py  = pred_pos
        tw, th  = self.kalman_bbox_size or (g_tmpl.shape[1], g_tmpl.shape[0])
        margin  = max(tw, th) * 2
        fh, fw  = g_frame.shape
        sx1 = max(0, int(px-margin));  sy1 = max(0, int(py-margin))
        sx2 = min(fw, int(px+margin)); sy2 = min(fh, int(py+margin))
        search  = g_frame[sy1:sy2, sx1:sx2]
        offset  = (sx1, sy1)
        best_val, best_box = 0.0, None
        for sc in (0.875, 1.0, 1.125):
            nh = max(8, int(g_tmpl.shape[0]*sc))
            nw = max(8, int(g_tmpl.shape[1]*sc))
            if nh >= search.shape[0] or nw >= search.shape[1]:
                continue
            rt  = cv2.resize(g_tmpl, (nw, nh))
            res = cv2.matchTemplate(search, rt, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if mv > best_val:
                best_val = mv
                best_box = (ml[0]+offset[0], ml[1]+offset[1], nw, nh)
        if best_val >= REDETECT_THRESHOLD and best_box:
            return best_box, best_val
        return None

    # ─────────────────────────────────────────────────────────────────────────
    #  3. GRABCUT SEPARATION
    # ─────────────────────────────────────────────────────────────────────────

    def _grabcut_separate(self, frame: np.ndarray,
                          bbox: tuple, pred_pos: tuple):
        """
        Run GrabCut inside bbox to segment foreground blobs.
        If 2+ significant blobs found (multi-object overlap), pick the one
        that best matches the appearance model + Kalman prediction.

        Returns a refined (x,y,w,h) or None if separation failed / not needed.
        """
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        # Clamp + size check
        x = max(1, min(x, fw-2));  y = max(1, min(y, fh-2))
        w = min(w, fw-x-1);        h = min(h, fh-y-1)
        if w < GRABCUT_MIN_SIDE or h < GRABCUT_MIN_SIDE:
            return None

        patch   = frame[y:y+h, x:x+w].copy()
        mask    = np.zeros(patch.shape[:2], dtype=np.uint8)
        bgd_mdl = np.zeros((1, 65), np.float64)
        fgd_mdl = np.zeros((1, 65), np.float64)

        # Init rectangle: small inner margin
        mg   = max(2, min(w, h) // 10)
        rect = (mg, mg, w - 2*mg, h - 2*mg)

        try:
            cv2.grabCut(patch, mask, rect,
                        bgd_mdl, fgd_mdl,
                        GRABCUT_ITERS, cv2.GC_INIT_WITH_RECT)
        except Exception as e:
            print(f"[GRABCUT] Error: {e}")
            return None

        # Binary foreground mask
        fg = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        # Morphological clean-up (remove tiny specks)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg     = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel)
        fg     = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        min_area = w * h * GRABCUT_BLOB_MIN_RATIO
        blobs    = [c for c in contours if cv2.contourArea(c) > min_area]

        if len(blobs) < 2:
            # Only one foreground region — no separation needed
            return None

        print(f"[GRABCUT] {len(blobs)} blobs in bbox. Selecting best match.")

        best_blob, best_score = None, -1.0
        diag = float(np.hypot(w, h)) + 1e-6

        for blob in blobs:
            bx, by, bw, bh = cv2.boundingRect(blob)
            # Guard tiny blobs
            if bw < 6 or bh < 6:
                continue
            bcx = x + bx + bw / 2
            bcy = y + by + bh / 2

            # Appearance score for this blob
            blob_patch = self._get_patch(frame, (x+bx, y+by, bw, bh))
            app  = self.appearance.score(blob_patch)

            # Proximity score (closer to Kalman pred → better)
            if pred_pos:
                dist  = float(np.hypot(bcx - pred_pos[0], bcy - pred_pos[1]))
                prox  = 1.0 - min(dist / diag, 1.0)
            else:
                prox  = 0.5   # no prediction available

            # Weighted combination: appearance 65%, proximity 35%
            combined = app * 0.65 + prox * 0.35

            if combined > best_score:
                best_score = combined
                best_blob  = (x + bx, y + by, bw, bh)

        if best_blob and best_score > 0.25:
            print(f"[GRABCUT] Chose blob at {best_blob}, score={best_score:.3f}")
            return best_blob

        return None

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP — one-frame update
    # ─────────────────────────────────────────────────────────────────────────

    def _step(self, frame: np.ndarray):
        if self.track_state == self.TS_NONE:
            return None

        pred_pos = self.kalman.predict()
        ok, raw  = self.tracker.update(frame)
        ov       = self._overlap(raw, frame.shape) if ok else 0.0
        fh, fw   = frame.shape[:2]

        # ── Out-of-frame check (V5) ───────────────────────────────────────
        is_out = (ok and ov < 0.5)
        if pred_pos:
            px, py = pred_pos
            if px < 5 or px > fw-5 or py < 5 or py > fh-5:
                is_out = True
        if is_out:
            self.track_state = self.TS_OUT_FRAME
            self.playing     = False
            return self.bbox

        # ── 2. Appearance check (NEW V6) ─────────────────────────────────
        app_score = 1.0
        if ok and ov > 0.25:
            patch     = self._get_patch(frame, raw)
            app_score = self.appearance.score(patch)
            self.app_score = app_score

        # ── Multi-object suspicion (NEW V6) ──────────────────────────────
        # Triggered by: appearance drop  OR  bbox area spike
        ref_area  = (self.kalman_bbox_size[0] * self.kalman_bbox_size[1]
                     if self.kalman_bbox_size else 1)
        cur_area  = raw[2] * raw[3] if ok else 0
        area_ratio = cur_area / max(ref_area, 1)

        multi_suspected = ok and (
            app_score < APPEARANCE_THRESHOLD or
            area_ratio > APPEARANCE_AREA_SPIKE
        )

        if multi_suspected:
            reason = ("appearance drop" if app_score < APPEARANCE_THRESHOLD
                      else "area spike")
            print(f"[MULTI-OBJ] Suspected ({reason}): "
                  f"app={app_score:.2f}  area_ratio={area_ratio:.2f}")

            # 3. GrabCut separation (NEW V6)
            refined = self._grabcut_separate(frame, raw, pred_pos)
            if refined:
                self.multi_obj_flash_t = time.time()
                self._init_tracking(frame, refined)
                return refined

            # GrabCut couldn't help → coast on Kalman
            self.lost_count += 1
            if pred_pos and self.kalman_bbox_size:
                pw, ph = self.kalman_bbox_size
                k_box  = (int(pred_pos[0]-pw/2), int(pred_pos[1]-ph/2),
                          int(pw), int(ph))
                if self.trail:
                    self.trail.append(pred_pos)
                self.track_state = self.TS_OCCLUDED
                return k_box
            self.track_state = self.TS_LOST
            return self.bbox

        # ── Kick / trajectory anomaly detection (V5) ─────────────────────
        anomaly_detected = False
        is_kick_event    = False

        if ok and ov > 0.1 and self.track_state in [self.TS_TRACKING, self.TS_COASTING]:
            cx, cy = self._center(raw)
            if pred_pos and self.kalman_bbox_size:
                px, py = pred_pos
                w_, h_ = self.kalman_bbox_size
                dist_pred = float(np.hypot(cx-px, cy-py))
                max_pred  = max(w_, h_) * TRAJECTORY_TOLERANCE
                if dist_pred > max_pred:
                    last_cx, last_cy = self.trail[-1] if self.trail else (cx, cy)
                    dist_last = float(np.hypot(cx-last_cx, cy-last_cy))
                    if dist_last < max(w_, h_) * MAX_KICK_DISTANCE:
                        is_kick_event = True
                        print(f"[KICK] Sharp snap ({dist_pred:.1f}px off-path)")
                    else:
                        anomaly_detected = True
                        print(f"[ANOMALY] Teleport {dist_pred:.1f}px → coast")

        # ── A: Good visual tracking ───────────────────────────────────────
        if ok and not anomaly_detected:
            self.bbox = raw
            cx, cy    = self._center(raw)

            if is_kick_event:
                vx, vy = self.kalman.get_velocity()
                # Re-init Kalman with kick direction hint
                self.kalman.init(cx, cy, vx * 0.6, vy * 0.6)
                self.kick_flash_timer = time.time()
            else:
                self.kalman.correct(cx, cy)

            self.kalman_bbox_size = (raw[2], raw[3])
            self.lost_count       = 0

            # Template update (grass-gated, V5)
            if (self.frame_idx - self.template_fidx) >= TEMPLATE_UPDATE_INT:
                self._save_template(frame, raw)

            # Appearance model soft-update (V6)
            patch = self._get_patch(frame, raw)
            self.appearance.soft_update(patch, app_score)

            self.track_state = self.TS_TRACKING
            self.trail.append((cx, cy))
            return raw

        # ── C/D: Coast on Kalman + try re-detection (V5) ─────────────────
        self.lost_count += 1
        k_box = None
        if pred_pos and self.kalman_bbox_size:
            pw, ph = self.kalman_bbox_size
            k_box  = (int(pred_pos[0]-pw/2), int(pred_pos[1]-ph/2),
                      int(pw), int(ph))
            if self.trail:
                self.trail.append(pred_pos)

        found = self._redetect(frame, pred_pos)
        if found:
            rb, conf = found
            print(f"[RECOVERED] conf={conf:.2f}")
            self._init_tracking(frame, rb)
            return rb

        if k_box:
            self.track_state = self.TS_COASTING
            return k_box

        self.track_state = self.TS_LOST
        return self.bbox

    # ─────────────────────────────────────────────────────────────────────────
    #  DRAWING
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_bbox(self, frame: np.ndarray, bbox: tuple, state: str):
        if not bbox:
            return
        x, y, w, h = [int(v) for v in bbox]
        now = time.time()

        if now - self.kick_flash_timer < 0.5:
            color = CLR_KICK
        elif now - self.multi_obj_flash_t < 0.7:
            color = CLR_MULTIOBJ
        else:
            color = {
                self.TS_TRACKING:  CLR_OK,
                self.TS_COASTING:  CLR_COAST,
                self.TS_OCCLUDED:  CLR_WARN,
                self.TS_OUT_FRAME: CLR_LOST,
                self.TS_LOST:      CLR_LOST,
            }.get(state, CLR_SELECT)

        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        c = max(6, min(w,h)//4)
        for (px_, py_), (dx, dy) in [
            ((x,   y),   ( 1,  1)), ((x+w, y),   (-1,  1)),
            ((x,   y+h), ( 1, -1)), ((x+w, y+h), (-1, -1)),
        ]:
            cv2.line(frame, (px_,py_), (px_+dx*c, py_), color, 3)
            cv2.line(frame, (px_,py_), (px_, py_+dy*c), color, 3)
        cv2.drawMarker(frame, (x+w//2, y+h//2),
                       color, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)

    def _draw_trail(self, frame: np.ndarray):
        pts = list(self.trail)
        for i in range(1, len(pts)):
            a  = i / len(pts)
            c  = tuple(int(v*a) for v in CLR_TRAIL)
            cv2.line(frame,
                     (int(pts[i-1][0]), int(pts[i-1][1])),
                     (int(pts[i][0]),   int(pts[i][1])),
                     c, max(1, int(2*a)), cv2.LINE_AA)

    def _draw_pip(self, frame: np.ndarray, bbox: tuple):
        if not bbox:
            return
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        px_, py_ = int(w*0.4), int(h*0.4)
        x1, y1  = max(0, x-px_), max(0, y-py_)
        x2, y2  = min(fw, x+w+px_), min(fh, y+h+py_)
        if x2<=x1 or y2<=y1:
            return
        roi    = frame[y1:y2, x1:x2].copy()
        pip_w  = max(120, int(fw*0.28))
        aspect = (y2-y1) / max(x2-x1, 1)
        pip_h  = max(80, min(int(pip_w*aspect), int(fh*0.35)))
        zoomed = cv2.resize(roi, (pip_w, pip_h))
        m      = 10
        tx, ty = fw-pip_w-m, m
        ov     = frame.copy()
        cv2.rectangle(ov, (tx-4,ty-22), (tx+pip_w+4,ty+pip_h+4), (15,15,15), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        frame[ty:ty+pip_h, tx:tx+pip_w] = zoomed
        pip_col = CLR_FROZEN if self.template_frozen else CLR_PIP
        cv2.rectangle(frame, (tx-2,ty-2), (tx+pip_w+2,ty+pip_h+2), pip_col, 2)
        cv2.putText(frame, "TARGET VIEW", (tx, ty-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, pip_col, 1, cv2.LINE_AA)

    def _draw_roi_overlay(self, frame: np.ndarray):
        if not (self.roi_pt1 and self.roi_pt2):
            return
        x1 = min(self.roi_pt1[0], self.roi_pt2[0])
        y1 = min(self.roi_pt1[1], self.roi_pt2[1])
        x2 = max(self.roi_pt1[0], self.roi_pt2[0])
        y2 = max(self.roi_pt1[1], self.roi_pt2[1])
        cv2.rectangle(frame, (x1,y1), (x2,y2), CLR_SELECT, 2)
        ov = frame.copy()
        cv2.rectangle(ov, (x1,y1), (x2,y2), CLR_SELECT, -1)
        cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)

    def _draw_appearance_bar(self, frame: np.ndarray):
        """Small horizontal bar showing live appearance similarity score."""
        if self.track_state in (self.TS_NONE, self.TS_OUT_FRAME, self.TS_LOST):
            return
        fh, fw = frame.shape[:2]
        bar_x, bar_y = 8, 95
        bar_w, bar_h = 150, 10
        score = max(0.0, min(1.0, self.app_score))

        # Background
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x+bar_w, bar_y+bar_h), (40,40,40), -1)
        # Fill (green→orange→red based on score)
        fill = int(bar_w * score)
        r = int(255 * (1 - score))
        g = int(255 * score)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x+fill, bar_y+bar_h), (0, g, r), -1)
        # Border
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x+bar_w, bar_y+bar_h), (120,120,120), 1)
        # Label
        lbl = f"Appear: {score:.2f}"
        col = (0,200,80) if score >= APPEARANCE_THRESHOLD else CLR_MULTIOBJ
        cv2.putText(frame, lbl, (bar_x, bar_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray):
        fh, fw = frame.shape[:2]
        now = time.time()

        # Top-left info panel
        cv2.rectangle(frame, (0,0), (200, 92), (0,0,0), -1)
        cv2.putText(frame, f"FPS: {self.fps_value:5.1f}", (8,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, CLR_OK, 2, cv2.LINE_AA)

        tmpl_txt = "TMPL: FROZEN"   if self.template_frozen else "TMPL: LEARNING"
        tmpl_col = CLR_FROZEN       if self.template_frozen else CLR_OK
        cv2.putText(frame, tmpl_txt, (8,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, tmpl_col, 1, cv2.LINE_AA)

        pb_txt = "PLAYING" if self.playing else "PAUSED"
        pb_col = CLR_OK   if self.playing  else (100,100,255)
        cv2.putText(frame, f"STAT: {pb_txt}", (8,72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, pb_col, 1, cv2.LINE_AA)

        # Appearance bar (V6)
        self._draw_appearance_bar(frame)

        # Bottom hint
        hint = "ENTER:start  SPACE:pause  R:retarget  P:PiP  Q:quit"
        cv2.putText(frame, hint, (8, fh-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150,150,150), 1, cv2.LINE_AA)

        # Selection prompt
        if self.track_state == self.TS_NONE:
            for i, (msg, sc, col) in enumerate([
                ("Draw a box around the target", 0.78, CLR_SELECT),
                ("then press ENTER to start",    0.60, (170,170,170)),
            ]):
                (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
                sx = fw//2 - tw//2;  sy = fh//2 - 30 + i*40
                cv2.rectangle(frame, (sx-10,sy-th-4), (sx+tw+10,sy+6), (0,0,0), -1)
                cv2.putText(frame, msg, (sx,sy),
                            cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)
            return

        # Kick flash
        if now - self.kick_flash_timer < 0.5:
            cv2.putText(frame, "[ KICK ! ]", (fw//2-60, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, CLR_KICK, 3, cv2.LINE_AA)

        # Multi-object / GrabCut flash
        if now - self.multi_obj_flash_t < 0.7:
            msg = "[ GRABCUT: re-locked ]"
            (tw,_),_ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
            cv2.putText(frame, msg, (fw//2-tw//2, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, CLR_MULTIOBJ, 2, cv2.LINE_AA)

        # State badge
        state_cfg = {
            self.TS_TRACKING: ("[  TRACKING  ]",           CLR_OK,    False),
            self.TS_COASTING: ("[ COASTING (Physics) ]",   CLR_COAST, True),
            self.TS_OCCLUDED: ("[ OCCLUDED - Kalman ]",    CLR_WARN,  True),
        }
        if self.track_state in state_cfg:
            label, color, blink = state_cfg[self.track_state]
            if not (blink and int(now*3) % 2 == 0):
                (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
                sx = fw//2-tw//2;  sy = fh-26
                cv2.rectangle(frame, (sx-8,sy-th-4), (sx+tw+8,sy+6), (0,0,0), -1)
                cv2.putText(frame, label, (sx,sy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)

        if self.track_state in (self.TS_LOST, self.TS_OUT_FRAME):
            ov = frame.copy()
            cv2.rectangle(ov, (0,0), (fw,fh), (0,0,80), -1)
            cv2.addWeighted(ov, 0.20, frame, 0.80, 0, frame)
            msg1 = ("STOPPED: Target Out Of Frame"
                    if self.track_state == self.TS_OUT_FRAME
                    else "STOPPED: Target Lost")
            msg2 = "Press R to select a new target"
            for i, (msg, sc, col) in enumerate([(msg1,0.8,CLR_WARN),(msg2,0.6,CLR_LOST)]):
                (mw,_),_ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
                cv2.putText(frame, msg, (fw//2-mw//2, fh//2-20+i*40),
                            cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        print("=" * 60)
        print("  SPORTS TRACKER V6")
        print("  New: 6-state Kalman | Appearance Model | GrabCut Separation")
        print(f"  Video: {self.video_path}")
        print("=" * 60)

        while True:
            advance = (self.playing and
                       self.track_state not in
                       [self.TS_NONE, self.TS_OUT_FRAME, self.TS_LOST])

            if advance:
                ret, raw = self.cap.read()
                if not ret:
                    self._show_end_screen(); break
                self.cur_frame  = self._resize(raw)
                self.frame_idx += 1

                self._fps_cnt += 1
                now = time.time()
                if now - self._fps_t0 >= 0.4:
                    self.fps_value = self._fps_cnt / (now - self._fps_t0)
                    self._fps_cnt  = 0
                    self._fps_t0   = now

                disp_bbox = self._step(self.cur_frame)

                if self.lost_count > MAX_LOST_FRAMES:
                    print("[STOP] Target lost too long.")
                    self.playing     = False
                    self.track_state = self.TS_LOST
            else:
                disp_bbox      = self.bbox
                self._fps_t0   = time.time()
                self._fps_cnt  = 0

            display = self.cur_frame.copy()
            self._draw_trail(display)
            if self.pip_enabled and disp_bbox and self.track_state == self.TS_TRACKING:
                self._draw_pip(display, disp_bbox)
            if disp_bbox and self.track_state != self.TS_NONE:
                self._draw_bbox(display, disp_bbox, self.track_state)
            self._draw_roi_overlay(display)
            self._draw_hud(display)

            cv2.imshow(self.WIN, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == 13:   # ENTER
                if self.track_state in [self.TS_TRACKING, self.TS_COASTING, self.TS_OCCLUDED]:
                    self.playing = True; self._fps_t0 = time.time(); self._fps_cnt = 0
                elif self.track_state in [self.TS_LOST, self.TS_OUT_FRAME]:
                    print("[!] Cannot resume — target lost. Press R to re-select.")
                else:
                    print("[!] Select a target first.")
            elif key == ord(' '):
                if self.playing:
                    self.playing = False
                elif self.track_state in [self.TS_TRACKING, self.TS_COASTING, self.TS_OCCLUDED]:
                    self.playing = True; self._fps_t0 = time.time(); self._fps_cnt = 0
            elif key == ord('r'):
                self.playing = False
                self.track_state = self.TS_NONE
                self.tracker  = None;  self.bbox = None
                self.trail.clear();    self.lost_count = 0
                self.roi_pt1  = None;  self.roi_pt2 = None
                self.template_frozen = False
                self.app_score = 1.0
                print("[RESET] Draw a new target, then press ENTER")
            elif key == ord('p'):
                self.pip_enabled = not self.pip_enabled

        self.cap.release()
        cv2.destroyAllWindows()
        print("[EXIT] Done.")

    def _show_end_screen(self):
        if self.cur_frame is None: return
        frame = self.cur_frame.copy()
        msg = "VIDEO ENDED"
        (tw,_),_ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.putText(frame, msg,
                    (frame.shape[1]//2-tw//2, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,200), 3, cv2.LINE_AA)
        cv2.imshow(self.WIN, frame)
        cv2.waitKey(3000)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python sports_tracker_v6.py <video_file>")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"[ERROR] File not found: '{path}'")
        sys.exit(1)
    try:
        SportsTracker(path).run()
    except IOError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
