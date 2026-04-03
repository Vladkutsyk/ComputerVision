"""
SPORTS VIDEO TRACKER V8 — Semantic Attributes & Adaptive CSRT
=============================================================
Built on V7 (Kalman + GrabCut + Appearance Model).

NEW in V8:
  1. Semantic Object Types: Prompts user to define target as Ball (B) or Player (P).
  2. Adaptive CSRT Params: Modifies padding, template size, and scale params 
     based on the object type (tight for balls, wide for players).
  3. Dominant Color Protection (Anti-Team Swap): Automatically extracts the player's 
     jersey color (ignoring the green pitch) and strictly validates it every frame.
     If the color shifts heavily (e.g., crossed paths with opposing team), the
     visual track is rejected.

Controls:
    Draw ROI  — hold LMB + drag
    B / P     — define target as Ball (B) or Player (P) after drawing
    ENTER     — start / resume
    SPACE     — pause / resume
    R         — re-select target
    Q / ESC   — quit
"""

import cv2
import numpy as np
import sys
import os
import time
from collections import deque

# ── General tracking ─────────────────────────────────────────────────────────
MAX_LOST_FRAMES     = 45
REDETECT_THRESHOLD  = 0.50
TEMPLATE_UPDATE_INT = 8
MAX_FRAME_WIDTH     = 1280

# ── Trajectory / kick detection ──────────────────────────────────────────────
TRAJECTORY_TOLERANCE = 1.15
MAX_KICK_DISTANCE    = 4.5

# ── Appearance & Validation ──────────────────────────────────────────────────
APPEARANCE_BINS          = 32
APPEARANCE_THRESHOLD     = 0.42
APPEARANCE_UPDATE_THRESH = 0.68
APPEARANCE_AREA_SPIKE    = 1.80

HUE_SHIFT_TOLERANCE      = 20    # Max allowed shift in dominant Hue (0-180)

# ── GrabCut ──────────────────────────────────────────────────────────────────
GRABCUT_ITERS          = 3
GRABCUT_BLOB_MIN_RATIO = 0.10
GRABCUT_MIN_SIDE       = 24

# ── Colours ──────────────────────────────────────────────────────────────────
CLR_OK       = (0,   220,  80)
CLR_WARN     = (0,   165, 255)
CLR_COAST    = (255, 105, 180)
CLR_LOST     = (50,   50, 220)
CLR_SELECT   = (0,   220, 255)
CLR_PIP      = (0,   220, 220)
CLR_TRAIL    = (0,   180, 255)
CLR_FROZEN   = (255,  50,  50)
CLR_KICK     = (255, 255,   0)
CLR_MULTIOBJ = (0,   100, 255)


# ─────────────────────────────────────────────────────────────────────────────
#  KALMAN PREDICTOR (6-State)
# ─────────────────────────────────────────────────────────────────────────────

class KalmanPredictor:
    MAX_SPEED = 130 

    def __init__(self):
        dt = 1.0
        kf = cv2.KalmanFilter(6, 2)
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0,  0.5*dt*dt, 0        ],
            [0, 1, 0,  dt, 0,         0.5*dt*dt],
            [0, 0, 1,  0,  dt,        0        ],
            [0, 0, 0,  1,  0,         dt       ],
            [0, 0, 0,  0,  1,         0        ],
            [0, 0, 0,  0,  0,         1        ],
        ], dtype=np.float32)

        kf.measurementMatrix = np.zeros((2, 6), dtype=np.float32)
        kf.measurementMatrix[0, 0] = 1.0
        kf.measurementMatrix[1, 1] = 1.0

        kf.processNoiseCov = np.diag([1e-3, 1e-3, 5e-2, 5e-2, 5e-1, 5e-1]).astype(np.float32)
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
        if not self.initialized: return None
        p = self.kf.predict()
        return float(p[0, 0]), float(p[1, 0])

    def correct(self, cx: float, cy: float):
        if not self.initialized:
            self.init(cx, cy)
            return
        self.kf.correct(np.array([[cx], [cy]], dtype=np.float32))
        s     = self.kf.statePost
        speed = float(np.hypot(s[2, 0], s[3, 0]))
        if speed > self.MAX_SPEED:
            ratio    = self.MAX_SPEED / speed
            s[2, 0] *= ratio
            s[3, 0] *= ratio
            self.kf.statePost = s

    def get_velocity(self):
        if not self.initialized: return 0., 0.
        s = self.kf.statePost
        return float(s[2, 0]), float(s[3, 0])


# ─────────────────────────────────────────────────────────────────────────────
#  APPEARANCE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class AppearanceModel:
    def __init__(self, bins: int = APPEARANCE_BINS):
        self.bins = bins
        self.hist = None

    def _make_hist(self, patch: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [self.bins, self.bins], [0, 180, 0, 256])
        cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
        return h

    def build(self, patch: np.ndarray):
        if patch.size == 0: return
        self.hist = self._make_hist(patch)

    def score(self, patch: np.ndarray) -> float:
        if self.hist is None or patch.size == 0: return 1.0 
        h = self._make_hist(patch)
        corr = cv2.compareHist(self.hist, h, cv2.HISTCMP_CORREL)
        return max(0.0, float(corr))

    def soft_update(self, patch: np.ndarray, score: float):
        if score >= APPEARANCE_UPDATE_THRESH:
            self.build(patch)


# ─────────────────────────────────────────────────────────────────────────────
#  SPORTS TRACKER V8
# ─────────────────────────────────────────────────────────────────────────────

class SportsTracker:
    TS_NONE         = "NONE"
    TS_TYPE_SELECT  = "TYPE_SELECT"  # New state: waiting for B or P
    TS_TRACKING     = "TRACKING"
    TS_OCCLUDED     = "OCCLUDED"
    TS_COASTING     = "COASTING"
    TS_OUT_FRAME    = "OUT_FRAME"
    TS_LOST         = "LOST"

    OBJ_BALL   = "BALL"
    OBJ_PLAYER = "PLAYER"

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened(): raise IOError(f"Cannot open video: {video_path}")

        ok, first = self.cap.read()
        if not ok: raise IOError("Cannot read first frame")
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

        self.obj_type         = None
        self.dominant_hue     = None

        self.template         = None
        self.template_fidx    = 0
        self.template_frozen  = False
        self.kick_flash_timer = 0.0

        self.appearance          = AppearanceModel()
        self.app_score           = 1.0   
        self.multi_obj_flash_t   = 0.0  

        self.trail = deque(maxlen=60)
        self.pip_enabled = True

        self.drawing = False
        self.roi_pt1 = None
        self.roi_pt2 = None

        self.fps_value = 0.0
        self._fps_t0   = time.time()
        self._fps_cnt  = 0

        self.WIN = "Sports Tracker V8"
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        h, w = self.cur_frame.shape[:2]
        cv2.resizeWindow(self.WIN, min(w, 1280), min(h, 720))
        cv2.setMouseCallback(self.WIN, self._mouse_cb)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w > MAX_FRAME_WIDTH:
            return cv2.resize(frame, (MAX_FRAME_WIDTH, int(h * MAX_FRAME_WIDTH / w)))
        return frame

    def _win_to_frame(self, wx: int, wy: int):
        try:
            _, _, dw, dh = cv2.getWindowImageRect(self.WIN)
            if dw > 0 and dh > 0 and self.cur_frame is not None:
                fh, fw = self.cur_frame.shape[:2]
                return int(wx * fw / dw), int(wy * fh / dh)
        except Exception:
            pass
        return wx, wy

    def _mouse_cb(self, event, x, y, flags, param):
        if self.track_state == self.TS_TYPE_SELECT: return # Block drawing while prompting
        
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
                    self.bbox = (x1, y1, x2-x1, y2-y1)
                    self.track_state = self.TS_TYPE_SELECT # Wait for attribute input

    # ── V8: Adaptive CSRT & Attributes ────────────────────────────────────────

    def _get_adaptive_csrt_params(self):
        """Modifies CSRT parameters based on semantic object type."""
        params = cv2.TrackerCSRT_Params()
        
        if self.obj_type == self.OBJ_BALL:
            # Ball: Fast scale changes, tight padding to avoid learning feet
            params.padding = 1.5 
            params.template_size = 100
            params.scale_sigma_factor = 0.50 
            params.use_color_names = False # Balls often motion-blur into gray/white streaks
            params.psr_threshold = 0.035
        else:
            # Player: Needs context, relies on color/HOG
            params.padding = 3.0
            params.template_size = 200
            params.scale_sigma_factor = 0.25
            params.use_color_names = True
            params.psr_threshold = 0.035

        return params

    def _extract_dominant_hue(self, patch: np.ndarray):
        """Extracts the dominant non-green hue from a patch (Jersey color)."""
        if patch.size == 0: return None
        
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        
        # Mask out the green pitch
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_fg = cv2.bitwise_not(mask_green)
        
        # Calculate Hue histogram only for foreground
        hist = cv2.calcHist([hsv], [0], mask_fg, [180], [0, 180])
        
        # Find the peak hue
        _, max_val, _, max_loc = cv2.minMaxLoc(hist)
        
        if max_val > 5: # Ensure we actually have meaningful foreground pixels
            return max_loc[1]
        return None

    def _hue_distance(self, h1, h2):
        """Circular distance for Hue (0-180)."""
        if h1 is None or h2 is None: return 0
        diff = abs(h1 - h2)
        return min(diff, 180 - diff)

    def _init_tracking(self, frame: np.ndarray):
        x, y, w, h = [int(v) for v in self.bbox]
        fh, fw = frame.shape[:2]
        x = max(0, min(x, fw - 2));  y = max(0, min(y, fh - 2))
        w = max(10, min(w, fw - x)); h = max(10, min(h, fh - y))
        self.bbox = (x, y, w, h)

        # 1. Setup adaptive tracker
        params = self._get_adaptive_csrt_params()
        self.tracker = cv2.TrackerCSRT_create(params)
        self.tracker.init(frame, self.bbox)
        
        self.lost_count       = 0
        self.track_state      = self.TS_TRACKING
        self.kalman_bbox_size = (w, h)

        self.kalman = KalmanPredictor()
        self.kalman.init(x + w / 2, y + h / 2)

        self._save_template(frame, self.bbox, force=True)

        patch = self._get_patch(frame, self.bbox)
        if patch.size > 0:
            self.appearance.build(patch)
            
            # 2. Extract semantic color attribute if player
            if self.obj_type == self.OBJ_PLAYER:
                self.dominant_hue = self._extract_dominant_hue(patch)
                print(f"[ATTRIBUTES] Player target. Dominant Hue locked at: {self.dominant_hue}")

        self.app_score = 1.0
        self.trail.clear()
        self.kick_flash_timer  = 0.0
        self.multi_obj_flash_t = 0.0
        print(f"[TRACKER] Init ({self.obj_type}) bbox={self.bbox}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_open_grass(self, frame: np.ndarray, bbox: tuple) -> bool:
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        mx, my = max(10, w), max(10, h)
        x1, y1 = max(0, x-mx), max(0, y-my)
        x2, y2 = min(fw, x+w+mx), min(fh, y+h+my)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return False
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
        if ix2<=ix1 or iy2<=iy1: return 0.0
        return (ix2-ix1)*(iy2-iy1) / max(w*h, 1)

    def _get_patch(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x1, y1 = max(0,x), max(0,y)
        x2, y2 = min(fw,x+w), min(fh,y+h)
        if x2 <= x1 or y2 <= y1: return np.zeros((1,1,3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]

    def _redetect(self, frame: np.ndarray, pred_pos: tuple):
        if self.template is None or pred_pos is None: return None
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
            if nh >= search.shape[0] or nw >= search.shape[1]: continue
            rt  = cv2.resize(g_tmpl, (nw, nh))
            res = cv2.matchTemplate(search, rt, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if mv > best_val:
                best_val = mv
                best_box = (ml[0]+offset[0], ml[1]+offset[1], nw, nh)
        if best_val >= REDETECT_THRESHOLD and best_box:
            return best_box, best_val
        return None

    def _grabcut_separate(self, frame: np.ndarray, bbox: tuple, pred_pos: tuple):
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x = max(1, min(x, fw-2));  y = max(1, min(y, fh-2))
        w = min(w, fw-x-1);        h = min(h, fh-y-1)
        if w < GRABCUT_MIN_SIDE or h < GRABCUT_MIN_SIDE: return None

        patch   = frame[y:y+h, x:x+w].copy()
        mask    = np.zeros(patch.shape[:2], dtype=np.uint8)
        bgd_mdl = np.zeros((1, 65), np.float64)
        fgd_mdl = np.zeros((1, 65), np.float64)

        mg   = max(2, min(w, h) // 10)
        rect = (mg, mg, w - 2*mg, h - 2*mg)

        try:
            cv2.grabCut(patch, mask, rect, bgd_mdl, fgd_mdl, GRABCUT_ITERS, cv2.GC_INIT_WITH_RECT)
        except Exception:
            return None

        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg     = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel)
        fg     = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = w * h * GRABCUT_BLOB_MIN_RATIO
        blobs    = [c for c in contours if cv2.contourArea(c) > min_area]

        if len(blobs) < 2: return None

        best_blob, best_score = None, -1.0
        diag = float(np.hypot(w, h)) + 1e-6

        for blob in blobs:
            bx, by, bw, bh = cv2.boundingRect(blob)
            if bw < 6 or bh < 6: continue
            bcx = x + bx + bw / 2
            bcy = y + by + bh / 2

            blob_patch = self._get_patch(frame, (x+bx, y+by, bw, bh))
            app  = self.appearance.score(blob_patch)

            # V8: Also strictly check jersey color during grabcut selection
            color_penalty = 0.0
            if self.obj_type == self.OBJ_PLAYER and self.dominant_hue is not None:
                cur_hue = self._extract_dominant_hue(blob_patch)
                if self._hue_distance(self.dominant_hue, cur_hue) > HUE_SHIFT_TOLERANCE:
                    color_penalty = 0.5 # Severely penalize blob if it's the wrong team color

            prox  = 1.0 - min(float(np.hypot(bcx - pred_pos[0], bcy - pred_pos[1])) / diag, 1.0) if pred_pos else 0.5
            combined = (app * 0.65 + prox * 0.35) - color_penalty

            if combined > best_score:
                best_score = combined
                best_blob  = (x + bx, y + by, bw, bh)

        if best_blob and best_score > 0.25:
            return best_blob
        return None

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 
    # ─────────────────────────────────────────────────────────────────────────

    def _step(self, frame: np.ndarray):
        if self.track_state in [self.TS_NONE, self.TS_TYPE_SELECT]: return None

        pred_pos = self.kalman.predict()
        ok, raw  = self.tracker.update(frame)
        ov       = self._overlap(raw, frame.shape) if ok else 0.0
        fh, fw   = frame.shape[:2]

        is_out = (ok and ov < 0.5)
        if pred_pos:
            px, py = pred_pos
            if px < 5 or px > fw-5 or py < 5 or py > fh-5: is_out = True
        if is_out:
            self.track_state = self.TS_OUT_FRAME
            self.playing     = False
            return self.bbox

        app_score = 1.0
        color_swap_detected = False

        if ok and ov > 0.25:
            patch     = self._get_patch(frame, raw)
            app_score = self.appearance.score(patch)
            self.app_score = app_score
            
            # V8: Dominant Color Protection Check
            if self.obj_type == self.OBJ_PLAYER and self.dominant_hue is not None:
                cur_hue = self._extract_dominant_hue(patch)
                if self._hue_distance(self.dominant_hue, cur_hue) > HUE_SHIFT_TOLERANCE:
                    color_swap_detected = True
                    print(f"[COLOR ALERT] Wrong team! Expected H:{self.dominant_hue}, Got H:{cur_hue}")

        ref_area  = (self.kalman_bbox_size[0] * self.kalman_bbox_size[1] if self.kalman_bbox_size else 1)
        cur_area  = raw[2] * raw[3] if ok else 0
        area_ratio = cur_area / max(ref_area, 1)

        multi_suspected = ok and (app_score < APPEARANCE_THRESHOLD or area_ratio > APPEARANCE_AREA_SPIKE or color_swap_detected)

        if multi_suspected:
            refined = self._grabcut_separate(frame, raw, pred_pos)
            if refined:
                self.multi_obj_flash_t = time.time()
                # Re-init purely visually, keep our semantic attributes
                self.bbox = refined
                self.tracker = cv2.TrackerCSRT_create(self._get_adaptive_csrt_params())
                self.tracker.init(frame, self.bbox)
                return refined

            self.lost_count += 1
            if pred_pos and self.kalman_bbox_size:
                pw, ph = self.kalman_bbox_size
                k_box  = (int(pred_pos[0]-pw/2), int(pred_pos[1]-ph/2), int(pw), int(ph))
                if self.trail: self.trail.append(pred_pos)
                self.track_state = self.TS_OCCLUDED
                return k_box
            self.track_state = self.TS_LOST
            return self.bbox

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
                    
                    # Players don't "kick" themselves, so limit teleport acceptance
                    kick_limit = MAX_KICK_DISTANCE if self.obj_type == self.OBJ_BALL else 2.0
                    
                    if dist_last < max(w_, h_) * kick_limit:
                        is_kick_event = True
                    else:
                        anomaly_detected = True

        if ok and not anomaly_detected:
            self.bbox = raw
            cx, cy    = self._center(raw)

            if is_kick_event:
                vx, vy = self.kalman.get_velocity()
                self.kalman.init(cx, cy, vx * 0.6, vy * 0.6)
                if self.obj_type == self.OBJ_BALL: self.kick_flash_timer = time.time()
            else:
                self.kalman.correct(cx, cy)

            self.kalman_bbox_size = (raw[2], raw[3])
            self.lost_count       = 0

            if (self.frame_idx - self.template_fidx) >= TEMPLATE_UPDATE_INT:
                self._save_template(frame, raw)

            patch = self._get_patch(frame, raw)
            self.appearance.soft_update(patch, app_score)

            self.track_state = self.TS_TRACKING
            self.trail.append((cx, cy))
            return raw

        self.lost_count += 1
        k_box = None
        if pred_pos and self.kalman_bbox_size:
            pw, ph = self.kalman_bbox_size
            k_box  = (int(pred_pos[0]-pw/2), int(pred_pos[1]-ph/2), int(pw), int(ph))
            if self.trail: self.trail.append(pred_pos)

        found = self._redetect(frame, pred_pos)
        if found:
            rb, conf = found
            # Re-init visually
            self.bbox = rb
            self.tracker = cv2.TrackerCSRT_create(self._get_adaptive_csrt_params())
            self.tracker.init(frame, self.bbox)
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
        if not bbox: return
        x, y, w, h = [int(v) for v in bbox]
        now = time.time()

        if now - self.kick_flash_timer < 0.5: color = CLR_KICK
        elif now - self.multi_obj_flash_t < 0.7: color = CLR_MULTIOBJ
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
        for (px_, py_), (dx, dy) in [((x,y),(1,1)), ((x+w,y),(-1,1)), ((x,y+h),(1,-1)), ((x+w,y+h),(-1,-1))]:
            cv2.line(frame, (px_,py_), (px_+dx*c, py_), color, 3)
            cv2.line(frame, (px_,py_), (px_, py_+dy*c), color, 3)

    def _draw_trail(self, frame: np.ndarray):
        pts = list(self.trail)
        for i in range(1, len(pts)):
            a = i / len(pts)
            c = tuple(int(v*a) for v in CLR_TRAIL)
            cv2.line(frame, (int(pts[i-1][0]), int(pts[i-1][1])), (int(pts[i][0]), int(pts[i][1])), c, max(1, int(2*a)), cv2.LINE_AA)

    def _draw_appearance_bar(self, frame: np.ndarray):
        if self.track_state in (self.TS_NONE, self.TS_TYPE_SELECT, self.TS_OUT_FRAME, self.TS_LOST): return
        bar_x, bar_y, bar_w, bar_h = 8, 95, 150, 10
        score = max(0.0, min(1.0, self.app_score))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (40,40,40), -1)
        fill = int(bar_w * score)
        r, g = int(255 * (1 - score)), int(255 * score)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill, bar_y+bar_h), (0, g, r), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (120,120,120), 1)
        col = (0,200,80) if score >= APPEARANCE_THRESHOLD else CLR_MULTIOBJ
        cv2.putText(frame, f"Appear: {score:.2f}", (bar_x, bar_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray):
        fh, fw = frame.shape[:2]
        now = time.time()

        cv2.rectangle(frame, (0,0), (200, 92), (0,0,0), -1)
        cv2.putText(frame, f"FPS: {self.fps_value:5.1f}", (8,25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, CLR_OK, 2, cv2.LINE_AA)
        
        type_str = f"TYPE: {self.obj_type}" if self.obj_type else "TYPE: UNSET"
        cv2.putText(frame, type_str, (8,50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_SELECT, 1, cv2.LINE_AA)
        
        pb_txt = "PLAYING" if self.playing else "PAUSED"
        cv2.putText(frame, f"STAT: {pb_txt}", (8,72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_OK if self.playing else (100,100,255), 1, cv2.LINE_AA)

        self._draw_appearance_bar(frame)
        cv2.putText(frame, "ENTER:start  SPACE:pause  R:retarget  Q:quit", (8, fh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150,150,150), 1, cv2.LINE_AA)

        if self.track_state == self.TS_NONE:
            msg = "Draw a box around the target"
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)
            cv2.putText(frame, msg, (fw//2 - tw//2, fh//2), cv2.FONT_HERSHEY_SIMPLEX, 0.78, CLR_SELECT, 2, cv2.LINE_AA)
            return

        # V8: Prompt for semantic type
        if self.track_state == self.TS_TYPE_SELECT:
            ov = frame.copy()
            cv2.rectangle(ov, (0,0), (fw,fh), (0,0,0), -1)
            cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
            
            x, y, w, h = self.bbox
            cv2.rectangle(frame, (x,y), (x+w, y+h), CLR_SELECT, 2)
            
            msg1 = "Press 'B' if target is a BALL"
            msg2 = "Press 'P' if target is a PLAYER"
            
            for i, msg in enumerate([msg1, msg2]):
                (mw,_),_ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.putText(frame, msg, (fw//2-mw//2, fh//2 - 20 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CLR_SELECT, 2, cv2.LINE_AA)
            return

        if self.track_state in [self.TS_LOST, self.TS_OUT_FRAME]:
            msg1 = "STOPPED: Target Out Of Frame" if self.track_state == self.TS_OUT_FRAME else "STOPPED: Target Lost"
            msg2 = "Press R to select a new target"
            for i, (msg, sc, col) in enumerate([(msg1,0.8,CLR_WARN),(msg2,0.6,CLR_LOST)]):
                (mw,_),_ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
                cv2.putText(frame, msg, (fw//2-mw//2, fh//2-20+i*40), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        print("=" * 60)
        print("  SPORTS TRACKER V8 (Semantic Attributes)")
        print(f"  Video: {self.video_path}")
        print("=" * 60)

        while True:
            advance = (self.playing and self.track_state not in [self.TS_NONE, self.TS_TYPE_SELECT, self.TS_OUT_FRAME, self.TS_LOST])

            if advance:
                ret, raw = self.cap.read()
                if not ret: break
                self.cur_frame = self._resize(raw)
                self.frame_idx += 1

                self._fps_cnt += 1
                now = time.time()
                if now - self._fps_t0 >= 0.4:
                    self.fps_value = self._fps_cnt / (now - self._fps_t0)
                    self._fps_cnt, self._fps_t0 = 0, now

                disp_bbox = self._step(self.cur_frame)
                if self.lost_count > MAX_LOST_FRAMES: self.playing, self.track_state = False, self.TS_LOST
            else:
                disp_bbox, self._fps_cnt, self._fps_t0 = self.bbox, 0, time.time()

            display = self.cur_frame.copy()
            self._draw_trail(display)
            if disp_bbox and self.track_state not in [self.TS_NONE, self.TS_TYPE_SELECT]:
                self._draw_bbox(display, disp_bbox, self.track_state)
            
            # Draw ROI box dynamically
            if self.drawing and self.roi_pt1 and self.roi_pt2:
                x1, y1 = min(self.roi_pt1[0], self.roi_pt2[0]), min(self.roi_pt1[1], self.roi_pt2[1])
                x2, y2 = max(self.roi_pt1[0], self.roi_pt2[0]), max(self.roi_pt1[1], self.roi_pt2[1])
                cv2.rectangle(display, (x1,y1), (x2,y2), CLR_SELECT, 2)

            self._draw_hud(display)
            cv2.imshow(self.WIN, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27): break
            
            # V8: Semantic Type Input
            elif self.track_state == self.TS_TYPE_SELECT:
                if key in [ord('b'), ord('B')]:
                    self.obj_type = self.OBJ_BALL
                    self._init_tracking(self.cur_frame)
                elif key in [ord('p'), ord('P')]:
                    self.obj_type = self.OBJ_PLAYER
                    self._init_tracking(self.cur_frame)
                elif key == ord('r'):
                    self.track_state = self.TS_NONE
                    
            elif key == 13: # ENTER
                if self.track_state in [self.TS_TRACKING, self.TS_COASTING, self.TS_OCCLUDED]:
                    self.playing = True; self._fps_t0, self._fps_cnt = time.time(), 0
            elif key == ord(' '):
                self.playing = not self.playing if self.track_state in [self.TS_TRACKING, self.TS_COASTING, self.TS_OCCLUDED] else False
            elif key == ord('r'):
                self.playing = False
                self.track_state = self.TS_NONE
                self.tracker, self.bbox, self.obj_type, self.dominant_hue = None, None, None, None
                self.trail.clear(); self.lost_count = 0

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    if len(sys.argv) < 2:
        sys.exit(1)
    try:
        SportsTracker(sys.argv[1]).run()
    except IOError as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()