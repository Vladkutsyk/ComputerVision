"""
SPORTS VIDEO TRACKER  —  CSRT + Kalman + Re-Detection
======================================================
Dependencies:
    pip install opencv-contrib-python numpy

Usage:
    python sports_tracker.py <video_file>
    python sports_tracker.py match.mov

Controls:
    Draw ROI  — hold left mouse button, drag rectangle around target
    ENTER     — start / resume playback
    SPACE     — pause / resume
    R         — re-select target (pauses video, pick new box, press ENTER)
    P         — toggle Picture-in-Picture
    Q / ESC   — quit
"""

import cv2
import numpy as np
import sys
import os
import time
from collections import deque

# ── CSRT tuning for sports (fast motion, occlusions, similar backgrounds) ──
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

MAX_LOST_FRAMES     = 45    # frames before giving up on lost target
REDETECT_THRESHOLD  = 0.50  # template-match confidence threshold
TEMPLATE_UPDATE_INT = 8     # update reference template every N frames
MAX_FRAME_WIDTH     = 1280  # auto-downscale for performance

# BGR colours
CLR_OK     = (0,   220,  80)
CLR_WARN   = (0,   165, 255)
CLR_LOST   = (50,   50, 220)
CLR_SELECT = (0,   220, 255)
CLR_PIP    = (0,   220, 220)
CLR_TRAIL  = (0,   180, 255)


# ─────────────────────────────────────────────────────────────────────────────
#  KALMAN PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

class KalmanPredictor:
    """Constant-velocity Kalman: state=[cx,cy,vx,vy], measurement=[cx,cy]."""

    def __init__(self):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32)
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32)
        kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self.kf          = kf
        self.initialized = False

    def init(self, cx: float, cy: float):
        self.kf.statePre  = np.array([[cx], [cy], [0.], [0.]], dtype=np.float32)
        self.kf.statePost = self.kf.statePre.copy()
        self.initialized  = True

    def predict(self):
        if not self.initialized:
            return None
        p = self.kf.predict()
        # Extract the scalar at row 0, column 0 and row 1, column 0
        return float(p[0, 0]), float(p[1, 0])

    def correct(self, cx: float, cy: float):
        if not self.initialized:
            self.init(cx, cy)
            return
        self.kf.correct(np.array([[cx], [cy]], dtype=np.float32))


# ─────────────────────────────────────────────────────────────────────────────
#  SPORTS TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class SportsTracker:
    TS_NONE      = "NONE"
    TS_TRACKING  = "TRACKING"
    TS_OCCLUDED  = "OCCLUDED"
    TS_OUT_FRAME = "OUT_FRAME"
    TS_LOST      = "LOST"

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        # ── Read first frame, rewind ─────────────────────────────────────────
        ok, first = self.cap.read()
        if not ok:
            raise IOError("Cannot read first frame")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cur_frame = self._resize(first)

        # ── Playback ─────────────────────────────────────────────────────────
        self.playing   = False   # video starts PAUSED
        self.frame_idx = 0

        # ── Tracker state ────────────────────────────────────────────────────
        self.tracker          = None
        self.bbox             = None    # (x, y, w, h) in frame pixels
        self.track_state      = self.TS_NONE
        self.lost_count       = 0
        self.kalman           = KalmanPredictor()
        self.kalman_bbox_size = None
        self.template         = None
        self.template_fidx    = 0

        # ── Trail ────────────────────────────────────────────────────────────
        self.trail = deque(maxlen=60)

        # ── PiP ──────────────────────────────────────────────────────────────
        self.pip_enabled = True

        # ── ROI drawing — coords stored in FRAME space ───────────────────────
        self.drawing = False
        self.roi_pt1 = None
        self.roi_pt2 = None

        # ── FPS ──────────────────────────────────────────────────────────────
        self.fps_value = 0.0
        self._fps_t0   = time.time()
        self._fps_cnt  = 0

        # ── Window ───────────────────────────────────────────────────────────
        self.WIN = "Sports Tracker"
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        h, w = self.cur_frame.shape[:2]
        cv2.resizeWindow(self.WIN, min(w, 1280), min(h, 720))
        cv2.setMouseCallback(self.WIN, self._mouse_cb)

    # ── Resize helper ─────────────────────────────────────────────────────────

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w > MAX_FRAME_WIDTH:
            return cv2.resize(frame,
                              (MAX_FRAME_WIDTH, int(h * MAX_FRAME_WIDTH / w)))
        return frame

    # ── Window ↔ frame coordinate conversion ─────────────────────────────────
    #
    #  cv2.getWindowImageRect(name) returns (x, y, disp_w, disp_h)
    #  where disp_w/disp_h are the pixels the image occupies inside the window.
    #  Mouse event coordinates are in that same display space (0..disp_w, 0..disp_h).
    #  We scale them to the original frame size.

    def _win_to_frame(self, wx: int, wy: int):
        """Window display coords → frame pixel coords."""
        try:
            _, _, dw, dh = cv2.getWindowImageRect(self.WIN)
            if dw > 0 and dh > 0 and self.cur_frame is not None:
                fh, fw = self.cur_frame.shape[:2]
                return int(wx * fw / dw), int(wy * fh / dh)
        except Exception:
            pass
        return wx, wy

    # ── Mouse callback ────────────────────────────────────────────────────────

    def _mouse_cb(self, event, x, y, flags, param):
        # Convert from window display coords to actual frame coords
        fx, fy = self._win_to_frame(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.roi_pt1 = (fx, fy)
            self.roi_pt2 = (fx, fy)

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
                    self._init_tracking(self.cur_frame,
                                        (x1, y1, x2 - x1, y2 - y1))

    # ── Tracker initialisation ────────────────────────────────────────────────

    def _make_csrt(self):
        params = cv2.TrackerCSRT_Params()
        for k, v in CSRT_PARAMS.items():
            if hasattr(params, k):
                try:
                    setattr(params, k, v)
                except Exception:
                    pass
        return cv2.TrackerCSRT_create(params)

    def _init_tracking(self, frame: np.ndarray, bbox: tuple):
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        # Clamp to frame
        x = max(0, min(x, fw - 2))
        y = max(0, min(y, fh - 2))
        w = max(10, min(w, fw - x))
        h = max(10, min(h, fh - y))
        bbox = (x, y, w, h)

        self.tracker = self._make_csrt()
        self.tracker.init(frame, bbox)
        self.bbox             = bbox
        self.lost_count       = 0
        self.track_state      = self.TS_TRACKING
        self.kalman_bbox_size = (w, h)
        self.kalman.init(x + w / 2, y + h / 2)
        self._save_template(frame, bbox)
        self.trail.clear()
        print(f"[TRACKER] Initialised  bbox={bbox}")

    def _save_template(self, frame: np.ndarray, bbox: tuple):
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        if x2 > x1 and y2 > y1:
            self.template      = frame[y1:y2, x1:x2].copy()
            self.template_fidx = self.frame_idx

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _center(bbox):
        x, y, w, h = bbox
        return x + w / 2, y + h / 2

    @staticmethod
    def _overlap(bbox, fshape):
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = fshape[:2]
        ix1, iy1 = max(0, x), max(0, y)
        ix2, iy2 = min(fw, x + w), min(fh, y + h)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1) / max(w * h, 1)

    # ── Re-detection via template matching ───────────────────────────────────

    def _redetect(self, frame: np.ndarray):
        if self.template is None:
            return None
        g_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g_tmpl  = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        # Narrow search to Kalman-predicted region first
        pred = self.kalman.predict()
        if pred and self.kalman_bbox_size:
            px, py   = pred
            tw, th   = self.kalman_bbox_size
            margin   = max(tw, th) * 3
            fh, fw   = g_frame.shape
            sx1 = max(0, int(px - margin));  sy1 = max(0, int(py - margin))
            sx2 = min(fw, int(px + margin)); sy2 = min(fh, int(py + margin))
            search   = g_frame[sy1:sy2, sx1:sx2]
            offset   = (sx1, sy1)
        else:
            search, offset = g_frame, (0, 0)

        best_val, best_box = 0.0, None
        for sc in (0.75, 0.875, 1.0, 1.125, 1.25):
            nh = max(8, int(g_tmpl.shape[0] * sc))
            nw = max(8, int(g_tmpl.shape[1] * sc))
            if nh >= search.shape[0] or nw >= search.shape[1]:
                continue
            rt  = cv2.resize(g_tmpl, (nw, nh))
            res = cv2.matchTemplate(search, rt, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if mv > best_val:
                best_val = mv
                best_box = (ml[0] + offset[0], ml[1] + offset[1], nw, nh)

        if best_val >= REDETECT_THRESHOLD and best_box:
            return best_box, best_val
        return None

    # ── One-frame tracker update ──────────────────────────────────────────────

    def _step(self, frame: np.ndarray):
        """Update tracker for one frame. Returns display bbox."""
        if self.track_state == self.TS_NONE:
            return None

        ok, raw = self.tracker.update(frame)
        ov = self._overlap(raw, frame.shape) if ok else 0.0

        # A: Good tracking
        if ok and ov > 0.6:
            self.bbox = raw
            cx, cy = self._center(raw)
            self.kalman.correct(cx, cy)
            self.kalman_bbox_size = (raw[2], raw[3])
            self.lost_count = 0
            if (self.frame_idx - self.template_fidx) >= TEMPLATE_UPDATE_INT:
                self._save_template(frame, raw)
            self.track_state = self.TS_TRACKING
            self.trail.append((cx, cy))
            return raw

        # B: Partially outside frame
        if ok and 0.1 < ov <= 0.6:
            self.bbox = raw
            self.kalman.correct(*self._center(raw))
            self.lost_count += 1
            self.track_state = self.TS_OUT_FRAME
            return raw

        # C: Fully outside frame
        if ok and ov <= 0.1:
            self.lost_count += 1
            self.track_state = self.TS_OUT_FRAME
            pred = self.kalman.predict()
            if pred and self.kalman_bbox_size:
                pw, ph = self.kalman_bbox_size
                return (int(pred[0]-pw/2), int(pred[1]-ph/2), int(pw), int(ph))
            return self.bbox

        # D: CSRT failed — Kalman prediction + template re-detection
        self.lost_count += 1

        k_box = None
        pred = self.kalman.predict()
        if pred and self.kalman_bbox_size:
            pw, ph = self.kalman_bbox_size
            k_box = (int(pred[0]-pw/2), int(pred[1]-ph/2), int(pw), int(ph))

        found = self._redetect(frame)
        if found:
            rb, conf = found
            print(f"[REDETECT] frame={self.frame_idx}  conf={conf:.2f}  box={rb}")
            self._init_tracking(frame, rb)
            return rb

        if k_box:
            self.track_state = self.TS_OCCLUDED
            return k_box

        self.track_state = self.TS_LOST
        return self.bbox

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_bbox(self, frame: np.ndarray, bbox: tuple, state: str):
        if not bbox:
            return
        x, y, w, h = [int(v) for v in bbox]
        color = {
            self.TS_TRACKING:  CLR_OK,
            self.TS_OCCLUDED:  CLR_WARN,
            self.TS_OUT_FRAME: CLR_WARN,
            self.TS_LOST:      CLR_LOST,
        }.get(state, CLR_SELECT)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Corner brackets
        c = max(6, min(w, h) // 4)
        for (px, py), (dx, dy) in [
            ((x,   y),   ( 1,  1)),
            ((x+w, y),   (-1,  1)),
            ((x,   y+h), ( 1, -1)),
            ((x+w, y+h), (-1, -1)),
        ]:
            cv2.line(frame, (px, py), (px + dx*c, py), color, 3)
            cv2.line(frame, (px, py), (px, py + dy*c), color, 3)

        cv2.drawMarker(frame, (x + w//2, y + h//2),
                       color, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)

    def _draw_trail(self, frame: np.ndarray):
        pts = list(self.trail)
        for i in range(1, len(pts)):
            a  = i / len(pts)
            c  = tuple(int(v * a) for v in CLR_TRAIL)
            p1 = (int(pts[i-1][0]), int(pts[i-1][1]))
            p2 = (int(pts[i][0]),   int(pts[i][1]))
            cv2.line(frame, p1, p2, c, max(1, int(2*a)), cv2.LINE_AA)

    def _draw_pip(self, frame: np.ndarray, bbox: tuple):
        if not bbox:
            return
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        px, py = int(w * 0.4), int(h * 0.4)
        x1, y1 = max(0, x - px), max(0, y - py)
        x2, y2 = min(fw, x + w + px), min(fh, y + h + py)
        if x2 <= x1 or y2 <= y1:
            return
        roi    = frame[y1:y2, x1:x2].copy()
        pip_w  = max(120, int(fw * 0.28))
        aspect = (y2 - y1) / max(x2 - x1, 1)
        pip_h  = max(80, min(int(pip_w * aspect), int(fh * 0.35)))
        zoomed = cv2.resize(roi, (pip_w, pip_h))
        m      = 10
        tx, ty = fw - pip_w - m, m

        ov = frame.copy()
        cv2.rectangle(ov, (tx-4, ty-22), (tx+pip_w+4, ty+pip_h+4), (15,15,15), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        frame[ty:ty+pip_h, tx:tx+pip_w] = zoomed
        cv2.rectangle(frame, (tx-2, ty-2), (tx+pip_w+2, ty+pip_h+2), CLR_PIP, 2)
        cv2.putText(frame, "TARGET VIEW", (tx, ty - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_PIP, 1, cv2.LINE_AA)

    def _draw_roi_overlay(self, frame: np.ndarray):
        """Draw in-progress selection rectangle (frame-space coords → correct at any zoom)."""
        if not (self.roi_pt1 and self.roi_pt2):
            return
        x1 = min(self.roi_pt1[0], self.roi_pt2[0])
        y1 = min(self.roi_pt1[1], self.roi_pt2[1])
        x2 = max(self.roi_pt1[0], self.roi_pt2[0])
        y2 = max(self.roi_pt1[1], self.roi_pt2[1])
        cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_SELECT, 2)
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), CLR_SELECT, -1)
        cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)

    def _draw_hud(self, frame: np.ndarray):
        fh, fw = frame.shape[:2]

        # FPS
        cv2.rectangle(frame, (0, 0), (155, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {self.fps_value:5.1f}", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, CLR_OK, 2, cv2.LINE_AA)

        # PLAYING / PAUSED
        pb_txt = "PLAYING" if self.playing else "PAUSED"
        pb_col = CLR_OK    if self.playing else (100, 100, 255)
        cv2.putText(frame, pb_txt, (8, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, pb_col, 1, cv2.LINE_AA)

        # PiP indicator
        pip_txt = "PiP: ON " if self.pip_enabled else "PiP: OFF"
        pip_col = CLR_PIP   if self.pip_enabled else (90, 90, 90)
        cv2.putText(frame, pip_txt, (8, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, pip_col, 1, cv2.LINE_AA)

        # Controls (bottom)
        hint = "ENTER:start  SPACE:pause  R:retarget  P:PiP  Q:quit"
        cv2.putText(frame, hint, (8, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1, cv2.LINE_AA)

        # ── No tracker → show selection instructions in centre ─────────────
        if self.track_state == self.TS_NONE:
            lines = [
                ("Draw a box around the target", 0.78, CLR_SELECT),
                ("then press ENTER to start",    0.60, (170, 170, 170)),
            ]
            for i, (msg, sc, col) in enumerate(lines):
                (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
                sx = fw // 2 - tw // 2
                sy = fh // 2 - 30 + i * 40
                cv2.rectangle(frame, (sx-10, sy-th-4), (sx+tw+10, sy+6), (0,0,0), -1)
                cv2.putText(frame, msg, (sx, sy),
                            cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)
            return

        # ── Tracker state badge (bottom-centre) ───────────────────────────
        state_cfg = {
            self.TS_TRACKING:  ("[  TRACKING  ]",        CLR_OK,   False),
            self.TS_OCCLUDED:  ("[ OCCLUDED - Kalman ]", CLR_WARN, True),
            self.TS_OUT_FRAME: ("[ OUT OF FRAME ]",      CLR_WARN, True),
            self.TS_LOST:      ("[ TARGET LOST ]",       CLR_LOST, True),
        }
        label, color, blink = state_cfg.get(self.track_state,
                                             ("", (255,255,255), False))
        if not label:
            return
        if blink and (int(time.time() * 3) % 2 == 0):
            return   # blinking effect

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
        sx = fw // 2 - tw // 2
        sy = fh - 26
        cv2.rectangle(frame, (sx-8, sy-th-4), (sx+tw+8, sy+6), (0,0,0), -1)
        cv2.putText(frame, label, (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)

        # Full-lost dim overlay
        if self.track_state == self.TS_LOST:
            ov = frame.copy()
            cv2.rectangle(ov, (0,0), (fw, fh), (0, 0, 80), -1)
            cv2.addWeighted(ov, 0.20, frame, 0.80, 0, frame)
            msg = "TARGET LOST — Press R to select a new target"
            (mw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.putText(frame, msg, (fw//2 - mw//2, fh//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, CLR_LOST, 2, cv2.LINE_AA)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        print("=" * 55)
        print("  SPORTS TRACKER  ready")
        print(f"  Video : {self.video_path}")
        print("  1. Draw a box around the target")
        print("  2. Press ENTER to start playback")
        print("  SPACE=pause  R=retarget  P=PiP  Q=quit")
        print("=" * 55)

        while True:
            # Advance video only when playing AND a tracker is active
            advance = self.playing and self.track_state != self.TS_NONE

            if advance:
                ret, raw = self.cap.read()
                if not ret:
                    print("[END] Video finished.")
                    self._show_end_screen()
                    break
                self.cur_frame  = self._resize(raw)
                self.frame_idx += 1

                # FPS counter
                self._fps_cnt += 1
                now = time.time()
                if now - self._fps_t0 >= 0.4:
                    self.fps_value = self._fps_cnt / (now - self._fps_t0)
                    self._fps_cnt  = 0
                    self._fps_t0   = now

                # Run tracker
                disp_bbox = self._step(self.cur_frame)

                # Too many lost frames → stop, ask user to re-select
                if self.lost_count > MAX_LOST_FRAMES:
                    print("[STOP] Target lost too long. Press R to re-select.")
                    self.playing     = False
                    self.track_state = self.TS_NONE
                    self.tracker     = None
                    self.bbox        = None

            else:
                disp_bbox = self.bbox
                # Reset FPS timer while paused (no fake spike on resume)
                self._fps_t0  = time.time()
                self._fps_cnt = 0

            # ── Build display ─────────────────────────────────────────────
            display = self.cur_frame.copy()

            self._draw_trail(display)

            if (self.pip_enabled
                    and disp_bbox
                    and self.track_state == self.TS_TRACKING):
                self._draw_pip(display, disp_bbox)

            if disp_bbox and self.track_state != self.TS_NONE:
                self._draw_bbox(display, disp_bbox, self.track_state)

            # Selection rectangle (always in frame-space → correct at any window size)
            self._draw_roi_overlay(display)

            self._draw_hud(display)

            cv2.imshow(self.WIN, display)

            # ── Keys ──────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):            # Q / ESC — quit
                break

            elif key == 13:                      # ENTER — start / resume
                if self.track_state != self.TS_NONE:
                    self.playing  = True
                    self._fps_t0  = time.time()
                    self._fps_cnt = 0
                    print("[PLAY] Started")
                else:
                    print("[!] Select a target first, then press ENTER")

            elif key == ord(' '):                # SPACE — pause / resume
                if self.playing:
                    self.playing = False
                    print("[PAUSE]")
                elif self.track_state != self.TS_NONE:
                    self.playing  = True
                    self._fps_t0  = time.time()
                    self._fps_cnt = 0
                    print("[RESUME]")

            elif key == ord('r'):                # R — re-select target
                self.playing     = False
                self.track_state = self.TS_NONE
                self.tracker     = None
                self.bbox        = None
                self.trail.clear()
                self.lost_count  = 0
                self.roi_pt1     = None
                self.roi_pt2     = None
                print("[RESET] Draw a new target, then press ENTER")

            elif key == ord('p'):                # P — toggle PiP
                self.pip_enabled = not self.pip_enabled
                print(f"[PiP] {'ON' if self.pip_enabled else 'OFF'}")

        self.cap.release()
        cv2.destroyAllWindows()
        print("[EXIT] Tracker closed.")

    def _show_end_screen(self):
        if self.cur_frame is None:
            return
        frame = self.cur_frame.copy()
        msg = "VIDEO ENDED"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.putText(frame, msg,
                    (frame.shape[1]//2 - tw//2, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 200), 3, cv2.LINE_AA)
        cv2.imshow(self.WIN, frame)
        cv2.waitKey(3000)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage:   python sports_tracker.py <video_file>")
        print("Example: python sports_tracker.py match.mov")
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
