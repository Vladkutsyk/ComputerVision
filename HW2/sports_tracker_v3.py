"""
SPORTS VIDEO TRACKER V3 — CSRT + Kalman + strict Color Validation
=================================================================
Dependencies:
    pip install opencv-contrib-python numpy

Usage:
    python sports_tracker_v3.py <video_file>
"""

import cv2
import numpy as np
import sys
import os
import time
from collections import deque

# ── CSRT tuning for sports ──
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

MAX_FRAME_WIDTH      = 1280  

# ── STRICT VALIDATION SETTINGS ──
# Distance threshold for color mismatch (Bhattacharyya distance: 0.0=identical, 1.0=no overlap)
# 0.55 is generally a good threshold to distinguish a player from grass or a different colored player.
COLOR_DIST_THRESHOLD = 0.55  
# Multiplier for trajectory jump limit
TRAJECTORY_TOLERANCE = 1.15  

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
    """Constant-velocity Kalman predicting physical trajectory."""

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
        self.lost_reason      = ""
        
        self.kalman           = KalmanPredictor()
        self.kalman_bbox_size = None
        
        self.target_hist      = None  # Stores the original object's color profile

        self.trail = deque(maxlen=60)
        self.pip_enabled = True

        self.drawing = False
        self.roi_pt1 = None
        self.roi_pt2 = None

        self.fps_value = 0.0
        self._fps_t0   = time.time()
        self._fps_cnt  = 0

        self.WIN = "Sports Tracker V3 (Strict)"
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        h, w = self.cur_frame.shape[:2]
        cv2.resizeWindow(self.WIN, min(w, 1280), min(h, 720))
        cv2.setMouseCallback(self.WIN, self._mouse_cb)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w > MAX_FRAME_WIDTH:
            return cv2.resize(frame,
                              (MAX_FRAME_WIDTH, int(h * MAX_FRAME_WIDTH / w)))
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

    def _make_csrt(self):
        params = cv2.TrackerCSRT_Params()
        for k, v in CSRT_PARAMS.items():
            if hasattr(params, k):
                try:
                    setattr(params, k, v)
                except Exception:
                    pass
        return cv2.TrackerCSRT_create(params)

    def _extract_color_hist(self, frame: np.ndarray, bbox: tuple):
        """Calculates a 2D Hue-Saturation histogram for the given bounding box."""
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 32 bins for Hue, 32 bins for Saturation
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def _init_tracking(self, frame: np.ndarray, bbox: tuple):
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x = max(0, min(x, fw - 2))
        y = max(0, min(y, fh - 2))
        w = max(10, min(w, fw - x))
        h = max(10, min(h, fh - y))
        bbox = (x, y, w, h)

        self.tracker = self._make_csrt()
        self.tracker.init(frame, bbox)
        self.bbox             = bbox
        self.track_state      = self.TS_TRACKING
        self.lost_reason      = ""
        self.kalman_bbox_size = (w, h)
        
        self.kalman = KalmanPredictor()
        self.kalman.init(x + w / 2, y + h / 2)
        
        # Lock in the target's color profile
        self.target_hist = self._extract_color_hist(frame, bbox)
        
        self.trail.clear()
        print(f"[TRACKER] Initialised  bbox={bbox}")

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

    def _trigger_lost(self, reason: str):
        """Halts the tracker and demands user input."""
        print(f"[STOP] {reason}")
        self.track_state = self.TS_LOST
        self.lost_reason = reason
        self.playing = False
        return self.bbox

    # ── Strict Validation & Tracking Update ──────────────────────────────────
    def _step(self, frame: np.ndarray):
        if self.track_state == self.TS_NONE:
            return None

        pred_pos = self.kalman.predict()
        ok, raw = self.tracker.update(frame)
        ov = self._overlap(raw, frame.shape) if ok else 0.0

        # --- VALIDATION 1: Left Frame or Visual Tracker internal failure ---
        if not ok or ov <= 0.1:
            return self._trigger_lost("Object left the frame or visual lock failed.")

        if ok and ov > 0.1:
            # --- VALIDATION 2: Color Integrity (Grass/Wall/Wrong Player check) ---
            current_hist = self._extract_color_hist(frame, raw)
            if self.target_hist is not None and current_hist is not None:
                color_dist = cv2.compareHist(self.target_hist, current_hist, cv2.HISTCMP_BHATTACHARYYA)
                if color_dist > COLOR_DIST_THRESHOLD:
                    return self._trigger_lost(f"Color shift! Tracking wrong object or background (dist: {color_dist:.2f})")

            # --- VALIDATION 3: Kinematic Trajectory Integrity (Jump detection) ---
            cx, cy = self._center(raw)
            if pred_pos and self.kalman_bbox_size:
                px, py = pred_pos
                w, h = self.kalman_bbox_size
                dist = np.hypot(cx - px, cy - py)
                max_dist = max(w, h) * TRAJECTORY_TOLERANCE

                if dist > max_dist:
                    return self._trigger_lost(f"Trajectory jump! ID switch detected ({dist:.1f}px jump)")

        # --- TRACKING SUCCESS ---
        self.bbox = raw
        cx, cy = self._center(raw)
        self.kalman.correct(cx, cy)
        self.kalman_bbox_size = (raw[2], raw[3])
        self.track_state = self.TS_TRACKING
        self.trail.append((cx, cy))
        
        return raw

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_bbox(self, frame: np.ndarray, bbox: tuple, state: str):
        if not bbox:
            return
        x, y, w, h = [int(v) for v in bbox]
        color = CLR_OK if state == self.TS_TRACKING else CLR_LOST

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        c = max(6, min(w, h) // 4)
        for (px, py), (dx, dy) in [
            ((x,   y),   ( 1,  1)), ((x+w, y),   (-1,  1)),
            ((x,   y+h), ( 1, -1)), ((x+w, y+h), (-1, -1)),
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

        cv2.rectangle(frame, (0, 0), (155, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {self.fps_value:5.1f}", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, CLR_OK, 2, cv2.LINE_AA)

        pb_txt = "PLAYING" if self.playing else "PAUSED"
        pb_col = CLR_OK    if self.playing else (100, 100, 255)
        cv2.putText(frame, pb_txt, (8, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, pb_col, 1, cv2.LINE_AA)

        pip_txt = "PiP: ON " if self.pip_enabled else "PiP: OFF"
        pip_col = CLR_PIP   if self.pip_enabled else (90, 90, 90)
        cv2.putText(frame, pip_txt, (8, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, pip_col, 1, cv2.LINE_AA)

        hint = "ENTER:start  SPACE:pause  R:retarget  P:PiP  Q:quit"
        cv2.putText(frame, hint, (8, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1, cv2.LINE_AA)

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

        if self.track_state == self.TS_TRACKING:
            label = "[  TRACKING  ]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
            sx = fw // 2 - tw // 2
            sy = fh - 26
            cv2.rectangle(frame, (sx-8, sy-th-4), (sx+tw+8, sy+6), (0,0,0), -1)
            cv2.putText(frame, label, (sx, sy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, CLR_OK, 2, cv2.LINE_AA)

        # Draw intense "Lost" warning screen
        if self.track_state == self.TS_LOST:
            ov = frame.copy()
            cv2.rectangle(ov, (0,0), (fw, fh), (0, 0, 80), -1)
            cv2.addWeighted(ov, 0.20, frame, 0.80, 0, frame)
            
            lines = [
                f"STOPPED: {self.lost_reason}",
                "Press R to select a new target"
            ]
            for i, msg in enumerate(lines):
                sc = 0.8 if i == 0 else 0.6
                col = CLR_WARN if i == 0 else CLR_LOST
                (mw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
                cv2.putText(frame, msg, (fw//2 - mw//2, fh//2 - 20 + i*40),
                            cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)

    def run(self):
        print("=" * 55)
        print("  SPORTS TRACKER V3 (Strict Validation) ready")
        print(f"  Video : {self.video_path}")
        print("=" * 55)

        while True:
            advance = self.playing and self.track_state == self.TS_TRACKING

            if advance:
                ret, raw = self.cap.read()
                if not ret:
                    print("[END] Video finished.")
                    self._show_end_screen()
                    break
                self.cur_frame  = self._resize(raw)
                self.frame_idx += 1

                self._fps_cnt += 1
                now = time.time()
                if now - self._fps_t0 >= 0.4:
                    self.fps_value = self._fps_cnt / (now - self._fps_t0)
                    self._fps_cnt  = 0
                    self._fps_t0   = now

                disp_bbox = self._step(self.cur_frame)

            else:
                disp_bbox = self.bbox
                self._fps_t0  = time.time()
                self._fps_cnt = 0

            display = self.cur_frame.copy()

            self._draw_trail(display)

            if (self.pip_enabled and disp_bbox and self.track_state == self.TS_TRACKING):
                self._draw_pip(display, disp_bbox)

            if disp_bbox and self.track_state != self.TS_NONE:
                self._draw_bbox(display, disp_bbox, self.track_state)

            self._draw_roi_overlay(display)
            self._draw_hud(display)

            cv2.imshow(self.WIN, display)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):            
                break
            elif key == 13:                      
                if self.track_state == self.TS_TRACKING:
                    self.playing  = True
                    self._fps_t0  = time.time()
                    self._fps_cnt = 0
                elif self.track_state == self.TS_LOST:
                    print("[!] Cannot resume. Target lost. Press R to re-select.")
                else:
                    print("[!] Select a target first, then press ENTER")
            elif key == ord(' '):                
                if self.playing:
                    self.playing = False
                elif self.track_state == self.TS_TRACKING:
                    self.playing  = True
                    self._fps_t0  = time.time()
                    self._fps_cnt = 0
            elif key == ord('r'):                
                self.playing     = False
                self.track_state = self.TS_NONE
                self.tracker     = None
                self.bbox        = None
                self.trail.clear()
                self.roi_pt1     = None
                self.roi_pt2     = None
                self.target_hist = None

            elif key == ord('p'):                
                self.pip_enabled = not self.pip_enabled

        self.cap.release()
        cv2.destroyAllWindows()
        print("[EXIT] Tracker closed.")

    def _show_end_screen(self):
        if self.cur_frame is None: return
        frame = self.cur_frame.copy()
        msg = "VIDEO ENDED"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.putText(frame, msg,
                    (frame.shape[1]//2 - tw//2, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 200), 3, cv2.LINE_AA)
        cv2.imshow(self.WIN, frame)
        cv2.waitKey(3000)

def main():
    if len(sys.argv) < 2:
        print("Usage: python sports_tracker_v3.py <video_file>")
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