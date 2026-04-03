
"""
BALL TRACKER V2 — Ball-only sports tracker
=========================================

Designed for tracking ONLY the ball.

Main ideas:
- Faster Kalman motion model for ball speed
- Trajectory-aware recovery when the ball is briefly hidden
- Circle/shape checks to avoid drifting to legs/players
- Background/grass tolerance: grass alone never causes an immediate stop
- Latency tolerance: waits through short misses before stopping

Dependencies:
    pip install opencv-contrib-python numpy

Usage:
    python ball_tracker_v2.py <video_file>

Controls:
    Draw a box around the ball, then press ENTER to start.
    SPACE: pause/resume
    R: retarget
    P: toggle picture-in-picture
    Q / ESC: quit
"""

from __future__ import annotations

import cv2
import numpy as np
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple


# -----------------------------------------------------------------------------
# Parameters tuned for a fast sports ball
# -----------------------------------------------------------------------------
MAX_FRAME_WIDTH = 1280

MAX_LOST_FRAMES = 40          # tolerate short invisibility / latency
TRACK_CONFIDENCE = 0.58
RECOVER_CONFIDENCE = 0.50
STOP_CONFIDENCE = 0.20        # only used after repeated failures

# Ball shape gates
BALL_MIN_CIRCULARITY = 0.42
BALL_MIN_CIRCULARITY_HARD = 0.30
BALL_ASPECT_MIN = 0.70
BALL_ASPECT_MAX = 1.45

# Search / recovery
BALL_SEARCH_MARGIN_MULT = 4.0
TEMPLATE_SCALE_SET = (0.80, 0.90, 1.00, 1.10, 1.20)
MAX_HOUGH_CANDIDATES = 10

# Grass/background handling
GRASS_HUE_LOW = 35
GRASS_HUE_HIGH = 85
GRASS_SAT_MIN = 30
GRASS_VAL_MIN = 30
GRASS_STOP_INSIDE_THRESHOLD = 0.84   # very permissive: grass is allowed
GRASS_STOP_OBJECTNESS_THRESHOLD = 0.18
GRASS_STOP_COLOR_THRESHOLD = 0.18

# Ball physics
TRAJECTORY_TOLERANCE = 1.45      # allow a lot of motion between frames
KICK_MULTIPLIER = 5.2            # sudden snap still possible
MAX_PRED_SEARCH_SCALE = 2.8

# Visuals
CLR_OK = (0, 220, 80)
CLR_WARN = (0, 165, 255)
CLR_COAST = (255, 105, 180)
CLR_LOST = (50, 50, 220)
CLR_SELECT = (0, 220, 255)
CLR_PIP = (0, 220, 220)
CLR_TRAIL = (0, 180, 255)
CLR_KICK = (255, 255, 0)
CLR_INFO = (220, 220, 220)

# CSRT tuning
CSRT_PARAMS = {
    "admm_iterations": 4,
    "background_ratio": 2,
    "num_hog_channels_used": 18,
    "padding": 3.0,
    "template_size": 200,
    "gsl_sigma": 1.0,
    "hog_orientations": 9,
    "num_scales": 33,
    "scale_step": 1.02,
    "scale_sigma_factor": 0.25,
    "psr_threshold": 0.035,
    "use_channel_weights": True,
    "use_color_names": True,
    "use_gray": True,
    "use_hog": True,
}


# -----------------------------------------------------------------------------
# Small helper structures
# -----------------------------------------------------------------------------
@dataclass
class CandidateStats:
    confidence: float = 0.0
    circularity: float = 0.0
    aspect_ratio: float = 0.0
    grass_inside: float = 1.0
    edge_density: float = 0.0


class KalmanPredictor:
    """
    Constant-velocity model.
    Good enough for a ball if process noise is kept relatively high.
    """
    def __init__(self):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 5e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf = kf
        self.initialized = False

    def init(self, cx: float, cy: float):
        self.kf.statePre = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        self.kf.statePost = self.kf.statePre.copy()
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


# -----------------------------------------------------------------------------
# Ball tracker
# -----------------------------------------------------------------------------
class BallTracker:
    TS_NONE = "NONE"
    TS_TRACKING = "TRACKING"
    TS_COASTING = "COASTING"
    TS_OCCLUDED = "OCCLUDED"
    TS_OUT_FRAME = "OUT_FRAME"
    TS_LOST = "LOST"

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

        self.playing = False
        self.frame_idx = 0

        self.tracker = None
        self.bbox = None
        self.track_state = self.TS_NONE
        self.lost_count = 0
        self.stop_reason = ""

        self.kalman = KalmanPredictor()
        self.kalman_bbox_size = None

        self.template = None
        self.template_fidx = 0
        self.template_frozen = False

        self.last_good_bbox = None
        self.last_good_conf = 0.0

        self.trail = deque(maxlen=60)
        self.pip_enabled = True
        self.kick_flash_timer = 0.0

        self.drawing = False
        self.roi_pt1 = None
        self.roi_pt2 = None

        self.fps_value = 0.0
        self._fps_t0 = time.time()
        self._fps_cnt = 0

        self.WIN = "Ball Tracker V2"
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        h, w = self.cur_frame.shape[:2]
        cv2.resizeWindow(self.WIN, min(w, 1280), min(h, 720))
        cv2.setMouseCallback(self.WIN, self._mouse_cb)

    # -------------------------------------------------------------------------
    # Basic helpers
    # -------------------------------------------------------------------------
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

    @staticmethod
    def _center(bbox):
        x, y, w, h = bbox
        return x + w / 2.0, y + h / 2.0

    @staticmethod
    def _overlap(bbox, fshape):
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = fshape[:2]
        ix1, iy1 = max(0, x), max(0, y)
        ix2, iy2 = min(fw, x + w), min(fh, y + h)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1) / max(w * h, 1)

    def _clamp_bbox(self, bbox: Tuple[int, int, int, int], shape) -> Tuple[int, int, int, int]:
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = shape[:2]
        x = max(0, min(x, fw - 2))
        y = max(0, min(y, fh - 2))
        w = max(8, min(w, fw - x))
        h = max(8, min(h, fh - y))
        return x, y, w, h

    def _extract_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy(), (x1, y1, x2 - x1, y2 - y1)

    def _make_csrt(self):
        params = cv2.TrackerCSRT_Params()
        for k, v in CSRT_PARAMS.items():
            if hasattr(params, k):
                try:
                    setattr(params, k, v)
                except Exception:
                    pass
        return cv2.TrackerCSRT_create(params)

    def _reset_trackers(self):
        self.tracker = None
        self.bbox = None
        self.kalman = KalmanPredictor()
        self.kalman_bbox_size = None
        self.template = None
        self.template_fidx = 0
        self.template_frozen = False
        self.lost_count = 0
        self.last_good_bbox = None
        self.last_good_conf = 0.0
        self.trail.clear()
        self.stop_reason = ""

    # -------------------------------------------------------------------------
    # Ball appearance / confidence
    # -------------------------------------------------------------------------
    def _inside_grass_ratio(self, roi_bgr: np.ndarray) -> float:
        if roi_bgr.size == 0:
            return 1.0
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([GRASS_HUE_LOW, GRASS_SAT_MIN, GRASS_VAL_MIN]),
            np.array([GRASS_HUE_HIGH, 255, 255]),
        )
        return float(cv2.countNonZero(mask)) / float(mask.size)

    def _edge_density(self, roi_bgr: np.ndarray) -> float:
        if roi_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        return float(cv2.countNonZero(edges)) / float(edges.size)

    def _circularity(self, roi_bgr: np.ndarray) -> float:
        if roi_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # threshold on local contrast; ball edges often survive this
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if area <= 1e-3 or peri <= 1e-3:
            return 0.0
        val = float(4.0 * np.pi * area / (peri * peri))
        return max(0.0, min(1.0, val))

    def _ball_stats(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> CandidateStats:
        roi_pack = self._extract_roi(frame, bbox)
        if roi_pack is None:
            return CandidateStats()

        roi, _ = roi_pack
        x, y, w, h = bbox
        aspect = w / max(h, 1)
        grass_inside = self._inside_grass_ratio(roi)
        edge_density = self._edge_density(roi)
        circularity = self._circularity(roi)

        # Confidence is a blend: circularity + near-square aspect + edge signal
        ar_score = 1.0 - min(1.0, abs(aspect - 1.0) / 0.55)
        circ_score = circularity
        edge_score = max(0.0, min(1.0, edge_density / 0.10))
        grass_penalty = max(0.0, 1.0 - grass_inside)

        conf = 0.36 * ar_score + 0.34 * circ_score + 0.20 * edge_score + 0.10 * grass_penalty

        # Allow ball on grass; grass is a weak negative only.
        if grass_inside > GRASS_STOP_INSIDE_THRESHOLD:
            conf *= 0.88

        return CandidateStats(
            confidence=float(max(0.0, min(1.0, conf))),
            circularity=float(circularity),
            aspect_ratio=float(aspect),
            grass_inside=float(grass_inside),
            edge_density=float(edge_density),
        )

    def _is_ball_like(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        x, y, w, h = bbox
        if w < 5 or h < 5:
            return False
        stats = self._ball_stats(frame, bbox)
        aspect = w / max(h, 1)

        if not (BALL_ASPECT_MIN <= aspect <= BALL_ASPECT_MAX):
            return False
        if stats.circularity < BALL_MIN_CIRCULARITY_HARD:
            return False
        if stats.confidence < 0.18:
            return False
        return True

    def _reference_confidence(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        # For ball, use objectness only; grass is not a stop condition by itself.
        stats = self._ball_stats(frame, bbox)
        return stats.confidence

    # -------------------------------------------------------------------------
    # Recovery methods
    # -------------------------------------------------------------------------
    def _save_template(self, frame: np.ndarray, bbox: tuple, force=False):
        roi_pack = self._extract_roi(frame, bbox)
        if roi_pack is None:
            return
        roi, _ = roi_pack
        if not force and self._inside_grass_ratio(roi) > 0.58:
            self.template_frozen = True
            return

        self.template_frozen = False
        self.template = roi.copy()
        self.template_fidx = self.frame_idx

    def _template_match_search(self, frame: np.ndarray, pred_pos: tuple):
        if self.template is None or pred_pos is None or self.kalman_bbox_size is None:
            return None

        g_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g_tmpl = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        px, py = pred_pos
        tw, th = self.kalman_bbox_size
        margin = max(tw, th) * BALL_SEARCH_MARGIN_MULT
        fh, fw = g_frame.shape
        sx1 = max(0, int(px - margin))
        sy1 = max(0, int(py - margin))
        sx2 = min(fw, int(px + margin))
        sy2 = min(fh, int(py + margin))
        search = g_frame[sy1:sy2, sx1:sx2]
        if search.size == 0:
            return None
        offset = (sx1, sy1)

        best_score = 0.0
        best_box = None

        for sc in TEMPLATE_SCALE_SET:
            nh = max(8, int(g_tmpl.shape[0] * sc))
            nw = max(8, int(g_tmpl.shape[1] * sc))
            if nh >= search.shape[0] or nw >= search.shape[1]:
                continue
            rt = cv2.resize(g_tmpl, (nw, nh))
            res = cv2.matchTemplate(search, rt, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            cand = (ml[0] + offset[0], ml[1] + offset[1], nw, nh)
            conf = self._reference_confidence(frame, cand)
            score = 0.42 * float(max(0.0, mv)) + 0.58 * conf
            if score > best_score:
                best_score = score
                best_box = cand

        if best_box is not None and best_score >= RECOVER_CONFIDENCE:
            return best_box, best_score
        return None

    def _hough_circle_search(self, frame: np.ndarray, pred_pos: tuple):
        if pred_pos is None or self.kalman_bbox_size is None:
            return None

        px, py = pred_pos
        tw, th = self.kalman_bbox_size
        ref_r = max(4, int(max(tw, th) / 2))
        margin = max(tw, th) * BALL_SEARCH_MARGIN_MULT

        fh, fw = frame.shape[:2]
        sx1 = max(0, int(px - margin))
        sy1 = max(0, int(py - margin))
        sx2 = min(fw, int(px + margin))
        sy2 = min(fh, int(py + margin))
        roi = frame[sy1:sy2, sx1:sx2]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(10, ref_r),
            param1=90,
            param2=18,
            minRadius=max(2, int(ref_r * 0.65)),
            maxRadius=max(3, int(ref_r * 1.55)),
        )
        if circles is None:
            return None

        circles = np.squeeze(circles, axis=0)
        best_box = None
        best_score = 0.0

        for c in circles[:MAX_HOUGH_CANDIDATES]:
            cx, cy, r = float(c[0]), float(c[1]), float(c[2])
            cand = (int(sx1 + cx - r), int(sy1 + cy - r), int(2 * r), int(2 * r))
            stats = self._ball_stats(frame, cand)
            # Circle proposal is useful only if the ball-like score is reasonable.
            score = stats.confidence + 0.08 * stats.circularity
            if score > best_score:
                best_score = score
                best_box = cand

        if best_box is not None and best_score >= RECOVER_CONFIDENCE:
            return best_box, best_score
        return None

    def _search_near_prediction(self, frame: np.ndarray, pred_pos: tuple):
        # Prefer the faster / more robust circle search for the ball,
        # then template matching.
        found = self._hough_circle_search(frame, pred_pos)
        if found is not None:
            return found
        return self._template_match_search(frame, pred_pos)

    # -------------------------------------------------------------------------
    # Stop condition
    # -------------------------------------------------------------------------
    def _grass_object_gate(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Return True only if the region is mostly grass AND there is very weak
        object evidence. This avoids stopping when the ball is simply resting
        on grass.
        """
        roi_pack = self._extract_roi(frame, bbox)
        if roi_pack is None:
            return False
        roi, _ = roi_pack
        grass_inside = self._inside_grass_ratio(roi)
        objectness = self._reference_confidence(frame, bbox)

        return (
            grass_inside >= GRASS_STOP_INSIDE_THRESHOLD and
            objectness <= GRASS_STOP_OBJECTNESS_THRESHOLD
        )

    def _stop_tracking(self, reason: str):
        self.playing = False
        self.track_state = self.TS_LOST
        self.stop_reason = reason
        print(f"[STOP] {reason}")

    # -------------------------------------------------------------------------
    # Tracking initialization
    # -------------------------------------------------------------------------
    def _init_tracking(self, frame: np.ndarray, bbox: tuple):
        bbox = self._clamp_bbox(bbox, frame.shape)
        x, y, w, h = bbox

        self.tracker = self._make_csrt()
        self.tracker.init(frame, bbox)
        self.bbox = bbox

        self.track_state = self.TS_TRACKING
        self.lost_count = 0
        self.stop_reason = ""
        self.kalman = KalmanPredictor()
        self.kalman.init(x + w / 2.0, y + h / 2.0)
        self.kalman_bbox_size = (w, h)
        self._save_template(frame, bbox, force=True)

        self.last_good_bbox = bbox
        self.last_good_conf = 1.0
        self.trail.clear()
        self.kick_flash_timer = 0.0

        print(f"[TRACKER] Initialized bbox={bbox}")

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
                if (x2 - x1) > 6 and (y2 - y1) > 6:
                    self._init_tracking(self.cur_frame, (x1, y1, x2 - x1, y2 - y1))

    # -------------------------------------------------------------------------
    # Update step
    # -------------------------------------------------------------------------
    def _step(self, frame: np.ndarray):
        if self.track_state == self.TS_NONE:
            return None

        pred_pos = self.kalman.predict()
        ok, raw = self.tracker.update(frame) if self.tracker is not None else (False, None)
        ov = self._overlap(raw, frame.shape) if (ok and raw is not None) else 0.0
        fh, fw = frame.shape[:2]

        # Out of frame check
        is_out_of_frame = False
        if ok and ov < 0.5:
            is_out_of_frame = True
        if pred_pos:
            px, py = pred_pos
            if px < 5 or px > fw - 5 or py < 5 or py > fh - 5:
                is_out_of_frame = True

        if is_out_of_frame:
            self.track_state = self.TS_OUT_FRAME
            self._stop_tracking("Object went out of frame.")
            return self.bbox

        # Score current candidate
        candidate_conf = 0.0
        if ok and raw is not None:
            candidate_conf = self._reference_confidence(frame, raw)

        # Ball guardrails: do not drift to players/legs.
        if ok and raw is not None and self.kalman_bbox_size is not None:
            x, y, w, h = raw
            ar = w / max(h, 1)
            if not (BALL_ASPECT_MIN <= ar <= BALL_ASPECT_MAX):
                ok = False
            elif candidate_conf < 0.16:
                ok = False

        # Trajectory sanity
        is_kick_event = False
        anomaly_detected = False
        if ok and raw is not None and pred_pos and self.kalman_bbox_size:
            cx, cy = self._center(raw)
            px, py = pred_pos
            w, h = self.kalman_bbox_size
            dist_pred = np.hypot(cx - px, cy - py)
            max_pred_dist = max(w, h) * TRAJECTORY_TOLERANCE

            if dist_pred > max_pred_dist:
                last_cx, last_cy = self.trail[-1] if self.trail else (cx, cy)
                dist_last = np.hypot(cx - last_cx, cy - last_cy)
                max_jump = max(w, h) * KICK_MULTIPLIER
                if dist_last < max_jump:
                    is_kick_event = True
                    print(f"[KICK] Ball snap detected, off-path by {dist_pred:.1f}px")
                else:
                    anomaly_detected = True

        # Good tracking
        if ok and not anomaly_detected and raw is not None:
            self.bbox = raw
            cx, cy = self._center(raw)

            if is_kick_event:
                self.kalman.init(cx, cy)
                self.kick_flash_timer = time.time()
            else:
                self.kalman.correct(cx, cy)

            self.kalman_bbox_size = (raw[2], raw[3])
            self.lost_count = 0
            self.last_good_bbox = raw
            self.last_good_conf = candidate_conf

            if (self.frame_idx - self.template_fidx) >= 8:
                self._save_template(frame, raw)

            self.track_state = self.TS_TRACKING if candidate_conf >= TRACK_CONFIDENCE else self.TS_OCCLUDED
            self.trail.append((cx, cy))
            return raw

        # Recovery / coasting
        self.lost_count += 1

        if pred_pos and self.kalman_bbox_size is not None:
            pw, ph = self.kalman_bbox_size
            k_box = (int(pred_pos[0] - pw / 2), int(pred_pos[1] - ph / 2), int(pw), int(ph))
            self.trail.append((pred_pos[0], pred_pos[1]))
        else:
            k_box = self.bbox

        # Try to recover from the trajectory first
        found = self._search_near_prediction(frame, pred_pos)
        if found:
            rb, conf = found
            self._init_tracking(frame, rb)
            self.last_good_conf = conf
            print(f"[RECOVERED] Ball reacquired, conf={conf:.2f}")
            return rb

        # Grass never stops us immediately. Only stop if repeated misses plus
        # very weak object evidence, otherwise keep coasting.
        if k_box is not None:
            if self.lost_count >= 3 and self._grass_object_gate(frame, k_box):
                self._stop_tracking("Ball region looks like empty grass with weak object evidence.")
                return self.bbox

            if self.lost_count > MAX_LOST_FRAMES:
                self._stop_tracking("Ball lost for too long. Press R to re-select.")
                return self.bbox

            self.track_state = self.TS_COASTING
            return k_box

        if self.lost_count > MAX_LOST_FRAMES:
            self._stop_tracking("Ball lost for too long. Press R to re-select.")
        return self.bbox

    # -------------------------------------------------------------------------
    # Drawing
    # -------------------------------------------------------------------------
    def _draw_bbox(self, frame: np.ndarray, bbox: tuple, state: str):
        if not bbox:
            return

        x, y, w, h = [int(v) for v in bbox]
        if time.time() - self.kick_flash_timer < 0.5:
            color = CLR_KICK
        else:
            color = {
                self.TS_TRACKING: CLR_OK,
                self.TS_COASTING: CLR_COAST,
                self.TS_OCCLUDED: CLR_WARN,
                self.TS_OUT_FRAME: CLR_LOST,
                self.TS_LOST: CLR_LOST,
            }.get(state, CLR_SELECT)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.drawMarker(
            frame, (x + w // 2, y + h // 2), color,
            cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA
        )

    def _draw_trail(self, frame: np.ndarray):
        pts = list(self.trail)
        for i in range(1, len(pts)):
            a = i / len(pts)
            c = tuple(int(v * a) for v in CLR_TRAIL)
            p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
            p2 = (int(pts[i][0]), int(pts[i][1]))
            cv2.line(frame, p1, p2, c, max(1, int(2 * a)), cv2.LINE_AA)

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

        roi = frame[y1:y2, x1:x2].copy()
        pip_w = max(120, int(fw * 0.28))
        aspect = (y2 - y1) / max(x2 - x1, 1)
        pip_h = max(80, min(int(pip_w * aspect), int(fh * 0.35)))
        zoomed = cv2.resize(roi, (pip_w, pip_h))
        m = 10
        tx, ty = fw - pip_w - m, m

        ov = frame.copy()
        cv2.rectangle(ov, (tx - 4, ty - 22), (tx + pip_w + 4, ty + pip_h + 4), (15, 15, 15), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        frame[ty:ty + pip_h, tx:tx + pip_w] = zoomed

        cv2.rectangle(frame, (tx - 2, ty - 2), (tx + pip_w + 2, ty + pip_h + 2), CLR_PIP, 2)
        cv2.putText(frame, "BALL VIEW", (tx, ty - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_PIP, 1, cv2.LINE_AA)

    def _draw_roi_overlay(self, frame: np.ndarray):
        if not (self.roi_pt1 and self.roi_pt2):
            return
        x1 = min(self.roi_pt1[0], self.roi_pt2[0])
        y1 = min(self.roi_pt1[1], self.roi_pt2[1])
        x2 = max(self.roi_pt1[0], self.roi_pt2[0])
        y2 = max(self.roi_pt1[1], self.roi_pt2[1])
        cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_SELECT, 2)

    def _draw_hud(self, frame: np.ndarray):
        fh, fw = frame.shape[:2]

        cv2.rectangle(frame, (0, 0), (270, 110), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {self.fps_value:5.1f}", (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, CLR_OK, 2, cv2.LINE_AA)
        cv2.putText(frame, f"CONF: {self.last_good_conf:0.2f}", (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, CLR_INFO, 1, cv2.LINE_AA)

        pb_txt = "PLAYING" if self.playing else "PAUSED"
        pb_col = CLR_OK if self.playing else (100, 100, 255)
        cv2.putText(frame, f"STAT: {pb_txt}", (8, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, pb_col, 1, cv2.LINE_AA)
        cv2.putText(frame, "OBJ: BALL ONLY", (8, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, CLR_INFO, 1, cv2.LINE_AA)

        hint = "ENTER:start  SPACE:pause  R:retarget  P:PiP  Q:quit"
        cv2.putText(frame, hint, (8, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1, cv2.LINE_AA)

        if self.track_state == self.TS_NONE:
            lines = [
                ("Draw a box around the ball", 0.78, CLR_SELECT),
                ("then press ENTER to start", 0.60, (170, 170, 170)),
            ]
            for i, (msg, sc, col) in enumerate(lines):
                (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
                sx = fw // 2 - tw // 2
                sy = fh // 2 - 30 + i * 40
                cv2.rectangle(frame, (sx - 10, sy - th - 4), (sx + tw + 10, sy + 6), (0, 0, 0), -1)
                cv2.putText(frame, msg, (sx, sy),
                            cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)
            return

        if time.time() - self.kick_flash_timer < 0.5:
            cv2.putText(frame, "[ KICK ! ]", (fw // 2 - 60, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, CLR_KICK, 3, cv2.LINE_AA)

        state_cfg = {
            self.TS_TRACKING: ("[ TRACKING ]", CLR_OK, False),
            self.TS_COASTING: ("[ COASTING ]", CLR_COAST, True),
            self.TS_OCCLUDED: ("[ OCCLUDED ]", CLR_WARN, True),
        }
        if self.track_state in state_cfg:
            label, color, blink = state_cfg[self.track_state]
            if not (blink and (int(time.time() * 3) % 2 == 0)):
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
                sx = fw // 2 - tw // 2
                sy = fh - 26
                cv2.rectangle(frame, (sx - 8, sy - th - 4), (sx + tw + 8, sy + 6), (0, 0, 0), -1)
                cv2.putText(frame, label, (sx, sy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)

        if self.track_state in [self.TS_LOST, self.TS_OUT_FRAME]:
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (fw, fh), (0, 0, 80), -1)
            cv2.addWeighted(ov, 0.20, frame, 0.80, 0, frame)

            msg1 = self.stop_reason if self.stop_reason else "STOPPED"
            msg2 = "Press R to select a new ball"
            for i, msg in enumerate([msg1, msg2]):
                sc = 0.8 if i == 0 else 0.6
                col = CLR_WARN if i == 0 else CLR_LOST
                (mw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
                cv2.putText(frame, msg, (fw // 2 - mw // 2, fh // 2 - 20 + i * 40),
                            cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    def run(self):
        print("=" * 60)
        print(" BALL TRACKER V2")
        print(f" Video : {self.video_path}")
        print("=" * 60)

        while True:
            advance = self.playing and self.track_state not in [self.TS_NONE, self.TS_OUT_FRAME, self.TS_LOST]

            if advance:
                ret, raw = self.cap.read()
                if not ret:
                    print("[END] Video finished.")
                    self._show_end_screen()
                    break

                self.cur_frame = self._resize(raw)
                self.frame_idx += 1

                self._fps_cnt += 1
                now = time.time()
                if now - self._fps_t0 >= 0.4:
                    self.fps_value = self._fps_cnt / (now - self._fps_t0)
                    self._fps_cnt = 0
                    self._fps_t0 = now

                disp_bbox = self._step(self.cur_frame)
            else:
                disp_bbox = self.bbox
                self._fps_t0 = time.time()
                self._fps_cnt = 0

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
            elif key == 13:
                if self.track_state in [self.TS_TRACKING, self.TS_COASTING, self.TS_OCCLUDED]:
                    self.playing = True
                    self._fps_t0 = time.time()
                    self._fps_cnt = 0
                elif self.track_state in [self.TS_LOST, self.TS_OUT_FRAME]:
                    print("[!] Cannot resume. Press R to re-select the ball.")
                else:
                    print("[!] Select the ball first, then press ENTER")
            elif key == ord(' '):
                if self.playing:
                    self.playing = False
                elif self.track_state in [self.TS_TRACKING, self.TS_COASTING, self.TS_OCCLUDED]:
                    self.playing = True
                    self._fps_t0 = time.time()
                    self._fps_cnt = 0
            elif key == ord('r'):
                self.playing = False
                self.track_state = self.TS_NONE
                self._reset_trackers()
                self.roi_pt1 = None
                self.roi_pt2 = None
                self.template_frozen = False
            elif key == ord('p'):
                self.pip_enabled = not self.pip_enabled

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
                    (frame.shape[1] // 2 - tw // 2, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 200), 3, cv2.LINE_AA)
        cv2.imshow(self.WIN, frame)
        cv2.waitKey(3000)


def main():
    if len(sys.argv) < 2:
        print("Usage: python ball_tracker_v2.py <video_file>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"[ERROR] File not found: '{path}'")
        sys.exit(1)

    try:
        BallTracker(path).run()
    except IOError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
