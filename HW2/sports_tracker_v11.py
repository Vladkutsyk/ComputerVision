"""SPORTS VIDEO TRACKER V6
===========================
CSRT + Kalman + Appearance Model + Object Classification + Occlusion Guard

Features
--------
1) Stores an initial reference model for the selected target:
   - HSV histogram
   - ball/player classification
   - player jersey color (upper-body dominant color)
   - shape/aspect reference
2) Detects grass-only / wrong-object drift and stops tracking with a message.
3) Reduces ball drift onto legs/heads/other players by using ball-specific
   appearance constraints and optional circle-based recovery.
4) Keeps the same interaction model as V5.

Dependencies
------------
    pip install opencv-contrib-python numpy

Usage
-----
    python sports_tracker_v6.py <video_file>
"""

from __future__ import annotations

import cv2
import numpy as np
import os
import sys
import time
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────────────
MAX_FRAME_WIDTH = 1280
MAX_LOST_FRAMES = 28
REDETECT_THRESHOLD = 0.52
TEMPLATE_UPDATE_INT = 8
MIN_TEMPLATE_SIZE = 10

# Confidence gates
STOP_CONFIDENCE = 0.34
OCCLUDE_CONFIDENCE = 0.48
TRACK_CONFIDENCE = 0.60

# Search / recovery
SEARCH_MARGIN_MULT = 2.2
BALL_SEARCH_MARGIN_MULT = 3.0
MAX_REDTECT_CANDIDATES = 12

# grass rejection
GRASS_HUE_LOW = 35
GRASS_HUE_HIGH = 85
GRASS_SAT_MIN = 35
GRASS_VAL_MIN = 35
GRASS_HALO_THRESHOLD = 0.40
GRASS_INSIDE_THRESHOLD = 0.68

# When a region looks like grass, only stop if object evidence is also weak.
GRASS_STOP_INSIDE_THRESHOLD = 0.72
GRASS_STOP_OBJECTNESS_THRESHOLD = 0.30
GRASS_STOP_COLOR_THRESHOLD = 0.26


# Shape gates
BALL_MIN_CIRCULARITY = 0.50
BALL_ASPECT_MIN = 0.72
BALL_ASPECT_MAX = 1.38
PLAYER_ASPECT_MIN = 1.18
PLAYER_ASPECT_MAX = 6.50

# BGR colors
CLR_OK = (0, 220, 80)
CLR_WARN = (0, 165, 255)
CLR_COAST = (255, 105, 180)
CLR_LOST = (50, 50, 220)
CLR_SELECT = (0, 220, 255)
CLR_PIP = (0, 220, 220)
CLR_TRAIL = (0, 180, 255)
CLR_FROZEN = (255, 50, 50)
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

# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AppearanceModel:
    label: str = "unknown"  # "ball" or "player"
    bbox_size: Tuple[int, int] = (0, 0)
    aspect_ratio: float = 1.0
    hsv_hist: Optional[np.ndarray] = None
    upper_hist: Optional[np.ndarray] = None
    jersey_bgr: Optional[np.ndarray] = None
    circularity: float = 0.0
    mean_bgr: Optional[np.ndarray] = None


class KalmanPredictor:
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
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
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


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────

class SportsTracker:
    TS_NONE = "NONE"
    TS_TRACKING = "TRACKING"
    TS_OCCLUDED = "OCCLUDED"
    TS_COASTING = "COASTING"
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
        self.kick_flash_timer = 0.0

        self.appearance = AppearanceModel()
        self.last_good_bbox = None
        self.last_good_conf = 0.0

        self.trail = deque(maxlen=60)
        self.pip_enabled = True

        self.drawing = False
        self.roi_pt1 = None
        self.roi_pt2 = None

        self.fps_value = 0.0
        self._fps_t0 = time.time()
        self._fps_cnt = 0

        self.WIN = "Sports Tracker V6"
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        h, w = self.cur_frame.shape[:2]
        cv2.resizeWindow(self.WIN, min(w, 1280), min(h, 720))
        cv2.setMouseCallback(self.WIN, self._mouse_cb)

    # ────────────────────────────────────────────────────────────────────────
    # Utilities
    # ────────────────────────────────────────────────────────────────────────
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

    def _clamp_bbox(self, bbox: Tuple[int, int, int, int], shape) -> Tuple[int, int, int, int]:
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = shape[:2]
        x = max(0, min(x, fw - 2))
        y = max(0, min(y, fh - 2))
        w = max(MIN_TEMPLATE_SIZE, min(w, fw - x))
        h = max(MIN_TEMPLATE_SIZE, min(h, fh - y))
        return x, y, w, h

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

    def _extract_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy(), (x1, y1, x2 - x1, y2 - y1)

    def _build_hist(self, roi_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def _hist_similarity(self, hist_a: Optional[np.ndarray], hist_b: Optional[np.ndarray]) -> float:
        if hist_a is None or hist_b is None:
            return 0.0
        val = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
        # map roughly [-1, 1] -> [0, 1]
        return max(0.0, min(1.0, (val + 1.0) / 2.0))

    def _mean_bgr(self, roi_bgr: np.ndarray, exclude_grass: bool = True) -> np.ndarray:
        if roi_bgr.size == 0:
            return np.array([0, 0, 0], dtype=np.float32)
        if not exclude_grass:
            return roi_bgr.reshape(-1, 3).mean(axis=0).astype(np.float32)
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(
            hsv,
            np.array([GRASS_HUE_LOW, GRASS_SAT_MIN, GRASS_VAL_MIN]),
            np.array([GRASS_HUE_HIGH, 255, 255]),
        )
        mask_non_green = cv2.bitwise_not(mask_green)
        pixels = roi_bgr[mask_non_green > 0]
        if pixels.size == 0:
            return roi_bgr.reshape(-1, 3).mean(axis=0).astype(np.float32)
        return pixels.reshape(-1, 3).mean(axis=0).astype(np.float32)

    def _upper_roi(self, roi_bgr: np.ndarray) -> np.ndarray:
        if roi_bgr.shape[0] < 4:
            return roi_bgr
        cut = max(1, int(roi_bgr.shape[0] * 0.45))
        return roi_bgr[:cut, :].copy()

    def _circularity(self, roi_bgr: np.ndarray) -> float:
        if roi_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 40, 120)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if area <= 1e-3 or peri <= 1e-3:
            return 0.0
        circ = float(4.0 * np.pi * area / (peri * peri))
        return max(0.0, min(1.0, circ))

    def _grass_ratio(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]

        margin_x = max(10, w)
        margin_y = max(10, h)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(fw, x + w + margin_x)
        y2 = min(fh, y + h + margin_y)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([GRASS_HUE_LOW, GRASS_SAT_MIN, GRASS_VAL_MIN]),
            np.array([GRASS_HUE_HIGH, 255, 255]),
        )
        ix1 = x - x1
        iy1 = y - y1
        if 0 <= ix1 < mask.shape[1] and 0 <= iy1 < mask.shape[0]:
            ix2 = min(mask.shape[1], ix1 + w)
            iy2 = min(mask.shape[0], iy1 + h)
            mask[iy1:iy2, ix1:ix2] = 0

        total = max(1, (x2 - x1) * (y2 - y1) - (w * h))
        return float(cv2.countNonZero(mask)) / float(total)

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

    def _objectness_score(self, roi_bgr: np.ndarray) -> float:
        if roi_bgr.size == 0:
            return 0.0
        grass_inside = self._inside_grass_ratio(roi_bgr)
        edge_density = self._edge_density(roi_bgr)
        circularity = self._circularity(roi_bgr)
        edge_score = max(0.0, min(1.0, edge_density / 0.08))
        circularity_score = max(0.0, circularity)
        grass_penalty = max(0.0, 1.0 - grass_inside)
        return float(max(0.0, min(1.0, 0.45 * edge_score + 0.35 * circularity_score + 0.20 * grass_penalty)))

    def _classify_object(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> AppearanceModel:
        roi_pack = self._extract_roi(frame, bbox)
        if roi_pack is None:
            return AppearanceModel()

        roi, bbox = roi_pack
        x, y, w, h = bbox
        aspect = h / max(w, 1)
        mean_bgr = self._mean_bgr(roi, exclude_grass=True)
        full_hist = self._build_hist(roi)
        upper = self._upper_roi(roi)
        upper_hist = self._build_hist(upper) if upper.size else None
        jersey_bgr = self._mean_bgr(upper, exclude_grass=True) if upper.size else mean_bgr
        circularity = self._circularity(roi)

        # Coarse heuristic: ball = small and near-square, player = taller shape.
        small = (w * h) < (frame.shape[0] * frame.shape[1] * 0.012)
        ball_likeness = 0.0
        player_likeness = 0.0

        if 0.70 <= (w / max(h, 1)) <= 1.45:
            ball_likeness += 0.42
        if circularity >= BALL_MIN_CIRCULARITY:
            ball_likeness += 0.36
        if small:
            ball_likeness += 0.22

        if aspect >= PLAYER_ASPECT_MIN:
            player_likeness += 0.50
        if (w * h) >= (frame.shape[0] * frame.shape[1] * 0.008):
            player_likeness += 0.20
        if jersey_bgr is not None and np.linalg.norm(jersey_bgr.astype(np.float32) - np.array([0, 255, 0], dtype=np.float32)) > 55:
            player_likeness += 0.12
        if upper_hist is not None and upper_hist.size > 0:
            player_likeness += 0.18

        label = "ball" if ball_likeness >= player_likeness else "player"
        return AppearanceModel(
            label=label,
            bbox_size=(w, h),
            aspect_ratio=float(aspect),
            hsv_hist=full_hist,
            upper_hist=upper_hist,
            jersey_bgr=jersey_bgr,
            circularity=float(circularity),
            mean_bgr=mean_bgr,
        )

    def _reference_confidence(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        roi_pack = self._extract_roi(frame, bbox)
        if roi_pack is None:
            return 0.0
        roi, _ = roi_pack
        grass_inside = self._inside_grass_ratio(roi)
        return max(0.0, 1.0 - grass_inside)

    def _score_candidate(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        roi_pack = self._extract_roi(frame, bbox)
        if roi_pack is None or self.appearance.hsv_hist is None:
            return 0.0

        roi, bbox = roi_pack
        x, y, w, h = bbox
        aspect = h / max(w, 1)
        grass_inside = self._inside_grass_ratio(roi)
        grass_halo = self._grass_ratio(frame, bbox)
        hist_full = self._build_hist(roi)
        hist_sim = self._hist_similarity(hist_full, self.appearance.hsv_hist)

        shape_score = 0.5
        if self.appearance.label == "ball":
            ar = w / max(h, 1)
            ar_score = 1.0 - min(1.0, abs(ar - 1.0) / 0.55)
            circ = self._circularity(roi)
            circ_score = circ
            size_ref = max(1, self.appearance.bbox_size[0] * self.appearance.bbox_size[1])
            size_now = w * h
            size_score = 1.0 - min(1.0, abs(size_now - size_ref) / float(size_ref) * 1.5)
            shape_score = max(0.0, 0.45 * ar_score + 0.30 * circ_score + 0.25 * size_score)
        else:
            ar = h / max(w, 1)
            ar_score = 1.0 - min(1.0, abs(ar - self.appearance.aspect_ratio) / max(self.appearance.aspect_ratio, 1.0) * 0.55)
            size_ref = max(1, self.appearance.bbox_size[0] * self.appearance.bbox_size[1])
            size_now = w * h
            size_score = 1.0 - min(1.0, abs(size_now - size_ref) / float(size_ref) * 0.9)
            upper_score = 0.5
            if self.appearance.upper_hist is not None and roi.shape[0] > 2:
                upper = self._upper_roi(roi)
                upper_hist = self._build_hist(upper)
                upper_score = self._hist_similarity(upper_hist, self.appearance.upper_hist)
            color_score = 0.5
            if self.appearance.jersey_bgr is not None:
                cdist = float(np.linalg.norm(self._mean_bgr(self._upper_roi(roi), exclude_grass=True) - self.appearance.jersey_bgr))
                color_score = max(0.0, 1.0 - min(1.0, cdist / 120.0))
            shape_score = max(0.0, 0.32 * ar_score + 0.20 * size_score + 0.24 * upper_score + 0.24 * color_score)

        grass_score = max(0.0, 1.0 - max(grass_inside, grass_halo))
        motion_score = 0.5
        pred = self.kalman.predict()
        if pred and self.kalman_bbox_size:
            px, py = pred
            cx, cy = self._center(bbox)
            dist = np.hypot(cx - px, cy - py)
            scale = max(self.kalman_bbox_size) * (2.5 if self.appearance.label == "ball" else 1.8)
            motion_score = max(0.0, 1.0 - min(1.0, dist / max(scale, 1.0)))

        # More strict for ball: prevent drifting onto players.
        if self.appearance.label == "ball":
            candidate = 0.34 * hist_sim + 0.34 * shape_score + 0.22 * motion_score + 0.10 * grass_score
        else:
            candidate = 0.30 * hist_sim + 0.32 * shape_score + 0.18 * motion_score + 0.20 * grass_score

        # Guardrail: strong grass means invalid regardless of the score.
        if grass_inside > GRASS_INSIDE_THRESHOLD:
            candidate *= 0.18
        return float(max(0.0, min(1.0, candidate)))

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
        self.trail.clear()
        self.lost_count = 0
        self.last_good_bbox = None
        self.last_good_conf = 0.0
        self.appearance = AppearanceModel()

    # ────────────────────────────────────────────────────────────────────────
    # Tracking lifecycle
    # ────────────────────────────────────────────────────────────────────────
    def _init_tracking(self, frame: np.ndarray, bbox: tuple):
        bbox = self._clamp_bbox(bbox, frame.shape)
        x, y, w, h = bbox

        self.appearance = self._classify_object(frame, bbox)
        self.tracker = self._make_csrt()
        self.tracker.init(frame, bbox)
        self.bbox = bbox
        self.lost_count = 0
        self.track_state = self.TS_TRACKING
        self.kalman_bbox_size = (w, h)
        self.kalman = KalmanPredictor()
        self.kalman.init(x + w / 2.0, y + h / 2.0)

        self._save_template(frame, bbox, force=True)
        self.trail.clear()
        self.kick_flash_timer = 0.0
        self.stop_reason = ""
        self.last_good_bbox = bbox
        self.last_good_conf = 1.0

        print(f"[TRACKER] Initialised bbox={bbox} class={self.appearance.label} aspect={self.appearance.aspect_ratio:.2f}")

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
                    self._init_tracking(self.cur_frame, (x1, y1, x2 - x1, y2 - y1))

    def _save_template(self, frame: np.ndarray, bbox: tuple, force=False):
        roi_pack = self._extract_roi(frame, bbox)
        if roi_pack is None:
            return
        roi, bbox = roi_pack
        if not force and self._inside_grass_ratio(roi) > 0.45:
            self.template_frozen = True
            return

        self.template_frozen = False
        self.template = roi.copy()
        self.template_fidx = self.frame_idx

    def _redetect_with_template(self, frame: np.ndarray, pred_pos: tuple):
        if self.template is None:
            return None
        g_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g_tmpl = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        if pred_pos and self.kalman_bbox_size:
            px, py = pred_pos
            tw, th = self.kalman_bbox_size
            margin_mult = BALL_SEARCH_MARGIN_MULT if self.appearance.label == "ball" else SEARCH_MARGIN_MULT
            margin = max(tw, th) * margin_mult
            fh, fw = g_frame.shape
            sx1 = max(0, int(px - margin))
            sy1 = max(0, int(py - margin))
            sx2 = min(fw, int(px + margin))
            sy2 = min(fh, int(py + margin))
            search = g_frame[sy1:sy2, sx1:sx2]
            offset = (sx1, sy1)
        else:
            return None

        if search.size == 0:
            return None

        best_val, best_box = 0.0, None
        scales = (0.85, 1.0, 1.15) if self.appearance.label == "player" else (0.80, 0.92, 1.0, 1.08, 1.18)
        for sc in scales:
            nh = max(8, int(g_tmpl.shape[0] * sc))
            nw = max(8, int(g_tmpl.shape[1] * sc))
            if nh >= search.shape[0] or nw >= search.shape[1]:
                continue
            rt = cv2.resize(g_tmpl, (nw, nh))
            res = cv2.matchTemplate(search, rt, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            cand = (ml[0] + offset[0], ml[1] + offset[1], nw, nh)
            conf = self._score_candidate(frame, cand)
            score = 0.45 * max(0.0, float(mv)) + 0.55 * conf
            if score > best_val:
                best_val = score
                best_box = cand

        if best_box is not None and best_val >= REDETECT_THRESHOLD:
            return best_box, best_val
        return None

    def _redetect_ball_circles(self, frame: np.ndarray, pred_pos: tuple):
        if self.appearance.label != "ball" or pred_pos is None or self.kalman_bbox_size is None:
            return None

        px, py = pred_pos
        tw, th = self.kalman_bbox_size
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
        gray = cv2.medianBlur(gray, 5)
        ref_r = int(max(tw, th) / 2)
        min_r = max(2, int(ref_r * 0.65))
        max_r = max(min_r + 1, int(ref_r * 1.45))

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(10, ref_r),
            param1=90,
            param2=18,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is None:
            return None

        circles = np.squeeze(circles, axis=0)
        best = None
        best_score = 0.0
        for c in circles[:MAX_REDTECT_CANDIDATES]:
            cx, cy, r = float(c[0]), float(c[1]), float(c[2])
            x = int(sx1 + cx - r)
            y = int(sy1 + cy - r)
            w = int(2 * r)
            h = int(2 * r)
            cand = (x, y, w, h)
            conf = self._score_candidate(frame, cand)
            # Ball guard: circle-based suggestion must be reasonably strong.
            if conf > best_score:
                best_score = conf
                best = cand

        if best is not None and best_score >= REDETECT_THRESHOLD:
            return best, best_score
        return None

    def _stop_tracking(self, reason: str):
        self.playing = False
        self.track_state = self.TS_LOST
        self.stop_reason = reason
        print(f"[STOP] {reason}")

    def _step(self, frame: np.ndarray):
        if self.track_state == self.TS_NONE:
            return None

        pred_pos = self.kalman.predict()
        ok, raw = self.tracker.update(frame) if self.tracker is not None else (False, None)
        ov = self._overlap(raw, frame.shape) if (ok and raw is not None) else 0.0
        fh, fw = frame.shape[:2]

        # Hard out-of-frame check
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

        anomaly_detected = False
        is_kick_event = False

        # Validate current tracker output against the reference model.
        candidate_conf = 0.0
        if ok and raw is not None:
            candidate_conf = self._score_candidate(frame, raw)

        # Ball-specific protection: do not let it lock onto legs/heads/other players.
        if ok and raw is not None and self.appearance.label == "ball":
            x, y, w, h = raw
            ar = w / max(h, 1)
            size_ref = max(1, self.appearance.bbox_size[0] * self.appearance.bbox_size[1])
            size_now = w * h
            size_ratio = size_now / float(size_ref)
            # too large, too elongated, or low appearance score => likely a wrong object.
            if not (BALL_ASPECT_MIN <= ar <= BALL_ASPECT_MAX) or size_ratio > 2.4 or candidate_conf < STOP_CONFIDENCE:
                ok = False
                anomaly_detected = True

        # Player-specific protection: don't learn grass/occlusion as the player.
        if ok and raw is not None and self.appearance.label == "player":
            grass_inside = self._inside_grass_ratio(self._extract_roi(frame, raw)[0]) if self._extract_roi(frame, raw) else 1.0
            if grass_inside > GRASS_INSIDE_THRESHOLD and candidate_conf < OCCLUDE_CONFIDENCE:
                ok = False
                anomaly_detected = True

        # Trajectory sanity
        if ok and raw is not None and ov > 0.1 and self.track_state in [self.TS_TRACKING, self.TS_COASTING]:
            cx, cy = self._center(raw)
            if pred_pos and self.kalman_bbox_size:
                px, py = pred_pos
                w, h = self.kalman_bbox_size
                dist_pred = np.hypot(cx - px, cy - py)
                max_pred_dist = max(w, h) * (1.1 if self.appearance.label == "ball" else 1.25)
                if dist_pred > max_pred_dist:
                    last_cx, last_cy = self.trail[-1] if self.trail else (cx, cy)
                    dist_last = np.hypot(cx - last_cx, cy - last_cy)
                    max_jump = max(w, h) * (3.8 if self.appearance.label == "ball" else 4.6)
                    if dist_last < max_jump:
                        is_kick_event = True
                        print(f"[KICK DETECTED] Vector snap off-path by {dist_pred:.1f}px")
                    else:
                        anomaly_detected = True
                        print(f"[ANOMALY] Teleportation detected. Dist={dist_pred:.1f}px")

        # A: good visual tracking
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

            if (self.frame_idx - self.template_fidx) >= TEMPLATE_UPDATE_INT:
                self._save_template(frame, raw)

            # state naming: strong match => tracking, weaker but acceptable => occluded
            self.track_state = self.TS_TRACKING if candidate_conf >= TRACK_CONFIDENCE else self.TS_OCCLUDED
            self.trail.append((cx, cy))
            return raw

        # B: attempt recovery while coasting with Kalman prediction.
        self.lost_count += 1
        k_box = None
        if pred_pos and self.kalman_bbox_size:
            pw, ph = self.kalman_bbox_size
            k_box = (int(pred_pos[0] - pw / 2), int(pred_pos[1] - ph / 2), int(pw), int(ph))
            self.trail.append((pred_pos[0], pred_pos[1]))

        # Try ball-specific circle recovery first, then template recovery.
        found = self._redetect_ball_circles(frame, pred_pos)
        if found is None:
            found = self._redetect_with_template(frame, pred_pos)

        if found:
            rb, conf = found
            self._init_tracking(frame, rb)
            self.last_good_conf = conf
            print(f"[RECOVERED] Target found. conf={conf:.2f} class={self.appearance.label}")
            return rb

        # If the search region is mostly grass, stop only when the region also
        # lacks real object evidence. This keeps ball-on-grass recoverable.
        if pred_pos:
            probe_box = k_box if k_box else self.bbox
            if probe_box is not None:
                roi_pack = self._extract_roi(frame, probe_box)
                if roi_pack is not None:
                    roi, _ = roi_pack
                    grass_inside = self._inside_grass_ratio(roi)
                    objectness = self._objectness_score(roi)
                    color_signal = self._reference_confidence(frame, probe_box)
                    if (grass_inside >= GRASS_STOP_INSIDE_THRESHOLD and
                        objectness <= GRASS_STOP_OBJECTNESS_THRESHOLD and
                        color_signal <= GRASS_STOP_COLOR_THRESHOLD and
                        self.lost_count >= 2):
                        self._stop_tracking("Grass dominates and object evidence is too weak.")
                        return self.bbox

        if self.lost_count > MAX_LOST_FRAMES:
            self._stop_tracking("Target lost too long. Press R to re-select.")
            return self.bbox

        if k_box is not None:
            self.track_state = self.TS_COASTING
            return k_box

        self._stop_tracking("Target lost.")
        return self.bbox

    # ────────────────────────────────────────────────────────────────────────
    # Drawing
    # ────────────────────────────────────────────────────────────────────────
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
        c = max(6, min(w, h) // 4)
        for (px, py), (dx, dy) in [
            ((x, y), (1, 1)), ((x + w, y), (-1, 1)),
            ((x, y + h), (1, -1)), ((x + w, y + h), (-1, -1)),
        ]:
            cv2.line(frame, (px, py), (px + dx * c, py), color, 3)
            cv2.line(frame, (px, py), (px, py + dy * c), color, 3)

        cv2.drawMarker(frame, (x + w // 2, y + h // 2), color, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)

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

        pip_color = CLR_FROZEN if self.template_frozen else CLR_PIP
        cv2.rectangle(frame, (tx - 2, ty - 2), (tx + pip_w + 2, ty + pip_h + 2), pip_color, 2)
        cv2.putText(frame, "TARGET VIEW", (tx, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pip_color, 1, cv2.LINE_AA)

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

        cv2.rectangle(frame, (0, 0), (270, 122), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {self.fps_value:5.1f}", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, CLR_OK, 2, cv2.LINE_AA)
        cv2.putText(frame, f"OBJ: {self.appearance.label.upper()}", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, CLR_INFO, 1, cv2.LINE_AA)
        tmpl_txt = "TMPL: FROZEN" if self.template_frozen else "TMPL: LEARNING"
        tmpl_col = CLR_FROZEN if self.template_frozen else CLR_OK
        cv2.putText(frame, tmpl_txt, (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.50, tmpl_col, 1, cv2.LINE_AA)
        pb_txt = "PLAYING" if self.playing else "PAUSED"
        pb_col = CLR_OK if self.playing else (100, 100, 255)
        cv2.putText(frame, f"STAT: {pb_txt}", (8, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.50, pb_col, 1, cv2.LINE_AA)
        cv2.putText(frame, f"CONF: {self.last_good_conf:0.2f}", (130, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, CLR_INFO, 1, cv2.LINE_AA)

        hint = "ENTER:start  SPACE:pause  R:retarget  P:PiP  Q:quit"
        cv2.putText(frame, hint, (8, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1, cv2.LINE_AA)

        if self.track_state == self.TS_NONE:
            lines = [
                ("Draw a box around the target", 0.78, CLR_SELECT),
                ("then press ENTER to start", 0.60, (170, 170, 170)),
            ]
            for i, (msg, sc, col) in enumerate(lines):
                (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
                sx = fw // 2 - tw // 2
                sy = fh // 2 - 30 + i * 40
                cv2.rectangle(frame, (sx - 10, sy - th - 4), (sx + tw + 10, sy + 6), (0, 0, 0), -1)
                cv2.putText(frame, msg, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)
            return

        if time.time() - self.kick_flash_timer < 0.5:
            cv2.putText(frame, "[ KICK ! ]", (fw // 2 - 60, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, CLR_KICK, 3, cv2.LINE_AA)

        state_cfg = {
            self.TS_TRACKING: ("[  TRACKING  ]", CLR_OK, False),
            self.TS_COASTING: ("[ COASTING (Trajectory) ]", CLR_COAST, True),
            self.TS_OCCLUDED: ("[ OCCLUDED - Guarded ]", CLR_WARN, True),
        }
        if self.track_state in state_cfg:
            label, color, blink = state_cfg[self.track_state]
            if not (blink and (int(time.time() * 3) % 2 == 0)):
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
                sx = fw // 2 - tw // 2
                sy = fh - 26
                cv2.rectangle(frame, (sx - 8, sy - th - 4), (sx + tw + 8, sy + 6), (0, 0, 0), -1)
                cv2.putText(frame, label, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)

        if self.track_state in [self.TS_LOST, self.TS_OUT_FRAME]:
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (fw, fh), (0, 0, 80), -1)
            cv2.addWeighted(ov, 0.20, frame, 0.80, 0, frame)

            msg1 = self.stop_reason if self.stop_reason else ("STOPPED: Target Out Of Frame" if self.track_state == self.TS_OUT_FRAME else "STOPPED: Target Lost")
            msg2 = "Press R to select a new target"
            for i, msg in enumerate([msg1, msg2]):
                sc = 0.8 if i == 0 else 0.6
                col = CLR_WARN if i == 0 else CLR_LOST
                (mw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
                cv2.putText(frame, msg, (fw // 2 - mw // 2, fh // 2 - 20 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)

        if self.appearance.label == "player" and self.appearance.jersey_bgr is not None:
            jersey = np.clip(self.appearance.jersey_bgr, 0, 255).astype(int)
            cv2.rectangle(frame, (fw - 145, 10), (fw - 10, 50), (0, 0, 0), -1)
            cv2.rectangle(frame, (fw - 140, 15), (fw - 100, 45), tuple(int(v) for v in jersey[::-1]), -1)
            cv2.putText(frame, "JERSEY", (fw - 95, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_INFO, 1, cv2.LINE_AA)

    # ────────────────────────────────────────────────────────────────────────
    # Main loop
    # ────────────────────────────────────────────────────────────────────────
    def run(self):
        print("=" * 60)
        print(" SPORTS TRACKER V6")
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
                    print("[!] Cannot resume. Target lost/out of frame. Press R to re-select.")
                else:
                    print("[!] Select a target first, then press ENTER")
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
                self.stop_reason = ""
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
        cv2.putText(frame, msg, (frame.shape[1] // 2 - tw // 2, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 200), 3, cv2.LINE_AA)
        cv2.imshow(self.WIN, frame)
        cv2.waitKey(3000)


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
