"""
SPORTS VIDEO TRACKER V8
========================
Built on V6. New in V8:

  1. Adaptive CSRT — per-type profiles (ball / player) + runtime adaptation
                     based on current speed class (fast/normal/static).
                     Tracker is re-initialised with updated params every
                     CSRT_REINIT_INTERVAL frames when tracking is stable.

  2. MotionMap — rolling weighted-velocity history. Trajectory-aware
                 template search: search region is an ellipse stretched
                 along the motion direction instead of a fixed circle.

  3. Target Profile — two new object attributes:
       a) target_type : "ball" | "player" | "unknown"
                        Auto-classified at ROI selection (size + shape).
                        Press B / F / U to override manually.
       b) jersey_hue  : dominant jersey colour hue (0-179) for players.
                        Stored as a dedicated upper-body AppearanceModel.
                        Any CSRT result whose jersey colour does not match
                        is immediately rejected (wrong player guard).

Controls:
    Draw ROI  — LMB drag
    ENTER     — start / resume
    SPACE     — pause / resume
    R         — re-select target
    B/F/U     — force type: Ball / Football-player / Unknown
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

# ── Target type tokens ────────────────────────────────────────────────────────
TARGET_UNKNOWN = "unknown"
TARGET_BALL    = "ball"
TARGET_PLAYER  = "player"

# ── Ball auto-detection thresholds ───────────────────────────────────────────
BALL_AREA_MAX         = 4000   # px² — balls are small
BALL_ASPECT_RANGE     = (0.65, 1.55)  # w/h must be near-square

# ── Jersey colour ─────────────────────────────────────────────────────────────
JERSEY_HUE_TOLERANCE  = 25    # ±degrees for hue match
JERSEY_MIN_SAT        = 45    # ignore grey/white/black pixels
JERSEY_COLOR_THRESH   = 0.32  # min appearance correlation → same team
JERSEY_UPPER_FRAC     = 0.60  # top fraction of bbox used for jersey check

# ── CSRT profiles per target type ────────────────────────────────────────────
_CSRT_COMMON = dict(
    admm_iterations       = 4,
    hog_orientations      = 9,
    scale_step            = 1.02,
    scale_sigma_factor    = 0.25,
    use_channel_weights   = True,
    use_color_names       = True,
    use_gray              = True,
    use_hog               = True,
)
CSRT_PROFILE = {
    TARGET_BALL: dict(**_CSRT_COMMON,
        padding              = 4.2,   # wide context (ball is tiny, bg matters)
        template_size        = 120,
        num_hog_channels_used= 18,
        num_scales           = 48,    # more scales: ball changes apparent size fast
        background_ratio     = 3,
        psr_threshold        = 0.022, # lenient: ball disappears a lot
        gsl_sigma            = 1.2,
    ),
    TARGET_PLAYER: dict(**_CSRT_COMMON,
        padding              = 2.6,
        template_size        = 200,
        num_hog_channels_used= 18,
        num_scales           = 28,
        background_ratio     = 2,
        psr_threshold        = 0.042,
        gsl_sigma            = 1.0,
    ),
    TARGET_UNKNOWN: dict(**_CSRT_COMMON,
        padding              = 3.0,
        template_size        = 175,
        num_hog_channels_used= 18,
        num_scales           = 33,
        background_ratio     = 2,
        psr_threshold        = 0.035,
        gsl_sigma            = 1.0,
    ),
}

# Speed-class adaptive deltas (applied to psr_threshold / padding)
#   fast   : speed > 18 px/frame
#   normal : 4..18
#   static : < 4
SPEED_FAST_PSR_DELTA    = -0.012  # more lenient when target flies
SPEED_FAST_PAD_DELTA    = +0.60
SPEED_STATIC_PSR_DELTA  = +0.018  # stricter when target is still
SPEED_CLASS_FAST        = 18.0
SPEED_CLASS_STATIC      = 4.0

# Re-init CSRT with updated params every N stable tracking frames
CSRT_REINIT_INTERVAL = 30

# ── General tracking ─────────────────────────────────────────────────────────
MAX_LOST_FRAMES      = 45
TEMPLATE_UPDATE_INT  = 8
MAX_FRAME_WIDTH      = 1280
TRAJECTORY_TOLERANCE = 1.15
MAX_KICK_DISTANCE    = 4.5

# ── Appearance ────────────────────────────────────────────────────────────────
APPEARANCE_BINS          = 32
APPEARANCE_THRESHOLD     = 0.42
APPEARANCE_UPDATE_THRESH = 0.68
APPEARANCE_AREA_SPIKE    = 1.80

# ── GrabCut ───────────────────────────────────────────────────────────────────
GRABCUT_ITERS          = 3
GRABCUT_BLOB_MIN_RATIO = 0.10
GRABCUT_MIN_SIDE       = 24

# ── MotionMap ─────────────────────────────────────────────────────────────────
MOTION_HISTORY         = 18   # frames kept in velocity buffer
MOTION_CONF_THR        = 5.0  # min speed (px/frame) to use directional search
MOTION_REDETECT_THR    = 0.46 # template match confidence for trajectory search

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
CLR_MULTIOBJ = (0,   100, 255)
CLR_JERSEY   = (255, 140,   0)   # jersey mismatch warning


# ═════════════════════════════════════════════════════════════════════════════
#  KALMAN PREDICTOR  (6-state, from V6)
# ═════════════════════════════════════════════════════════════════════════════

class KalmanPredictor:
    """Constant-acceleration Kalman: state=[cx,cy,vx,vy,ax,ay], meas=[cx,cy]."""

    MAX_SPEED_PLAYER = 80   # px/frame
    MAX_SPEED_BALL   = 160

    def __init__(self, target_type: str = TARGET_UNKNOWN):
        self.max_speed = (self.MAX_SPEED_BALL if target_type == TARGET_BALL
                          else self.MAX_SPEED_PLAYER)
        dt = 1.0
        kf = cv2.KalmanFilter(6, 2)
        kf.transitionMatrix = np.array([
            [1,0,dt,0, .5*dt*dt,0],
            [0,1,0, dt,0,       .5*dt*dt],
            [0,0,1, 0, dt,      0],
            [0,0,0, 1, 0,       dt],
            [0,0,0, 0, 1,       0],
            [0,0,0, 0, 0,       1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.zeros((2,6), dtype=np.float32)
        kf.measurementMatrix[0,0] = 1.0
        kf.measurementMatrix[1,1] = 1.0
        kf.processNoiseCov = np.diag(
            [1e-3, 1e-3, 5e-2, 5e-2, 5e-1, 5e-1]).astype(np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
        kf.errorCovPost        = np.eye(6, dtype=np.float32) * 0.1
        self.kf = kf
        self.initialized = False

    def init(self, cx, cy, vx=0., vy=0.):
        s = np.array([[cx],[cy],[vx],[vy],[0.],[0.]], dtype=np.float32)
        self.kf.statePre  = s.copy()
        self.kf.statePost = s.copy()
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 0.1
        self.initialized = True

    def predict(self):
        if not self.initialized: return None
        p = self.kf.predict()
        return float(p[0,0]), float(p[1,0])

    def correct(self, cx, cy):
        if not self.initialized:
            self.init(cx, cy); return
        self.kf.correct(np.array([[cx],[cy]], dtype=np.float32))
        s = self.kf.statePost
        speed = float(np.hypot(s[2,0], s[3,0]))
        if speed > self.max_speed:
            r = self.max_speed / speed
            s[2,0] *= r; s[3,0] *= r
            self.kf.statePost = s

    def get_velocity(self):
        if not self.initialized: return 0., 0.
        s = self.kf.statePost
        return float(s[2,0]), float(s[3,0])


# ═════════════════════════════════════════════════════════════════════════════
#  APPEARANCE MODEL  (V6, unchanged)
# ═════════════════════════════════════════════════════════════════════════════

class AppearanceModel:
    def __init__(self, bins: int = APPEARANCE_BINS):
        self.bins = bins
        self.hist = None

    def _make_hist(self, patch):
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv],[0,1],None,[self.bins,self.bins],[0,180,0,256])
        cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
        return h

    def build(self, patch):
        if patch.size == 0: return
        self.hist = self._make_hist(patch)

    def score(self, patch) -> float:
        if self.hist is None or patch.size == 0: return 1.0
        h    = self._make_hist(patch)
        corr = cv2.compareHist(self.hist, h, cv2.HISTCMP_CORREL)
        return max(0.0, float(corr))

    def soft_update(self, patch, score):
        if score >= APPEARANCE_UPDATE_THRESH: self.build(patch)


# ═════════════════════════════════════════════════════════════════════════════
#  MOTION MAP  — rolling velocity history for trajectory-aware search
# ═════════════════════════════════════════════════════════════════════════════

class MotionMap:
    """
    Stores the last N velocity observations and computes a recency-weighted
    smoothed velocity.  Used to bias template-matching search towards the
    predicted motion direction, which dramatically reduces false matches
    when players from the same team are nearby.
    """

    def __init__(self, history: int = MOTION_HISTORY):
        self._vx = deque(maxlen=history)
        self._vy = deque(maxlen=history)

    def update(self, vx: float, vy: float):
        self._vx.append(vx)
        self._vy.append(vy)

    def smooth_velocity(self):
        if not self._vx: return 0., 0.
        n = len(self._vx)
        w = np.linspace(0.4, 1.0, n, dtype=np.float64)
        w /= w.sum()
        vx = float(np.dot(list(self._vx), w))
        vy = float(np.dot(list(self._vy), w))
        return vx, vy

    def speed(self) -> float:
        vx, vy = self.smooth_velocity()
        return float(np.hypot(vx, vy))

    def search_region(self, cx: float, cy: float,
                      base_r: float, fw: int, fh: int):
        """
        Returns (x1,y1,x2,y2) search region for template matching.
        Along motion vector: stretched up to 3× base_r.
        Perpendicular: base_r (not stretched).
        Falls back to a circle when speed is low.
        """
        vx, vy = self.smooth_velocity()
        spd    = float(np.hypot(vx, vy))

        if spd < MOTION_CONF_THR:
            # Low speed → symmetric circle
            x1 = int(cx - base_r); y1 = int(cy - base_r)
            x2 = int(cx + base_r); y2 = int(cy + base_r)
        else:
            nx, ny   = vx/spd, vy/spd
            fwd_ext  = min(base_r * 2.5, spd * 4)
            perp_ext = base_r * 0.85
            # Centre of search is shifted forward along motion
            sc_x = cx + nx * fwd_ext * 0.4
            sc_y = cy + ny * fwd_ext * 0.4
            # Bounding box of the oriented ellipse
            bw = abs(nx)*fwd_ext + abs(ny)*perp_ext
            bh = abs(ny)*fwd_ext + abs(nx)*perp_ext
            x1 = int(sc_x - bw); y1 = int(sc_y - bh)
            x2 = int(sc_x + bw); y2 = int(sc_y + bh)

        return (max(0,x1), max(0,y1), min(fw,x2), min(fh,y2))


# ═════════════════════════════════════════════════════════════════════════════
#  SPORTS TRACKER
# ═════════════════════════════════════════════════════════════════════════════

class SportsTracker:
    TS_NONE      = "NONE"
    TS_TRACKING  = "TRACKING"
    TS_OCCLUDED  = "OCCLUDED"
    TS_COASTING  = "COASTING"
    TS_OUT_FRAME = "OUT_FRAME"
    TS_LOST      = "LOST"

    # ── init ─────────────────────────────────────────────────────────────────

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open: {video_path}")
        ok, first = self.cap.read()
        if not ok: raise IOError("Cannot read first frame")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cur_frame = self._resize(first)

        self.playing   = False
        self.frame_idx = 0

        # tracker core
        self.tracker          = None
        self.bbox             = None
        self.track_state      = self.TS_NONE
        self.lost_count       = 0

        # Kalman (will be re-created with correct type on _init_tracking)
        self.kalman           = KalmanPredictor()
        self.kalman_bbox_size = None

        # template
        self.template         = None
        self.template_fidx    = 0
        self.template_frozen  = False
        self.kick_flash_timer = 0.0

        # appearance
        self.appearance        = AppearanceModel()
        self.jersey_appearance = AppearanceModel()   # upper-body only
        self.app_score         = 1.0
        self.multi_obj_flash_t = 0.0
        self.jersey_warn_t     = 0.0   # timestamp of last jersey mismatch

        # ── V8: target profile ───────────────────────────────────────────
        self.target_type      = TARGET_UNKNOWN    # "ball" | "player" | "unknown"
        self.jersey_hue       = None              # int 0-179 or None
        self.jersey_color_name = ""
        self.jersey_bgr       = (128,128,128)     # for HUD swatch

        # ── V8: adaptive CSRT ────────────────────────────────────────────
        self._csrt_params      = CSRT_PROFILE[TARGET_UNKNOWN].copy()
        self._speed_class      = "normal"         # "fast"|"normal"|"static"

        # ── V8: motion map ───────────────────────────────────────────────
        self.motion_map        = MotionMap()

        # trail
        self.trail = deque(maxlen=60)

        # PiP
        self.pip_enabled = True

        # ROI drawing
        self.drawing = False
        self.roi_pt1 = None
        self.roi_pt2 = None

        # FPS
        self.fps_value = 0.0
        self._fps_t0   = time.time()
        self._fps_cnt  = 0

        self.WIN = "Sports Tracker V8"
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        h, w = self.cur_frame.shape[:2]
        cv2.resizeWindow(self.WIN, min(w,1280), min(h,720))
        cv2.setMouseCallback(self.WIN, self._mouse_cb)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _resize(self, frame):
        h, w = frame.shape[:2]
        if w > MAX_FRAME_WIDTH:
            return cv2.resize(frame,(MAX_FRAME_WIDTH, int(h*MAX_FRAME_WIDTH/w)))
        return frame

    def _win_to_frame(self, wx, wy):
        try:
            _, _, dw, dh = cv2.getWindowImageRect(self.WIN)
            if dw > 0 and dh > 0 and self.cur_frame is not None:
                fh, fw = self.cur_frame.shape[:2]
                return int(wx*fw/dw), int(wy*fh/dh)
        except Exception: pass
        return wx, wy

    @staticmethod
    def _center(bbox):
        x,y,w,h = bbox; return x+w/2, y+h/2

    @staticmethod
    def _overlap(bbox, fshape):
        x,y,w,h = [int(v) for v in bbox]
        fh,fw   = fshape[:2]
        ix1,iy1 = max(0,x), max(0,y)
        ix2,iy2 = min(fw,x+w), min(fh,y+h)
        if ix2<=ix1 or iy2<=iy1: return 0.0
        return (ix2-ix1)*(iy2-iy1)/max(w*h,1)

    def _get_patch(self, frame, bbox):
        x,y,w,h = [int(v) for v in bbox]
        fh,fw   = frame.shape[:2]
        x1,y1   = max(0,x), max(0,y)
        x2,y2   = min(fw,x+w), min(fh,y+h)
        if x2<=x1 or y2<=y1:
            return np.zeros((1,1,3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]

    # ── V8: target type classification ───────────────────────────────────────

    def _classify_target(self, frame, bbox) -> str:
        x,y,w,h = [int(v) for v in bbox]
        area    = w * h
        aspect  = w / max(h, 1)

        # Size + aspect check for ball
        if (area < BALL_AREA_MAX and
                BALL_ASPECT_RANGE[0] < aspect < BALL_ASPECT_RANGE[1]):
            patch = self._get_patch(frame, bbox)
            gray  = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            gray  = cv2.GaussianBlur(gray, (5,5), 0)
            r_min = max(3, min(w,h)//5)
            r_max = max(r_min+1, max(w,h)//2)
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1,
                minDist=min(w,h)//2,
                param1=40, param2=12,
                minRadius=r_min, maxRadius=r_max)
            if circles is not None:
                return TARGET_BALL

        # Player: tall, large
        if area > BALL_AREA_MAX or aspect < 0.55 or aspect > 2.2:
            return TARGET_PLAYER

        return TARGET_UNKNOWN

    # ── V8: jersey colour ────────────────────────────────────────────────────

    @staticmethod
    def _hue_to_name(hue: int) -> str:
        if hue is None: return "unknown"
        h = hue % 180
        if h < 8  or h > 167: return "red"
        if h < 22:             return "orange"
        if h < 38:             return "yellow"
        if h < 85:             return "green"
        if h < 100:            return "cyan"
        if h < 130:            return "blue"
        if h < 148:            return "purple"
        if h < 167:            return "pink"
        return "red"

    @staticmethod
    def _hue_to_bgr(hue: int):
        if hue is None: return (128,128,128)
        px  = np.array([[[hue, 210, 210]]], dtype=np.uint8)
        bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)
        return tuple(int(v) for v in bgr[0,0])

    def _detect_jersey_hue(self, patch) -> int | None:
        """
        Extract dominant jersey hue from the upper JERSEY_UPPER_FRAC of patch.
        Excludes grass green, near-black shadows, and near-white pixels.
        Returns hue int (0-179) or None if no colour found.
        """
        if patch.size == 0: return None
        h = patch.shape[0]
        upper = patch[:max(1, int(h * JERSEY_UPPER_FRAC)), :]
        if upper.size == 0: return None

        hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

        # Exclusion masks
        grass  = (H > 35) & (H < 85) & (S > 40)
        dark   = V < 40
        bright = S < 30
        exclude = grass | dark | bright

        hues = H[~exclude & (S > JERSEY_MIN_SAT)]
        if len(hues) < 40: return None

        # Circular histogram smoothing to find dominant hue
        hist = np.bincount(hues.astype(np.int32), minlength=180).astype(np.float32)
        smoothed = np.zeros(180, dtype=np.float32)
        k = 10
        for offset in range(-k, k+1):
            smoothed += np.roll(hist, offset)
        dominant = int(np.argmax(smoothed))
        return dominant

    def _check_jersey_color(self, frame, bbox) -> bool:
        """
        Returns True if the current bbox looks like it has the expected
        jersey colour.  Only active for players with a known jersey_hue.
        Uses the dedicated jersey_appearance model (upper-body H+S histogram).
        """
        if self.target_type != TARGET_PLAYER or self.jersey_hue is None:
            return True
        patch = self._get_patch(frame, bbox)
        upper = patch[:max(1, int(patch.shape[0]*JERSEY_UPPER_FRAC)), :]
        score = self.jersey_appearance.score(upper)
        return score >= JERSEY_COLOR_THRESH

    # ── V8: adaptive CSRT ────────────────────────────────────────────────────

    def _compute_csrt_params(self) -> dict:
        """Build the ideal CSRT params dict for the current state."""
        base   = CSRT_PROFILE[self.target_type].copy()
        speed  = self.motion_map.speed()

        if speed > SPEED_CLASS_FAST:
            base["psr_threshold"] = max(0.010,
                base["psr_threshold"] + SPEED_FAST_PSR_DELTA)
            base["padding"] = base["padding"] + SPEED_FAST_PAD_DELTA
            new_class = "fast"
        elif speed < SPEED_CLASS_STATIC:
            base["psr_threshold"] = min(0.080,
                base["psr_threshold"] + SPEED_STATIC_PSR_DELTA)
            new_class = "static"
        else:
            new_class = "normal"

        self._speed_class = new_class
        return base

    def _make_csrt_from_params(self, params: dict):
        p = cv2.TrackerCSRT_Params()
        for k, v in params.items():
            if hasattr(p, k):
                try: setattr(p, k, v)
                except Exception: pass
        return cv2.TrackerCSRT_create(p)

    def _maybe_reinit_csrt(self, frame):
        """
        Re-initialise CSRT with updated adaptive params every
        CSRT_REINIT_INTERVAL frames (only when tracking is stable).
        """
        new_params = self._compute_csrt_params()

        # Check if anything changed significantly
        psr_diff = abs(new_params.get("psr_threshold",0) -
                       self._csrt_params.get("psr_threshold",0))
        pad_diff = abs(new_params.get("padding",0) -
                       self._csrt_params.get("padding",0))

        if psr_diff < 0.003 and pad_diff < 0.15:
            return   # not worth re-init

        print(f"[CSRT-ADAPT] type={self.target_type}  speed={self._speed_class}"
              f"  psr={new_params['psr_threshold']:.3f}"
              f"  pad={new_params['padding']:.2f}")

        self._csrt_params = new_params
        self.tracker = self._make_csrt_from_params(new_params)
        self.tracker.init(frame, self.bbox)

    # ── tracker init / core ───────────────────────────────────────────────────

    def _init_tracking(self, frame, bbox):
        x,y,w,h = [int(v) for v in bbox]
        fh,fw   = frame.shape[:2]
        x = max(0,min(x,fw-2)); y = max(0,min(y,fh-2))
        w = max(10,min(w,fw-x)); h = max(10,min(h,fh-y))
        bbox = (x,y,w,h)

        # ── Classify target (only on first init or manual override) ──────
        if self.track_state == self.TS_NONE:
            self.target_type = self._classify_target(frame, bbox)
            print(f"[CLASSIFY] type={self.target_type}")

        # ── Jersey colour (players only) ─────────────────────────────────
        patch = self._get_patch(frame, bbox)
        if self.target_type == TARGET_PLAYER:
            upper = patch[:max(1,int(patch.shape[0]*JERSEY_UPPER_FRAC)), :]
            self.jersey_hue        = self._detect_jersey_hue(patch)
            self.jersey_color_name = self._hue_to_name(self.jersey_hue)
            self.jersey_bgr        = self._hue_to_bgr(self.jersey_hue)
            self.jersey_appearance.build(upper)
            print(f"[JERSEY] hue={self.jersey_hue}  name={self.jersey_color_name}")
        else:
            self.jersey_hue = None
            self.jersey_color_name = ""
            self.jersey_bgr = (128,128,128)

        # ── Kalman (type-aware max speed) ─────────────────────────────────
        self.kalman           = KalmanPredictor(self.target_type)
        self.kalman.init(x+w/2, y+h/2)
        self.kalman_bbox_size = (w,h)

        # ── CSRT ─────────────────────────────────────────────────────────
        self._csrt_params = self._compute_csrt_params()
        self.tracker      = self._make_csrt_from_params(self._csrt_params)
        self.tracker.init(frame, bbox)

        # ── Appearance ───────────────────────────────────────────────────
        self.appearance.build(patch)
        self.app_score = 1.0

        # ── Rest ─────────────────────────────────────────────────────────
        self.bbox              = bbox
        self.lost_count        = 0
        self.track_state       = self.TS_TRACKING
        self.motion_map        = MotionMap()
        self.trail.clear()
        self.kick_flash_timer  = 0.0
        self.multi_obj_flash_t = 0.0
        self.jersey_warn_t     = 0.0
        self._save_template(frame, bbox, force=True)
        print(f"[TRACKER] Init  bbox={bbox}")

    # ── grass freeze (V5) ────────────────────────────────────────────────────

    def _is_open_grass(self, frame, bbox) -> bool:
        x,y,w,h = [int(v) for v in bbox]
        fh,fw   = frame.shape[:2]
        mx,my   = max(10,w), max(10,h)
        x1,y1   = max(0,x-mx), max(0,y-my)
        x2,y2   = min(fw,x+w+mx), min(fh,y+h+my)
        roi     = frame[y1:y2, x1:x2]
        if roi.size == 0: return False
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35,40,40]), np.array([85,255,255]))
        mask[y-y1:y-y1+h, x-x1:x-x1+w] = 0
        total = (x2-x1)*(y2-y1) - w*h
        return total > 0 and cv2.countNonZero(mask)/total > 0.40

    def _save_template(self, frame, bbox, force=False):
        if not force and not self._is_open_grass(frame, bbox):
            self.template_frozen = True; return
        self.template_frozen = False
        x,y,w,h = [int(v) for v in bbox]
        fh,fw   = frame.shape[:2]
        x1,y1   = max(0,x), max(0,y)
        x2,y2   = min(fw,x+w), min(fh,y+h)
        if x2>x1 and y2>y1:
            self.template      = frame[y1:y2,x1:x2].copy()
            self.template_fidx = self.frame_idx

    # ── V8: trajectory-aware template search ─────────────────────────────────

    def _trajectory_search(self, frame, pred_pos):
        """
        Template matching inside a direction-biased search region.
        Replaces the old fixed-circle _redetect.
        Returns (bbox, confidence) or None.
        """
        if self.template is None or pred_pos is None:
            return None
        g_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g_tmpl  = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        pw = g_tmpl.shape[1]; ph = g_tmpl.shape[0]
        fh, fw  = g_frame.shape

        base_r  = max(pw, ph) * 2.2
        x1,y1,x2,y2 = self.motion_map.search_region(
            pred_pos[0], pred_pos[1], base_r, fw, fh)

        search  = g_frame[y1:y2, x1:x2]
        offset  = (x1, y1)

        best_val, best_box = 0.0, None
        for sc in (0.85, 1.0, 1.15):
            nh = max(8, int(ph*sc)); nw = max(8, int(pw*sc))
            if nh >= search.shape[0] or nw >= search.shape[1]: continue
            rt  = cv2.resize(g_tmpl, (nw,nh))
            res = cv2.matchTemplate(search, rt, cv2.TM_CCOEFF_NORMED)
            _,mv,_,ml = cv2.minMaxLoc(res)
            if mv > best_val:
                best_val = mv
                best_box = (ml[0]+offset[0], ml[1]+offset[1], nw, nh)

        if best_val >= MOTION_REDETECT_THR and best_box:
            # Additional jersey-colour guard before accepting re-detection
            if not self._check_jersey_color(frame, best_box):
                print(f"[TRAJ-SEARCH] Found but jersey mismatch — skipping")
                return None
            return best_box, best_val
        return None

    # ── GrabCut separation (V6) ───────────────────────────────────────────────

    def _grabcut_separate(self, frame, bbox, pred_pos):
        x,y,w,h = [int(v) for v in bbox]
        fh,fw   = frame.shape[:2]
        x = max(1,min(x,fw-2)); y = max(1,min(y,fh-2))
        w = min(w,fw-x-1);      h = min(h,fh-y-1)
        if w < GRABCUT_MIN_SIDE or h < GRABCUT_MIN_SIDE: return None

        patch   = frame[y:y+h, x:x+w].copy()
        mask    = np.zeros(patch.shape[:2], dtype=np.uint8)
        bgd_m   = np.zeros((1,65), np.float64)
        fgd_m   = np.zeros((1,65), np.float64)
        mg      = max(2, min(w,h)//10)
        rect    = (mg, mg, w-2*mg, h-2*mg)
        try:
            cv2.grabCut(patch, mask, rect, bgd_m, fgd_m,
                        GRABCUT_ITERS, cv2.GC_INIT_WITH_RECT)
        except Exception: return None

        fg  = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),255,0).astype(np.uint8)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        fg  = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  ker)
        fg  = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, ker)

        contours,_ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area   = w*h*GRABCUT_BLOB_MIN_RATIO
        blobs      = [c for c in contours if cv2.contourArea(c) > min_area]
        if len(blobs) < 2: return None

        print(f"[GRABCUT] {len(blobs)} blobs → selecting best")
        diag       = float(np.hypot(w,h))+1e-6
        best_blob, best_score = None, -1.0
        for blob in blobs:
            bx,by,bw,bh = cv2.boundingRect(blob)
            if bw<6 or bh<6: continue
            bcx = x+bx+bw/2; bcy = y+by+bh/2
            bp  = self._get_patch(frame,(x+bx,y+by,bw,bh))
            app = self.appearance.score(bp)
            prx = (1.0 - min(float(np.hypot(bcx-pred_pos[0],bcy-pred_pos[1]))/diag,1.0)
                   if pred_pos else 0.5)
            # Jersey guard on each blob
            jer = 1.0 if self._check_jersey_color(frame,(x+bx,y+by,bw,bh)) else 0.2
            combined = app*0.55 + prx*0.25 + jer*0.20
            if combined > best_score:
                best_score = combined; best_blob = (x+bx, y+by, bw, bh)

        if best_blob and best_score > 0.25:
            print(f"[GRABCUT] Chose {best_blob}  score={best_score:.3f}")
            return best_blob
        return None

    # ── mouse ─────────────────────────────────────────────────────────────────

    def _mouse_cb(self, event, x, y, flags, param):
        fx,fy = self._win_to_frame(x,y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True; self.roi_pt1 = self.roi_pt2 = (fx,fy)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.roi_pt2 = (fx,fy)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False; self.roi_pt2 = (fx,fy)
            if self.cur_frame is not None:
                x1=min(self.roi_pt1[0],self.roi_pt2[0])
                y1=min(self.roi_pt1[1],self.roi_pt2[1])
                x2=max(self.roi_pt1[0],self.roi_pt2[0])
                y2=max(self.roi_pt1[1],self.roi_pt2[1])
                if (x2-x1)>8 and (y2-y1)>8:
                    self._init_tracking(self.cur_frame,(x1,y1,x2-x1,y2-y1))

    # ── STEP ─────────────────────────────────────────────────────────────────

    def _step(self, frame):
        if self.track_state == self.TS_NONE:
            return None

        # ── 1. Kalman predict + motion map ───────────────────────────────
        pred_pos = self.kalman.predict()
        vx,vy    = self.kalman.get_velocity()
        self.motion_map.update(vx, vy)

        # ── 2. Adaptive CSRT re-init (stable tracking only) ──────────────
        if (self.frame_idx % CSRT_REINIT_INTERVAL == 0
                and self.track_state == self.TS_TRACKING
                and self.bbox is not None):
            self._maybe_reinit_csrt(frame)

        # ── 3. CSRT update ───────────────────────────────────────────────
        ok, raw = self.tracker.update(frame)
        ov      = self._overlap(raw, frame.shape) if ok else 0.0
        fh,fw   = frame.shape[:2]

        # ── 4. Out-of-frame ──────────────────────────────────────────────
        is_out = ok and ov < 0.5
        if pred_pos:
            px,py = pred_pos
            if px<5 or px>fw-5 or py<5 or py>fh-5: is_out=True
        if is_out:
            self.track_state = self.TS_OUT_FRAME
            self.playing     = False
            return self.bbox

        # ── 5. Appearance check ──────────────────────────────────────────
        app_score = 1.0
        if ok and ov > 0.25:
            patch     = self._get_patch(frame, raw)
            app_score = self.appearance.score(patch)
            self.app_score = app_score

            # ── 5b. Jersey colour guard (V8) ─────────────────────────────
            if self.target_type == TARGET_PLAYER and self.jersey_hue is not None:
                if not self._check_jersey_color(frame, raw):
                    print("[JERSEY] Wrong colour — rejecting CSRT result")
                    self.jersey_warn_t = time.time()
                    app_score = 0.0
                    self.app_score = 0.0

        # ── 6. Multi-object / GrabCut ────────────────────────────────────
        ref_area   = (self.kalman_bbox_size[0]*self.kalman_bbox_size[1]
                      if self.kalman_bbox_size else 1)
        area_ratio = (raw[2]*raw[3]/max(ref_area,1)) if ok else 1.0
        multi_sus  = ok and (app_score < APPEARANCE_THRESHOLD
                             or area_ratio > APPEARANCE_AREA_SPIKE)

        if multi_sus:
            reason = "appearance" if app_score < APPEARANCE_THRESHOLD else "area-spike"
            print(f"[MULTI] {reason}  app={app_score:.2f}  ar={area_ratio:.2f}")
            refined = self._grabcut_separate(frame, raw, pred_pos)
            if refined:
                self.multi_obj_flash_t = time.time()
                self._init_tracking(frame, refined)
                return refined
            # Coast
            self.lost_count += 1
            if pred_pos and self.kalman_bbox_size:
                pw,ph = self.kalman_bbox_size
                k_box = (int(pred_pos[0]-pw/2),int(pred_pos[1]-ph/2),int(pw),int(ph))
                self.trail.append(pred_pos)
                self.track_state = self.TS_OCCLUDED
                return k_box
            self.track_state = self.TS_LOST
            return self.bbox

        # ── 7. Kick / anomaly detection (V5) ─────────────────────────────
        anomaly   = False
        is_kick   = False
        if ok and ov > 0.1 and self.track_state in [self.TS_TRACKING, self.TS_COASTING]:
            cx,cy = self._center(raw)
            if pred_pos and self.kalman_bbox_size:
                px,py   = pred_pos
                w_,h_   = self.kalman_bbox_size
                dist_p  = float(np.hypot(cx-px,cy-py))
                max_p   = max(w_,h_) * TRAJECTORY_TOLERANCE
                if dist_p > max_p:
                    last  = self.trail[-1] if self.trail else (cx,cy)
                    dl    = float(np.hypot(cx-last[0],cy-last[1]))
                    if dl < max(w_,h_) * MAX_KICK_DISTANCE:
                        is_kick = True
                        print(f"[KICK] {dist_p:.1f}px off-path")
                    else:
                        anomaly = True
                        print(f"[ANOMALY] teleport {dist_p:.1f}px")

        # ── 8. Good visual tracking ───────────────────────────────────────
        if ok and not anomaly:
            self.bbox = raw
            cx,cy     = self._center(raw)
            if is_kick:
                vx_,vy_ = self.kalman.get_velocity()
                self.kalman.init(cx,cy,vx_*0.6,vy_*0.6)
                self.kick_flash_timer = time.time()
            else:
                self.kalman.correct(cx,cy)
            self.kalman_bbox_size = (raw[2],raw[3])
            self.lost_count       = 0
            if (self.frame_idx - self.template_fidx) >= TEMPLATE_UPDATE_INT:
                self._save_template(frame, raw)
            patch = self._get_patch(frame, raw)
            self.appearance.soft_update(patch, app_score)
            # Jersey appearance soft-update
            if self.target_type == TARGET_PLAYER and app_score >= APPEARANCE_UPDATE_THRESH:
                upper = patch[:max(1,int(patch.shape[0]*JERSEY_UPPER_FRAC)),:]
                self.jersey_appearance.soft_update(upper, app_score)
            self.track_state = self.TS_TRACKING
            self.trail.append((cx,cy))
            return raw

        # ── 9. Coast: Kalman + trajectory search ─────────────────────────
        self.lost_count += 1
        k_box = None
        if pred_pos and self.kalman_bbox_size:
            pw,ph = self.kalman_bbox_size
            k_box = (int(pred_pos[0]-pw/2),int(pred_pos[1]-ph/2),int(pw),int(ph))
            self.trail.append(pred_pos)

        found = self._trajectory_search(frame, pred_pos)
        if found:
            rb,conf = found
            print(f"[RECOVERED] traj-search conf={conf:.2f}")
            self._init_tracking(frame, rb)
            return rb

        if k_box:
            self.track_state = self.TS_COASTING; return k_box
        self.track_state = self.TS_LOST
        return self.bbox

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw_bbox(self, frame, bbox, state):
        if not bbox: return
        x,y,w,h = [int(v) for v in bbox]
        now = time.time()
        if now - self.kick_flash_timer < 0.5:
            color = CLR_KICK
        elif now - self.multi_obj_flash_t < 0.7:
            color = CLR_MULTIOBJ
        elif now - self.jersey_warn_t < 0.6:
            color = CLR_JERSEY
        else:
            color = {self.TS_TRACKING:CLR_OK, self.TS_COASTING:CLR_COAST,
                     self.TS_OCCLUDED:CLR_WARN, self.TS_OUT_FRAME:CLR_LOST,
                     self.TS_LOST:CLR_LOST}.get(state, CLR_SELECT)

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        c = max(6,min(w,h)//4)
        for (px_,py_),(dx,dy) in [
            ((x,y),(1,1)),((x+w,y),(-1,1)),
            ((x,y+h),(1,-1)),((x+w,y+h),(-1,-1))]:
            cv2.line(frame,(px_,py_),(px_+dx*c,py_),color,3)
            cv2.line(frame,(px_,py_),(px_,py_+dy*c),color,3)
        cv2.drawMarker(frame,(x+w//2,y+h//2),color,cv2.MARKER_CROSS,14,1,cv2.LINE_AA)

        # Motion direction arrow
        if self.motion_map.speed() > MOTION_CONF_THR:
            vx,vy = self.motion_map.smooth_velocity()
            spd   = float(np.hypot(vx,vy)) + 1e-6
            ar_len = min(max(w,h)//2, 40)
            ax1,ay1 = x+w//2, y+h//2
            ax2 = int(ax1 + vx/spd * ar_len)
            ay2 = int(ay1 + vy/spd * ar_len)
            cv2.arrowedLine(frame,(ax1,ay1),(ax2,ay2),color,2,
                            tipLength=0.35, line_type=cv2.LINE_AA)

    def _draw_trail(self, frame):
        pts = list(self.trail)
        for i in range(1,len(pts)):
            a  = i/len(pts)
            c  = tuple(int(v*a) for v in CLR_TRAIL)
            cv2.line(frame,(int(pts[i-1][0]),int(pts[i-1][1])),
                           (int(pts[i][0]),  int(pts[i][1])),
                     c,max(1,int(2*a)),cv2.LINE_AA)

    def _draw_pip(self, frame, bbox):
        if not bbox: return
        x,y,w,h = [int(v) for v in bbox]
        fh,fw   = frame.shape[:2]
        px_,py_ = int(w*0.4), int(h*0.4)
        x1,y1   = max(0,x-px_), max(0,y-py_)
        x2,y2   = min(fw,x+w+px_), min(fh,y+h+py_)
        if x2<=x1 or y2<=y1: return
        roi    = frame[y1:y2,x1:x2].copy()
        pip_w  = max(120,int(fw*0.28))
        aspect = (y2-y1)/max(x2-x1,1)
        pip_h  = max(80,min(int(pip_w*aspect),int(fh*0.35)))
        zoomed = cv2.resize(roi,(pip_w,pip_h))
        m = 10; tx,ty = fw-pip_w-m, m
        ov = frame.copy()
        cv2.rectangle(ov,(tx-4,ty-22),(tx+pip_w+4,ty+pip_h+4),(15,15,15),-1)
        cv2.addWeighted(ov,0.65,frame,0.35,0,frame)
        frame[ty:ty+pip_h,tx:tx+pip_w] = zoomed
        pip_col = CLR_FROZEN if self.template_frozen else CLR_PIP
        cv2.rectangle(frame,(tx-2,ty-2),(tx+pip_w+2,ty+pip_h+2),pip_col,2)
        cv2.putText(frame,"TARGET VIEW",(tx,ty-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,pip_col,1,cv2.LINE_AA)

    def _draw_roi_overlay(self, frame):
        if not (self.roi_pt1 and self.roi_pt2): return
        x1=min(self.roi_pt1[0],self.roi_pt2[0]); y1=min(self.roi_pt1[1],self.roi_pt2[1])
        x2=max(self.roi_pt1[0],self.roi_pt2[0]); y2=max(self.roi_pt1[1],self.roi_pt2[1])
        cv2.rectangle(frame,(x1,y1),(x2,y2),CLR_SELECT,2)
        ov=frame.copy(); cv2.rectangle(ov,(x1,y1),(x2,y2),CLR_SELECT,-1)
        cv2.addWeighted(ov,0.15,frame,0.85,0,frame)

    def _draw_hud(self, frame):
        fh,fw = frame.shape[:2]
        now   = time.time()

        # ── top-left panel ───────────────────────────────────────────────
        cv2.rectangle(frame,(0,0),(215,145),(0,0,0),-1)

        # FPS
        cv2.putText(frame,f"FPS: {self.fps_value:5.1f}",(8,26),
                    cv2.FONT_HERSHEY_SIMPLEX,0.85,CLR_OK,2,cv2.LINE_AA)

        # Playback
        pb_col = CLR_OK if self.playing else (100,100,255)
        cv2.putText(frame,f"{'PLAYING' if self.playing else 'PAUSED'}",(8,50),
                    cv2.FONT_HERSHEY_SIMPLEX,0.50,pb_col,1,cv2.LINE_AA)

        # Template
        tmpl_col = CLR_FROZEN if self.template_frozen else CLR_OK
        cv2.putText(frame,f"TMPL:{'FROZEN' if self.template_frozen else 'LEARN'}",(8,70),
                    cv2.FONT_HERSHEY_SIMPLEX,0.48,tmpl_col,1,cv2.LINE_AA)

        # ── V8: Target type ──────────────────────────────────────────────
        type_lbl = {"ball":"[BALL]","player":"[PLAYER]","unknown":"[?TYPE]"
                    }.get(self.target_type,"[?]")
        type_col = {TARGET_BALL:(0,200,255),TARGET_PLAYER:(80,255,80),
                    TARGET_UNKNOWN:(160,160,160)}.get(self.target_type,(160,160,160))
        cv2.putText(frame,type_lbl,(8,92),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,type_col,2,cv2.LINE_AA)

        # ── V8: Jersey colour swatch ─────────────────────────────────────
        if self.target_type == TARGET_PLAYER and self.jersey_hue is not None:
            sw_x,sw_y,sw_w,sw_h = 8,102,18,14
            cv2.rectangle(frame,(sw_x,sw_y),(sw_x+sw_w,sw_y+sw_h),
                          self.jersey_bgr,-1)
            cv2.rectangle(frame,(sw_x,sw_y),(sw_x+sw_w,sw_y+sw_h),(200,200,200),1)
            cv2.putText(frame,f"Kit:{self.jersey_color_name}",
                        (sw_x+sw_w+4,sw_y+12),
                        cv2.FONT_HERSHEY_SIMPLEX,0.42,self.jersey_bgr,1,cv2.LINE_AA)

        # ── V8: CSRT speed class ─────────────────────────────────────────
        sc_col = {
            "fast":  (0,100,255),
            "normal":CLR_OK,
            "static":(160,160,160),
        }.get(self._speed_class, CLR_OK)
        cv2.putText(frame,f"CSRT:{self._speed_class.upper()}",(8,130),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,sc_col,1,cv2.LINE_AA)

        # ── appearance bar ────────────────────────────────────────────────
        if self.track_state not in (self.TS_NONE,self.TS_OUT_FRAME,self.TS_LOST):
            bx,by,bw,bh = 8,148,150,10
            score = max(0.,min(1.,self.app_score))
            cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(40,40,40),-1)
            fill = int(bw*score)
            r_= int(255*(1-score)); g_=int(255*score)
            cv2.rectangle(frame,(bx,by),(bx+fill,by+bh),(0,g_,r_),-1)
            cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(120,120,120),1)
            col_ = CLR_OK if score>=APPEARANCE_THRESHOLD else CLR_MULTIOBJ
            cv2.putText(frame,f"Appear:{score:.2f}",(bx,by-3),
                        cv2.FONT_HERSHEY_SIMPLEX,0.38,col_,1,cv2.LINE_AA)

        # ── bottom hint ───────────────────────────────────────────────────
        cv2.putText(frame,"ENTER:start  SPACE:pause  R:retarget  B/F/U:type  P:PiP  Q:quit",
                    (8,fh-10),cv2.FONT_HERSHEY_SIMPLEX,0.38,(140,140,140),1,cv2.LINE_AA)

        # ── selection prompt ──────────────────────────────────────────────
        if self.track_state == self.TS_NONE:
            for i,(msg,sc,col) in enumerate([
                ("Draw a box around the target",0.78,CLR_SELECT),
                ("then press ENTER to start",   0.60,(170,170,170)),
            ]):
                (tw,th),_ = cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,sc,2)
                sx=fw//2-tw//2; sy=fh//2-30+i*40
                cv2.rectangle(frame,(sx-10,sy-th-4),(sx+tw+10,sy+6),(0,0,0),-1)
                cv2.putText(frame,msg,(sx,sy),cv2.FONT_HERSHEY_SIMPLEX,sc,col,2,cv2.LINE_AA)
            return

        # ── flash labels ─────────────────────────────────────────────────
        if now-self.kick_flash_timer < 0.5:
            cv2.putText(frame,"[ KICK ! ]",(fw//2-60,55),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,CLR_KICK,3,cv2.LINE_AA)
        if now-self.multi_obj_flash_t < 0.7:
            msg="[ GRABCUT: re-locked ]"
            (tw,_),_=cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,0.70,2)
            cv2.putText(frame,msg,(fw//2-tw//2,55),
                        cv2.FONT_HERSHEY_SIMPLEX,0.70,CLR_MULTIOBJ,2,cv2.LINE_AA)
        if now-self.jersey_warn_t < 0.6:
            msg="[ WRONG JERSEY - coast ]"
            (tw,_),_=cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,0.65,2)
            cv2.putText(frame,msg,(fw//2-tw//2,55),
                        cv2.FONT_HERSHEY_SIMPLEX,0.65,CLR_JERSEY,2,cv2.LINE_AA)

        # ── state badge ───────────────────────────────────────────────────
        state_cfg = {
            self.TS_TRACKING: ("[  TRACKING  ]",        CLR_OK,    False),
            self.TS_COASTING: ("[ COASTING (Physics) ]",CLR_COAST, True),
            self.TS_OCCLUDED: ("[ OCCLUDED - Kalman ]", CLR_WARN,  True),
        }
        if self.track_state in state_cfg:
            label,color,blink = state_cfg[self.track_state]
            if not (blink and int(now*3)%2==0):
                (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.72,2)
                sx=fw//2-tw//2; sy=fh-26
                cv2.rectangle(frame,(sx-8,sy-th-4),(sx+tw+8,sy+6),(0,0,0),-1)
                cv2.putText(frame,label,(sx,sy),
                            cv2.FONT_HERSHEY_SIMPLEX,0.72,color,2,cv2.LINE_AA)

        if self.track_state in (self.TS_LOST,self.TS_OUT_FRAME):
            ov=frame.copy()
            cv2.rectangle(ov,(0,0),(fw,fh),(0,0,80),-1)
            cv2.addWeighted(ov,0.20,frame,0.80,0,frame)
            msg1=("STOPPED: Out Of Frame" if self.track_state==self.TS_OUT_FRAME
                  else "STOPPED: Target Lost")
            for i,(msg,sc,col) in enumerate([(msg1,0.8,CLR_WARN),
                                             ("Press R to select a new target",0.6,CLR_LOST)]):
                (mw,_),_=cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,sc,2)
                cv2.putText(frame,msg,(fw//2-mw//2,fh//2-20+i*40),
                            cv2.FONT_HERSHEY_SIMPLEX,sc,col,2,cv2.LINE_AA)

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        print("="*62)
        print("  SPORTS TRACKER V8")
        print("  Adaptive CSRT | MotionMap | Ball/Player | Jersey Guard")
        print(f"  Video: {self.video_path}")
        print("  B=ball  F=player  U=unknown  P=PiP  R=retarget  Q=quit")
        print("="*62)

        while True:
            advance = (self.playing and self.track_state not in
                       [self.TS_NONE, self.TS_OUT_FRAME, self.TS_LOST])

            if advance:
                ret,raw = self.cap.read()
                if not ret:
                    self._show_end_screen(); break
                self.cur_frame  = self._resize(raw)
                self.frame_idx += 1

                self._fps_cnt += 1
                now = time.time()
                if now - self._fps_t0 >= 0.4:
                    self.fps_value = self._fps_cnt/(now-self._fps_t0)
                    self._fps_cnt=0; self._fps_t0=now

                disp_bbox = self._step(self.cur_frame)

                if self.lost_count > MAX_LOST_FRAMES:
                    print("[STOP] Lost too long.")
                    self.playing=False; self.track_state=self.TS_LOST
            else:
                disp_bbox    = self.bbox
                self._fps_t0 = time.time(); self._fps_cnt=0

            display = self.cur_frame.copy()
            self._draw_trail(display)
            if self.pip_enabled and disp_bbox and self.track_state==self.TS_TRACKING:
                self._draw_pip(display, disp_bbox)
            if disp_bbox and self.track_state != self.TS_NONE:
                self._draw_bbox(display, disp_bbox, self.track_state)
            self._draw_roi_overlay(display)
            self._draw_hud(display)
            cv2.imshow(self.WIN, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'),27): break

            elif key == 13:   # ENTER
                if self.track_state in [self.TS_TRACKING,self.TS_COASTING,self.TS_OCCLUDED]:
                    self.playing=True; self._fps_t0=time.time(); self._fps_cnt=0
                elif self.track_state in [self.TS_LOST,self.TS_OUT_FRAME]:
                    print("[!] Target lost — press R to re-select.")
                else:
                    print("[!] Select a target first.")

            elif key == ord(' '):
                if self.playing: self.playing=False
                elif self.track_state in [self.TS_TRACKING,self.TS_COASTING,self.TS_OCCLUDED]:
                    self.playing=True; self._fps_t0=time.time(); self._fps_cnt=0

            elif key == ord('r'):
                self.playing=False; self.track_state=self.TS_NONE
                self.tracker=None; self.bbox=None
                self.trail.clear(); self.lost_count=0
                self.roi_pt1=None; self.roi_pt2=None
                self.template_frozen=False; self.app_score=1.0
                self.target_type=TARGET_UNKNOWN
                self.jersey_hue=None; self.jersey_color_name=""
                print("[RESET] Draw new target, then press ENTER")

            elif key == ord('p'):
                self.pip_enabled = not self.pip_enabled

            # ── V8: manual target type override ──────────────────────────
            elif key == ord('b'):
                self.target_type = TARGET_BALL
                self._csrt_params = self._compute_csrt_params()
                if self.bbox and self.cur_frame is not None:
                    self.tracker = self._make_csrt_from_params(self._csrt_params)
                    self.tracker.init(self.cur_frame, self.bbox)
                    self.kalman.max_speed = KalmanPredictor.MAX_SPEED_BALL
                print("[TYPE] Forced BALL")

            elif key == ord('f'):
                self.target_type = TARGET_PLAYER
                if self.bbox and self.cur_frame is not None:
                    patch = self._get_patch(self.cur_frame, self.bbox)
                    self.jersey_hue        = self._detect_jersey_hue(patch)
                    self.jersey_color_name = self._hue_to_name(self.jersey_hue)
                    self.jersey_bgr        = self._hue_to_bgr(self.jersey_hue)
                    upper = patch[:max(1,int(patch.shape[0]*JERSEY_UPPER_FRAC)),:]
                    self.jersey_appearance.build(upper)
                    self._csrt_params = self._compute_csrt_params()
                    self.tracker = self._make_csrt_from_params(self._csrt_params)
                    self.tracker.init(self.cur_frame, self.bbox)
                    self.kalman.max_speed = KalmanPredictor.MAX_SPEED_PLAYER
                print(f"[TYPE] Forced PLAYER  jersey={self.jersey_color_name}")

            elif key == ord('u'):
                self.target_type=TARGET_UNKNOWN; self.jersey_hue=None
                self.jersey_color_name=""; self.jersey_bgr=(128,128,128)
                print("[TYPE] Forced UNKNOWN")

        self.cap.release()
        cv2.destroyAllWindows()
        print("[EXIT] Done.")

    def _show_end_screen(self):
        if self.cur_frame is None: return
        frame = self.cur_frame.copy()
        msg = "VIDEO ENDED"
        (tw,_),_ = cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,1.4,3)
        cv2.putText(frame,msg,(frame.shape[1]//2-tw//2,frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,0,200),3,cv2.LINE_AA)
        cv2.imshow(self.WIN,frame); cv2.waitKey(3000)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python sports_tracker_v8.py <video_file>")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"[ERROR] File not found: '{path}'")
        sys.exit(1)
    try:
        SportsTracker(path).run()
    except IOError as e:
        print(f"[ERROR] {e}"); sys.exit(1)

if __name__ == "__main__":
    main()
