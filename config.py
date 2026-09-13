"""All tunable settings for the AI Blind Assistant.

Path constants are absolute so entry points work from any current directory.
Existing constant names are retained for compatibility.
"""

from pathlib import Path


# Project paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"
MEMORY_STORE_PATH = PROJECT_ROOT / "memory_store.json"

# Diagnostics ---------------------------------------------------------------
DEBUG = True
# Enable periodic latency/state measurements when DEBUG is enabled.
DEBUG_PERFORMANCE = True
DEBUG_LOG_INTERVAL_SECONDS = 5.0
LOG_FORMAT = "%(levelname)s %(message)s"

# Camera and vision ---------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WINDOW_NAME = "AI Blind Assistant"
DETECTION_INTERVAL_SECONDS = 0
# On macOS, a camera opened via cv2.VideoCapture sometimes isn't ready to
# deliver a frame on the very first read (the device is still initializing).
# Previously a single failed read ended the whole capture session, forcing a
# full app restart every time this happened. Now a failed read during
# startup is retried briefly before giving up for real.
CAMERA_WARMUP_RETRIES = 15
CAMERA_WARMUP_RETRY_DELAY_SECONDS = 0.2

# Replace this one path when deploying a custom-trained YOLO model.
YOLO_MODEL = str(MODELS_DIR / "blind_assistant_v4.pt")
# Custom tracker config (see bytetrack_custom.yaml) instead of the
# ultralytics default. The default's track_buffer/match_thresh were too
# strict for this camera's frame rate and caused constant track-ID churn on
# perfectly stationary objects, which broke scene description, navigation
# alerts, and room inference (all keyed off a stable track ID).
BYTE_TRACKER_CONFIG = str(PROJECT_ROOT / "bytetrack_custom.yaml")
CONFIDENCE_THRESHOLD = 0.65

# Startup health checks are advisory: failures are logged but never prevent
# the assistant from starting, allowing partially available hardware/services.
HEALTH_CHECKS_ENABLED = True
HEALTH_CHECK_TIMEOUT_SECONDS = 1.0

# Depth and distance --------------------------------------------------------
# Relative MiDaS depth only; calibration can convert it to metric distance later.
DEPTH_CAUTION_VALUE = 0.55
DEPTH_DANGER_VALUE = 0.75
DEPTH_APPROACH_RATE_THRESHOLD = 0.02
# Focal length in pixels for this specific camera; required for
# SizeBasedDistanceEstimator. None until calibrated — run
# calibrate_focal_length.py once to compute it.
FOCAL_LENGTH_PX: float | None = 1419.39
TTC_DANGER_SECONDS = 2.5
TTC_CAUTION_SECONDS = 5.0

# Two YOLO/ByteTrack boxes of the same class overlapping above this ratio
# are treated as the same physical object (duplicate track), keeping only
# the higher-confidence one. See VisionAssistant._deduplicate.
DUPLICATE_DETECTION_IOU_THRESHOLD = 0.6

# Temporal tracking and scene geometry -------------------------------------
TRACK_CONFIRMATION_FRAMES = 2
TRACK_REMOVAL_FRAMES = 4
TRACK_STABILITY_FRAMES = 8
MOTION_STATIONARY_SPEED_PX_PER_SECOND = 12.0
MOTION_CROSSING_SPEED_PX_PER_SECOND = 20.0
MOTION_POSITION_SMOOTHING_ALPHA = 0.35  # higher = follows raw motion faster, lower = smoother/laggier
MOTION_DEPTH_SMOOTHING_ALPHA = 0.25     # depth flickers more than position, so smooth it harder
# Raw per-frame distance_cm flickers with bbox width noise the same way raw
# depth does; smooth it with the same reasoning as MOTION_DEPTH_SMOOTHING_ALPHA.
# This reduces frame-to-frame jitter for a subject holding roughly still; it
# does not and should not prevent the number from tracking real movement
# (e.g. the user walking/turning quickly), which is expected behavior for a
# live navigation aid.
DISTANCE_CM_SMOOTHING_ALPHA = 0.25
WALKING_CORRIDOR_LEFT_RATIO = 0.32
WALKING_CORRIDOR_RIGHT_RATIO = 0.68
# A box must substantially overlap this central corridor before it can block
# the user's walking path. A centre point alone is too imprecise for wide boxes.
WALKING_CORRIDOR_MIN_OVERLAP = 0.35
NEAR_OBJECT_PIXEL_DISTANCE = 180.0
RELATION_VERTICAL_GAP_PIXELS = 80.0
SCENE_HISTORY_LIMIT = 100
# If the live scene becomes momentarily empty (a tracking flicker, or a
# voice question happening to be answered on the one frame where nothing
# was confirmed yet), keep answering "what do you see" style questions from
# the most recent non-empty scene for this long before actually saying
# nothing is visible. This is short on purpose — it smooths over single-frame
# gaps, it does not paper over genuinely empty scenes.
SCENE_EMPTY_GRACE_SECONDS = 1.5

# Room inference must be consistently observed before it changes the active
# scene or is spoken. This avoids flicker between an estimated room and None.
ROOM_CHANGE_CONFIRMATION_FRAMES = 4
ROOM_ANNOUNCEMENT_CONFIDENCE_THRESHOLD = 0.60
# How long to keep the last confidently known room after room-relevant
# evidence disappears (e.g. all room-relevant objects briefly leave frame,
# or a tracking flicker drops them). Without this grace period, a single
# gap in detections reset the room to "unknown" within a second, which
# produced wrong/flip-flopping "you appear to be in a ..." announcements
# and made "where am I" answers unreliable moments after being correct.
ROOM_UNKNOWN_GRACE_SECONDS = 4.0

# Risk assessment weights; final score is clamped to 0–100.
RISK_ALERT_THRESHOLD = 70.0
RISK_DEPTH_CAUTION_WEIGHT = 12.0
RISK_DEPTH_DANGER_WEIGHT = 24.0
RISK_AREA_WEIGHT = 30.0
RISK_PATH_WEIGHT = 22.0
RISK_CROSSING_WEIGHT = 16.0
RISK_POSITION_AHEAD_WEIGHT = 10.0
RISK_POSITION_SIDE_WEIGHT = 3.0
RISK_MOTION_WEIGHT = 12.0
RISK_TTC_CAUTION_WEIGHT = 12.0
RISK_TTC_DANGER_WEIGHT = 24.0
RISK_STABILITY_WEIGHT = 8.0
RISK_CONFIDENCE_WEIGHT = 8.0
RISK_REANNOUNCE_INCREASE = 12.0
# Never speak the same navigation state more often than this. A threat that
# disappears resets this state, so a later return is announced immediately.
NAVIGATION_ALERT_COOLDOWN_SECONDS = 6.0
NAVIGATION_STATE_CONFIRMATION_FRAMES = 2
NAVIGATION_ALERT_CONFIRMATION_FRAMES = 2
NAVIGATION_MIN_AREA_RATIO = 0.015
NAVIGATION_MIN_CONFIDENCE = 0.70
# Retain a temporarily missed ByteTrack state briefly, then discard it so no
# stale track can be announced after it has disappeared.
NAVIGATION_TRACK_STATE_TIMEOUT_SECONDS = 3.0
# These remain scene objects, but are navigation hazards only if genuinely
# close and well within the walking corridor.
NAVIGATION_SMALL_OBJECTS = {
    "bottle", "bowl", "cup", "handbag", "laptop", "plant", "remote",
    "cell phone", "phone", "book", "mouse", "keyboard", "fork", "spoon",
}
NAVIGATION_SMALL_OBJECT_CLOSE_DEPTH = 0.88
NAVIGATION_SMALL_OBJECT_MIN_CORRIDOR_OVERLAP = 0.60

# Voice I/O -----------------------------------------------------------------
VOICE_NAME = "Samantha"
VOICE_POLL_INTERVAL_SECONDS = 0.05
VOICE_COMMAND_WAKE_WORD = "assistant"
VOICE_LISTEN_TIMEOUT_SECONDS = 7
VOICE_FOLLOW_UP_LISTEN_TIMEOUT_SECONDS = 10
VOICE_PHRASE_LIMIT_SECONDS = 15
# Let a speaker pause briefly while forming a full command instead of cutting
# off questions such as "Where is the ... laptop?". Raised from 1.8 -> 2.5:
# the log showed real questions being truncated ("what do you [see]") because
# a normal thinking-pause mid-sentence exceeded the old threshold and the
# recognizer decided the phrase was already complete.
VOICE_PAUSE_THRESHOLD_SECONDS = 2.5
VOICE_MINIMUM_QUESTION_CHARACTERS = 4
VOICE_SPEECH_GUARD_SECONDS = 0.10

# Persistent memory ---------------------------------------------------------
MEMORY_HISTORY_LIMIT = 20
MAX_MEMORY_OBJECTS = 200
MEMORY_CONFIDENCE_HISTORY_LIMIT = MEMORY_HISTORY_LIMIT

# Dialogue and local LLM ----------------------------------------------------
# These limits constrain only the context sent to Ollama; permanent memory is
# never discarded because of them.
LLM_MODEL = "qwen2.5:3b"
LLM_HOST = "http://127.0.0.1:11434/api/generate"
LLM_TIMEOUT_SECONDS = 60
LLM_VISIBLE_CONTEXT_LIMIT = 15
LLM_MEMORY_CONTEXT_LIMIT = 12
LLM_MAX_RESPONSE_SENTENCES = 3
LLM_MAX_RESPONSE_CHARACTERS = 420
LLM_NUM_PREDICT = 48

# Weighted room evidence. Keys MUST match the *friendly* names produced by
# VisionAssistant._friendly_name (e.g. "tv" -> "television",
# "dining table" -> "table", "potted plant" -> "plant",
# "wardrobe" -> "closet"), since that is the name every downstream module
# (scene graph, memory, dialogue) actually sees. Custom detector labels
# work automatically as long as they follow the same post-translation
# naming; these optional labels merely add room context when available.
#
# Doors (closed_door/open_door/semi_door) are deliberately excluded here:
# a door is roughly equally likely in any room, so including it would add
# noise rather than signal to the room classifier.
#
# "table" was removed from kitchen: a table shows up in offices, living
# rooms, and dining areas just as often, so it was adding noise (false
# "kitchen" guesses) rather than real signal. oven/refrigerator/microwave/
# sink remain because they are genuinely kitchen-specific.
ROOM_VOTING_WEIGHTS = {
    "bedroom": {
        "bed": 5, "pillow": 4, "blanket": 4, "dresser": 4,
        "nightstand": 3, "closet": 3,
    },
    "kitchen": {"oven": 6, "refrigerator": 5, "microwave": 4, "sink": 4},
    "living room": {"couch": 5, "television": 3, "remote": 3, "plant": 1, "fan": 1},
    "office": {"laptop": 4, "keyboard": 4, "mouse": 3},
}