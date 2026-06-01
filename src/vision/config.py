"""
Configuration settings for the MOT Re-ID Tracker.
All model settings and constants live here.
Never hardcode values directly in logic files.
"""

# ── Model Settings ──────────────────────────────────────────
DEFAULT_MODEL_PATH = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_IMAGE_SIZE = 640

# ── COCO Class Names ─────────────────────────────────────────
VEHICLE_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ── Tracking Settings ────────────────────────────────────────
MAX_AGE = 30
MIN_HITS = 3
IOU_THRESHOLD = 0.3
