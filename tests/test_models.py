"""
Tests for vision data models.
Run with: pytest tests/test_models.py -v
"""

import time
from src.vision.models import BoundingBox, Detection, TrackedObject, FrameResult


class TestBoundingBox:
    """Tests for the BoundingBox class."""

    def test_width_calculation(self):
        # ARRANGE — create the object
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
        # ASSERT — check the result is correct
        # x2 - x1 = 300 - 100 = 200
        assert bbox.width == 200

    def test_height_calculation(self):
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
        # y2 - y1 = 400 - 200 = 200
        assert bbox.height == 200

    def test_area_calculation(self):
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
        # width * height = 200 * 200 = 40000
        assert bbox.area == 40000

    def test_center_calculation(self):
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
        # center_x = (100 + 300) / 2 = 200
        # center_y = (200 + 400) / 2 = 300
        assert bbox.center == (200.0, 300.0)

    def test_to_list(self):
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
        assert bbox.to_list() == [100, 200, 300, 400]


class TestDetection:
    """Tests for the Detection class."""

    def setup_method(self):
        """
        setup_method runs before EVERY test in this class.
        Creates a fresh Detection so every test starts clean.
        """
        self.bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
        self.detection = Detection(
            bbox=self.bbox, confidence=0.92, class_id=0, class_name="person"
        )

    def test_detection_creation(self):
        assert self.detection.class_name == "person"
        assert self.detection.confidence == 0.92
        assert self.detection.class_id == 0

    def test_timestamp_auto_set(self):
        # timestamp should be set automatically by __post_init__
        assert self.detection.timestamp is not None
        # timestamp should be recent — within last 5 seconds
        assert self.detection.timestamp <= time.time()

    def test_is_confident_above_threshold(self):
        # 0.92 >= 0.5 — should be True
        assert self.detection.is_confident(threshold=0.5) is True

    def test_is_confident_below_threshold(self):
        # 0.92 >= 0.95 — should be False
        assert self.detection.is_confident(threshold=0.95) is False

    def test_default_confidence_threshold(self):
        # default threshold is 0.5 — 0.92 >= 0.5 — True
        assert self.detection.is_confident() is True


class TestTrackedObject:
    """Tests for the TrackedObject class."""

    def setup_method(self):
        self.bbox = BoundingBox(x1=50, y1=60, x2=150, y2=200)
        self.tracked = TrackedObject(
            bbox=self.bbox, confidence=0.88, class_id=2, class_name="car", track_id=42
        )

    def test_track_id_assigned(self):
        assert self.tracked.track_id == 42

    def test_default_age_is_zero(self):
        # Every new track starts at age 0
        assert self.tracked.age == 0

    def test_default_hits_is_one(self):
        # Every new track starts with 1 hit
        assert self.tracked.hits == 1

    def test_not_confirmed_on_first_hit(self):
        # MIN_HITS is 3 — with only 1 hit this should not be confirmed
        assert self.tracked.is_confirmed is False

    def test_confirmed_after_min_hits(self):
        # Simulate 3 hits — now it should be confirmed
        self.tracked.hits = 3
        assert self.tracked.is_confirmed is True


class TestFrameResult:
    """Tests for FrameResult class."""

    def setup_method(self):
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        det = Detection(bbox=bbox, confidence=0.9, class_id=0, class_name="person")
        tracked = TrackedObject(
            bbox=bbox, confidence=0.9, class_id=0, class_name="person", track_id=1
        )
        self.result = FrameResult(
            frame_id=1,
            timestamp=time.time(),
            detections=[det],
            tracked_objects=[tracked],
            processing_time_ms=12.5,
        )

    def test_detection_count(self):
        assert self.result.detection_count == 1

    def test_track_count(self):
        assert self.result.track_count == 1

    def test_frame_id(self):
        assert self.result.frame_id == 1

    def test_processing_time(self):
        assert self.result.processing_time_ms == 12.5
