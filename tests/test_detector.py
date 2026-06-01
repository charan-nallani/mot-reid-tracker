"""
Tests for the Detector module.
Uses a synthetic test image so no real image file is needed.
Run with: pytest tests/test_detector.py -v
"""

import numpy as np
import pytest
from src.vision.detector import Detector


class TestDetector:
    """Tests for the Detector class."""

    def setup_method(self):
        """
        Create a Detector instance before each test.
        Uses CPU to keep tests fast and consistent.
        """
        self.detector = Detector(device="cpu")

    def test_detector_initializes(self):
        """Detector should initialize without errors."""
        assert self.detector is not None

    def test_device_is_set(self):
        """Device should be set correctly."""
        assert self.detector.device == "cpu"

    def test_model_is_loaded(self):
        """Model should be loaded after initialization."""
        assert self.detector.model is not None

    def test_detect_returns_list(self):
        """detect() should always return a list."""
        # Create a synthetic black image — 640x640 pixels, 3 color channels
        # np.zeros creates an array filled with zeros — black image
        fake_image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = self.detector.detect(fake_image)
        assert isinstance(result, list)

    def test_detect_empty_image_returns_empty_list(self):
        """A completely black image should return no detections."""
        fake_image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = self.detector.detect(fake_image)
        assert result == []

    def test_detect_from_file_raises_for_missing_file(self):
        """Should raise FileNotFoundError for non-existent files."""
        with pytest.raises(FileNotFoundError):
            self.detector.detect_from_file("non_existent_image.jpg")

    def test_visualize_returns_numpy_array(self):
        """visualize() should return a NumPy array."""
        fake_image = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = []
        result = self.detector.visualize(fake_image, detections)
        assert isinstance(result, np.ndarray)

    def test_visualize_does_not_modify_original(self):
        """visualize() should not modify the original image."""
        fake_image = np.zeros((640, 640, 3), dtype=np.uint8)
        original_copy = fake_image.copy()
        detections = []
        self.detector.visualize(fake_image, detections)
        # Original should be unchanged
        assert np.array_equal(fake_image, original_copy)
