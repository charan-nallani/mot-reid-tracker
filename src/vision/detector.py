"""
Detector module for MOT Re-ID Tracker.
Wraps YOLO11n to detect objects in images and video frames.
Converts raw YOLO output into our Detection and BoundingBox data models.
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.vision.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_PATH,
    VEHICLE_CLASSES,
)
from src.vision.models import BoundingBox, Detection

logger = logging.getLogger(__name__)


class Detector:
    """
    Wraps YOLO11n for object detection.

    Loads the model once on initialization and reuses it
    for every detection call. This is important for performance
    — loading a model takes seconds, inference takes milliseconds.

    Usage:
        detector = Detector()
        detections = detector.detect(image)
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        confidence: float = DEFAULT_CONFIDENCE,
        image_size: int = DEFAULT_IMAGE_SIZE,
        device: Optional[str] = None,
    ):
        self.confidence = confidence
        self.image_size = image_size

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Loading YOLO model from {model_path} on {self.device}")

        self.model = YOLO(model_path)
        self.model.to(self.device)

        logger.info(f"Model loaded successfully on {self.device}")

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run YOLO detection on a single image.

        Args:
            image: Image as a NumPy array in BGR format.
                   Shape should be (height, width, 3).

        Returns:
            List of Detection objects.
            Empty list if no objects detected.
        """
        results = self.model(
            image,
            verbose=False,
            conf=self.confidence,
            imgsz=self.image_size,
        )

        detections = []
        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls.item())

            if class_id not in VEHICLE_CLASSES:
                continue

            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            bbox = BoundingBox(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
            )

            class_name = VEHICLE_CLASSES[class_id]

            detection = Detection(
                bbox=bbox,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
            )

            detections.append(detection)

        logger.debug(f"Detected {len(detections)} objects")
        return detections

    def detect_from_file(self, image_path: str) -> List[Detection]:
        """
        Load an image from disk and run detection.

        Args:
            image_path: Path to the image file.

        Returns:
            List of Detection objects.

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the image cannot be loaded.
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(path))

        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        return self.detect(image)

    def visualize(
        self,
        image: np.ndarray,
        detections: List[Detection],
    ) -> np.ndarray:
        """
        Draw detection boxes on the image for debugging and demos.

        Args:
            image: Original image as NumPy array.
            detections: List of Detection objects to draw.

        Returns:
            New image with bounding boxes drawn on it.
        """
        output = image.copy()

        for detection in detections:
            bbox = detection.bbox
            x1, y1, x2, y2 = (
                int(bbox.x1),
                int(bbox.y1),
                int(bbox.x2),
                int(bbox.y2),
            )

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{detection.class_name} {detection.confidence:.2f}"

            cv2.rectangle(output, (x1, y1 - 20), (x2, y1), (0, 255, 0), -1)

            cv2.putText(
                output,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        return output
