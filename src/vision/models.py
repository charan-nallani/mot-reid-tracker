"""
Data models for the MOT Re-ID Tracker vision pipeline.
These classes define what a Detection, TrackedObject,
and FrameResult look like throughout the system.
"""

from dataclasses import dataclass
from typing import List, Optional
import time


@dataclass
class BoundingBox:
    """
    Represents a bounding box around a detected object.
    Uses xyxy format — top left corner and bottom right corner.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Width of the bounding box in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Height of the bounding box in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Area of the bounding box in square pixels."""
        return self.width * self.height

    @property
    def center(self):
        """Center point of the bounding box as (x, y)."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_list(self) -> list:
        """Return bounding box as a plain list [x1, y1, x2, y2]."""
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class Detection:
    """
    Represents a single object detected by YOLOv8 in one frame.
    Created fresh for every object detected in every frame.
    """

    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    timestamp: Optional[float] = None

    def __post_init__(self):
        """Runs automatically after __init__. Sets timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()

    def is_confident(self, threshold: float = 0.5) -> bool:
        """Check if this detection meets the confidence threshold."""
        return self.confidence >= threshold


@dataclass
class TrackedObject:
    """
    A detection that has been assigned a persistent track ID by ByteTrack.
    Once confirmed, this object is tracked across frames.
    """

    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    track_id: int
    age: int = 0
    hits: int = 1
    timestamp: Optional[float] = None

    def __post_init__(self):
        """Sets timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()

    @property
    def is_confirmed(self) -> bool:
        """Track is confirmed only after MIN_HITS detections."""
        from src.vision.config import MIN_HITS

        return self.hits >= MIN_HITS


@dataclass
class FrameResult:
    """
    Complete results for one processed video frame.
    Contains all detections and tracked objects for that frame.
    """

    frame_id: int
    timestamp: float
    detections: List[Detection]
    tracked_objects: List[TrackedObject]
    processing_time_ms: float

    @property
    def detection_count(self) -> int:
        """Number of raw detections in this frame."""
        return len(self.detections)

    @property
    def track_count(self) -> int:
        """Number of active tracked objects in this frame."""
        return len(self.tracked_objects)
