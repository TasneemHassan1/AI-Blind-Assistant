"""Small, dependency-free helpers shared by the assistant modules."""

from __future__ import annotations

from datetime import datetime, timezone
from math import hypot
from typing import Sequence


BoundingBox = tuple[float, float, float, float]
Point = tuple[float, float]


def utc_now() -> str:
    """Return an ISO-8601 timestamp suitable for memory records."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_box_center(box: Sequence[float]) -> Point:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_area_ratio(box: Sequence[float], frame_shape: Sequence[int]) -> float:
    """Return the fraction of the image occupied by a bounding box."""
    x1, y1, x2, y2 = box
    height, width = frame_shape[:2]
    if height <= 0 or width <= 0:
        return 0.0
    return max(0.0, (x2 - x1) * (y2 - y1)) / (height * width)


def get_position(center_x: float, image_width: int) -> str:
    third = image_width / 3
    if center_x < third:
        return "left"
    if center_x < third * 2:
        return "ahead"
    return "right"


def euclidean_distance(first: Point, second: Point) -> float:
    return hypot(first[0] - second[0], first[1] - second[1])


def format_distance_cm(distance_cm: float | None) -> str | None:
    """Human-friendly distance phrase, switching to meters above 100cm."""
    if distance_cm is None or distance_cm <= 0:
        return None
    if distance_cm >= 100:
        return f"about {distance_cm / 100:.1f} meters"
    return f"about {round(distance_cm)} centimeters"


def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Return intersection-over-union of two (x1, y1, x2, y2) boxes.

    Used to detect when YOLO/ByteTrack has assigned two different track IDs
    to what is really the same physical object (near-identical overlapping
    boxes of the same class), so the duplicate can be filtered out before it
    reaches distance estimation, navigation, or dialogue.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0