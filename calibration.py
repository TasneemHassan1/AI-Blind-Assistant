"""Isolated interface for future relative-depth to distance calibration."""

from __future__ import annotations

from typing import Protocol


class DepthCalibrator(Protocol):
    def to_meters(self, relative_depth: float | None) -> float | None: ...


class RelativeDepthCalibrator:
    """Default calibration: preserve MiDaS relative depth without false precision."""

    def to_meters(self, relative_depth: float | None) -> float | None:
        return None
    
# Rough average real-world widths (cm) for common COCO classes. These are
# population averages, not measurements of the user's specific objects, so
# expect roughly ±20-30% error per estimate — this is an approximation,
# not a precise measurement.
KNOWN_OBJECT_WIDTHS_CM: dict[str, float] = {
    "bottle": 7.0,
    "cup": 8.0,
    "phone": 7.5,
    "laptop": 34.0,
    "keyboard": 45.0,
    "mouse": 6.0,
    "book": 15.0,
    "chair": 45.0,
    "table": 120.0,
    "couch": 180.0,
    "bed": 140.0,
    "refrigerator": 70.0,
    "television": 100.0,
    "backpack": 30.0,
    "person": 45.0,  # shoulder width; unreliable due to pose/angle variation
    # Custom-trained classes. Keys use the friendly post-translation names
    # (see VisionAssistant._friendly_name), since that is the name the
    # size-based estimator is called with.
    "closed door": 80.0,
    "open door": 80.0,
    "semi door": 80.0,
    "closet": 100.0,  # wardrobe
}


class SizeBasedDistanceEstimator:
    """Estimate metric distance from bounding-box width, for object
    classes with a roughly known real-world size. Independent of MiDaS —
    MiDaS's relative depth has no reliable path to metric units (see
    RelativeDepthCalibrator above). Requires a one-time focal-length
    calibration for this specific camera (see calibrate_focal_length.py).
    """

    def __init__(self, focal_length_px: float | None) -> None:
        self.focal_length_px = focal_length_px

    def to_cm(self, class_name: str, bbox_width_px: float) -> float | None:
        if self.focal_length_px is None:
            return None
        real_width_cm = KNOWN_OBJECT_WIDTHS_CM.get(class_name)
        if real_width_cm is None or bbox_width_px <= 0:
            return None
        return (real_width_cm * self.focal_length_px) / bbox_width_px