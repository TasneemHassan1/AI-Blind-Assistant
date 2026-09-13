"""Vision pipeline: YOLO/ByteTrack detections enriched with MiDaS depth."""
from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from calibration import DepthCalibrator, RelativeDepthCalibrator, SizeBasedDistanceEstimator
from config import (
    BYTE_TRACKER_CONFIG, CONFIDENCE_THRESHOLD, DUPLICATE_DETECTION_IOU_THRESHOLD,
    FOCAL_LENGTH_PX, YOLO_MODEL,
)
from debug import debug_throttled, logger
from domain_models import Detection as DetectionModel
from utils import BoundingBox, box_area_ratio, compute_iou, get_box_center, get_position

# Public compatibility alias retained for callers that import this name.
Detection = dict[str, Any]


class VisionAssistant:
    """Produces structured, tracked detections for one camera frame."""

    def __init__(self, calibrator: DepthCalibrator | None = None) -> None:
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model: YOLO | None = None
        self.depth_model: Any | None = None
        self.transform: Any | None = None
        self._reported_unavailable_model = False
        try:
            self.model = YOLO(YOLO_MODEL)
        except Exception as error:
            logger.exception("YOLO model could not be loaded from %s: %s", YOLO_MODEL, error)
        try:
            self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.depth_model.to(self.device).eval()
            self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        except Exception as error:
            # Object detection remains useful when optional relative depth is unavailable.
            self.depth_model = None
            self.transform = None
            logger.warning("MiDaS could not be loaded; continuing without depth: %s", error)

        self.calibrator = calibrator or RelativeDepthCalibrator()

        # Size-based metric distance estimator: independent of MiDaS depth,
        # uses known real-world object widths + bounding-box width instead.
        # Only works for classes in KNOWN_OBJECT_WIDTHS_CM, and only once
        # FOCAL_LENGTH_PX has been calibrated for this camera (see
        # calibrate_focal_length.py). Returns None otherwise.
        self.distance_estimator = SizeBasedDistanceEstimator(FOCAL_LENGTH_PX)

        # Stabilized reference range for depth normalization. Using a pure
        # per-frame min/max makes the same physical distance produce very
        # different normalized values between consecutive frames (any small
        # scene change shifts the reference), which caused erratic alerts
        # and missed collisions right when they mattered most. An EMA-based
        # reference changes gradually, so normalized depth is comparable
        # frame-to-frame.
        self._depth_ref_min: float | None = None
        self._depth_ref_max: float | None = None
        self._depth_ref_alpha = 0.1

        self._friendly_names = {
            "cell phone": "phone",
            "tv": "television",
            "dining table": "table",
            "potted plant": "plant",
            "sports ball": "ball",
            "traffic light": "traffic light",
            # Custom-trained classes: underscore-to-space handles most of
            # these fine (closed_door -> "closed door"), but "wardrobe"
            # reads more naturally as "closet" for a voice assistant.
            "wardrobe": "closet",
        }

    def _friendly_name(self, name: str) -> str:
        return self._friendly_names.get(name, name.replace("_", " "))

    def create_depth_map(self, frame: np.ndarray) -> np.ndarray:
        if self.depth_model is None or self.transform is None:
            raise RuntimeError("MiDaS depth model is unavailable")
        started = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(rgb).to(self.device)

        with torch.inference_mode():
            prediction = self.depth_model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Use robust percentiles instead of true min/max so a single noisy
        # pixel can't distort the reference range.
        frame_min = float(np.percentile(depth_map, 2))
        frame_max = float(np.percentile(depth_map, 98))

        if self._depth_ref_min is None:
            # First frame: initialize directly, nothing to smooth yet.
            self._depth_ref_min = frame_min
            self._depth_ref_max = frame_max
        else:
            alpha = self._depth_ref_alpha
            self._depth_ref_min = (
                alpha * frame_min + (1 - alpha) * self._depth_ref_min
            )
            self._depth_ref_max = (
                alpha * frame_max + (1 - alpha) * self._depth_ref_max
            )

        depth_range = self._depth_ref_max - self._depth_ref_min

        if depth_range > 1e-6:
            depth_map = (depth_map - self._depth_ref_min) / depth_range
            depth_map = np.clip(depth_map, 0.0, 1.0)
        else:
            depth_map = np.zeros_like(depth_map)

        debug_throttled("midas_timing", seconds=round(time.perf_counter() - started, 3))
        return depth_map

    @staticmethod
    def estimate_depth(
        depth_map: np.ndarray,
        box: BoundingBox,
    ) -> float | None:

        x1, y1, x2, y2 = map(int, box)

        height, width = depth_map.shape[:2]

        roi = depth_map[
            max(0, y1):min(height, y2),
            max(0, x1):min(width, x2),
        ]

        if roi.size == 0:
            return None

        return float(np.median(roi))

    def detect(
        self,
        frame: np.ndarray,
    ) -> tuple[list[Detection], np.ndarray]:

        depth_map = None
        if self.depth_model is not None and self.transform is not None:
            try:
                depth_map = self.create_depth_map(frame)
            except Exception as error:
                logger.warning("Depth inference failed; continuing without depth: %s", error)

        if self.model is None:
            if not self._reported_unavailable_model:
                logger.error("YOLO inference skipped because the model is unavailable.")
                self._reported_unavailable_model = True
            return [], frame

        try:
            yolo_started = time.perf_counter()
            result = self.model.track(
                frame,
                persist=True,
                tracker=BYTE_TRACKER_CONFIG,
                verbose=False,
            )[0]

        except Exception as error:
            logger.exception("YOLO tracking failed: %s", error)
            return [], frame

        debug_throttled("yolo_timing", seconds=round(time.perf_counter() - yolo_started, 3))
        image_width = result.orig_shape[1]
        track_ids = result.boxes.id

        detections: list[Detection] = []

        for index, box in enumerate(result.boxes):

            confidence = float(box.conf.item())

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            if box.xyxy is None:
                continue

            coordinates = tuple(
                float(value)
                for value in box.xyxy[0].cpu().tolist()
            )

            bbox: BoundingBox = coordinates

            center = get_box_center(bbox)

            bbox_width_px = bbox[2] - bbox[0]

            track_id = (
                int(track_ids[index].item())
                if track_ids is not None
                else None
            )

            class_id = int(box.cls.item())

            depth = None

            if depth_map is not None:
                try:
                    depth = self.estimate_depth(depth_map, bbox)
                except Exception:
                    depth = None

            friendly_name = self._friendly_name(self.model.names[class_id])

            detection = DetectionModel(
                id=track_id,
                name=friendly_name,
                confidence=confidence,
                bounding_box=bbox,
                center=center,
                depth=depth,
                distance_meters=self.calibrator.to_meters(depth),
                distance_cm=self.distance_estimator.to_cm(friendly_name, bbox_width_px),
                position=get_position(center[0], image_width),
                area_ratio=box_area_ratio(bbox, frame.shape),
                tracked=track_id is not None,
            )
            # Keep the legacy dictionary contract until all consumers can use
            # typed models directly.
            detections.append({**detection.to_dict(), "class": friendly_name, "class_id": class_id})

        detections = self._deduplicate(detections)

        annotated = result.plot()

        debug_throttled(
            "detections",
            count=len(detections),
            names=", ".join(d["name"] for d in detections) if detections else "none",
        )

        return detections, annotated

    @staticmethod
    def _deduplicate(detections: list[Detection]) -> list[Detection]:
        """Drop duplicate detections of the same class with heavily overlapping boxes.

        YOLO/ByteTrack occasionally assigns two distinct track IDs to the
        same physical object (e.g. two near-identical laptop boxes), which
        then reports two different distance/position readings for what is
        really one object. Keep only the higher-confidence box per
        overlapping pair.
        """
        kept: list[Detection] = []
        for detection in sorted(detections, key=lambda d: d["confidence"], reverse=True):
            is_duplicate = any(
                detection["class_id"] == other["class_id"]
                and compute_iou(detection["bounding_box"], other["bounding_box"])
                >= DUPLICATE_DETECTION_IOU_THRESHOLD
                for other in kept
            )
            if not is_duplicate:
                kept.append(detection)
        return kept