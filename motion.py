"""Motion, walking-corridor, TTC, and multi-frame track validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from config import (
    DEPTH_APPROACH_RATE_THRESHOLD, DISTANCE_CM_SMOOTHING_ALPHA,
    MOTION_CROSSING_SPEED_PX_PER_SECOND, MOTION_DEPTH_SMOOTHING_ALPHA,
    MOTION_POSITION_SMOOTHING_ALPHA, MOTION_STATIONARY_SPEED_PX_PER_SECOND,
    TRACK_CONFIRMATION_FRAMES, TRACK_REMOVAL_FRAMES, WALKING_CORRIDOR_LEFT_RATIO,
    WALKING_CORRIDOR_RIGHT_RATIO, WALKING_CORRIDOR_MIN_OVERLAP,
)
from utils import Point, euclidean_distance, utc_now


@dataclass
class TrackState:
    first_seen: str
    center: Point
    depth: float | None
    timestamp: float
    observations: int = 1
    missed_frames: int = 0
    last_detection: dict[str, Any] = field(default_factory=dict)
    # Temporally smoothed versions of center/depth/distance, used only for
    # motion math (velocity, depth_rate, ttc, path checks) and for the
    # distance figure spoken to the user. Raw per-frame MiDaS depth, YOLO
    # box centers, and bbox-width-derived distance_cm all flicker frame to
    # frame even for a perfectly still object, which was producing velocity
    # spikes in the hundreds of px/s, spurious "approaching"/"crossing"
    # alerts, and a distance reading that jumped around for no real reason.
    smoothed_center: Point = field(default=(0.0, 0.0))
    smoothed_depth: float | None = None
    smoothed_distance_cm: float | None = None


class MotionAnalyzer:
    """Enrich ByteTrack IDs and expose only stable, recently observed objects."""

    def __init__(self) -> None:
        self._tracks: dict[int, TrackState] = {}

    def update(
        self, detections: Iterable[dict[str, Any]], timestamp: float, frame_width: int
    ) -> list[dict[str, Any]]:
        seen: set[int] = set()
        enriched: list[dict[str, Any]] = []
        for item in detections:
            track_id = item.get("id")
            if track_id is None:
                continue
            track_id = int(track_id)
            seen.add(track_id)
            state = self._tracks.get(track_id)
            if state is None:
                state = TrackState(utc_now(), item["center"], item["depth"], timestamp)
                state.smoothed_center = item["center"]
                state.smoothed_depth = item["depth"]
                state.smoothed_distance_cm = item.get("distance_cm")
                self._tracks[track_id] = state
            else:
                if state.missed_frames:
                    state.observations = 0
                state.observations += 1
                state.missed_frames = 0

            prev_smoothed_center = state.smoothed_center
            prev_smoothed_depth = state.smoothed_depth
            prev_smoothed_distance = state.smoothed_distance_cm

            new_smoothed_center = self._smooth_point(
                prev_smoothed_center, item["center"], MOTION_POSITION_SMOOTHING_ALPHA
            )
            new_smoothed_depth = self._smooth_depth(
                prev_smoothed_depth, item["depth"], MOTION_DEPTH_SMOOTHING_ALPHA
            )
            new_smoothed_distance = self._smooth_depth(
                prev_smoothed_distance, item.get("distance_cm"), DISTANCE_CM_SMOOTHING_ALPHA
            )

            updated = self._enrich(
                item,
                state,
                prev_smoothed_center,
                prev_smoothed_depth,
                new_smoothed_center,
                new_smoothed_depth,
                new_smoothed_distance,
                timestamp,
                frame_width,
            )

            state.center, state.depth, state.timestamp, state.last_detection = (
                item["center"], item["depth"], timestamp, updated,
            )
            state.smoothed_center = new_smoothed_center
            state.smoothed_depth = new_smoothed_depth
            state.smoothed_distance_cm = new_smoothed_distance

            if state.observations >= TRACK_CONFIRMATION_FRAMES:
                enriched.append(updated)
        for track_id, state in list(self._tracks.items()):
            if track_id not in seen:
                state.missed_frames += 1
                if state.missed_frames >= TRACK_REMOVAL_FRAMES:
                    del self._tracks[track_id]
        return enriched

    def _enrich(
        self,
        item: dict[str, Any],
        state: TrackState,
        prev_center: Point,
        prev_depth: float | None,
        new_center: Point,
        new_depth: float | None,
        new_distance_cm: float | None,
        now: float,
        width: int,
    ) -> dict[str, Any]:
        elapsed = max(now - state.timestamp, 1e-6)
        dx = new_center[0] - prev_center[0]
        speed = euclidean_distance(new_center, prev_center) / elapsed
        direction = "right" if dx > 0 else "left" if dx < 0 else "stationary"
        depth_rate = self._depth_rate(prev_depth, new_depth, elapsed)
        approaching = depth_rate >= DEPTH_APPROACH_RATE_THRESHOLD
        ttc = (1.0 - new_depth) / depth_rate if approaching and new_depth is not None else None
        corridor_overlap = self._corridor_overlap(item.get("bounding_box"), width)
        # A centre-point check incorrectly labels side objects with large or
        # offset boxes as blockers. Use the box's actual corridor overlap.
        in_path = corridor_overlap >= WALKING_CORRIDOR_MIN_OVERLAP
        was_in_path = WALKING_CORRIDOR_LEFT_RATIO * width <= prev_center[0] <= WALKING_CORRIDOR_RIGHT_RATIO * width
        crossing = not was_in_path and in_path and speed >= MOTION_CROSSING_SPEED_PX_PER_SECOND
        motion_state = "approaching" if approaching else "moving away" if depth_rate < 0 else "stationary"
        if speed < MOTION_STATIONARY_SPEED_PX_PER_SECOND:
            direction = "stationary"
        return {
            **item,
            "depth": new_depth,
            "distance_cm": new_distance_cm,
            "velocity": speed,
            "direction": direction,
            "motion_state": motion_state,
            "depth_rate": depth_rate,
            "ttc": ttc,
            "in_walking_path": in_path,
            "walking_corridor_overlap": corridor_overlap,
            "crossing_path": crossing,
            "track_stability": state.observations,
            "first_seen": state.first_seen,
            "last_seen": utc_now(),
        }

    @staticmethod
    def _corridor_overlap(
        bounding_box: tuple[float, float, float, float] | None,
        frame_width: int,
    ) -> float:
        """Return the fraction of one box lying in the walking corridor."""
        if not bounding_box or frame_width <= 0:
            return 0.0
        x1, _, x2, _ = bounding_box
        box_width = max(float(x2) - float(x1), 0.0)
        if box_width <= 0:
            return 0.0
        corridor_left = WALKING_CORRIDOR_LEFT_RATIO * frame_width
        corridor_right = WALKING_CORRIDOR_RIGHT_RATIO * frame_width
        overlap = max(0.0, min(float(x2), corridor_right) - max(float(x1), corridor_left))
        return min(overlap / box_width, 1.0)

    @staticmethod
    def _depth_rate(previous: float | None, current: float | None, elapsed: float) -> float:
        return (current - previous) / elapsed if previous is not None and current is not None else 0.0

    @staticmethod
    def _smooth_point(previous: Point, current: Point, alpha: float) -> Point:
        return (
            alpha * current[0] + (1 - alpha) * previous[0],
            alpha * current[1] + (1 - alpha) * previous[1],
        )

    @staticmethod
    def _smooth_depth(previous: float | None, current: float | None, alpha: float) -> float | None:
        if current is None:
            return previous
        if previous is None:
            return current
        return alpha * current + (1 - alpha) * previous