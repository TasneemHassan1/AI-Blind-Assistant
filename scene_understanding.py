"""Weighted, explainable room inference independent of detector model classes."""

from __future__ import annotations

import time
from typing import Any, Iterable

from config import ROOM_CHANGE_CONFIRMATION_FRAMES, ROOM_UNKNOWN_GRACE_SECONDS, ROOM_VOTING_WEIGHTS
from debug import debug


class SceneUnderstanding:
    """Room voting with temporal confirmation to prevent one-frame switches.

    A momentary lack of room-relevant evidence (e.g. a frame where tracking
    briefly drops the furniture that identifies the room, or no objects are
    visible for an instant) must not erase a room the assistant was just
    confident about. The previous behavior reset to None as soon as evidence
    was missing for a few consecutive frames; combined with track-ID churn
    that happened almost every second, this made the room flicker or
    disappear constantly, and "where am I" answers become wrong moments
    after being right. Now the last confidently known room is kept for
    ROOM_UNKNOWN_GRACE_SECONDS of continuous missing evidence, and a
    *different* room only takes over once it is itself confirmed for
    ROOM_CHANGE_CONFIRMATION_FRAMES with actual evidence.
    """

    def __init__(self) -> None:
        self._candidate: str | None = None
        self._candidate_frames = 0
        self._stable_room: str | None = None
        self._stable_confidence = 0.0
        self._last_evidence_at: float | None = None

    def infer_room(self, scene: dict[int, dict[str, Any]] | Iterable[dict[str, Any]]) -> dict[str, Any]:
        objects = scene.values() if isinstance(scene, dict) else scene
        names = {str(item["name"]).lower() for item in objects}
        scores = {room: sum(weight for label, weight in weights.items() if label in names)
                  for room, weights in ROOM_VOTING_WEIGHTS.items()}
        room, score = max(scores.items(), key=lambda item: item[1], default=("unknown", 0))
        total = sum(scores.values())
        raw_room = room if score else None
        now = time.monotonic()

        if raw_room is not None:
            self._last_evidence_at = now
            if raw_room == self._candidate:
                self._candidate_frames += 1
            else:
                self._candidate = raw_room
                self._candidate_frames = 1
            if self._candidate_frames >= ROOM_CHANGE_CONFIRMATION_FRAMES:
                self._stable_room = raw_room
                self._stable_confidence = round(score / total, 2) if total else 0.0
        else:
            # No room-relevant evidence this frame. Reset the confirmation
            # candidate so a later run of frames must re-confirm from
            # scratch, but only clear the *stable* room once the grace
            # period has genuinely elapsed with no evidence at all.
            self._candidate = None
            self._candidate_frames = 0
            if (
                self._stable_room is not None
                and self._last_evidence_at is not None
                and now - self._last_evidence_at >= ROOM_UNKNOWN_GRACE_SECONDS
            ):
                self._stable_room = None
                self._stable_confidence = 0.0

        result = {
            "estimated_room": self._stable_room,
            "confidence": self._stable_confidence if self._stable_room is not None else 0.0,
            "scores": scores,
            "raw_estimated_room": raw_room,
        }
        debug("room_prediction", **result)
        return result

    def current_room(self) -> dict[str, Any]:
        """Return the last frame-confirmed room without altering smoothing state."""
        return {
            "estimated_room": self._stable_room,
            "confidence": self._stable_confidence if self._stable_room else 0.0,
        }