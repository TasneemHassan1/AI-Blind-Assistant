"""Active stable-object graph, spatial relationships, and change history."""

from __future__ import annotations

import time
from copy import deepcopy
from threading import RLock
from typing import Any, Iterable

from config import (
    NEAR_OBJECT_PIXEL_DISTANCE,
    RELATION_VERTICAL_GAP_PIXELS,
    SCENE_EMPTY_GRACE_SECONDS,
    SCENE_HISTORY_LIMIT,
)
from utils import euclidean_distance, utc_now

SceneObject = dict[str, Any]


class SceneGraph:
    def __init__(self) -> None:
        self._objects: dict[int, SceneObject] = {}
        self._history: list[dict[str, Any]] = []
        self._lock = RLock()
        self._room: str | None = None
        # Most recent non-empty scene, kept briefly so a single frame with
        # zero confirmed objects (tracking flicker, or a voice question
        # happening to be answered on exactly that frame) doesn't make the
        # assistant claim nothing is visible when something was visible a
        # moment before and still plausibly is.
        self._last_nonempty_objects: dict[int, SceneObject] = {}
        self._last_nonempty_at: float | None = None

    def update(self, detections: Iterable[dict[str, Any]]) -> dict[int, SceneObject]:
        """Replace the active graph with stable tracked detections.

        Returned data remains a deep-copied dictionary for compatibility.
        """
        with self._lock:
            previous = self._objects
            current: dict[int, SceneObject] = {}
            for item in detections:
                if not item.get("tracked") or item.get("id") is None:
                    continue
                object_id = int(item["id"])
                previous_item = previous.get(object_id)
                current[object_id] = {
                    **item,
                    "id": object_id,
                    "track_id": object_id,
                    "tracked": True,
                    "near_objects": [],
                    "relationships": [],
                    "support_object": None,
                    "first_seen": previous_item["first_seen"] if previous_item else utc_now(),
                    "last_seen": utc_now(),
                }
            self._objects = current
            self._build_relationships()
            self._record_changes(previous)
            if self._objects:
                self._last_nonempty_objects = deepcopy(self._objects)
                self._last_nonempty_at = time.monotonic()
            return self.snapshot()

    def set_room(self, room: str | None) -> dict[int, SceneObject]:
        """Attach the current room estimate and record room changes.

        This additive helper leaves ``update`` and ``snapshot`` unchanged.
        """
        with self._lock:
            if room != self._room:
                now = utc_now()
                for item in self._objects.values():
                    self._add_event(now, "room changed", item, room=room)
                self._room = room
            for item in self._objects.values():
                item["estimated_room"] = room
            del self._history[:-SCENE_HISTORY_LIMIT]
            return self.snapshot()

    def _build_relationships(self) -> None:
        """Build deterministic, de-duplicated pairwise relationships."""
        objects = [self._objects[key] for key in sorted(self._objects)]
        support_candidates: dict[int, list[tuple[tuple[float, float, float, int], SceneObject]]] = {
            item["id"]: [] for item in objects
        }
        relationship_keys: dict[int, set[tuple[int, str]]] = {item["id"]: set() for item in objects}

        for index, first in enumerate(objects):
            for second in objects[index + 1:]:
                if euclidean_distance(first["center"], second["center"]) <= NEAR_OBJECT_PIXEL_DISTANCE:
                    self._add_near_relationship(first, second, relationship_keys)
                    self._add_near_relationship(second, first, relationship_keys)
                for item, candidate in ((first, second), (second, first)):
                    rank = self._support_rank(item, candidate)
                    if rank is not None:
                        support_candidates[item["id"]].append((rank, candidate))

        for item in objects:
            candidates = support_candidates[item["id"]]
            if not candidates:
                continue
            _, support = min(candidates, key=lambda value: value[0])
            item["support_object"] = support["name"]
            self._add_relationship(item, support, "on", relationship_keys)

    @staticmethod
    def _support_rank(item: SceneObject, candidate: SceneObject) -> tuple[float, float, float, int] | None:
        item_box = item.get("bounding_box")
        candidate_box = candidate.get("bounding_box")
        if item_box is None or candidate_box is None:
            return None
        vertical_gap = candidate_box[1] - item_box[3]
        item_width = item_box[2] - item_box[0]
        candidate_width = candidate_box[2] - candidate_box[0]
        overlap = max(0.0, min(item_box[2], candidate_box[2]) - max(item_box[0], candidate_box[0]))
        if not (0 <= vertical_gap <= RELATION_VERTICAL_GAP_PIXELS and candidate_width >= item_width and overlap > 0):
            return None
        # Closest vertical contact wins; then most overlap, widest support, stable ID.
        return (vertical_gap, -overlap, -candidate_width, candidate["id"])

    @staticmethod
    def _add_near_relationship(
        item: SceneObject, other: SceneObject, keys: dict[int, set[tuple[int, str]]]
    ) -> None:
        if other["name"] not in item["near_objects"]:
            item["near_objects"].append(other["name"])
        SceneGraph._add_relationship(item, other, "near", keys)

    @staticmethod
    def _add_relationship(
        item: SceneObject, other: SceneObject, relation: str, keys: dict[int, set[tuple[int, str]]]
    ) -> None:
        key = (other["id"], relation)
        if key not in keys[item["id"]]:
            keys[item["id"]].add(key)
            item["relationships"].append({"object": other["name"], "relation": relation})

    def _record_changes(self, previous: dict[int, SceneObject]) -> None:
        now = utc_now()
        for object_id in sorted(self._objects):
            item = self._objects[object_id]
            old = previous.get(object_id)
            if old is None:
                self._add_event(now, "entered", item)
                continue
            old_motion = old.get("motion_state")
            new_motion = item.get("motion_state")
            if new_motion != "stationary":
                self._add_event(now, "moved", item)
            if old_motion == "stationary" and new_motion != "stationary":
                self._add_event(now, "started moving", item)
            elif old_motion not in (None, "stationary") and new_motion == "stationary":
                self._add_event(now, "stopped moving", item)
                self._add_event(now, "stopped", item, legacy_event="stopped moving")
            if old.get("support_object") != item.get("support_object"):
                self._add_event(now, "support changed", item, support_object=item.get("support_object"))
                self._add_event(
                    now,
                    "support object changed",
                    item,
                    support_object=item.get("support_object"),
                    legacy_event="support changed",
                )
            if old.get("position") != item.get("position"):
                self._add_event(now, "position changed", item, position=item.get("position"))
            old_room = old.get("estimated_room", old.get("room"))
            new_room = item.get("estimated_room", item.get("room"))
            if old_room != new_room and new_room is not None:
                self._add_event(now, "room changed", item, room=new_room)
        for object_id in sorted(previous):
            if object_id not in self._objects:
                self._add_event(now, "disappeared", previous[object_id])
                self._add_event(now, "left", previous[object_id], legacy_event="disappeared")
        del self._history[:-SCENE_HISTORY_LIMIT]

    def _add_event(self, time: str, event: str, item: SceneObject, **extra: Any) -> None:
        self._history.append({"time": time, "event": event, "object": item["name"], **extra})

    def snapshot(self) -> dict[int, SceneObject]:
        with self._lock:
            return deepcopy(self._objects)

    def snapshot_for_dialogue(self) -> dict[int, SceneObject]:
        """Return the live scene, or a very recent one if momentarily empty.

        A voice question can happen to be answered on exactly the one frame
        where a track was between confirmations (tracking flicker) even
        though objects were clearly visible a fraction of a second before
        and after. Falling back to the last non-empty scene for a short
        grace window avoids the assistant wrongly saying nothing is visible
        in that situation, without masking a genuinely empty scene for long.
        """
        with self._lock:
            if self._objects:
                return deepcopy(self._objects)
            if (
                self._last_nonempty_at is not None
                and time.monotonic() - self._last_nonempty_at <= SCENE_EMPTY_GRACE_SECONDS
            ):
                return deepcopy(self._last_nonempty_objects)
            return {}

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._history)


_scene = SceneGraph()


def update_scene(detections: Iterable[dict[str, Any]]) -> dict[int, SceneObject]:
    return _scene.update(detections)


def get_scene() -> dict[int, SceneObject]:
    return _scene.snapshot()