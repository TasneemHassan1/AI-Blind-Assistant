"""Risk assessment and concise, state-aware navigation alerts."""

from __future__ import annotations

import time
from typing import Any, Iterable
from utils import format_distance_cm

from config import (
    DEPTH_CAUTION_VALUE,
    DEPTH_DANGER_VALUE,
    RISK_ALERT_THRESHOLD,
    RISK_AREA_WEIGHT,
    RISK_CONFIDENCE_WEIGHT,
    RISK_CROSSING_WEIGHT,
    RISK_DEPTH_CAUTION_WEIGHT,
    RISK_DEPTH_DANGER_WEIGHT,
    RISK_MOTION_WEIGHT,
    RISK_PATH_WEIGHT,
    RISK_POSITION_AHEAD_WEIGHT,
    RISK_POSITION_SIDE_WEIGHT,
    RISK_REANNOUNCE_INCREASE,
    NAVIGATION_ALERT_COOLDOWN_SECONDS,
    NAVIGATION_ALERT_CONFIRMATION_FRAMES,
    NAVIGATION_MIN_AREA_RATIO,
    NAVIGATION_MIN_CONFIDENCE,
    NAVIGATION_SMALL_OBJECT_CLOSE_DEPTH,
    NAVIGATION_SMALL_OBJECT_MIN_CORRIDOR_OVERLAP,
    NAVIGATION_SMALL_OBJECTS,
    NAVIGATION_STATE_CONFIRMATION_FRAMES,
    NAVIGATION_TRACK_STATE_TIMEOUT_SECONDS,
    RISK_STABILITY_WEIGHT,
    RISK_TTC_CAUTION_WEIGHT,
    RISK_TTC_DANGER_WEIGHT,
    TRACK_STABILITY_FRAMES,
    TTC_CAUTION_SECONDS,
    TTC_DANGER_SECONDS,
    WALKING_CORRIDOR_MIN_OVERLAP,
)

from debug import debug
from domain_models import NavigationAlert


class Navigation:

    def __init__(self) -> None:
        self._last_object_id: int | None = None
        self._last_risk = 0.0
        self._last_message: str | None = None
        self._last_announced_at = 0.0
        self._spoken_by_object: dict[int, dict[str, Any]] = {}
        self._state_candidates: dict[int, tuple[str, int]] = {}
        self._stable_states: dict[int, str] = {}

    @staticmethod
    def calculate_risk(item: dict[str, Any]) -> float:
        """Score travel-relevant evidence instead of using depth alone."""
        overlap = max(0.0, min(float(item.get("walking_corridor_overlap", 0.0)), 1.0))
        # Preserve sensible behavior for legacy direct callers that only
        # supplied the existing in_walking_path field.
        if overlap == 0.0 and item.get("in_walking_path"):
            overlap = 1.0
        corridor_factor = 0.35 + 0.65 * overlap
        score = min(
            item["area_ratio"] * RISK_AREA_WEIGHT / 0.3,
            RISK_AREA_WEIGHT,
        ) * corridor_factor

        depth = item.get("depth")

        if depth is not None:
            if depth >= DEPTH_DANGER_VALUE:
                score += RISK_DEPTH_DANGER_WEIGHT * corridor_factor
            elif depth >= DEPTH_CAUTION_VALUE:
                score += RISK_DEPTH_CAUTION_WEIGHT * corridor_factor

        if item.get("in_walking_path"):
            score += RISK_PATH_WEIGHT * overlap

        if item.get("crossing_path"):
            score += RISK_CROSSING_WEIGHT

        if item.get("position") == "ahead":
            score += RISK_POSITION_AHEAD_WEIGHT * corridor_factor
        else:
            score += RISK_POSITION_SIDE_WEIGHT * overlap

        if item.get("motion_state") == "approaching":
            score += RISK_MOTION_WEIGHT * corridor_factor

        ttc = item.get("ttc")

        if ttc is not None:
            if ttc <= TTC_DANGER_SECONDS:
                score += RISK_TTC_DANGER_WEIGHT * corridor_factor
            elif ttc <= TTC_CAUTION_SECONDS:
                score += RISK_TTC_CAUTION_WEIGHT * corridor_factor

        score += (
            min(
                item.get("track_stability", 0) / TRACK_STABILITY_FRAMES,
                1.0,
            )
            * RISK_STABILITY_WEIGHT
        )

        score += item.get("confidence", 0.0) * RISK_CONFIDENCE_WEIGHT

        return round(min(max(score, 0.0), 100.0), 1)

    def _build_message(self, obstacle: dict[str, Any], state: str | None = None) -> str:
        """Create concise, natural speech while preserving alert priority."""
        name = obstacle["name"].replace("_", " ")
        article = "An" if name[:1].lower() in "aeiou" else "A"
        position = obstacle.get("position", "ahead")
        distance_phrase = format_distance_cm(obstacle.get("distance_cm"))
        distance_suffix = f", {distance_phrase} away" if distance_phrase else ""

        if state == "collision" or self._priority(obstacle, state) == 3:
            return f"{article} {name} is very close ahead{distance_suffix}."

        if state == "blocking":
            return f"{article} {name} blocks your path{distance_suffix}."

        if state == "approaching":
            if position == "ahead":
                return f"{article} {name} is approaching ahead{distance_suffix}."
            return f"{article} {name} is approaching from your {position}{distance_suffix}."

        if state == "crossing":
            return f"{article} {name} is crossing your path{distance_suffix}."

        if position == "ahead":
            return f"{article} {name} is directly ahead{distance_suffix}."
        return f"{article} {name} is on your {position}{distance_suffix}."

    def generate_alert(
        self,
        objects: Iterable[dict[str, Any]],
    ) -> str | None:

        now = time.monotonic()
        candidates = []
        seen_ids: set[int] = set()
        for item in objects:
            object_id = item.get("id")
            if object_id is None:
                continue
            object_id = int(object_id)
            seen_ids.add(object_id)
            track = self._spoken_by_object.setdefault(
                object_id,
                {
                    "last_alert_time": 0.0,
                    "last_alert_type": None,
                    "consecutive_frames": 0,
                    "last_seen_time": now,
                },
            )
            track["last_seen_time"] = now
            track["consecutive_frames"] = max(
                int(track.get("consecutive_frames", 0)) + 1,
                int(item.get("track_stability", 0)),
            )
            if track["consecutive_frames"] < NAVIGATION_ALERT_CONFIRMATION_FRAMES:
                continue
            if not self._is_relevant(item):
                continue
            risk = self.calculate_risk(item)
            # Side objects are scene information, not navigation hazards.
            affects_user = bool(item.get("in_walking_path") or item.get("crossing_path"))
            if affects_user and risk >= RISK_ALERT_THRESHOLD:
                state = self._stabilize_state(object_id, self._raw_state(item))
                candidates.append((self._priority(item, state), risk, item, state))

        self._discard_missing_states(seen_ids, now)

        if not candidates:
            return None

        _, risk, obstacle, state = max(
            candidates,
            key=lambda candidate: (candidate[0], candidate[1]),
        )

        debug(
            "risk",
            object=obstacle["name"],
            risk=risk,
            depth=obstacle.get("depth"),
            velocity=obstacle.get("velocity"),
            ttc=obstacle.get("ttc"),
        )

        message = self._build_message(obstacle, state)
        object_id = int(obstacle["id"])
        previous = self._spoken_by_object.get(object_id)
        if not self._should_announce(previous, risk, obstacle, state, now):
            return None

        selected = NavigationAlert(
            object_id=object_id,
            object_name=str(obstacle["name"]),
            risk=risk,
            message=message,
        )
        self._last_object_id = selected.object_id
        self._last_risk = selected.risk
        self._last_message = selected.message
        self._last_announced_at = now
        self._spoken_by_object[object_id] = {
            "message": message,
            "last_alert_time": now,
            "last_alert_type": state,
            "risk": risk,
            "position": obstacle.get("position"),
            "consecutive_frames": previous.get("consecutive_frames", 0) if previous else 0,
            "last_seen_time": now,
        }

        debug(
            "selected_obstacle",
            object=obstacle["name"],
            risk=risk,
        )

        return selected.message

    def _reset_last_alert(self) -> None:
        self._last_object_id = None
        self._last_risk = 0.0
        self._last_message = None
        self._last_announced_at = 0.0
        self._spoken_by_object.clear()
        self._state_candidates.clear()
        self._stable_states.clear()

    @staticmethod
    def _priority(item: dict[str, Any], state: str | None = None) -> int:
        """Prioritize collision, blocking, person, motion, furniture, small."""
        if state is not None:
            if state == "collision":
                return 60
            if state == "blocking":
                return 50
            if str(item.get("name", "")).casefold() == "person":
                return 40
            if state in {"approaching", "crossing"}:
                return 30
            return 20 if Navigation._is_large_furniture(item) else 10
        ttc = item.get("ttc")
        if ttc is not None and ttc <= TTC_DANGER_SECONDS:
            return 60
        if item.get("in_walking_path"):
            return 50
        if str(item.get("name", "")).casefold() == "person":
            return 40
        if item.get("motion_state") == "approaching" or item.get("crossing_path"):
            return 30
        return 20 if Navigation._is_large_furniture(item) else 10

    @staticmethod
    def _is_relevant(item: dict[str, Any]) -> bool:
        """Filter out distant side objects before they become alerts."""
        ttc = item.get("ttc")
        immediate = ttc is not None and ttc <= TTC_DANGER_SECONDS
        overlap = float(item.get("walking_corridor_overlap", 1.0 if item.get("in_walking_path") else 0.0))
        in_corridor = bool(item.get("in_walking_path")) and overlap >= WALKING_CORRIDOR_MIN_OVERLAP
        if not in_corridor:
            return False
        name = str(item.get("name", "")).casefold()
        if name in NAVIGATION_SMALL_OBJECTS:
            depth = item.get("depth")
            close = isinstance(depth, (int, float)) and depth >= NAVIGATION_SMALL_OBJECT_CLOSE_DEPTH
            return immediate or (
                close
                and overlap >= NAVIGATION_SMALL_OBJECT_MIN_CORRIDOR_OVERLAP
                and item.get("confidence", 0.0) >= NAVIGATION_MIN_CONFIDENCE
            )
        return immediate or (
            item.get("confidence", 0.0) >= NAVIGATION_MIN_CONFIDENCE
            and item.get("area_ratio", 0.0) >= NAVIGATION_MIN_AREA_RATIO
        )

    @staticmethod
    def _is_large_furniture(item: dict[str, Any]) -> bool:
        return str(item.get("name", "")).casefold() in {
            "bed", "couch", "sofa", "table", "chair", "desk", "cabinet",
            "dresser", "refrigerator", "wardrobe", "closet",
        }

    @staticmethod
    def _raw_state(item: dict[str, Any]) -> str:
        ttc = item.get("ttc")
        if ttc is not None and ttc <= TTC_DANGER_SECONDS:
            return "collision"
        if item.get("in_walking_path"):
            return "blocking"
        if item.get("motion_state") == "approaching":
            return "approaching"
        return "crossing" if item.get("crossing_path") else "nearby"

    def _stabilize_state(self, object_id: int, raw_state: str) -> str:
        """Require a repeated state before changing spoken obstacle semantics."""
        stable = self._stable_states.get(object_id)
        candidate, count = self._state_candidates.get(object_id, (raw_state, 0))
        if candidate == raw_state:
            count += 1
        else:
            candidate, count = raw_state, 1
        self._state_candidates[object_id] = (candidate, count)
        if stable is None or count >= NAVIGATION_STATE_CONFIRMATION_FRAMES:
            self._stable_states[object_id] = raw_state
            return raw_state
        return stable

    @staticmethod
    def _should_announce(
        previous: dict[str, Any] | None,
        risk: float,
        obstacle: dict[str, Any],
        state: str,
        now: float,
    ) -> bool:
        if previous is None:
            return True
        last_alert_time = float(previous.get("last_alert_time", 0.0))
        if last_alert_time <= 0.0 or previous.get("last_alert_type") is None:
            return True
        elapsed = now - last_alert_time
        return (
            elapsed >= NAVIGATION_ALERT_COOLDOWN_SECONDS
            or risk >= float(previous["risk"]) + RISK_REANNOUNCE_INCREASE
            or obstacle.get("position") != previous.get("position")
            or state != previous.get("last_alert_type")
        )

    def _discard_missing_states(self, seen_ids: set[int], now: float) -> None:
        """Reset missing tracks and remove expired alert state completely."""
        tracked_ids = set(self._spoken_by_object) | set(self._state_candidates) | set(self._stable_states)
        for object_id in tracked_ids:
            if object_id not in seen_ids:
                track = self._spoken_by_object.get(object_id, {})
                track["consecutive_frames"] = 0
                last_seen = float(track.get("last_seen_time", 0.0))
                if now - last_seen >= NAVIGATION_TRACK_STATE_TIMEOUT_SECONDS:
                    self._spoken_by_object.pop(object_id, None)
                    self._state_candidates.pop(object_id, None)
                    self._stable_states.pop(object_id, None)