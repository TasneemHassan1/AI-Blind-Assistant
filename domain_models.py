"""Typed, serializable domain models used internally by the assistant.

The public APIs still exchange dictionaries so existing integrations continue to
work.  These models are deliberately small adapters at module boundaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class Detection:
    """A detector/tracker observation for one object in one frame."""

    id: int | None
    name: str
    confidence: float
    bounding_box: tuple[float, float, float, float]
    center: tuple[float, float]
    tracked: bool
    depth: float | None = None
    distance_meters: float | None = None
    distance_cm: float | None = None
    position: str | None = None
    area_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SceneObject:
    """A stable tracked object and its derived spatial properties."""

    id: int
    name: str
    first_seen: str
    last_seen: str
    position: str | None = None
    depth: float | None = None
    support_object: str | None = None
    near_objects: list[str] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class Observation:
    """One persistent-memory observation, safe to serialize as JSON."""

    track_id: int
    last_seen: str | None
    position: str | None
    depth: float | None
    confidence: float | None = None
    support_object: str | None = None
    nearby_objects: list[str] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemoryRecord:
    """Versioned semantic-memory record for one currently known track key."""

    track_id: int
    name: str
    created_at: str
    updated_at: str
    first_seen: str | None
    last_seen: str | None
    position: str | None = None
    depth: float | None = None
    estimated_room: str | None = None
    support_object: str | None = None
    nearby_objects: list[str] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    observation_count: int = 0
    confidence_history: list[float] = field(default_factory=list)
    last_confidence: float | None = None
    # Explicit additive alias for consumers that distinguish record creation
    # from its most recent mutation. ``updated_at`` remains authoritative.
    last_update_timestamp: str | None = None
    aliases: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], track_id: int) -> "MemoryRecord":
        observations = [
            Observation(
                track_id=int(item.get("track_id", track_id)),
                last_seen=item.get("last_seen"),
                position=item.get("position"),
                depth=item.get("depth"),
                confidence=item.get("confidence"),
                support_object=item.get("support_object"),
                nearby_objects=list(item.get("nearby_objects", [])),
                relationships=list(item.get("relationships", [])),
            )
            for item in data.get("observations", [])
            if isinstance(item, Mapping)
        ]
        confidence_history = [
            float(value) for value in data.get("confidence_history", [])
            if isinstance(value, (int, float))
        ]
        first_seen = data.get("first_seen")
        last_seen = data.get("last_seen")
        return cls(
            track_id=track_id,
            name=str(data.get("name", "unknown")),
            created_at=str(data.get("created_at") or first_seen or last_seen or ""),
            updated_at=str(data.get("updated_at") or last_seen or first_seen or ""),
            first_seen=first_seen,
            last_seen=last_seen,
            position=data.get("position"),
            depth=data.get("depth"),
            estimated_room=data.get("estimated_room"),
            support_object=data.get("support_object"),
            nearby_objects=list(data.get("nearby_objects", [])),
            relationships=list(data.get("relationships", [])),
            observations=observations,
            observation_count=int(data.get("observation_count", len(observations))),
            confidence_history=confidence_history,
            last_confidence=data.get("last_confidence"),
            last_update_timestamp=data.get("last_update_timestamp") or data.get("updated_at"),
            aliases=[str(alias) for alias in data.get("aliases", []) if isinstance(alias, str)],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NavigationAlert:
    """A selected navigation hazard; text remains the public output."""

    object_id: int
    object_name: str
    risk: float
    message: str
