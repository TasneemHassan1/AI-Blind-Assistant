"""Thread-safe, versioned long-term semantic memory."""

from __future__ import annotations

import json
import os
import shutil
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Mapping

from config import (
    MAX_MEMORY_OBJECTS,
    MEMORY_CONFIDENCE_HISTORY_LIMIT,
    MEMORY_HISTORY_LIMIT,
    MEMORY_STORE_PATH,
)
from debug import debug_throttled, logger
from domain_models import MemoryRecord, Observation
from utils import utc_now


MEMORY_SCHEMA_VERSION = 2


class SemanticMemory:
    """Permanent memory, keyed by object name for stability across track ID changes."""

    def __init__(self) -> None:
        self._records: dict[int, MemoryRecord] = {}
        self._lock = RLock()
        self._created_at = utc_now()
        self._updated_at = self._created_at

    def _find_record_by_name(self, name: str) -> tuple[int, MemoryRecord] | None:
        """Find a record by name, return (track_id, record) or None."""
        name_lower = name.casefold()
        for track_id, record in self._records.items():
            if record.name.casefold() == name_lower:
                return track_id, record
            # Check aliases too
            for alias in record.aliases:
                if alias.casefold() == name_lower:
                    return track_id, record
        return None

    def update(
        self,
        scene: dict[int, dict[str, Any]] | Iterable[dict[str, Any]],
        room: dict[str, Any] | None = None,
    ) -> None:
        objects = scene.values() if isinstance(scene, dict) else scene
        with self._lock:
            for item in objects:
                self._update_one(item, room)
            self._trim_records()
            self._updated_at = utc_now()
            debug_throttled("memory_size", records=len(self._records))

    def _update_one(self, item: Mapping[str, Any], room: Mapping[str, Any] | None) -> None:
        raw_track_id = item.get("id") or item.get("track_id")
        if raw_track_id is None:
            return
        
        name = str(item.get("name", "unknown"))
        now = utc_now()
        
        # Try to find existing record by name first
        existing = self._find_record_by_name(name)
        
        if existing is not None:
            track_id, record = existing
            # Update existing record
            record.updated_at = now
            record.last_update_timestamp = now
            record.last_seen = item.get("last_seen") or now
            record.position = item.get("position")
            record.depth = item.get("depth")
            record.support_object = item.get("support_object")
            record.nearby_objects = list(item.get("near_objects", []))
            record.relationships = list(item.get("relationships", []))
            if room and room.get("estimated_room"):
                record.estimated_room = room.get("estimated_room")
            
            # Update track_id if it changed
            new_track_id = int(raw_track_id)
            if new_track_id != track_id and new_track_id not in self._records:
                # Move record to new track_id
                del self._records[track_id]
                record.track_id = new_track_id
                self._records[new_track_id] = record
                track_id = new_track_id
        else:
            # Create new record
            track_id = int(raw_track_id)
            if track_id in self._records:
                # Update existing by track_id (fallback)
                record = self._records[track_id]
                record.updated_at = now
                record.last_update_timestamp = now
                record.last_seen = item.get("last_seen") or now
                record.position = item.get("position")
                record.depth = item.get("depth")
                record.support_object = item.get("support_object")
                record.nearby_objects = list(item.get("near_objects", []))
                record.relationships = list(item.get("relationships", []))
                if room and room.get("estimated_room"):
                    record.estimated_room = room.get("estimated_room")
                return
            else:
                record = MemoryRecord(
                    track_id=track_id,
                    name=name,
                    created_at=now,
                    updated_at=now,
                    last_update_timestamp=now,
                    first_seen=item.get("first_seen") or now,
                    last_seen=item.get("last_seen") or now,
                    estimated_room=(room.get("estimated_room") if room else None) or "unknown",
                )
                self._records[track_id] = record

        # Add observation
        confidence = item.get("confidence")
        observation = Observation(
            track_id=track_id,
            last_seen=record.last_seen,
            position=record.position,
            depth=record.depth,
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            support_object=record.support_object,
            nearby_objects=list(record.nearby_objects),
            relationships=list(record.relationships),
        )
        serialized = observation.to_dict()
        if not record.observations or record.observations[-1].to_dict() != serialized:
            record.observations.append(observation)
            record.observations = record.observations[-MEMORY_HISTORY_LIMIT:]
        record.observation_count += 1
        if observation.confidence is not None:
            record.last_confidence = observation.confidence
            record.confidence_history.append(observation.confidence)
            record.confidence_history = record.confidence_history[-MEMORY_CONFIDENCE_HISTORY_LIMIT:]

    def _trim_records(self) -> None:
        while len(self._records) > MAX_MEMORY_OBJECTS:
            oldest_id = min(
                self._records,
                key=lambda key: self._records[key].last_seen or "",
            )
            del self._records[oldest_id]

    def find(self, name: str) -> list[dict[str, Any]]:
        """Backward-compatible alias for :meth:`find_by_name`."""
        return self.find_by_name(name)

    def find_by_name(self, name: str) -> list[dict[str, Any]]:
        """Find exact names or user-provided aliases, most recent first."""
        query = name.casefold().strip()
        with self._lock:
            records = [
                r for r in self._records.values()
                if r.name.casefold() == query
                or any(alias.casefold() == query for alias in r.aliases)
            ]
            return self._export_many(records)

    def find_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return most recently observed records, newest first."""
        with self._lock:
            return self._export_many(self._records.values(), limit)

    def find_by_room(self, room: str) -> list[dict[str, Any]]:
        query = room.casefold().strip()
        with self._lock:
            records = [
                r for r in self._records.values()
                if (r.estimated_room or "").casefold() == query
            ]
            return self._export_many(records)

    def list_objects(self) -> list[dict[str, Any]]:
        """Return all records sorted by most recent observation."""
        with self._lock:
            return self._export_many(self._records.values())

    def has_seen(self, name: str) -> bool:
        return bool(self.find_by_name(name))

    def add_alias(self, track_id: int, alias: str) -> bool:
        """Attach a non-destructive, user-friendly name to one memory record."""
        cleaned = alias.strip()
        if not cleaned:
            return False
        with self._lock:
            record = self._records.get(int(track_id))
            if record is None:
                # Try to find by name
                found = self._find_record_by_name(alias)
                if found is not None:
                    _, record = found
                else:
                    return False
            if all(existing.casefold() != cleaned.casefold() for existing in record.aliases):
                record.aliases.append(cleaned)
                record.updated_at = utc_now()
                self._updated_at = record.updated_at
            return True

    def recent_observations(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the newest individual observations across permanent memory."""
        with self._lock:
            observations: list[dict[str, Any]] = []
            for record in self._records.values():
                for observation in record.observations:
                    observations.append({"name": record.name, **observation.to_dict()})
            observations.sort(key=lambda item: item.get("last_seen") or "", reverse=True)
            return deepcopy(observations[:max(0, limit)])

    def find_closest(
        self,
        name: str,
        room: str | None = None,
        position: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the best name/alias match, biased toward optional context."""
        query = name.casefold().strip()
        if not query:
            return None
        room_query = room.casefold().strip() if room else None
        position_query = position.casefold().strip() if position else None
        with self._lock:
            def score(record: MemoryRecord) -> tuple[float, str]:
                labels = [record.name, *record.aliases]
                lexical = max(SequenceMatcher(None, query, label.casefold()).ratio() for label in labels)
                if query in (label.casefold() for label in labels):
                    lexical += 1.0
                context = 0.2 if room_query and (record.estimated_room or "").casefold() == room_query else 0.0
                context += 0.1 if position_query and (record.position or "").casefold() == position_query else 0.0
                return lexical + context, record.last_seen or ""
            if not self._records:
                return None
            best = max(self._records.values(), key=score)
            return deepcopy(best.to_dict())

    def statistics(self) -> dict[str, Any]:
        """Return lightweight aggregate metadata without exposing mutable state."""
        with self._lock:
            rooms: dict[str, int] = {}
            observation_total = 0
            aliases = 0
            for record in self._records.values():
                room = record.estimated_room or "unknown"
                rooms[room] = rooms.get(room, 0) + 1
                observation_total += record.observation_count
                aliases += len(record.aliases)
            return {
                "schema_version": MEMORY_SCHEMA_VERSION,
                "record_count": len(self._records),
                "observation_count": observation_total,
                "alias_count": aliases,
                "rooms": dict(sorted(rooms.items())),
                "created_at": self._created_at,
                "updated_at": self._updated_at,
                "last_update_timestamp": self._updated_at,
            }

    def _export_many(
        self, records: Iterable[MemoryRecord], limit: int | None = None
    ) -> list[dict[str, Any]]:
        ordered = sorted(records, key=lambda r: r.last_seen or "", reverse=True)
        if limit is not None:
            ordered = ordered[:max(0, limit)]
        return deepcopy([record.to_dict() for record in ordered])

    def snapshot(self) -> dict[int, dict[str, Any]]:
        with self._lock:
            return deepcopy({key: record.to_dict() for key, record in self._records.items()})

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._updated_at = utc_now()

    def save(self, path: str | Path | None = None) -> bool:
        """Atomically persist memory and retain the prior valid version as `.bak`."""
        target = Path(path) if path is not None else MEMORY_STORE_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._updated_at = utc_now()
            payload = {
                "schema_version": MEMORY_SCHEMA_VERSION,
                "created_at": self._created_at,
                "updated_at": self._updated_at,
                "records": {str(key): record.to_dict() for key, record in self._records.items()},
            }
        temporary = target.with_suffix(target.suffix + ".tmp")
        backup = target.with_suffix(target.suffix + ".bak")
        try:
            if target.exists():
                shutil.copy2(target, backup)
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            logger.debug("Saved %d semantic-memory record(s) to %s.", len(payload["records"]), target)
            return True
        except OSError as error:
            logger.exception("Could not save semantic memory to %s: %s", target, error)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def load(self, path: str | Path | None = None) -> bool:
        """Load current or legacy files without discarding in-memory memory on error."""
        target = Path(path) if path is not None else MEMORY_STORE_PATH
        if not target.exists():
            return False
        try:
            with target.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            records, created_at, updated_at = self._decode_payload(raw)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            backup = target.with_name(f"{target.name}.corrupt-{utc_now().replace(':', '-')}.bak")
            try:
                shutil.copy2(target, backup)
                logger.error("Invalid memory file backed up to %s: %s", backup, error)
            except OSError:
                logger.exception("Invalid memory file could not be backed up: %s", error)
            return False
        with self._lock:
            self._records = records
            self._created_at = created_at or self._created_at
            self._updated_at = updated_at or self._updated_at
        logger.debug("Loaded %d semantic-memory record(s) from %s.", len(records), target)
        return True

    @staticmethod
    def _decode_payload(raw: Any) -> tuple[dict[int, MemoryRecord], str | None, str | None]:
        if not isinstance(raw, Mapping):
            raise ValueError("memory payload is not an object")
        # Version 1 stored the records directly at the root. Preserve it.
        source = raw.get("records") if "records" in raw else raw
        if not isinstance(source, Mapping):
            raise ValueError("memory records are not an object")
        records: dict[int, MemoryRecord] = {}
        for key, value in source.items():
            if not isinstance(value, Mapping):
                logger.warning("Skipping invalid memory record %r.", key)
                continue
            try:
                track_id = int(key)
                records[track_id] = MemoryRecord.from_mapping(value, track_id)
            except (TypeError, ValueError) as error:
                logger.warning("Skipping invalid memory record %r: %s", key, error)
        return records, raw.get("created_at"), raw.get("updated_at") or raw.get("last_update_timestamp")