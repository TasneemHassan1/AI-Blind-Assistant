"""Contracts for optional future sensor-fusion sources."""

from __future__ import annotations

from typing import Any, Protocol


class SensorProvider(Protocol):
    def read(self) -> dict[str, Any]:
        """Return normalized sensor data without changing vision/navigation code."""
