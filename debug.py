"""Optional, dependency-free diagnostic logging."""

from __future__ import annotations

import logging
import time
from typing import Any

from config import DEBUG, DEBUG_LOG_INTERVAL_SECONDS, DEBUG_PERFORMANCE, LOG_FORMAT

# Keep third-party libraries quiet even when project diagnostics are enabled.
logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)
logger = logging.getLogger("blind_assistant")
logger.setLevel(logging.DEBUG if DEBUG else logging.WARNING)
_last_events: dict[str, float] = {}


def debug(event: str, **values: Any) -> None:
    """Emit structured diagnostic events at the configured interval."""
    if DEBUG:
        now = time.monotonic()
        if now - _last_events.get(event, 0.0) < DEBUG_LOG_INTERVAL_SECONDS:
            return
        _last_events[event] = now
        details = " ".join(f"{key}={value}" for key, value in values.items())
        logger.debug("%s %s", event, details)


def debug_throttled(event: str, interval_seconds: float | None = None, **values: Any) -> None:
    """Log periodic performance/state data without flooding the console."""
    if not DEBUG or not DEBUG_PERFORMANCE:
        return
    interval = DEBUG_LOG_INTERVAL_SECONDS if interval_seconds is None else interval_seconds
    now = time.monotonic()
    if now - _last_events.get(event, 0.0) < interval:
        return
    _last_events[event] = now
    details = " ".join(f"{key}={value}" for key, value in values.items())
    logger.debug("%s %s", event, details)


import logging
from logging.handlers import RotatingFileHandler

from config import DEBUG, DEBUG_LOG_INTERVAL_SECONDS, DEBUG_PERFORMANCE, LOG_FORMAT, PROJECT_ROOT

logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)
logger = logging.getLogger("blind_assistant")
logger.setLevel(logging.DEBUG if DEBUG else logging.WARNING)

if DEBUG:
    file_handler = RotatingFileHandler(
        PROJECT_ROOT / "assistant.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)