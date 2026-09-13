"""Best-effort startup checks that never change application control flow."""

from __future__ import annotations

import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import cv2

from config import (
    CAMERA_INDEX,
    HEALTH_CHECKS_ENABLED,
    HEALTH_CHECK_TIMEOUT_SECONDS,
    LLM_HOST,
    MEMORY_STORE_PATH,
    VOICE_NAME,
    YOLO_MODEL,
)
from debug import logger


def run_startup_checks() -> dict[str, bool]:
    """Validate optional dependencies and log actionable status, never raising."""
    if not HEALTH_CHECKS_ENABLED:
        return {}
    checks = {
        "yolo_model": _model_available(),
        "memory_file": _memory_available(),
        "ollama": _ollama_available(),
        "camera": _camera_available(),
        "voice": _voice_available(),
    }
    unavailable = [name for name, available in checks.items() if not available]
    if unavailable:
        logger.warning("Startup health checks found unavailable component(s): %s", ", ".join(unavailable))
    else:
        logger.debug("Startup health checks passed.")
    return checks


def _model_available() -> bool:
    exists = Path(YOLO_MODEL).is_file()
    if not exists:
        logger.error("Configured YOLO model is missing: %s", YOLO_MODEL)
    return exists


def _memory_available() -> bool:
    exists = MEMORY_STORE_PATH.exists()
    if not exists:
        logger.info("Memory file does not exist yet; it will be created on shutdown: %s", MEMORY_STORE_PATH)
    return exists


def _ollama_available() -> bool:
    # The generate endpoint needs POST data; `/api/tags` is a cheap readiness probe.
    endpoint = LLM_HOST.split("/api/", 1)[0] + "/api/tags"
    try:
        with urllib.request.urlopen(endpoint, timeout=HEALTH_CHECK_TIMEOUT_SECONDS):
            return True
    except (urllib.error.URLError, OSError) as error:
        logger.warning("Ollama is not reachable at %s: %s", endpoint, error)
        return False


def _camera_available() -> bool:
    capture = cv2.VideoCapture(CAMERA_INDEX)
    try:
        available = capture.isOpened()
        if not available:
            logger.warning("Camera %s is unavailable during startup check.", CAMERA_INDEX)
        return available
    except cv2.error as error:
        logger.warning("Camera %s check failed: %s", CAMERA_INDEX, error)
        return False
    finally:
        capture.release()


def _voice_available() -> bool:
    executable = shutil.which("say")
    if executable is None:
        logger.warning("macOS 'say' command is unavailable; voice '%s' cannot be used.", VOICE_NAME)
        return False
    try:
        # An empty utterance validates the configured voice without audible output.
        result = subprocess.run(
            [executable, "-v", VOICE_NAME, ""],
            capture_output=True,
            text=True,
            timeout=HEALTH_CHECK_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        logger.warning("Voice '%s' health check failed: %s", VOICE_NAME, error)
        return False
    if result.returncode != 0:
        logger.warning("Voice '%s' is unavailable: %s", VOICE_NAME, result.stderr.strip())
        return False
    return True
