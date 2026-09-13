"""Camera lifecycle and presentation only; processing belongs to the controller.

This file replaces the version that dropped cv2.imshow()/cv2.waitKey()
entirely. That version broke app.py (the desktop entry point, used by the
voice-first assistant itself) because it silently stopped showing the
annotated window it always used to show, and it never explained that
trade-off to you.

Instead, the window is now optional (``show_window``), defaulting to True so
app.py behaves exactly as before with zero changes. web.py explicitly passes
``show_window=False`` and reads frames via ``get_latest_frame()`` for the
MJPEG stream. Both entry points share one implementation instead of forking
into two different cameras that can drift apart.
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Protocol

import cv2
import numpy as np

from config import (
    CAMERA_INDEX,
    CAMERA_WARMUP_RETRIES,
    CAMERA_WARMUP_RETRY_DELAY_SECONDS,
    CAMERA_WINDOW_NAME,
    DETECTION_INTERVAL_SECONDS,
)
from debug import debug_throttled, logger


class FrameController(Protocol):
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str | None]:
        ...


class Camera:
    def __init__(
        self,
        controller: FrameController,
        index: int = CAMERA_INDEX,
        show_window: bool = True,
    ) -> None:
        self.controller = controller
        self.index = index
        self.show_window = show_window

        # Latest annotated frame, kept for web streaming (Flask reads this
        # from a different thread than the one running the capture loop).
        self._latest_frame: np.ndarray | None = None
        self._frame_lock = Lock()

    def get_latest_frame(self) -> np.ndarray | None:
        """Thread-safe read of the most recent annotated frame, or None."""
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def run(self) -> None:
        capture = cv2.VideoCapture(self.index)

        if not capture.isOpened():
            logger.error("Camera %s could not be opened.", self.index)
            return

        annotated = None
        last_processed = 0.0
        # macOS cameras sometimes report "opened" before they can actually
        # deliver a frame yet. A single failed read used to end the whole
        # capture session immediately, forcing a full app restart every time
        # this happened at startup. Retry briefly instead of giving up on
        # the very first miss.
        failed_reads = 0

        print("Camera started.")

        try:
            while True:

                success, frame = capture.read()

                if not success:
                    failed_reads += 1
                    if failed_reads <= CAMERA_WARMUP_RETRIES:
                        time.sleep(CAMERA_WARMUP_RETRY_DELAY_SECONDS)
                        continue
                    logger.error("Camera %s failed to read a frame; stopping capture.", self.index)
                    break

                failed_reads = 0

                now = time.monotonic()

                if now - last_processed >= DETECTION_INTERVAL_SECONDS:

                    annotated, alert = self.controller.process_frame(frame)

                    debug_throttled("camera_frames", index=self.index)

                    if alert:
                        logger.info("ALERT: %s", alert)

                    last_processed = now

                output = annotated if annotated is not None else frame

                with self._frame_lock:
                    self._latest_frame = output

                if self.show_window:
                    cv2.imshow(CAMERA_WINDOW_NAME, output)
                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        break

        finally:
            capture.release()
            if self.show_window:
                try:
                    cv2.destroyAllWindows()
                except cv2.error as error:
                    logger.debug("Could not destroy OpenCV windows during cleanup: %s", error)