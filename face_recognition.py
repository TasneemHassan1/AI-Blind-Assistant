"""Extension contract for future privacy-aware face-recognition providers."""

from __future__ import annotations

from typing import Protocol
import numpy as np


class FaceRecognitionProvider(Protocol):
    def identify(self, frame: np.ndarray) -> list[str]: ...
