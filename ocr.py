"""Extension contract for future text-recognition providers."""

from __future__ import annotations

from typing import Protocol
import numpy as np


class OCRProvider(Protocol):
    def read_text(self, frame: np.ndarray) -> list[str]: ...
