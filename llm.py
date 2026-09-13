"""Provider-neutral contracts reserved for future LLM integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):

    @abstractmethod
    def generate_response(
        self,
        question: str,
        scene: dict[str, Any],
        memory: dict[str, Any],
    ) -> str:
        """Generate an answer grounded in the current scene and semantic memory."""
        raise NotImplementedError