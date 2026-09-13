"""Application coordinator; reusable from desktop, web, and API front ends."""

from __future__ import annotations


import time
from threading import RLock
from typing import Any, Callable

import numpy as np

from assistant import VisionAssistant
from config import ROOM_ANNOUNCEMENT_CONFIDENCE_THRESHOLD
from debug import debug_throttled, logger
from dialogue import DialogueManager
from health import run_startup_checks
from llm import BaseLLM
from memory import SemanticMemory
from motion import MotionAnalyzer
from navigation import Navigation
from ollama_llm import OllamaLLM
from scene_graph import SceneGraph
from scene_understanding import SceneUnderstanding
from voice import is_speaking, shutdown_voice, stop_speech

class AssistantController:
    def __init__(
        self,
        speak: Callable[[str | None, bool], None],
        llm: BaseLLM | None = None,
    ) -> None:
        self.health = run_startup_checks()
        self.vision = VisionAssistant()
        self.motion = MotionAnalyzer()

        self.scene_graph = SceneGraph()
        self.memory = SemanticMemory()
        self.memory.load()  # Restore previously persisted memory, if any.
        self.understanding = SceneUnderstanding()

        self.navigation = Navigation()

        # One LLM instance for the whole application.
        self.llm = llm or OllamaLLM()

        self.dialogue = DialogueManager(
            self.scene_graph,
            self.memory,
            self.understanding,
            self.llm,
        )

        self._speak = speak
        self._questions_active = False
        self._state_lock = RLock()
        self._shutdown = False

        # Keep latest processed scene.
        self._latest_scene: dict[int, dict] = {}

        # Prevent repeating the same navigation alert.
        self._last_alert: str | None = None
        self._last_announced_room: str | None = None

        # --- Dashboard state (additive; read-only from the web layer) ---
        # These exist purely so a monitoring UI (web.py) can show what the
        # voice-first assistant is already doing. Nothing here changes how
        # begin_question/handle_question/process_frame behave for voice.
        self._current_room: str | None = None
        self._last_question: str | None = None
        self._last_answer: str | None = None

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, str | None]:

        started = time.perf_counter()
        try:
            detections, annotated = self.vision.detect(frame)

            stable = self.motion.update(
                detections,
                time.monotonic(),
                frame.shape[1],
            )

            scene = self.scene_graph.update(stable)

            with self._state_lock:
                self._latest_scene = scene

            room = self.understanding.infer_room(scene)
            scene = self.scene_graph.set_room(room.get("estimated_room"))

            estimated_room = room.get("estimated_room")

            with self._state_lock:
                self._current_room = estimated_room

            room_announcement = None
            if (
                estimated_room
                and room.get("confidence", 0.0) >= ROOM_ANNOUNCEMENT_CONFIDENCE_THRESHOLD
                and estimated_room != self._last_announced_room
            ):
                room_announcement = f"You appear to be in a {estimated_room}."

            memory_started = time.perf_counter()
            self.memory.update(scene, room)
            debug_throttled("memory_update", seconds=round(time.perf_counter() - memory_started, 3))

            alert = None

            # Don't interrupt while answering user questions.
            with self._state_lock:
                questions_active = self._questions_active
            if not questions_active:
                navigation_started = time.perf_counter()
                alert = self.navigation.generate_alert(scene.values())
                debug_throttled("navigation_timing", seconds=round(time.perf_counter() - navigation_started, 3))

            # Speak only if alert changed.
            with self._state_lock:
                previous_alert = self._last_alert
            if alert and alert != previous_alert:
                with self._state_lock:
                    self._last_alert = alert
                self._speak(alert, False)

            # Navigation has priority over an informational room announcement.
            # Keep the room pending until a quiet frame, then announce it once.
            if room_announcement and alert is None and not questions_active:
                self._speak(room_announcement, False)
                self._last_announced_room = estimated_room

            # Clear remembered alert once danger disappears.
            if alert is None:
                with self._state_lock:
                    self._last_alert = None

            debug_throttled(
                "frame_processing",
                seconds=round(time.perf_counter() - started, 3),
                visible_objects=len(scene),
            )

            return annotated, alert

        except Exception as error:
            logger.exception(
                "Frame processing failed: %s",
                error,
            )
            return frame, None

    def begin_question(self) -> None:
        logger.debug("Voice question started.")
        stop_speech()
        with self._state_lock:
            self._questions_active = True

    def cancel_question(self) -> None:
        """Reset ask-mode when no question was actually heard.

        Without this, a wake word followed by silence/noise would leave
        _questions_active stuck at True forever, permanently disabling
        navigation alerts — unacceptable for a blind-assistance tool.
        """
        logger.debug("Voice question cancelled (nothing heard).")
        with self._state_lock:
            self._questions_active = False

    def handle_question(
        self,
        question: str,
    ) -> str:

        try:
            logger.info("[User question]: %s", question)

            answer = self.dialogue.answer(question)

            logger.info("[Assistant answer]: %s", answer)

            with self._state_lock:
                self._last_question = question
                self._last_answer = answer

            self._speak(answer, True)

            return answer

        except Exception as error:
            logger.exception(
                "Dialogue failed: %s",
                error,
            )

            return "Sorry, I couldn't answer that."

        finally:
            with self._state_lock:
                self._questions_active = False

    def get_dashboard_state(self) -> dict[str, Any]:
        """Read-only snapshot for the web dashboard. Never used by voice logic."""
        with self._state_lock:
            room = self._current_room
            alert = self._last_alert
            questions_active = self._questions_active
            last_question = self._last_question
            last_answer = self._last_answer

        if is_speaking():
            voice_status = "speaking"
        elif questions_active:
            voice_status = "listening"
        else:
            voice_status = "idle"

        return {
            "room": room,
            "alert": alert,
            "voice_status": voice_status,
            "last_question": last_question,
            "last_answer": last_answer,
            "timestamp": time.time(),
        }

    def shutdown(self) -> None:
        """Call this before the app exits to persist memory to disk."""
        with self._state_lock:
            if self._shutdown:
                return
            self._shutdown = True
        if not self.memory.save():
            logger.error("Semantic memory was not saved during shutdown.")
        shutdown_voice()