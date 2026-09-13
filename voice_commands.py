"""Wake-word based Ask Mode, isolated from navigation decisions."""

from __future__ import annotations

from threading import Event, Thread
from typing import Callable

import speech_recognition as sr

from config import (
    VOICE_COMMAND_WAKE_WORD,
    VOICE_FOLLOW_UP_LISTEN_TIMEOUT_SECONDS,
    VOICE_LISTEN_TIMEOUT_SECONDS,
    VOICE_MINIMUM_QUESTION_CHARACTERS,
    VOICE_PAUSE_THRESHOLD_SECONDS,
    VOICE_PHRASE_LIMIT_SECONDS,
    VOICE_SPEECH_GUARD_SECONDS,
)
from debug import logger
from voice import is_speaking


class VoiceCommandListener:
    def __init__(
        self,
        on_wake: Callable[[], None],
        on_question: Callable[[str], None],
        on_cancel: Callable[[], None] | None = None,
    ) -> None:

        self._on_wake = on_wake
        self._on_question = on_question
        self._on_cancel = on_cancel or (lambda: None)

        self._recognizer = sr.Recognizer()
        self._recognizer.pause_threshold = VOICE_PAUSE_THRESHOLD_SECONDS
        self._recognizer.dynamic_energy_threshold = True

        self._stop = Event()

    def start(self) -> Thread:
        thread = Thread(
            target=self._listen_forever,
            daemon=True,
            name="voice-commands",
        )
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()

    def _listen_forever(self) -> None:

        while not self._stop.is_set():

            if is_speaking():
                self._stop.wait(VOICE_SPEECH_GUARD_SECONDS)
                continue

            phrase = self.listen_once()

            if not phrase:
                continue

            logger.debug("Heard: %s", phrase)

            if VOICE_COMMAND_WAKE_WORD not in phrase:
                continue

            self._on_wake()

            question = phrase.replace(
                VOICE_COMMAND_WAKE_WORD,
                "",
                1,
            ).strip()

            # لو قال "assistant" فقط
            if not question:
                logger.debug("Waiting for follow-up question...")
                question = self.listen_once(VOICE_FOLLOW_UP_LISTEN_TIMEOUT_SECONDS)

            if not question:
                logger.debug("No question heard; cancelling ask mode.")
                self._on_cancel()
                continue

            question = question.strip()

            if not self._is_complete_question(question):
                logger.debug("Ignoring incomplete voice question: %s", question)
                self._on_cancel()
                continue

            logger.debug("Question: %s", question)

            try:
                self._on_question(question)
            except Exception as error:
                logger.exception(
                    "Question handling failed: %s",
                    error,
                )
                self._on_cancel()

    def listen_once(self, timeout_seconds: float | None = None) -> str | None:

        try:

            if is_speaking():
                return None

            with sr.Microphone() as source:

                self._recognizer.adjust_for_ambient_noise(
                    source,
                    duration=0.5,
                )

                audio = self._recognizer.listen(
                    source,
                    timeout=timeout_seconds if timeout_seconds is not None else VOICE_LISTEN_TIMEOUT_SECONDS,
                    phrase_time_limit=VOICE_PHRASE_LIMIT_SECONDS,
                )

            text = self._recognizer.recognize_google(
                audio,
                language="en-US",
            )

            return text.lower().strip()

        except sr.WaitTimeoutError:
            logger.debug("Listening timed out.")
            return None

        except sr.UnknownValueError:
            logger.debug("Speech not understood.")
            return None

        except sr.RequestError as error:
            logger.debug("Speech service unavailable: %s", error)
            return None

        except OSError as error:
            logger.debug("Microphone unavailable: %s", error)
            return None

        except Exception as error:
            logger.exception(
                "Unexpected voice error: %s",
                error,
            )
            return None

    @staticmethod
    def _is_complete_question(question: str) -> bool:
        """Reject only empty/noise fragments once the recognizer ends speech."""
        cleaned = question.strip()
        if len(cleaned) < VOICE_MINIMUM_QUESTION_CHARACTERS:
            return False
        return bool(cleaned.split())