"""Single-slot asynchronous speech, with newer messages replacing stale pending ones."""

from __future__ import annotations

from queue import Empty, Full, Queue
import subprocess
from threading import Event, Lock, Thread

from config import VOICE_NAME, VOICE_POLL_INTERVAL_SECONDS
from debug import logger


class VoiceOutput:
    def __init__(self) -> None:
        self._queue: Queue[str] = Queue(maxsize=1)
        self._last_message: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = Lock()
        self._stop = Event()
        self._speaking = Event()
        self._thread = Thread(target=self._run, daemon=True, name="voice-output")
        self._thread.start()

    def speak(self, message: str | None, dialogue: bool = False) -> None:
        if self._stop.is_set() or not message or (message == self._last_message and not dialogue):
            return
        self._last_message = message
        # Set before queueing so the listener cannot capture pending TTS.
        self._speaking.set()
        if dialogue:
            with self._lock:
                if self._process and self._process.poll() is None:
                    self._process.terminate()
        try:
            self._queue.put_nowait(message)
        except Full:
            try:
                self._queue.get_nowait()  # Replace pending, outdated speech.
            except Empty:
                pass
            self._queue.put_nowait(message)

    def stop_current_speech(self) -> None:
        """Interrupt any speech immediately, including navigation alerts."""
        with self._lock:
            if self._process and self._process.poll() is None:
                self._process.terminate()
        try:
            while True:
                self._queue.get_nowait()
        except Empty:
            pass
        self._speaking.clear()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                message = self._queue.get(timeout=VOICE_POLL_INTERVAL_SECONDS)
                with self._lock:
                    self._process = subprocess.Popen(["say", "-v", VOICE_NAME, message])
                self._process.wait()
                with self._lock:
                    self._process = None
                if self._queue.empty():
                    self._speaking.clear()
            except Empty:
                continue
            except OSError as error:
                logger.warning("Voice output unavailable: %s", error)
                if self._queue.empty():
                    self._speaking.clear()
            except Exception as error:
                # A worker failure must not terminate the only speech thread.
                logger.exception("Unexpected voice-output failure: %s", error)
                if self._queue.empty():
                    self._speaking.clear()

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop pending speech safely; safe to call more than once."""
        self._stop.set()
        self._speaking.clear()
        with self._lock:
            if self._process and self._process.poll() is None:
                self._process.terminate()
        self._thread.join(timeout=timeout)

    def is_speaking(self) -> bool:
        """Return whether speech is queued or currently playing."""
        return self._speaking.is_set()


_voice = VoiceOutput()


def speak(message: str | None, dialogue: bool = False) -> None:
    _voice.speak(message, dialogue)


def stop_speech() -> None:
    """Interrupt any in-progress or queued speech immediately."""
    _voice.stop_current_speech()


def shutdown_voice() -> None:
    """Release the module-level speech worker during application shutdown."""
    _voice.shutdown()


def is_speaking() -> bool:
    """Expose speech activity without changing the existing ``speak`` API."""
    return _voice.is_speaking()