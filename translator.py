"""
Translation module - translates English text to Korean using Google Translate.
Runs translations in a background thread to avoid blocking the main loop.
"""

import threading
import queue
from typing import NamedTuple

from deep_translator import GoogleTranslator


class TranslationResult(NamedTuple):
    english: str
    korean: str


class Translator:
    """Translates English text to Korean in a background thread."""

    def __init__(self):
        self._translator = GoogleTranslator(source="en", target="ko")
        self._input_queue: queue.Queue[str] = queue.Queue()
        self._output_queue: queue.Queue[TranslationResult] = queue.Queue()
        self._running = False
        self._thread = None

    def _worker(self):
        """Background worker that processes translation requests."""
        while self._running:
            try:
                english_text = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                korean_text = self._translator.translate(english_text)
                if korean_text is None:
                    korean_text = "(translation failed)"
            except Exception:
                korean_text = "(translation error)"

            self._output_queue.put(TranslationResult(english_text, korean_text))

    def start(self):
        """Start the translation worker thread."""
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the translation worker thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def submit(self, english_text: str):
        """Submit English text for translation."""
        self._input_queue.put(english_text)

    def get_result(self, timeout: float = None) -> TranslationResult | None:
        """Get a translation result. Returns None on timeout."""
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
