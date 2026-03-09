"""
Translation module - translates English text to Korean.

Supports two backends:
- "google": Google Translate via deep-translator (fast, online)
- "ai": Helsinki-NLP/opus-mt-en-ko MarianMT model (better quality, offline)

Runs translations in a background thread to avoid blocking the main loop.
"""

import threading
import queue
from typing import NamedTuple


class TranslationResult(NamedTuple):
    english: str
    korean: str


class _GoogleBackend:
    """Google Translate backend."""

    def __init__(self):
        from deep_translator import GoogleTranslator
        self._translator = GoogleTranslator(source="en", target="ko")

    def translate(self, text: str) -> str:
        result = self._translator.translate(text)
        return result if result else "(translation failed)"


class _MarianMTBackend:
    """Local AI translation using Helsinki-NLP/opus-mt-en-ko."""

    def __init__(self):
        from transformers import MarianMTModel, MarianTokenizer

        model_name = "Helsinki-NLP/opus-mt-en-ko"
        print("[INFO] Loading MarianMT translation model (first run downloads ~300MB)...")
        self._tokenizer = MarianTokenizer.from_pretrained(model_name)
        self._model = MarianMTModel.from_pretrained(model_name)
        print("[INFO] Translation model loaded.")

    def translate(self, text: str) -> str:
        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = self._model.generate(**inputs, num_beams=4, max_length=512)
        return self._tokenizer.decode(translated[0], skip_special_tokens=True)


class Translator:
    """Translates English text to Korean in a background thread."""

    def __init__(self, backend: str = "google"):
        """
        Args:
            backend: "google" for Google Translate, "ai" for local MarianMT model.
        """
        if backend == "ai":
            self._backend = _MarianMTBackend()
        else:
            self._backend = _GoogleBackend()

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
                korean_text = self._backend.translate(english_text)
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
