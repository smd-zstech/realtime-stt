"""
Translation module - translates English text to Korean.

Supports two backends:
- "google": Google Translate via deep-translator (fast, online)
- "ai": Facebook NLLB-200 model (better quality, offline, ~600MB)

Runs translations in a background thread to avoid blocking the main loop.
"""

import threading
import queue
from typing import NamedTuple


class TranslationResult(NamedTuple):
    english: str
    korean: str


class _GoogleBackend:
    """Google Translate backend with retry and rate-limit handling."""

    def __init__(self):
        from deep_translator import GoogleTranslator
        self._translator = GoogleTranslator(source="en", target="ko")
        self._last_request_time = 0.0

    def translate(self, text: str) -> str:
        import time

        max_retries = 3
        for attempt in range(max_retries):
            # Rate limit: ensure at least 0.5s between requests
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < 0.5:
                time.sleep(0.5 - elapsed)

            try:
                self._last_request_time = time.monotonic()
                result = self._translator.translate(text)
                return result if result else "(translation failed)"
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 2  # 2s, 4s backoff
                    print(f"[WARN] Translation retry {attempt + 1}/{max_retries} "
                          f"after {wait}s: {e}")
                    time.sleep(wait)
                    # Recreate translator instance in case of stale connection
                    from deep_translator import GoogleTranslator
                    self._translator = GoogleTranslator(source="en", target="ko")
                else:
                    print(f"[ERROR] Translation failed after {max_retries} retries: {e}")
                    return "(translation failed)"
        return "(translation failed)"


class _NLLBBackend:
    """Local AI translation using Facebook NLLB-200 (distilled, 600M)."""

    def __init__(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = "facebook/nllb-200-distilled-600M"
        print("[INFO] Loading NLLB-200 translation model (first run downloads ~600MB)...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._tokenizer.src_lang = "eng_Latn"
        self._ko_token_id = self._tokenizer.convert_tokens_to_ids("kor_Hang")
        print("[INFO] Translation model loaded.")

    def translate(self, text: str) -> str:
        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = self._model.generate(
            **inputs,
            forced_bos_token_id=self._ko_token_id,
            num_beams=4,
            max_length=512,
        )
        return self._tokenizer.decode(translated[0], skip_special_tokens=True)


class Translator:
    """Translates English text to Korean in a background thread."""

    def __init__(self, backend: str = "google"):
        """
        Args:
            backend: "google" for Google Translate, "ai" for local NLLB-200 model.
        """
        if backend == "ai":
            self._backend = _NLLBBackend()
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
