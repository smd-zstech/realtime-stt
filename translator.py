"""
Translation module - translates English text to Korean.

Supports two backends:
- "google": Google Translate via deep-translator (fast, online)
- "ai": Facebook NLLB-200 model (better quality, offline, ~600MB)

Provides both synchronous translate() and async submit()/get_result() APIs.
"""

import re
import threading
import time
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


# Patterns that indicate a broken/degenerate translation output.
_REPEATED_DOTS = re.compile(r"\.{5,}")  # 5+ consecutive dots
_REPEATED_CHAR = re.compile(r"(.)\1{9,}")  # same char 10+ times in a row


def _is_bad_translation(text: str) -> bool:
    """Detect degenerate translation output (dots, repeated chars/words)."""
    if not text or text == "(translation failed)":
        return False

    # Check for "....................................." patterns
    if _REPEATED_DOTS.search(text):
        return True

    # Check for single character repeated excessively (e.g. "ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ")
    if _REPEATED_CHAR.search(text):
        return True

    # Check for word-level repetition in translation
    # e.g. "번역 번역 번역 번역 번역 번역"
    words = text.split()
    if len(words) >= 4:
        # If any single word makes up 70%+ of all words, it's degenerate
        from collections import Counter
        counts = Counter(words)
        most_common_word, most_common_count = counts.most_common(1)[0]
        if most_common_count / len(words) >= 0.7 and most_common_count >= 4:
            return True

    return False


class Translator:
    """Translates English text to Korean. Thread-safe."""

    def __init__(self, backend: str = "google"):
        if backend == "ai":
            self._backend = _NLLBBackend()
        else:
            self._backend = _GoogleBackend()
        self._lock = threading.Lock()

    def translate(self, english_text: str) -> str:
        """Translate English text to Korean (blocking, thread-safe).

        Returns Korean text, or an error placeholder on failure.
        """
        with self._lock:
            try:
                result = self._backend.translate(english_text)
                if _is_bad_translation(result):
                    print(f"[WARN] Bad translation detected, retrying: "
                          f"{result[:60]}...")
                    # Retry once — transient API glitch
                    result = self._backend.translate(english_text)
                    if _is_bad_translation(result):
                        print(f"[WARN] Bad translation persists: {result[:60]}...")
                        return "(번역 실패 - 재시도 필요)"
                return result
            except Exception as e:
                print(f"[ERROR] Translation failed: {e}")
                return "(translation failed)"
