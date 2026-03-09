"""
Speech-to-text engine with context-aware inference.

Uses faster-whisper for transcription. Maintains a rolling context of recent
transcriptions so that Whisper can use prior sentences as a prompt, improving
accuracy for quiet or unclear words heard through a laptop microphone.
"""

from collections import deque

import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    """Transcribes audio segments using Whisper with context-based inference."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        language: str = "en",
        context_window: int = 5,
    ):
        """
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3).
                        Larger models are more accurate but slower.
            device: "cpu", "cuda", or "auto".
            compute_type: "default", "float16", "int8", etc.
            language: Language code for transcription.
            context_window: Number of recent sentences kept as context prompt.
        """
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
        self.language = language
        self._context: deque[str] = deque(maxlen=context_window)

    def _build_context_prompt(self) -> str | None:
        """Build a prompt from recent transcriptions for context inference."""
        if not self._context:
            return None
        return " ".join(self._context)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe an audio segment with context-aware inference.

        The previous transcriptions are passed as an `initial_prompt` to Whisper,
        which biases the model toward vocabulary and phrasing consistent with the
        ongoing conversation. This helps recover quiet or unclear words that a
        laptop microphone may not capture well.

        Args:
            audio: Float32 numpy array of audio samples.
            sample_rate: Sample rate of the audio (must be 16000 for Whisper).

        Returns:
            Transcribed text string.
        """
        context_prompt = self._build_context_prompt()

        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            initial_prompt=context_prompt,
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=300,
            ),
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        full_text = " ".join(text_parts).strip()

        if full_text:
            self._context.append(full_text)

        return full_text

    def reset_context(self):
        """Clear the context history."""
        self._context.clear()
