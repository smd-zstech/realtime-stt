"""
Speech-to-text engine with context-aware inference.

Uses faster-whisper (CTranslate2) or OpenVINO as backend depending on device.

Device selection:
- "cuda"         : NVIDIA GPU (requires CUDA toolkit, uses faster-whisper)
- "cpu"          : CPU (uses faster-whisper)
- "openvino-gpu" : Intel GPU via OpenVINO (e.g. Intel Arc)
- "openvino-npu" : Intel NPU via OpenVINO (e.g. Intel AI Boost)
- "auto"         : Tries CUDA -> OpenVINO GPU -> CPU
"""

from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    """Check if a working NVIDIA CUDA environment is present."""
    try:
        import ctranslate2
        return "cuda" in ctranslate2.get_supported_compute_types("cuda")
    except Exception:
        pass
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _openvino_available() -> bool:
    """Check if OpenVINO + optimum-intel are installed."""
    try:
        from optimum.intel import OVModelForSpeechSeq2Seq  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_device(device: str) -> str:
    """Resolve user-facing device string to an internal device key.

    Returns one of: "cuda", "cpu", "openvino-gpu", "openvino-npu".
    """
    if device in ("openvino-gpu", "openvino-npu"):
        if _openvino_available():
            return device
        print(f"[WARN] {device} requested but OpenVINO not installed. Falling back to CPU.")
        return "cpu"

    if device == "cuda":
        if _cuda_available():
            return "cuda"
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return "cpu"

    if device == "auto":
        if _cuda_available():
            print("[INFO] NVIDIA GPU detected. Using CUDA.")
            return "cuda"
        if _openvino_available():
            print("[INFO] OpenVINO detected. Using Intel GPU.")
            return "openvino-gpu"
        print("[INFO] Using CPU.")
        return "cpu"

    return "cpu"


# ---------------------------------------------------------------------------
# Model size -> HuggingFace model ID mapping
# ---------------------------------------------------------------------------

_WHISPER_HF_MODELS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large-v3": "openai/whisper-large-v3",
}


# ---------------------------------------------------------------------------
# Backend: faster-whisper (CUDA / CPU)
# ---------------------------------------------------------------------------

class _FasterWhisperBackend:
    """Wraps faster-whisper for CUDA and CPU inference."""

    def __init__(self, model_size: str, device: str, compute_type: str, beam_size: int = 1):
        from faster_whisper import WhisperModel

        if compute_type == "default":
            compute_type = "float16" if device == "cuda" else "int8"
        print(f"[INFO] faster-whisper device={device}, compute_type={compute_type}")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._beam_size = beam_size

    def transcribe(self, audio: np.ndarray, language: str, initial_prompt: str | None) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language=language,
            initial_prompt=initial_prompt,
            beam_size=self._beam_size,
            best_of=self._beam_size,
            temperature=0.0,
            condition_on_previous_text=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=250,
            ),
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


# ---------------------------------------------------------------------------
# Backend: OpenVINO (Intel GPU / NPU)
# ---------------------------------------------------------------------------

class _OpenVINOBackend:
    """Wraps optimum-intel OpenVINO pipeline for Intel GPU/NPU inference."""

    def __init__(self, model_size: str, ov_device: str):
        import os
        from pathlib import Path
        from optimum.intel import OVModelForSpeechSeq2Seq
        from transformers import AutoProcessor, pipeline

        model_id = _WHISPER_HF_MODELS.get(model_size, f"openai/whisper-{model_size}")
        # Local cache dir for exported IR models
        cache_dir = Path.home() / ".cache" / "realtime-stt-ov" / model_size
        print(f"[INFO] OpenVINO model={model_id}, device={ov_device}")

        # Check if we already have a cached IR model
        cached = cache_dir.exists() and (cache_dir / "openvino_encoder_model.xml").exists()

        if cached:
            print(f"[INFO] Loading cached IR model from {cache_dir}")
            try:
                model = OVModelForSpeechSeq2Seq.from_pretrained(
                    str(cache_dir), device=ov_device,
                )
                print(f"[INFO] Successfully loaded on {ov_device} from cache.")
            except Exception as e:
                print(f"[WARN] Failed to load cache on {ov_device}: {e}")
                print("[INFO] Retrying cached model on CPU...")
                model = OVModelForSpeechSeq2Seq.from_pretrained(
                    str(cache_dir), device="CPU",
                )
        else:
            print("[INFO] First run — exporting model (this may take a while)...")
            # Export on CPU first (more reliable), then save to cache
            try:
                model = OVModelForSpeechSeq2Seq.from_pretrained(
                    model_id, export=True, device="CPU",
                )
                # Save exported IR to cache for future GPU loading
                cache_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(cache_dir))
                print(f"[INFO] Cached IR model to {cache_dir}")

                # Now reload on target device if not CPU
                if ov_device != "CPU":
                    try:
                        model = OVModelForSpeechSeq2Seq.from_pretrained(
                            str(cache_dir), device=ov_device,
                        )
                        print(f"[INFO] Reloaded on {ov_device} successfully.")
                    except Exception as e:
                        print(f"[WARN] {ov_device} reload failed: {e}")
                        print("[INFO] Continuing on CPU.")
                        model = OVModelForSpeechSeq2Seq.from_pretrained(
                            str(cache_dir), device="CPU",
                        )
            except Exception as e:
                print(f"[WARN] OpenVINO export failed: {e}")
                raise

        processor = AutoProcessor.from_pretrained(model_id)

        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
        )

    def transcribe(self, audio: np.ndarray, language: str, initial_prompt: str | None) -> str:
        generate_kwargs = {"language": language, "task": "transcribe"}
        # Note: prompt_ids is intentionally not used here because the
        # HuggingFace pipeline passes them as numpy arrays internally,
        # which causes torch.cat to fail in whisper's generate().

        result = self._pipeline(
            {"raw": audio, "sampling_rate": 16000},
            generate_kwargs=generate_kwargs,
        )
        return result.get("text", "").strip()


# ---------------------------------------------------------------------------
# Public Transcriber class
# ---------------------------------------------------------------------------

_ACCENT_PROMPT = (
    "The following is a conversation that may include speakers with various "
    "English accents such as Indian, British, Australian, Singaporean, "
    "or other non-native English accents. Listen carefully for accent "
    "variations in pronunciation."
)


class Transcriber:
    """Transcribes audio segments using Whisper with context-based inference."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        language: str = "en",
        context_window: int = 5,
        beam_size: int = 3,
        accent_boost: bool = True,
    ):
        """
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3).
            device: "cpu", "cuda", "openvino-gpu", "openvino-npu", or "auto".
            compute_type: CTranslate2 compute type (ignored for OpenVINO).
            language: Language code for transcription.
            context_window: Number of recent sentences kept as context prompt.
            beam_size: Beam size for decoding (higher = more accurate, slower).
            accent_boost: Add accent-aware initial prompt for better recognition.
        """
        resolved = _resolve_device(device)

        if resolved.startswith("openvino"):
            ov_device = "GPU" if resolved == "openvino-gpu" else "NPU"
            try:
                self._backend = _OpenVINOBackend(model_size, ov_device)
            except Exception as e:
                print(f"[WARN] OpenVINO backend failed completely: {e}")
                print("[INFO] Falling back to faster-whisper on CPU.")
                self._backend = _FasterWhisperBackend(model_size, "cpu", compute_type, beam_size)
        else:
            self._backend = _FasterWhisperBackend(model_size, resolved, compute_type, beam_size)

        self.language = language
        self._accent_boost = accent_boost
        self._context: deque[str] = deque(maxlen=context_window)

    def _build_context_prompt(self) -> str | None:
        """Build a prompt from recent transcriptions for context inference."""
        parts = []
        if self._accent_boost:
            parts.append(_ACCENT_PROMPT)
        if self._context:
            parts.append(" ".join(self._context))
        return " ".join(parts) if parts else None

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe an audio segment with context-aware inference.

        Args:
            audio: Float32 numpy array of audio samples.
            sample_rate: Sample rate of the audio (must be 16000 for Whisper).

        Returns:
            Transcribed text string.
        """
        context_prompt = self._build_context_prompt()
        full_text = self._backend.transcribe(audio, self.language, context_prompt)

        if full_text:
            self._context.append(full_text)

        return full_text

    def reset_context(self):
        """Clear the context history."""
        self._context.clear()
