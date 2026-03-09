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

    def __init__(self, model_size: str, device: str, compute_type: str):
        from faster_whisper import WhisperModel

        if compute_type == "default":
            compute_type = "float16" if device == "cuda" else "int8"
        print(f"[INFO] faster-whisper device={device}, compute_type={compute_type}")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, language: str, initial_prompt: str | None) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language=language,
            initial_prompt=initial_prompt,
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
        return " ".join(seg.text.strip() for seg in segments).strip()


# ---------------------------------------------------------------------------
# Backend: OpenVINO (Intel GPU / NPU)
# ---------------------------------------------------------------------------

class _OpenVINOBackend:
    """Wraps optimum-intel OpenVINO pipeline for Intel GPU/NPU inference."""

    def __init__(self, model_size: str, ov_device: str):
        from optimum.intel import OVModelForSpeechSeq2Seq
        from transformers import AutoProcessor, pipeline

        model_id = _WHISPER_HF_MODELS.get(model_size, f"openai/whisper-{model_size}")
        # ov_device: "GPU" or "NPU"
        print(f"[INFO] OpenVINO model={model_id}, device={ov_device}")
        print("[INFO] Loading model (first run may take a while for export)...")

        model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id, export=True, device=ov_device,
        )
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
            device: "cpu", "cuda", "openvino-gpu", "openvino-npu", or "auto".
            compute_type: CTranslate2 compute type (ignored for OpenVINO).
            language: Language code for transcription.
            context_window: Number of recent sentences kept as context prompt.
        """
        resolved = _resolve_device(device)

        if resolved.startswith("openvino"):
            ov_device = "GPU" if resolved == "openvino-gpu" else "NPU"
            self._backend = _OpenVINOBackend(model_size, ov_device)
        else:
            self._backend = _FasterWhisperBackend(model_size, resolved, compute_type)

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
