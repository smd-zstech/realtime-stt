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

import re
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
        # Check OpenVINO first — its import is lightweight.
        # ctranslate2/torch import (needed for CUDA check) can be heavy or
        # even hang on some Windows setups, so we avoid it when possible.
        if _openvino_available():
            print("[INFO] Intel GPU detected. Using OpenVINO GPU.")
            return "openvino-gpu"
        if _cuda_available():
            print("[INFO] NVIDIA GPU detected. Using CUDA.")
            return "cuda"
        print("[INFO] Using CPU (faster-whisper with int8).")
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
            temperature=[0.0, 0.2, 0.4, 0.6],
            condition_on_previous_text=True,
            suppress_blank=True,
            no_speech_threshold=0.5,
            log_prob_threshold=-0.8,
            compression_ratio_threshold=2.4,
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

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = model
        self._tokenizer = self._processor.tokenizer

    def transcribe(self, audio: np.ndarray, language: str, initial_prompt: str | None) -> str:
        # Prepare input features (mel spectrogram)
        inputs = self._processor(
            audio, sampling_rate=16000, return_tensors="pt",
        )
        input_features = inputs.input_features

        # Note: prompt_ids is NOT used with OpenVINO backend because
        # OVModelForSpeechSeq2Seq.generate() does not properly handle it —
        # it outputs the prompt text itself instead of transcribing audio.
        predicted_ids = self._model.generate(
            input_features,
            language=language,
            task="transcribe",
            num_beams=3,
            return_timestamps=False,
        )
        text = self._tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True,
        )
        return text[0].strip() if text else ""


# ---------------------------------------------------------------------------
# Public Transcriber class
# ---------------------------------------------------------------------------

_ACCENT_PROMPT = (
    "The following is a conversation that may include speakers with various "
    "English accents such as Indian, British, Australian, Singaporean, "
    "or other non-native English accents. Listen carefully for accent "
    "variations in pronunciation."
)

# Domain-specific vocabulary hints for Whisper initial_prompt.
# Whisper uses the prompt as context — mentioning these terms biases the
# decoder toward recognizing them correctly instead of similar-sounding words.
_DOMAIN_VOCAB = (
    # Zscaler products and features
    "Zscaler, ZIA, ZPA, ZDX, ZCC, Zscaler Client Connector, "
    "Zscaler Internet Access, Zscaler Private Access, "
    "Zscaler Digital Experience, Zscaler Zero Trust Exchange, "
    "ZTNA, Zero Trust Network Access, "
    "LA, Limited Availability, GA, General Availability, "
    "App Connector, Service Edge, Cloud Connector, Branch Connector, "
    "CASB, DLP, SWG, Secure Web Gateway, "
    "Zscaler Deception, ZPA Private Service Edge, "
    "NSS, Nanolog Streaming Service, "
    # Networking and security
    "SSL inspection, TLS, HTTPS, SD-WAN, MPLS, BGP, OSPF, "
    "SASE, SSE, SaaS, IaaS, PaaS, "
    "firewall, proxy, VPN, IPsec, GRE tunnel, PAC file, "
    "CIDR, subnet, VLAN, DNS, DHCP, NAT, "
    "IdP, SAML, SCIM, OAuth, MFA, SSO, Active Directory, Okta, Azure AD, "
    "SIEM, SOAR, SOC, EDR, XDR, MDR, NDR, "
    "CVE, IOC, indicators of compromise, threat intelligence, "
    "malware, ransomware, phishing, lateral movement, "
    "microsegmentation, least privilege, "
    # Cloud and infrastructure
    "AWS, Azure, GCP, Kubernetes, Docker, "
    "EC2, S3, VPC, Lambda, "
    "IoT, OT, SCADA, ICS, "
    # Common IT/business terms
    "SLA, RFP, POC, POV, PoC, PoV, "
    "RSC, TSC, TAS, TAM, SA, PM, SE, "
    "SNAP, SNAPs, Artex, "
    "C-level, CIO, CISO, CTO, CSO"
)


# ---------------------------------------------------------------------------
# Post-processing: domain term correction
# ---------------------------------------------------------------------------
# Whisper often mis-transcribes domain-specific acronyms and product names.
# This corrector applies regex-based rules AFTER transcription to fix them.

# Tier 1: Exact case-insensitive word replacements.
# Maps lowercased token(s) → correct form.
_EXACT_CORRECTIONS: dict[str, str] = {
    # Zscaler products
    "zscaler": "Zscaler",
    "z-scaler": "Zscaler",
    "zscalar": "Zscaler",
    "zee scaler": "Zscaler",
    "zia": "ZIA",
    "z.i.a.": "ZIA",
    "zpa": "ZPA",
    "z.p.a.": "ZPA",
    "zdx": "ZDX",
    "z.d.x.": "ZDX",
    "zcc": "ZCC",
    "z.c.c.": "ZCC",
    "ztna": "ZTNA",
    "z.t.n.a.": "ZTNA",
    # Zscaler components
    "app connector": "App Connector",
    "service edge": "Service Edge",
    "cloud connector": "Cloud Connector",
    "branch connector": "Branch Connector",
    "client connector": "Client Connector",
    "nanolog": "Nanolog",
    # Security acronyms — phonetic misrecognitions
    "sassy": "SASE",
    "sase": "SASE",
    "sasi": "SASE",
    "sassi": "SASE",
    "sse": "SSE",
    "casb": "CASB",
    "kazb": "CASB",
    "cas b": "CASB",
    "dlp": "DLP",
    "swg": "SWG",
    "ciso": "CISO",
    "see so": "CISO",
    "seeso": "CISO",
    "c-so": "CISO",
    "siem": "SIEM",
    "seem": "SIEM",
    "soar": "SOAR",
    "soc": "SOC",
    "edr": "EDR",
    "xdr": "XDR",
    "mdr": "MDR",
    "ndr": "NDR",
    # Networking
    "sd-wan": "SD-WAN",
    "sd wan": "SD-WAN",
    "sdwan": "SD-WAN",
    "mpls": "MPLS",
    "bgp": "BGP",
    "ospf": "OSPF",
    "ipsec": "IPsec",
    "ip sec": "IPsec",
    "gre": "GRE",
    "pac file": "PAC file",
    "pack file": "PAC file",
    # Identity
    "saml": "SAML",
    "scim": "SCIM",
    "skim": "SCIM",
    "oauth": "OAuth",
    "mfa": "MFA",
    "sso": "SSO",
    "idp": "IdP",
    "okta": "Okta",
    "octa": "Okta",
    # Cloud
    "saas": "SaaS",
    "sass": "SaaS",
    "iaas": "IaaS",
    "paas": "PaaS",
    "kubernetes": "Kubernetes",
    "aws": "AWS",
    "gcp": "GCP",
    # Business terms
    "poc": "POC",
    "pov": "POV",
    "rfp": "RFP",
    "sla": "SLA",
    "rsc": "RSC",
    "tsc": "TSC",
    "tas": "TAS",
    "tam": "TAM",
    "nss": "NSS",
    "c-level": "C-level",
    "c level": "C-level",
    "sea level": "C-level",
    "cio": "CIO",
    "cto": "CTO",
    "cso": "CSO",
    # IoT / OT
    "iot": "IoT",
    "scada": "SCADA",
    "ics": "ICS",
    # Zscaler release terms
    "la": "LA",
    "ga": "GA",
}

# Tier 2: Regex patterns for multi-word or contextual corrections.
# Each tuple: (compiled_regex, replacement_string)
_REGEX_CORRECTIONS: list[tuple[re.Pattern, str]] = [
    # "zero trust network access" → proper casing
    (re.compile(r"\bzero\s+trust\s+network\s+access\b", re.I), "Zero Trust Network Access"),
    (re.compile(r"\bzero\s+trust\s+exchange\b", re.I), "Zero Trust Exchange"),
    (re.compile(r"\bzero\s+trust\b", re.I), "Zero Trust"),
    # "Zscaler Internet/Private/Digital ..."
    (re.compile(r"\bz(?:ee\s*)?scaler\s+internet\s+access\b", re.I), "Zscaler Internet Access"),
    (re.compile(r"\bz(?:ee\s*)?scaler\s+private\s+access\b", re.I), "Zscaler Private Access"),
    (re.compile(r"\bz(?:ee\s*)?scaler\s+digital\s+experience\b", re.I), "Zscaler Digital Experience"),
    (re.compile(r"\bz(?:ee\s*)?scaler\s+client\s+connector\b", re.I), "Zscaler Client Connector"),
    # "limited availability" / "general availability"
    (re.compile(r"\blimited\s+availab\w+\b", re.I), "Limited Availability"),
    (re.compile(r"\bgeneral\s+availab\w+\b", re.I), "General Availability"),
    # "secure web gateway"
    (re.compile(r"\bsecure\s+web\s+gateway\b", re.I), "Secure Web Gateway"),
    # "indicators of compromise"
    (re.compile(r"\bindicators?\s+of\s+compromise\b", re.I), "indicators of compromise"),
    # SSL/TLS inspection
    (re.compile(r"\bssl\s+inspection\b", re.I), "SSL inspection"),
    (re.compile(r"\btls\s+inspection\b", re.I), "TLS inspection"),
    # "active directory"
    (re.compile(r"\bactive\s+directory\b", re.I), "Active Directory"),
    # "azure ad" / "azure a.d."
    (re.compile(r"\bazure\s+a\.?d\.?\b", re.I), "Azure AD"),
    # "threat intelligence"
    (re.compile(r"\bthreat\s+intelligen\w+\b", re.I), "threat intelligence"),
    # "lateral movement"
    (re.compile(r"\blateral\s+movement\b", re.I), "lateral movement"),
    # "least privilege"
    (re.compile(r"\bleast\s+privilege\b", re.I), "least privilege"),
]

# Build a lookup for single-word and multi-word exact corrections.
# We need to match longest phrases first to avoid partial replacements.
_SORTED_EXACT_KEYS = sorted(_EXACT_CORRECTIONS.keys(), key=len, reverse=True)
_EXACT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(key) + r"\b", re.I), repl)
    for key, repl in
    sorted(_EXACT_CORRECTIONS.items(), key=lambda x: len(x[0]), reverse=True)
]


def _correct_domain_terms(text: str) -> str:
    """Apply domain-specific corrections to Whisper output."""
    if not text:
        return text

    # Apply regex (multi-word) corrections first — longest patterns first
    for pattern, replacement in _REGEX_CORRECTIONS:
        text = pattern.sub(replacement, text)

    # Apply exact word/phrase corrections
    for pattern, replacement in _EXACT_PATTERNS:
        text = pattern.sub(replacement, text)

    return text


def _is_repetitive(text: str, max_ratio: float = 0.4) -> bool:
    """Detect Whisper hallucination where the same phrase is repeated many times.

    Splits the text into words and checks if any short phrase (1-8 words)
    is repeated so often that it accounts for more than *max_ratio* of the
    total words.
    """
    words = text.split()
    if len(words) < 6:
        return False

    total = len(words)

    # Check single-word dominance (e.g. "the the the the the the")
    from collections import Counter
    word_counts = Counter(words)
    most_common_word, most_common_count = word_counts.most_common(1)[0]
    if most_common_count >= 6 and most_common_count / total >= 0.6:
        return True

    # Check for repeated n-grams (n = 2..8)
    for n in range(2, min(9, total // 2 + 1)):
        ngram_counts: dict[str, int] = {}
        for i in range(total - n + 1):
            gram = " ".join(words[i:i + n])
            ngram_counts[gram] = ngram_counts.get(gram, 0) + 1

        for gram, count in ngram_counts.items():
            if count >= 3 and (count * n) / total >= max_ratio:
                return True

    return False


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

        # Auto beam_size: GPU can afford higher beam, CPU needs speed
        if beam_size <= 0:
            beam_size = 3 if resolved == "cuda" else 1
            print(f"[INFO] Auto beam_size={beam_size} for device={resolved}")

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
        parts.append(_DOMAIN_VOCAB)
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

        # Filter out Whisper hallucinations (repeated phrases)
        if full_text and _is_repetitive(full_text):
            print(f"[WARN] Filtered repetitive hallucination: {full_text[:80]}...")
            self._context.clear()  # Reset context to break the loop
            return ""

        # Post-process: correct domain-specific terms
        if full_text:
            full_text = _correct_domain_terms(full_text)
            self._context.append(full_text)

        return full_text

    def reset_context(self):
        """Clear the context history."""
        self._context.clear()
