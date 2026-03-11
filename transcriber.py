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
from collections import Counter, deque

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
    # Zscaler products and platform
    "Zscaler, ZIA, ZPA, ZDX, ZCC, ZTNA, ZEN, ZTE, ZWS, "
    "Zscaler Client Connector, Zscaler Internet Access, "
    "Zscaler Private Access, Zscaler Digital Experience, "
    "Zscaler Zero Trust Exchange, Zscaler One, Zscaler Risk360, "
    "Zscaler Deception, Zscaler Posture Control, Zscaler Data Protection, "
    "Zscaler Workload Segmentation, Zscaler Cloud Protection, "
    "Zscaler Browser Isolation, Zscaler Cloud Browser Isolation, "
    "Zscaler SaaS Security, Zscaler AI Protect, ZscalerGov, "
    "Zscaler Device Segmentation, Zero Trust SD-WAN, "
    # Zscaler components & architecture
    "App Connector, App Connector Group, Connector Group, "
    "Service Edge, Public Service Edge, Private Service Edge, "
    "Cloud Connector, Branch Connector, "
    "Nanolog, Nanolog Streaming Service, NSS, Cloud NSS, NSS VM, "
    "Log Streaming Service, LSS, ECRS, Central Authority, "
    "Z-Tunnel, Z-Tunnel 1.0, Z-Tunnel 2.0, Microtunnel, "
    "App Segment, Segment Group, Server Group, "
    "Access Policy, Forwarding Profile, App Profile, "
    "CloudPath, URL category, Cloud App Control, Bandwidth Control, "
    "Trusted Network, Location, Sub-Location, "
    # Zscaler data protection & threat features
    "SSMA, Single Scan Multi-Action, "
    "EDM, Exact Data Match, IDM, Indexed Document Matching, "
    "CDR, Content Disarm and Reconstruction, OCR, "
    "Cloud Sandbox, Cloud Firewall, Cloud IPS, DNS Security, "
    "Advanced Threat Protection, ATP, "
    "Privileged Remote Access, PRA, Application Discovery, "
    "Surrogate, Device Posture, "
    # Zscaler AI features
    "AI Protect, AI Asset Management, Smart Isolation, "
    # Zscaler risk & posture management
    "CSPM, DSPM, SSPM, UVM, CNAPP, CWPP, CIEM, "
    "EASM, External Attack Surface Management, "
    "ITDR, Identity Threat Detection and Response, "
    "Breach Predictor, Risk Score, "
    # Zscaler partner program
    "Zenith, Alpine, Basecamp, Z-Flex, ZCCP, "
    "MSP, MSSP, VAR, ARR, "
    # Zscaler research & knowledge
    "ThreatLabz, Zpedia, Security Preview, Page Risk Index, "
    # Zscaler release & licensing terms
    "PSE, CBI, IaC, "
    "LA, Limited Availability, GA, General Availability, "
    "EOL, End of Life, EOS, End of Support, "
    "Business, Transformation, Unlimited, "
    # === SASE / SSE architecture ===
    "SASE, Secure Access Service Edge, SSE, Security Service Edge, "
    "CASB, Cloud Access Security Broker, DLP, Data Loss Prevention, "
    "SWG, Secure Web Gateway, FWaaS, Firewall as a Service, "
    "RBI, Remote Browser Isolation, "
    "ZTNA, Zero Trust Network Access, "
    "inline CASB, out-of-band CASB, forward proxy, reverse proxy, "
    "explicit proxy, transparent proxy, proxy chaining, "
    "SSL inspection, TLS inspection, SSL decryption, "
    "deep packet inspection, full SSL inspection, "
    "traffic steering, traffic forwarding, split tunnel, full tunnel, "
    "GRE tunnel, IPsec tunnel, PAC file, "
    "user-to-app, app-to-app, branch-to-internet, branch-to-cloud, "
    "east-west traffic, north-south traffic, "
    "backhauling, hair-pinning, direct-to-cloud, local breakout, "
    "internet breakout, cloud-first, "
    # Security operations & frameworks
    "SIEM, SOAR, SOC, SOC analyst, "
    "EDR, XDR, MDR, NDR, "
    "UEBA, User and Entity Behavior Analytics, "
    "NGFW, Next-Generation Firewall, "
    "WAF, Web Application Firewall, "
    "IDS, Intrusion Detection System, "
    "IPS, Intrusion Prevention System, "
    "DDoS, Distributed Denial of Service, "
    "PAM, Privileged Access Management, "
    "CISA, NIST, ISO 27001, SOC 2, FedRAMP, "
    "compliance, governance, audit, "
    # Threat & attack concepts
    "APT, Advanced Persistent Threat, "
    "CVE, IOC, IOCs, TTPs, MITRE ATT&CK, "
    "attack surface, security posture, Zero Trust, "
    "malware, ransomware, phishing, spear phishing, "
    "lateral movement, privilege escalation, "
    "data exfiltration, credential theft, credential stuffing, "
    "brute force, man-in-the-middle, MITM, "
    "supply chain attack, zero-day, zero day exploit, "
    "social engineering, business email compromise, BEC, "
    "DNS tunneling, DNS exfiltration, "
    "command and control, C2, beaconing, "
    "botnet, trojan, worm, rootkit, keylogger, "
    "sandbox evasion, polymorphic malware, fileless malware, "
    "insider threat, shadow IT, shadow AI, "
    "microsegmentation, least privilege, "
    # Encryption & certificates
    "PKI, Public Key Infrastructure, "
    "CA, Certificate Authority, "
    "TLS 1.2, TLS 1.3, mTLS, mutual TLS, "
    "certificate pinning, SSL certificate, "
    "encryption at rest, encryption in transit, "
    # Network security architecture
    "DMZ, demilitarized zone, "
    "air gap, network segmentation, "
    "perimeter security, perimeterless, "
    "hub-and-spoke, mesh architecture, "
    "overlay network, underlay network, "
    "SD-WAN overlay, SASE PoP, "
    "egress, ingress, "
    # Networking
    "SD-WAN, MPLS, BGP, OSPF, IPsec, GRE tunnel, PAC file, "
    "TCP, UDP, HTTP, HTTPS, SSL, TLS, DNS, DHCP, NAT, "
    "CIDR, subnet, VLAN, QoS, SNMP, SDN, NFV, "
    "Gbps, Mbps, Kbps, bandwidth, throughput, latency, jitter, packet loss, "
    "1 Gbps, 10 Gbps, 100 Gbps, "
    "firewall, proxy, VPN, "
    # Identity
    "IdP, Identity Provider, SAML, SCIM, OAuth, MFA, SSO, "
    "Active Directory, Azure AD, Entra ID, Okta, "
    "LDAP, RADIUS, RBAC, IAM, PAM, "
    "multi-factor authentication, single sign-on, "
    # Cloud and infrastructure
    "SaaS, IaaS, PaaS, IaC, "
    "AWS, Azure, GCP, Kubernetes, K8s, Docker, "
    "EC2, S3, VPC, Lambda, Terraform, "
    "on-premises, hybrid cloud, multi-cloud, public cloud, private cloud, "
    "DevOps, DevSecOps, CI/CD, API, REST API, SDK, "
    "IoT, OT, SCADA, ICS, PLC, "
    # Business / organizational
    "SLA, RFP, RFI, SOW, ROI, TCO, OPEX, CAPEX, KPI, ARR, "
    "POC, POV, PoC, PoV, "
    "RSC, TSC, TAS, TAM, SA, PM, SE, "
    "MSP, MSSP, VAR, "
    "SNAP, SNAPs, Artex, "
    "C-level, CIO, CISO, CTO, CSO, VP, SVP"
)


# ---------------------------------------------------------------------------
# Post-processing: domain term correction
# ---------------------------------------------------------------------------
# Whisper often mis-transcribes domain-specific acronyms and product names.
# This corrector applies regex-based rules AFTER transcription to fix them.

# Tier 1: Exact case-insensitive word replacements.
# Maps lowercased token(s) → correct form.
_EXACT_CORRECTIONS: dict[str, str] = {
    # === Zscaler Products & Platforms ===
    "zscaler": "Zscaler",
    "z-scaler": "Zscaler",
    "zscalar": "Zscaler",
    "zee scaler": "Zscaler",
    "z scaler": "Zscaler",
    "zed scaler": "Zscaler",
    "zia": "ZIA",
    "z.i.a.": "ZIA",
    "zpa": "ZPA",
    "z.p.a.": "ZPA",
    "zepa": "ZPA",
    "zdx": "ZDX",
    "z.d.x.": "ZDX",
    "zdex": "ZDX",
    "zcc": "ZCC",
    "z.c.c.": "ZCC",
    "ztna": "ZTNA",
    "z.t.n.a.": "ZTNA",
    "zws": "ZWS",
    "z.w.s.": "ZWS",
    "zte": "ZTE",
    "cbi": "CBI",
    # Zscaler components & connectors
    "app connector": "App Connector",
    "app connector group": "App Connector Group",
    "connector group": "Connector Group",
    "service edge": "Service Edge",
    "public service edge": "Public Service Edge",
    "private service edge": "Private Service Edge",
    "cloud connector": "Cloud Connector",
    "branch connector": "Branch Connector",
    "client connector": "Client Connector",
    "nanolog": "Nanolog",
    "nano log": "Nanolog",
    "zen": "ZEN",
    "zen node": "ZEN node",
    "z tunnel": "Z-Tunnel",
    "z-tunnel": "Z-Tunnel",
    "microtunnel": "Microtunnel",
    "micro tunnel": "Microtunnel",
    # Zscaler features & capabilities
    "browser isolation": "Browser Isolation",
    "cloud browser isolation": "Cloud Browser Isolation",
    "cloud firewall": "Cloud Firewall",
    "cloud ips": "Cloud IPS",
    "cloud sandbox": "Cloud Sandbox",
    "sandbox": "Sandbox",
    "data protection": "Data Protection",
    "workload segmentation": "Workload Segmentation",
    "workload communications": "Workload Communications",
    "device segmentation": "Device Segmentation",
    "posture control": "Posture Control",
    "risk score": "Risk Score",
    "risk 360": "Risk360",
    "risk360": "Risk360",
    "breach predictor": "Breach Predictor",
    "cloud security posture management": "Cloud Security Posture Management",
    "cspm": "CSPM",
    "dspm": "DSPM",
    "sspm": "SSPM",
    "uvm": "UVM",
    "cnapp": "CNAPP",
    "cwpp": "CWPP",
    "ciem": "CIEM",
    "digital experience monitoring": "Digital Experience Monitoring",
    "cloud path": "CloudPath",
    "cloudpath": "CloudPath",
    "app segmentation": "App Segmentation",
    "app segment": "App Segment",
    "server group": "Server Group",
    "segment group": "Segment Group",
    "access policy": "Access Policy",
    "forwarding profile": "Forwarding Profile",
    "app profile": "App Profile",
    "url category": "URL category",
    "cloud app control": "Cloud App Control",
    "bandwidth control": "Bandwidth Control",
    "dns security": "DNS Security",
    "idp": "IdP",
    "ecrs": "ECRS",
    "surrogate": "Surrogate",
    # Zscaler data protection features
    "edm": "EDM",
    "idm": "IDM",
    "ocr": "OCR",
    "cdr": "CDR",
    "ssma": "SSMA",
    "sigma": "SSMA",
    # Zscaler AI features
    "ai protect": "AI Protect",
    "ai asset management": "AI Asset Management",
    "smart isolation": "Smart Isolation",
    # Zscaler risk & posture
    "easm": "EASM",
    "itdr": "ITDR",
    # Zscaler partner program
    "zenith": "Zenith",
    "alpine": "Alpine",
    "basecamp": "Basecamp",
    "z-flex": "Z-Flex",
    "z flex": "Z-Flex",
    "zee flex": "Z-Flex",
    "zed flex": "Z-Flex",
    "zccp": "ZCCP",
    "z.c.c.p.": "ZCCP",
    # Zscaler research & knowledge
    "threat labs": "ThreatLabz",
    "threatlabz": "ThreatLabz",
    "threat labz": "ThreatLabz",
    "zpedia": "Zpedia",
    "z pedia": "Zpedia",
    "zscalergov": "ZscalerGov",
    # Zscaler deployment / release
    "zscaler one": "Zscaler One",
    "z1": "Z1",
    "pse": "PSE",
    "lss": "LSS",
    "cloud nss": "Cloud NSS",
    "nss vm": "NSS VM",
    "pop": "PoP",

    # === Security Acronyms & Terms ===
    # SASE / SSE
    "sassy": "SASE",
    "sase": "SASE",
    "sasi": "SASE",
    "sassi": "SASE",
    "sse": "SSE",
    # SASE / SSE architecture components
    "casb": "CASB",
    "kazb": "CASB",
    "cas b": "CASB",
    "cazb": "CASB",
    "dlp": "DLP",
    "swg": "SWG",
    "atp": "ATP",
    "ips": "IPS",
    "ids": "IDS",
    "fwaas": "FWaaS",
    "rbi": "RBI",
    "ngfw": "NGFW",
    "waf": "WAF",
    "ueba": "UEBA",
    "forward proxy": "forward proxy",
    "reverse proxy": "reverse proxy",
    "explicit proxy": "explicit proxy",
    "transparent proxy": "transparent proxy",
    "proxy chaining": "proxy chaining",
    "inline casb": "inline CASB",
    "out-of-band casb": "out-of-band CASB",
    "out of band casb": "out-of-band CASB",
    "split tunnel": "split tunnel",
    "full tunnel": "full tunnel",
    "local breakout": "local breakout",
    "internet breakout": "internet breakout",
    "backhauling": "backhauling",
    "back hauling": "backhauling",
    "hair-pinning": "hair-pinning",
    "hairpinning": "hair-pinning",
    "direct-to-cloud": "direct-to-cloud",
    "direct to cloud": "direct-to-cloud",
    # Traffic direction
    "east-west": "east-west",
    "east west": "east-west",
    "north-south": "north-south",
    "north south": "north-south",
    "egress": "egress",
    "ingress": "ingress",
    # SOC / SIEM / SOAR
    "ciso": "CISO",
    "see so": "CISO",
    "seeso": "CISO",
    "c-so": "CISO",
    "siem": "SIEM",
    "seem": "SIEM",
    "soar": "SOAR",
    "soc": "SOC",
    # EDR / XDR / NDR
    "edr": "EDR",
    "xdr": "XDR",
    "mdr": "MDR",
    "ndr": "NDR",
    # Threat & attack terms
    "apt": "APT",
    "cve": "CVE",
    "ioc": "IOC",
    "iocs": "IOCs",
    "ttps": "TTPs",
    "ttp": "TTP",
    "mitre": "MITRE",
    "mitre attack": "MITRE ATT&CK",
    "mitre att&ck": "MITRE ATT&CK",
    "c2": "C2",
    "c&c": "C&C",
    "command and control": "command and control",
    "ddos": "DDoS",
    "mitm": "MITM",
    "man-in-the-middle": "man-in-the-middle",
    "man in the middle": "man-in-the-middle",
    "bec": "BEC",
    "zero-day": "zero-day",
    "zero day": "zero-day",
    "data exfiltration": "data exfiltration",
    "credential stuffing": "credential stuffing",
    "privilege escalation": "privilege escalation",
    "dns tunneling": "DNS tunneling",
    "shadow it": "shadow IT",
    "shadow ai": "shadow AI",
    "insider threat": "insider threat",
    "fileless malware": "fileless malware",
    # Compliance & frameworks
    "nist": "NIST",
    "cisa": "CISA",
    "fedramp": "FedRAMP",
    "fed ramp": "FedRAMP",
    "soc 2": "SOC 2",
    "soc2": "SOC 2",
    "iso 27001": "ISO 27001",
    # Encryption & certificates
    "pki": "PKI",
    "mtls": "mTLS",
    "mutual tls": "mutual TLS",
    "tls 1.2": "TLS 1.2",
    "tls 1.3": "TLS 1.3",
    # Network architecture
    "dmz": "DMZ",
    "hub-and-spoke": "hub-and-spoke",
    "hub and spoke": "hub-and-spoke",

    # === Networking ===
    # Protocols & standards
    "sd-wan": "SD-WAN",
    "sd wan": "SD-WAN",
    "sdwan": "SD-WAN",
    "mpls": "MPLS",
    "bgp": "BGP",
    "ospf": "OSPF",
    "ipsec": "IPsec",
    "ip sec": "IPsec",
    "gre": "GRE",
    "tcp": "TCP",
    "udp": "UDP",
    "http": "HTTP",
    "https": "HTTPS",
    "ssl": "SSL",
    "tls": "TLS",
    "dns": "DNS",
    "dhcp": "DHCP",
    "nat": "NAT",
    "vlan": "VLAN",
    "cidr": "CIDR",
    "qos": "QoS",
    "snmp": "SNMP",
    "sdn": "SDN",
    "nfv": "NFV",
    "pac file": "PAC file",
    "pack file": "PAC file",
    # Units — Whisper often spells out abbreviations
    "gig": "Gbps",
    "gigs": "Gbps",
    "gbps": "Gbps",
    "mbps": "Mbps",
    "kbps": "Kbps",
    "gigabit": "Gigabit",
    "gigabyte": "Gigabyte",
    "terabyte": "Terabyte",
    "petabyte": "Petabyte",
    "tb": "TB",
    "gb": "GB",
    "mb": "MB",

    # === Identity & Access ===
    "saml": "SAML",
    "samuel": "SAML",
    "samel": "SAML",
    "scim": "SCIM",
    "skim": "SCIM",
    "oauth": "OAuth",
    "mfa": "MFA",
    "sso": "SSO",
    "okta": "Okta",
    "octa": "Okta",
    "ldap": "LDAP",
    "radius": "RADIUS",
    "rbac": "RBAC",
    "abac": "ABAC",
    "pam": "PAM",
    "iam": "IAM",
    "entra id": "Entra ID",
    "pra": "PRA",

    # === Cloud & Infrastructure ===
    "saas": "SaaS",
    "sass": "SaaS",
    "iaas": "IaaS",
    "paas": "PaaS",
    "kubernetes": "Kubernetes",
    "k8s": "K8s",
    "aws": "AWS",
    "gcp": "GCP",
    "vpc": "VPC",
    "ec2": "EC2",
    "lambda": "Lambda",
    "terraform": "Terraform",
    "ansible": "Ansible",
    "devops": "DevOps",
    "devsecops": "DevSecOps",
    "ci cd": "CI/CD",
    "ci/cd": "CI/CD",
    "api": "API",
    "rest api": "REST API",
    "sdk": "SDK",
    "iac": "IaC",
    # IoT / OT
    "iot": "IoT",
    "scada": "SCADA",
    "ics": "ICS",
    "ot": "OT",
    "plc": "PLC",

    # === Business / Organizational ===
    "poc": "POC",
    "pov": "POV",
    "rfp": "RFP",
    "rfi": "RFI",
    "sla": "SLA",
    "sow": "SOW",
    "roi": "ROI",
    "tco": "TCO",
    "opex": "OPEX",
    "capex": "CAPEX",
    "kpi": "KPI",
    "arr": "ARR",
    "rsc": "RSC",
    "tsc": "TSC",
    "tas": "TAS",
    "tam": "TAM",
    "nss": "NSS",
    "se": "SE",
    "sa": "SA",
    "msp": "MSP",
    "mssp": "MSSP",
    "var": "VAR",
    "c-level": "C-level",
    "c level": "C-level",
    "sea level": "C-level",
    "cio": "CIO",
    "cto": "CTO",
    "cso": "CSO",
    "vp": "VP",
    "svp": "SVP",
    "evp": "EVP",
    # Zscaler release terms
    "la": "LA",
    "ga": "GA",
    "eol": "EOL",
    "eos": "EOS",
    # Zscaler licensing (only specific bundle names, not generic words)
    "business bundle": "Business bundle",
    "transformation bundle": "Transformation bundle",
    "unlimited bundle": "Unlimited bundle",
}

# Tier 2: Regex patterns for multi-word or contextual corrections.
# Each tuple: (compiled_regex, replacement_string)
_REGEX_CORRECTIONS: list[tuple[re.Pattern, str]] = [
    # === Zscaler full product names (longest first) ===
    (re.compile(r"\bz(?:ee\s*)?scal[ae]r\s+internet\s+access\b", re.I), "Zscaler Internet Access"),
    (re.compile(r"\bz(?:ee\s*)?scal[ae]r\s+private\s+access\b", re.I), "Zscaler Private Access"),
    (re.compile(r"\bz(?:ee\s*)?scal[ae]r\s+digital\s+experience\b", re.I), "Zscaler Digital Experience"),
    (re.compile(r"\bz(?:ee\s*)?scal[ae]r\s+client\s+connector\b", re.I), "Zscaler Client Connector"),
    (re.compile(r"\bz(?:ee\s*)?scal[ae]r\s+cloud\s+protection\b", re.I), "Zscaler Cloud Protection"),
    (re.compile(r"\bz(?:ee\s*)?scal[ae]r\s+workload\s+segmentation\b", re.I), "Zscaler Workload Segmentation"),
    (re.compile(r"\bz(?:ee\s*)?scal[ae]r\s+workload\s+communications\b", re.I), "Zscaler Workload Communications"),
    (re.compile(r"\bz(?:ee\s*)?scal[ae]r\s+data\s+protection\b", re.I), "Zscaler Data Protection"),
    (re.compile(r"\bz(?:ee\s*)?scal[ae]r\s+deception\b", re.I), "Zscaler Deception"),
    (re.compile(r"\bz(?:ee\s*)?scal[ae]r\s+posture\s+control\b", re.I), "Zscaler Posture Control"),
    (re.compile(r"\bz(?:ee\s*)?scal[ae]r\s+risk\s*360\b", re.I), "Zscaler Risk360"),

    # === Zscaler additional product names ===
    (re.compile(r"\bz(?:ee\s*)?scal[ae]r\s+cloud\s+browser\s+isolation\b", re.I), "Zscaler Cloud Browser Isolation"),
    (re.compile(r"\bz(?:ee\s*)?scal[ae]r\s+saas\s+security\b", re.I), "Zscaler SaaS Security"),
    (re.compile(r"\bz(?:ee\s*)?scal[ae]r\s+device\s+segmentation\b", re.I), "Zscaler Device Segmentation"),
    (re.compile(r"\bz(?:ee\s*)?scal[ae]r\s+ai\s+protect\b", re.I), "Zscaler AI Protect"),
    (re.compile(r"\bz(?:ee\s*)?scal[ae]r\s+ai\s+security\b", re.I), "Zscaler AI Security"),
    (re.compile(r"\bzero\s+trust\s+sd[- ]?wan\b", re.I), "Zero Trust SD-WAN"),

    # === Zero Trust phrases ===
    (re.compile(r"\bzero\s+trust\s+network\s+access\b", re.I), "Zero Trust Network Access"),
    (re.compile(r"\bzero\s+trust\s+exchange\b", re.I), "Zero Trust Exchange"),
    (re.compile(r"\bzero\s+trust\s+architecture\b", re.I), "Zero Trust Architecture"),
    (re.compile(r"\bzero\s+trust\b", re.I), "Zero Trust"),

    # === Availability / Release ===
    (re.compile(r"\blimited\s+availab\w+\b", re.I), "Limited Availability"),
    (re.compile(r"\bgeneral\s+availab\w+\b", re.I), "General Availability"),
    (re.compile(r"\bend\s+of\s+life\b", re.I), "End of Life"),
    (re.compile(r"\bend\s+of\s+support\b", re.I), "End of Support"),

    # === Security phrases ===
    (re.compile(r"\bsecure\s+web\s+gateway\b", re.I), "Secure Web Gateway"),
    (re.compile(r"\bcloud\s+access\s+security\s+broker\b", re.I), "Cloud Access Security Broker"),
    (re.compile(r"\bdata\s+loss\s+prevention\b", re.I), "Data Loss Prevention"),
    (re.compile(r"\bdata\s+leakage\s+prevention\b", re.I), "Data Leakage Prevention"),
    (re.compile(r"\bdata\s+security\s+posture\s+management\b", re.I), "Data Security Posture Management"),
    (re.compile(r"\bsaas\s+security\s+posture\s+management\b", re.I), "SaaS Security Posture Management"),
    (re.compile(r"\bcloud\s+security\s+posture\s+management\b", re.I), "Cloud Security Posture Management"),
    (re.compile(r"\bexternal\s+attack\s+surface\s+management\b", re.I), "External Attack Surface Management"),
    (re.compile(r"\basset\s+exposure\s+management\b", re.I), "Asset Exposure Management"),
    (re.compile(r"\bidentity\s+threat\s+detection\s+and\s+response\b", re.I), "Identity Threat Detection and Response"),
    (re.compile(r"\bindicators?\s+of\s+compromise\b", re.I), "indicators of compromise"),
    (re.compile(r"\badvanced\s+threat\s+protection\b", re.I), "Advanced Threat Protection"),
    (re.compile(r"\badvanced\s+persistent\s+threat\b", re.I), "Advanced Persistent Threat"),
    (re.compile(r"\bthreat\s+intelligen\w+\b", re.I), "threat intelligence"),
    (re.compile(r"\blateral\s+movement\b", re.I), "lateral movement"),
    (re.compile(r"\bleast\s+privilege\b", re.I), "least privilege"),
    (re.compile(r"\bsecurity\s+service\s+edge\b", re.I), "Security Service Edge"),
    (re.compile(r"\bsecure\s+access\s+service\s+edge\b", re.I), "Secure Access Service Edge"),
    (re.compile(r"\bcontent\s+disarm\s+and\s+reconstruction\b", re.I), "Content Disarm and Reconstruction"),
    (re.compile(r"\bexact\s+data\s+match\b", re.I), "Exact Data Match"),
    (re.compile(r"\bindexed\s+document\s+matching\b", re.I), "Indexed Document Matching"),
    (re.compile(r"\bsingle\s+scan\s+multi[- ]?action\b", re.I), "Single Scan Multi-Action"),
    (re.compile(r"\bprivileged\s+remote\s+access\b", re.I), "Privileged Remote Access"),
    (re.compile(r"\bnanolog\s+streaming\s+service\b", re.I), "Nanolog Streaming Service"),
    (re.compile(r"\blog\s+streaming\s+service\b", re.I), "Log Streaming Service"),

    # === SASE / SSE architecture phrases ===
    (re.compile(r"\bfirewall\s+as\s+a\s+service\b", re.I), "Firewall as a Service"),
    (re.compile(r"\bremote\s+browser\s+isolation\b", re.I), "Remote Browser Isolation"),
    (re.compile(r"\bnext[- ]?gen(?:eration)?\s+firewall\b", re.I), "Next-Generation Firewall"),
    (re.compile(r"\bweb\s+application\s+firewall\b", re.I), "Web Application Firewall"),
    (re.compile(r"\bintrusion\s+detection\s+system\b", re.I), "Intrusion Detection System"),
    (re.compile(r"\bintrusion\s+prevention\s+system\b", re.I), "Intrusion Prevention System"),
    (re.compile(r"\buser\s+and\s+entity\s+behavio(?:u)?r\s+analytics\b", re.I), "User and Entity Behavior Analytics"),
    (re.compile(r"\bdistributed\s+denial\s+of\s+service\b", re.I), "Distributed Denial of Service"),
    (re.compile(r"\bprivileged\s+access\s+management\b", re.I), "Privileged Access Management"),
    (re.compile(r"\bbusiness\s+email\s+compromise\b", re.I), "business email compromise"),
    (re.compile(r"\bsupply\s+chain\s+attack\b", re.I), "supply chain attack"),
    (re.compile(r"\bman[- ]?in[- ]?the[- ]?middle\b", re.I), "man-in-the-middle"),
    (re.compile(r"\bspear\s+phishing\b", re.I), "spear phishing"),
    (re.compile(r"\bprivilege\s+escalation\b", re.I), "privilege escalation"),
    (re.compile(r"\bdata\s+exfiltration\b", re.I), "data exfiltration"),
    (re.compile(r"\bcredential\s+stuffing\b", re.I), "credential stuffing"),
    (re.compile(r"\bcredential\s+theft\b", re.I), "credential theft"),
    (re.compile(r"\bsocial\s+engineering\b", re.I), "social engineering"),
    (re.compile(r"\bdns\s+tunneling\b", re.I), "DNS tunneling"),
    (re.compile(r"\bdns\s+exfiltration\b", re.I), "DNS exfiltration"),
    (re.compile(r"\bsandbox\s+evasion\b", re.I), "sandbox evasion"),
    (re.compile(r"\bfileless\s+malware\b", re.I), "fileless malware"),
    (re.compile(r"\binsider\s+threat\b", re.I), "insider threat"),
    (re.compile(r"\bshadow\s+it\b", re.I), "shadow IT"),
    (re.compile(r"\bshadow\s+ai\b", re.I), "shadow AI"),
    (re.compile(r"\btraffic\s+steering\b", re.I), "traffic steering"),
    (re.compile(r"\btraffic\s+forwarding\b", re.I), "traffic forwarding"),
    (re.compile(r"\bsplit\s+tunnel(?:ing)?\b", re.I), "split tunnel"),
    (re.compile(r"\bfull\s+tunnel(?:ing)?\b", re.I), "full tunnel"),
    (re.compile(r"\blocal\s+breakout\b", re.I), "local breakout"),
    (re.compile(r"\binternet\s+breakout\b", re.I), "internet breakout"),
    (re.compile(r"\bdirect[- ]?to[- ]?cloud\b", re.I), "direct-to-cloud"),
    (re.compile(r"\bback[- ]?haul(?:ing)?\b", re.I), "backhauling"),
    (re.compile(r"\bhair[- ]?pin(?:ning)?\b", re.I), "hair-pinning"),
    (re.compile(r"\beast[- ]?west\s+traffic\b", re.I), "east-west traffic"),
    (re.compile(r"\bnorth[- ]?south\s+traffic\b", re.I), "north-south traffic"),
    (re.compile(r"\buser[- ]?to[- ]?app\b", re.I), "user-to-app"),
    (re.compile(r"\bapp[- ]?to[- ]?app\b", re.I), "app-to-app"),
    (re.compile(r"\bbranch[- ]?to[- ]?(?:internet|cloud)\b", re.I),
     lambda m: f"branch-to-{m.group(0).rsplit('to', 1)[-1].strip(' -').lower()}"),
    (re.compile(r"\bnetwork\s+segmentation\b", re.I), "network segmentation"),
    (re.compile(r"\bperimeter\s+security\b", re.I), "perimeter security"),
    (re.compile(r"\bhub[- ]?and[- ]?spoke\b", re.I), "hub-and-spoke"),
    (re.compile(r"\boverlay\s+network\b", re.I), "overlay network"),
    (re.compile(r"\bunderlay\s+network\b", re.I), "underlay network"),
    (re.compile(r"\bpublic\s+key\s+infrastructure\b", re.I), "Public Key Infrastructure"),
    (re.compile(r"\bcertificate\s+authority\b", re.I), "Certificate Authority"),
    (re.compile(r"\bmutual\s+tls\b", re.I), "mutual TLS"),
    (re.compile(r"\bencryption\s+at\s+rest\b", re.I), "encryption at rest"),
    (re.compile(r"\bencryption\s+in\s+transit\b", re.I), "encryption in transit"),
    (re.compile(r"\bzero[- ]?day\s+exploit\b", re.I), "zero-day exploit"),

    # === SSL / TLS / inspection ===
    (re.compile(r"\bssl\s+inspection\b", re.I), "SSL inspection"),
    (re.compile(r"\btls\s+inspection\b", re.I), "TLS inspection"),
    (re.compile(r"\bfull\s+ssl\s+inspection\b", re.I), "full SSL inspection"),
    (re.compile(r"\bssl\s+decryption\b", re.I), "SSL decryption"),
    (re.compile(r"\bdeep\s+packet\s+inspection\b", re.I), "deep packet inspection"),

    # === Identity ===
    (re.compile(r"\bactive\s+directory\b", re.I), "Active Directory"),
    (re.compile(r"\bazure\s+a\.?d\.?\b", re.I), "Azure AD"),
    (re.compile(r"\bmicrosoft\s+entra\b", re.I), "Microsoft Entra"),
    (re.compile(r"\bentra\s+id\b", re.I), "Entra ID"),
    (re.compile(r"\bidentity\s+provider\b", re.I), "Identity Provider"),
    (re.compile(r"\bmulti[- ]?factor\s+auth\w*\b", re.I), "multi-factor authentication"),
    (re.compile(r"\bsingle\s+sign[- ]?on\b", re.I), "single sign-on"),
    (re.compile(r"\bdevice\s+posture\b", re.I), "device posture"),

    # === Networking units — "10 gig" → "10 Gbps", "one gig" → "1 Gbps" ===
    (re.compile(r"\b(\d+)\s*gig(?:s|abit)?(?:\s+per\s+second)?\b", re.I),
     lambda m: f"{m.group(1)} Gbps"),
    (re.compile(r"\b(\d+)\s*meg(?:s|abit)?(?:\s+per\s+second)?\b", re.I),
     lambda m: f"{m.group(1)} Mbps"),
    # "one gig", "a gig" (without number)
    (re.compile(r"\bone\s+gig\b", re.I), "1 Gbps"),
    (re.compile(r"\ba\s+gig\b", re.I), "1 Gbps"),
    # "100 meg pipe" / "10 gig pipe" → "100 Mbps pipe" / "10 Gbps pipe"
    (re.compile(r"\b(\d+)\s*gig\s+(pipe|link|line|port|connection|interface)\b", re.I),
     lambda m: f"{m.group(1)} Gbps {m.group(2)}"),
    (re.compile(r"\b(\d+)\s*meg\s+(pipe|link|line|port|connection|interface)\b", re.I),
     lambda m: f"{m.group(1)} Mbps {m.group(2)}"),

    # === Cloud / infra phrases ===
    (re.compile(r"\bvirtual\s+private\s+cloud\b", re.I), "Virtual Private Cloud"),
    (re.compile(r"\bpublic\s+cloud\b", re.I), "public cloud"),
    (re.compile(r"\bprivate\s+cloud\b", re.I), "private cloud"),
    (re.compile(r"\bhybrid\s+cloud\b", re.I), "hybrid cloud"),
    (re.compile(r"\bmulti[- ]?cloud\b", re.I), "multi-cloud"),
    (re.compile(r"\bon[- ]?prem(?:ise)?s?\b", re.I), "on-premises"),

    # === Bandwidth / throughput expressions ===
    (re.compile(r"\bbandwidth\b", re.I), "bandwidth"),
    (re.compile(r"\bthroughput\b", re.I), "throughput"),
    (re.compile(r"\blatency\b", re.I), "latency"),
    (re.compile(r"\bjitter\b", re.I), "jitter"),
    (re.compile(r"\bpacket\s+loss\b", re.I), "packet loss"),
]

# Build a single combined regex for all exact corrections.
# Sorting by length (longest first) ensures multi-word phrases match before
# their sub-phrases (e.g. "cloud browser isolation" before "browser isolation").
# This is O(1) regex passes instead of O(n) individual pattern scans.
_EXACT_LOOKUP: dict[str, str] = {k.lower(): v for k, v in _EXACT_CORRECTIONS.items()}
_EXACT_COMBINED = re.compile(
    r"\b(?:" + "|".join(
        re.escape(key) for key in
        sorted(_EXACT_CORRECTIONS.keys(), key=len, reverse=True)
    ) + r")\b",
    re.I,
)


def _exact_replacer(m: re.Match) -> str:
    """Callback for the combined exact-match regex."""
    return _EXACT_LOOKUP.get(m.group(0).lower(), m.group(0))


def _correct_domain_terms(text: str) -> str:
    """Apply domain-specific corrections to Whisper output."""
    if not text:
        return text

    # Apply regex (multi-word) corrections first — longest patterns first
    for pattern, replacement in _REGEX_CORRECTIONS:
        text = pattern.sub(replacement, text)

    # Apply all exact word/phrase corrections in a single pass
    text = _EXACT_COMBINED.sub(_exact_replacer, text)

    return text


# Known Whisper hallucination phrases.
# Whisper often outputs these when there is silence, background noise, or
# very short unclear audio. Case-insensitive, stripped of punctuation.
_HALLUCINATION_PHRASES: set[str] = {
    "thank you",
    "thanks",
    "thanks for watching",
    "thanks for listening",
    "thank you for watching",
    "thank you for listening",
    "thank you very much",
    "please subscribe",
    "subscribe",
    "like and subscribe",
    "see you next time",
    "bye",
    "bye bye",
    "goodbye",
    "you",
    "okay",
    "ok",
    "so",
    "um",
    "uh",
    "hmm",
    "ah",
    "oh",
    "yeah",
    "yes",
    "no",
    "right",
    "well",
    "the end",
    "music",
    "applause",
    "laughter",
    "silence",
}

# Pre-compiled pattern to strip punctuation for hallucination check
_PUNCT_STRIP = re.compile(r"[^\w\s]", re.UNICODE)


def _is_hallucination(text: str) -> bool:
    """Check if the text is a known Whisper hallucination phrase."""
    cleaned = _PUNCT_STRIP.sub("", text).strip().lower()
    return cleaned in _HALLUCINATION_PHRASES


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

        # Filter out Whisper hallucinations
        if full_text:
            if _is_hallucination(full_text):
                print(f"[WARN] Filtered hallucination: \"{full_text}\"")
                return ""
            if _is_repetitive(full_text):
                print(f"[WARN] Filtered repetitive hallucination: {full_text[:80]}...")
                self._context.clear()
                return ""

        # Post-process: correct domain-specific terms
        if full_text:
            full_text = _correct_domain_terms(full_text)
            self._context.append(full_text)

        return full_text

    def reset_context(self):
        """Clear the context history."""
        self._context.clear()
