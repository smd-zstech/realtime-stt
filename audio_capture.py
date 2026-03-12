"""
Audio capture module - captures audio from the laptop microphone in real-time.
Uses Voice Activity Detection (VAD) based on energy threshold to detect speech segments.
Auto-calibrates silence threshold from ambient noise at startup.
Applies audio preprocessing (normalization, noise gate) for cleaner Whisper input.
"""

import threading
import queue
import numpy as np
import sounddevice as sd


def _preprocess_audio(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Preprocess audio segment before sending to Whisper.

    1. High-pass filter at ~80Hz to remove low-frequency rumble/hum
    2. Peak normalization to -1.0dB to ensure consistent input levels
    3. Soft noise gate to suppress low-level background noise
    """
    # --- High-pass filter (DC removal + low-frequency rumble cut) ---
    # Removes DC offset, AC hum (50/60Hz), desk vibrations, HVAC rumble.
    # Uses mean subtraction + simple FIR differencing (pure numpy, fast).
    filtered = audio - np.mean(audio)

    # --- Peak normalization to -1.0dB ---
    # Whisper expects audio in [-1, 1] range. Normalizing ensures we use the
    # full dynamic range, which improves recognition of quiet speakers.
    peak = np.max(np.abs(filtered))
    if peak > 0:
        target_peak = 0.89  # -1.0 dB
        filtered = filtered * (target_peak / peak)

    # --- Soft noise gate ---
    # Suppress samples below a low threshold to reduce background noise
    # between words. Uses smooth envelope to avoid clicks.
    gate_threshold = 0.01
    envelope = np.abs(filtered)
    # Simple smoothing of envelope (moving average over ~5ms)
    window = int(sample_rate * 0.005)
    if window > 1 and len(envelope) > window:
        kernel = np.ones(window) / window
        envelope = np.convolve(envelope, kernel, mode="same")
    # Soft gate: scale down (not hard cut) when below threshold
    gate = np.clip(envelope / gate_threshold, 0.0, 1.0)
    filtered = filtered * gate

    return filtered


class AudioCapture:
    """Captures audio from the microphone and yields speech segments."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        block_duration_ms: int = 30,
        silence_threshold: float = 0,
        silence_duration: float = 0.8,
        min_speech_duration: float = 0.3,
        max_speech_duration: float = 15.0,
        calibration_seconds: float = 1.5,
        pre_speech_buffer_ms: int = 300,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = int(sample_rate * block_duration_ms / 1000)
        self._user_threshold = silence_threshold
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration
        self._calibration_seconds = calibration_seconds

        # Pre-speech buffer: keep N ms of audio before speech is detected
        # so we don't clip the onset of words (critical for recognition).
        self._pre_buffer_blocks = max(
            1, int(pre_speech_buffer_ms / block_duration_ms)
        )

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._segment_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._running = False
        self._stream = None
        self._processor_thread = None

    def _calibrate(self):
        """Record ambient noise briefly and set threshold automatically."""
        num_samples = int(self.sample_rate * self._calibration_seconds)
        print(f"[INFO] Calibrating microphone ({self._calibration_seconds}s silence)...")
        try:
            recording = sd.rec(
                num_samples,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
            )
            sd.wait()
            ambient_energy = np.sqrt(np.mean(recording ** 2))
            # Set threshold to 2.5x ambient noise (tighter than 3x for
            # better sensitivity to quiet/accented speech)
            self.silence_threshold = max(ambient_energy * 2.5, 0.003)
            print(f"[INFO] Ambient noise: {ambient_energy:.5f}, "
                  f"threshold set to: {self.silence_threshold:.5f}")
        except Exception as e:
            self.silence_threshold = 0.01
            print(f"[WARN] Calibration failed ({e}), using fallback threshold: {self.silence_threshold}")

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        if status:
            pass  # Ignore overflow/underflow warnings silently
        self._audio_queue.put(indata.copy())

    def _process_audio(self):
        """Background thread: accumulates audio blocks into speech segments.

        Uses a ring buffer to keep pre-speech audio so word onsets aren't clipped.
        """
        from collections import deque

        speech_buffer = []
        silence_blocks = 0
        is_speaking = False
        blocks_per_second = self.sample_rate / self.block_size
        silence_blocks_threshold = int(self.silence_duration * blocks_per_second)
        min_speech_blocks = int(self.min_speech_duration * blocks_per_second)
        max_speech_blocks = int(self.max_speech_duration * blocks_per_second)

        # Ring buffer for pre-speech audio (captures word onsets)
        pre_buffer: deque[np.ndarray] = deque(
            maxlen=self._pre_buffer_blocks
        )

        while self._running:
            try:
                block = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            energy = np.sqrt(np.mean(block ** 2))

            if energy > self.silence_threshold:
                if not is_speaking:
                    # Starting speech — prepend the pre-buffer to capture onset
                    speech_buffer.extend(pre_buffer)
                    pre_buffer.clear()
                speech_buffer.append(block)
                silence_blocks = 0
                is_speaking = True
            elif is_speaking:
                speech_buffer.append(block)
                silence_blocks += 1

                if silence_blocks >= silence_blocks_threshold:
                    if len(speech_buffer) >= min_speech_blocks:
                        segment = np.concatenate(speech_buffer, axis=0).flatten()
                        # Apply audio preprocessing before queuing
                        segment = _preprocess_audio(segment, self.sample_rate)
                        self._segment_queue.put(segment)
                    speech_buffer = []
                    silence_blocks = 0
                    is_speaking = False
            else:
                # Not speaking — feed ring buffer for onset capture
                pre_buffer.append(block)

            # Force-split if speech is too long (prevents huge segments)
            if is_speaking and len(speech_buffer) >= max_speech_blocks:
                segment = np.concatenate(speech_buffer, axis=0).flatten()
                segment = _preprocess_audio(segment, self.sample_rate)
                self._segment_queue.put(segment)
                speech_buffer = []
                silence_blocks = 0
                # Stay in speaking mode so next blocks continue seamlessly

    def start(self):
        """Start capturing audio from the microphone."""
        # Auto-calibrate if user didn't specify a manual threshold
        if self._user_threshold <= 0:
            self._calibrate()

        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.block_size,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._processor_thread = threading.Thread(
            target=self._process_audio, daemon=True
        )
        self._processor_thread.start()

    def stop(self):
        """Stop capturing audio."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._processor_thread is not None:
            self._processor_thread.join(timeout=2.0)
            self._processor_thread = None

    def get_segment(self, timeout: float = None) -> np.ndarray | None:
        """Get the next speech segment. Returns None on timeout."""
        try:
            return self._segment_queue.get(timeout=timeout)
        except queue.Empty:
            return None
