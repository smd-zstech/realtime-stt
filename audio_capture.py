"""
Audio capture module - captures audio from the laptop microphone in real-time.
Uses Voice Activity Detection (VAD) based on energy threshold to detect speech segments.
"""

import threading
import queue
import numpy as np
import sounddevice as sd


class AudioCapture:
    """Captures audio from the microphone and yields speech segments."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        block_duration_ms: int = 30,
        silence_threshold: float = 0.01,
        silence_duration: float = 0.6,
        min_speech_duration: float = 0.3,
        max_speech_duration: float = 10.0,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = int(sample_rate * block_duration_ms / 1000)
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._segment_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._running = False
        self._stream = None
        self._processor_thread = None

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        if status:
            pass  # Ignore overflow/underflow warnings silently
        self._audio_queue.put(indata.copy())

    def _process_audio(self):
        """Background thread: accumulates audio blocks into speech segments."""
        speech_buffer = []
        silence_blocks = 0
        is_speaking = False
        blocks_per_second = self.sample_rate / self.block_size
        silence_blocks_threshold = int(self.silence_duration * blocks_per_second)
        min_speech_blocks = int(self.min_speech_duration * blocks_per_second)
        max_speech_blocks = int(self.max_speech_duration * blocks_per_second)

        while self._running:
            try:
                block = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            energy = np.sqrt(np.mean(block ** 2))

            if energy > self.silence_threshold:
                speech_buffer.append(block)
                silence_blocks = 0
                is_speaking = True
            elif is_speaking:
                speech_buffer.append(block)
                silence_blocks += 1

                if silence_blocks >= silence_blocks_threshold:
                    if len(speech_buffer) >= min_speech_blocks:
                        segment = np.concatenate(speech_buffer, axis=0).flatten()
                        self._segment_queue.put(segment)
                    speech_buffer = []
                    silence_blocks = 0
                    is_speaking = False

            # Force-split if speech is too long (prevents huge segments)
            if is_speaking and len(speech_buffer) >= max_speech_blocks:
                segment = np.concatenate(speech_buffer, axis=0).flatten()
                self._segment_queue.put(segment)
                speech_buffer = []
                silence_blocks = 0
                # Stay in speaking mode so next blocks continue seamlessly

    def start(self):
        """Start capturing audio from the microphone."""
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
