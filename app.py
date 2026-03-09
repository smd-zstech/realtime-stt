"""
Main application - Real-time Speech-to-Text with Korean Translation.

Integrates all modules:
- AudioCapture: microphone input with VAD
- Transcriber: Whisper-based STT with context inference
- Translator: English-to-Korean translation
- FileSaver: real-time log file output
- GUI: tkinter-based display showing English + Korean in real-time
"""

import argparse
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext, font as tkfont

from audio_capture import AudioCapture
from transcriber import Transcriber
from translator import Translator, TranslationResult
from file_saver import FileSaver


class App:
    """Main application with tkinter GUI."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        output_dir: str = None,
        silence_threshold: float = 0.01,
        beam_size: int = 3,
    ):
        self.audio = AudioCapture(silence_threshold=silence_threshold)
        self.transcriber = Transcriber(
            model_size=model_size, device=device, beam_size=beam_size,
        )
        self.translator = Translator()
        self.saver = FileSaver(output_dir=output_dir)
        self._running = False
        self._transcription_thread = None
        self._display_thread = None
        self._display_queue = queue.Queue()

        # --- tkinter GUI setup ---
        self.root = tk.Tk()
        self.root.title("Realtime STT - English / Korean")
        self.root.geometry("800x600")
        self.root.configure(bg="#1e1e2e")

        # Status bar
        self.status_var = tk.StringVar(value="Ready - Press Start to begin")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#313244",
            fg="#a6adc8",
            anchor="w",
            padx=10,
            pady=5,
        )
        status_bar.pack(fill="x", side="top")

        # Main text area
        default_font = tkfont.Font(family="Consolas", size=12)
        self.text_area = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            bg="#1e1e2e",
            fg="#cdd6f4",
            insertbackground="#cdd6f4",
            font=default_font,
            state=tk.DISABLED,
            padx=10,
            pady=10,
        )
        self.text_area.pack(fill="both", expand=True)

        # Tag styles for English and Korean
        self.text_area.tag_configure("timestamp", foreground="#6c7086")
        self.text_area.tag_configure("english", foreground="#89b4fa")
        self.text_area.tag_configure("korean", foreground="#a6e3a1")
        self.text_area.tag_configure("separator", foreground="#45475a")

        # Button frame
        btn_frame = tk.Frame(self.root, bg="#1e1e2e", pady=8)
        btn_frame.pack(fill="x", side="bottom")

        self.start_btn = tk.Button(
            btn_frame,
            text="Start",
            command=self._on_start,
            bg="#89b4fa",
            fg="#1e1e2e",
            width=12,
            font=("Consolas", 11, "bold"),
        )
        self.start_btn.pack(side="left", padx=10)

        self.stop_btn = tk.Button(
            btn_frame,
            text="Stop",
            command=self._on_stop,
            bg="#f38ba8",
            fg="#1e1e2e",
            width=12,
            state=tk.DISABLED,
            font=("Consolas", 11, "bold"),
        )
        self.stop_btn.pack(side="left", padx=10)

        self.file_label = tk.Label(
            btn_frame,
            text="Log: (not started)",
            bg="#1e1e2e",
            fg="#6c7086",
            font=("Consolas", 9),
        )
        self.file_label.pack(side="right", padx=10)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _append_text(self, timestamp: str, english: str, korean: str):
        """Append a transcription result to the text area (thread-safe)."""
        self.text_area.configure(state=tk.NORMAL)
        self.text_area.insert(tk.END, f"[{timestamp}]\n", "timestamp")
        self.text_area.insert(tk.END, f"EN: {english}\n", "english")
        self.text_area.insert(tk.END, f"KO: {korean}\n", "korean")
        self.text_area.insert(tk.END, "-" * 60 + "\n", "separator")
        self.text_area.configure(state=tk.DISABLED)
        self.text_area.see(tk.END)

    def _transcription_worker(self):
        """
        Transcription worker (background thread):
        Grabs audio segments, transcribes them, and submits for translation.
        Does NOT wait for translation — keeps processing audio immediately.
        """
        from datetime import datetime

        while self._running:
            segment = self.audio.get_segment(timeout=0.5)
            if segment is None:
                continue

            self.root.after(
                0, lambda: self.status_var.set("Transcribing...")
            )

            english_text = self.transcriber.transcribe(segment)
            if not english_text:
                self.root.after(
                    0, lambda: self.status_var.set("Listening...")
                )
                continue

            timestamp = datetime.now().strftime("%H:%M:%S")
            self.translator.submit(english_text)
            # Store timestamp with the submission for the display worker
            self._display_queue.put((timestamp, english_text))

            self.root.after(
                0, lambda: self.status_var.set("Listening...")
            )

    def _display_worker(self):
        """
        Display worker (background thread):
        Picks up translation results and displays them.
        Runs independently from transcription so it never blocks audio processing.
        """
        import queue as _queue

        while self._running:
            try:
                timestamp, english_text = self._display_queue.get(timeout=0.5)
            except _queue.Empty:
                continue

            result = self.translator.get_result(timeout=10.0)

            if result is None:
                korean_text = "(translation timeout)"
            else:
                korean_text = result.korean

            self.saver.save(english_text, korean_text)

            self.root.after(
                0,
                lambda ts=timestamp, en=english_text, ko=korean_text: (
                    self._append_text(ts, en, ko),
                    self.status_var.set("Listening..."),
                ),
            )

    def _on_start(self):
        """Start the real-time transcription pipeline."""
        self._running = True
        self.audio.start()
        self.translator.start()
        self.saver.start_session()
        self.transcriber.reset_context()

        # Clear text area for new session
        self.text_area.configure(state=tk.NORMAL)
        self.text_area.delete("1.0", tk.END)
        self.text_area.configure(state=tk.DISABLED)

        self.file_label.configure(text=f"Log: {self.saver.get_file_path()}")

        self._transcription_thread = threading.Thread(
            target=self._transcription_worker, daemon=True
        )
        self._transcription_thread.start()

        self._display_thread = threading.Thread(
            target=self._display_worker, daemon=True
        )
        self._display_thread.start()

        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("Listening...")

    def _on_stop(self):
        """Stop the transcription pipeline and save session log."""
        self._running = False
        self.audio.stop()
        self.translator.stop()

        if self._transcription_thread is not None:
            self._transcription_thread.join(timeout=3.0)
            self._transcription_thread = None
        if self._display_thread is not None:
            self._display_thread.join(timeout=3.0)
            self._display_thread = None

        # Finalize session: append English summary and open in Notepad
        saved_path = self.saver.finalize_session(open_in_editor=True)
        if saved_path:
            self.status_var.set(f"Saved: {saved_path}")
        else:
            self.status_var.set("Stopped (no transcriptions)")

        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)

    def _on_close(self):
        """Handle window close."""
        self._on_stop()
        self.root.destroy()

    def run(self):
        """Launch the application."""
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="Realtime Speech-to-Text with Korean Translation"
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base). Larger = more accurate but slower.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "openvino-gpu", "openvino-npu"],
        help="Compute device (default: auto). "
             "cuda=NVIDIA GPU, openvino-gpu=Intel GPU, openvino-npu=Intel NPU.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save transcript files (default: ~/realtime-stt/transcripts/).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Silence threshold for voice detection (default: 0.01). "
             "Lower = more sensitive to quiet sounds.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=3,
        help="Beam size for decoding (default: 3). "
             "Higher = better accuracy for accented speech but slower.",
    )
    args = parser.parse_args()

    app = App(
        model_size=args.model,
        device=args.device,
        output_dir=args.output_dir,
        silence_threshold=args.threshold,
        beam_size=args.beam_size,
    )
    app.run()


if __name__ == "__main__":
    main()
