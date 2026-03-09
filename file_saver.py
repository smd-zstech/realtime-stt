"""
File saving module - saves transcription results to a log file in real-time.
Each entry includes a timestamp, the English transcription, and the Korean translation.
"""

import os
from datetime import datetime
from pathlib import Path


class FileSaver:
    """Saves transcription results to a file in real-time."""

    def __init__(self, output_dir: str = None):
        """
        Args:
            output_dir: Directory to save transcript files.
                        Defaults to ~/realtime-stt/transcripts/
        """
        if output_dir is None:
            output_dir = os.path.join(
                os.path.expanduser("~"), "realtime-stt", "transcripts"
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = self.output_dir / f"transcript_{timestamp}.txt"

        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(f"=== Realtime Transcription Log ===\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 40}\n\n")

    def save(self, english: str, korean: str):
        """Append a transcription entry to the log file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}]\n")
            f.write(f"EN: {english}\n")
            f.write(f"KO: {korean}\n\n")

    def get_file_path(self) -> str:
        """Return the path to the current transcript file."""
        return str(self.file_path)
