"""
File saving module - saves transcription results to a log file in real-time.
Each entry includes a timestamp, the English transcription, and the Korean translation.

On session finalize (Stop), appends a polished English summary at the end of the log.
"""

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _clean_and_merge_english(sentences: list[str]) -> str:
    """Merge English transcript fragments into a clean, readable paragraph.

    - Removes consecutive duplicate sentences
    - Ensures proper punctuation and capitalization
    - Joins into flowing paragraphs
    """
    if not sentences:
        return ""

    # Remove consecutive duplicates (exact or near-match)
    deduped = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if deduped and s.lower() == deduped[-1].lower():
            continue
        deduped.append(s)

    if not deduped:
        return ""

    # Ensure each sentence ends with proper punctuation
    cleaned = []
    for s in deduped:
        # Remove leading/trailing whitespace
        s = s.strip()
        # Remove repeated whitespace
        s = re.sub(r"\s+", " ", s)
        # Capitalize first letter
        if s and s[0].islower():
            s = s[0].upper() + s[1:]
        # Add period if no ending punctuation
        if s and s[-1] not in ".!?;:":
            s += "."
        cleaned.append(s)

    return " ".join(cleaned)


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

        self.file_path = None
        self._session_entries: list[tuple[str, str, str]] = []  # (timestamp, en, ko)

    def start_session(self):
        """Start a new session log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = self.output_dir / f"transcript_{timestamp}.txt"
        self._session_entries = []

        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write("=== Realtime Transcription Log ===\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 40}\n\n")

    def save(self, english: str, korean: str):
        """Append a transcription entry to the log file."""
        if self.file_path is None:
            self.start_session()

        timestamp = datetime.now().strftime("%H:%M:%S")
        self._session_entries.append((timestamp, english, korean))

        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}]\n")
            f.write(f"EN: {english}\n")
            f.write(f"KO: {korean}\n\n")

    def finalize_session(self, open_in_editor: bool = True):
        """Finalize the session: append merged English summary and open in editor.

        Args:
            open_in_editor: If True, open the log file in Notepad (Windows).
        """
        if self.file_path is None or not self._session_entries:
            return None

        # Append the polished English summary at the end
        english_sentences = [en for _, en, _ in self._session_entries]
        merged = _clean_and_merge_english(english_sentences)

        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 40}\n")
            f.write(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 40}\n\n")
            f.write("=== English Summary (merged) ===\n\n")
            f.write(merged + "\n")

        saved_path = str(self.file_path)

        # Open in default text editor
        if open_in_editor:
            try:
                if sys.platform == "win32":
                    os.startfile(saved_path)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", saved_path])
                else:
                    subprocess.Popen(["xdg-open", saved_path])
            except Exception as e:
                print(f"[WARN] Could not open log file: {e}")

        # Reset for next session
        self._session_entries = []
        self.file_path = None

        return saved_path

    def get_file_path(self) -> str:
        """Return the path to the current transcript file."""
        if self.file_path is None:
            return str(self.output_dir / "(not started)")
        return str(self.file_path)
