"""Audio format utilities and duration splitting (for Gnani REST 60s limit)."""
import os
import wave
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}

def validate_audio_file(file_path: Path) -> Tuple[bool, str]:
    """Validate audio file existence, extension, and non-empty size."""
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return False, f"File does not exist: {file_path}"
    if p.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        return False, f"Unsupported audio format: {p.suffix}. Supported: {SUPPORTED_AUDIO_EXTENSIONS}"
    if p.stat().st_size == 0:
        return False, "Audio file is empty (0 bytes)."
    return True, "Valid"

def get_wav_duration_seconds(file_path: Path) -> float:
    """Read duration of standard WAV file using standard library."""
    try:
        with wave.open(str(file_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception:
        pass
    return 0.0

def split_wav_under_60s(file_path: Path, max_segment_sec: float = 55.0) -> List[Path]:
    """Split WAV audio file into segments under 60 seconds if duration exceeds limit."""
    p = Path(file_path)
    if p.suffix.lower() != ".wav":
        return [p]

    try:
        with wave.open(str(p), "rb") as wf:
            params = wf.getparams()
            n_frames = wf.getnframes()
            framerate = wf.getframerate()
            duration = n_frames / float(framerate) if framerate else 0

            if duration <= 59.0:
                return [p]

            frames_per_segment = int(max_segment_sec * framerate)
            segments = []
            segment_idx = 0
            
            while wf.tell() < n_frames:
                chunk_frames = wf.readframes(frames_per_segment)
                if not chunk_frames:
                    break
                tmp = tempfile.NamedTemporaryFile(suffix=f"_seg{segment_idx}.wav", delete=False)
                tmp_path = Path(tmp.name)
                tmp.close()
                
                with wave.open(str(tmp_path), "wb") as out_wf:
                    out_wf.setparams(params)
                    out_wf.writeframes(chunk_frames)
                segments.append(tmp_path)
                segment_idx += 1
                
            return segments if segments else [p]
    except Exception:
        return [p]
