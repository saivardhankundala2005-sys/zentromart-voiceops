"""Gnani Speech-to-Text integration (strictly gnani-vachana SDK)."""
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from gnani.stt import GnaniSTTClient
from gnani.stt.exceptions import GnaniSTTError, APIError, InvalidAudioError

from voiceops.config import GNANI_API_KEY, STT_LANG
from voiceops.voice.audio_utils import validate_audio_file, split_wav_under_60s

class GnaniSTTWrapper:
    """Encapsulates Gnani STT client with validation, auto-splitting, and error recovery."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GNANI_API_KEY
        if not self.api_key:
            raise ValueError("GNANI_API_KEY is not configured.")
        self.client = GnaniSTTClient(api_key=self.api_key)

    def transcribe(
        self,
        audio_path: Path,
        language_code: Optional[str] = None,
        auto_hi_fallback: bool = True
    ) -> Dict[str, Any]:
        """Transcribe an audio file using Gnani STT REST.
        
        Returns dict with:
        - success: bool
        - transcript: str
        - is_empty: bool
        - error: Optional[str]
        - language_code: str
        """
        p = Path(audio_path)
        valid, msg = validate_audio_file(p)
        if not valid:
            return {
                "success": False,
                "transcript": "",
                "is_empty": True,
                "error": msg,
                "language_code": language_code or STT_LANG,
            }

        lang = language_code or STT_LANG
        segments = split_wav_under_60s(p)
        transcripts = []

        try:
            for seg in segments:
                res = self.client.transcribe(str(seg), language_code=lang)
                text = (res.get("transcript") or "").strip()
                if text:
                    transcripts.append(text)
                # Clean up temporary split files
                if seg != p and seg.exists():
                    try:
                        seg.unlink()
                    except Exception:
                        pass
        except APIError as exc:
            return {
                "success": False,
                "transcript": "",
                "is_empty": True,
                "error": f"Gnani STT API error ({exc.status_code}): {exc.body}",
                "language_code": lang,
            }
        except GnaniSTTError as exc:
            return {
                "success": False,
                "transcript": "",
                "is_empty": True,
                "error": f"Gnani STT SDK error: {exc}",
                "language_code": lang,
            }

        full_transcript = " ".join(transcripts).strip()

        # If transcript is empty and fallback is enabled, try hi-IN once
        if not full_transcript and auto_hi_fallback and lang == "en-IN":
            try:
                res_hi = self.client.transcribe(str(p), language_code="hi-IN")
                full_transcript = (res_hi.get("transcript") or "").strip()
                if full_transcript:
                    lang = "hi-IN"
            except Exception:
                pass

        is_empty = (len(full_transcript) == 0)
        return {
            "success": True if full_transcript else False,
            "transcript": full_transcript,
            "is_empty": is_empty,
            "error": "Unintelligible or silent audio." if is_empty else None,
            "language_code": lang,
        }

# Global singleton
_STT_INSTANCE: Optional[GnaniSTTWrapper] = None

def get_stt_client() -> GnaniSTTWrapper:
    global _STT_INSTANCE
    if _STT_INSTANCE is None:
        _STT_INSTANCE = GnaniSTTWrapper()
    return _STT_INSTANCE
