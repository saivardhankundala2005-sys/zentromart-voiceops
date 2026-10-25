"""Gnani Text-to-Speech integration (strictly gnani-vachana SDK)."""
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
from gnani.tts import GnaniTTSClient
from gnani.tts.exceptions import GnaniTTSError, APIError

from voiceops.config import GNANI_API_KEY, TTS_MODEL, TTS_VOICE, STT_LANG

# Mapping from internal language codes to Gnani TTS language codes
TTS_LANG_MAP = {
    "en": "en-IN",
    "en-IN": "en-IN",
    "hi": "hi-IN",
    "hi-IN": "hi-IN",
    "hi-en": "hi-en",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "pa": "pa-IN",
    "ta": "ta-IN",
    "te": "te-IN",
}

class GnaniTTSWrapper:
    """Encapsulates Gnani TTS client with explicit model specification and error handling."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GNANI_API_KEY
        if not self.api_key:
            raise ValueError("GNANI_API_KEY is not configured.")
        self.client = GnaniTTSClient(api_key=self.api_key)

    def synthesize(
        self,
        text: str,
        output_file: Path,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synthesize spoken text into output WAV file.
        
        CRITICAL: Always passes explicit model ('timbre-v2.5' by default) per API requirement.
        """
        if not text or not text.strip():
            return {
                "success": False,
                "output_file": None,
                "error": "Empty text provided for synthesis.",
                "bytes_len": 0,
            }

        target_voice = voice or TTS_VOICE
        target_model = model or TTS_MODEL
        raw_lang = language or STT_LANG
        target_lang = TTS_LANG_MAP.get(raw_lang, "en-IN")

        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(3):
            try:
                audio_bytes = self.client.synthesize(
                    text=text,
                    voice=target_voice,
                    model=target_model, # Explicitly specified!
                    language=target_lang,
                    output_file=str(out_path),
                )
                return {
                    "success": True,
                    "output_file": str(out_path),
                    "bytes_len": len(audio_bytes) if audio_bytes else out_path.stat().st_size,
                    "error": None,
                    "voice": target_voice,
                    "model": target_model,
                    "language": target_lang,
                }
            except (APIError, GnaniTTSError) as exc:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    return {
                        "success": False,
                        "output_file": None,
                        "bytes_len": 0,
                        "error": f"Gnani TTS failed after 3 attempts: {exc}",
                        "voice": target_voice,
                        "model": target_model,
                        "language": target_lang,
                    }

# Global singleton
_TTS_INSTANCE: Optional[GnaniTTSWrapper] = None

def get_tts_client() -> GnaniTTSWrapper:
    global _TTS_INSTANCE
    if _TTS_INSTANCE is None:
        _TTS_INSTANCE = GnaniTTSWrapper()
    return _TTS_INSTANCE
