"""Unit and integration tests for Gnani voice integration and speech normalization."""
import os
import wave
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from voiceops.voice.speech_text import normalize_for_speech
from voiceops.voice.audio_utils import validate_audio_file
from voiceops.voice.stt import GnaniSTTWrapper
from voiceops.voice.tts import GnaniTTSWrapper
from voiceops.voice.speech import process_voice_turn
from voiceops.config import GNANI_API_KEY

def test_speech_normalization():
    # 1. Date conversion
    text = "Your refund was initiated on 2026-10-27."
    normalized = normalize_for_speech(text)
    assert "27 October 2026" in normalized
    assert "2026-10-27" not in normalized

    # 2. Currency expansion
    text2 = "A reverse pickup fee of ₹49 applies."
    norm2 = normalize_for_speech(text2)
    assert "49 rupees" in norm2

    # 3. Order ID chunking
    text3 = "Order 89782837 is on its way."
    norm3 = normalize_for_speech(text3)
    assert "8978 2837" in norm3

    # 4. Acronyms & Markdown removal
    text4 = "**Note**: COD fee is ₹25 for [RET-2] returns."
    norm4 = normalize_for_speech(text4)
    assert "cash on delivery" in norm4
    assert "25 rupees" in norm4
    assert "**" not in norm4
    assert "[RET-2]" not in norm4

def test_audio_file_validation(tmp_path):
    # Non-existent file
    v1, msg1 = validate_audio_file(tmp_path / "missing.wav")
    assert v1 is False

    # Unsupported extension
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("not audio")
    v2, msg2 = validate_audio_file(txt_file)
    assert v2 is False
    assert "Unsupported" in msg2

    # Empty WAV
    empty_wav = tmp_path / "empty.wav"
    empty_wav.touch()
    v3, msg3 = validate_audio_file(empty_wav)
    assert v3 is False
    assert "empty" in msg3

def test_empty_transcript_recovery(tmp_path):
    """Assert empty transcript returns recovery without calling LLM."""
    dummy_wav = tmp_path / "dummy.wav"
    # Create valid silent wav
    with wave.open(str(dummy_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 3200)

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = {"success": False, "transcript": "", "is_empty": True}
    
    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = {"success": True}

    with patch("voiceops.voice.speech.get_stt_client", return_value=mock_stt), \
         patch("voiceops.voice.speech.get_tts_client", return_value=mock_tts), \
         patch("voiceops.voice.speech.get_support_agent") as mock_agent:

        res = process_voice_turn(dummy_wav, conversation_id="test_silent", output_dir=tmp_path)
        assert res["success"] is False
        assert "didn't catch that" in res["reply_text"].lower()
        # Ensure LLM agent was NOT called on silent audio
        mock_agent.assert_not_called()

def test_explicit_tts_model_passed(tmp_path):
    """Assert model='timbre-v2.5' is explicitly passed on all TTS calls."""
    wrapper = GnaniTTSWrapper(api_key="mock_key")
    wrapper.client = MagicMock()
    wrapper.client.synthesize.return_value = b"RIFFmockwav"

    out_file = tmp_path / "test_out.wav"
    res = wrapper.synthesize("Hello", output_file=out_file)

    assert res["success"] is True
    # Verify client was called with explicit model
    wrapper.client.synthesize.assert_called_once()
    call_kwargs = wrapper.client.synthesize.call_args[1]
    assert call_kwargs["model"] == "timbre-v2.5"
    assert call_kwargs["voice"] == "Kaveri"

@pytest.mark.skipif(not GNANI_API_KEY, reason="Live Gnani API key not available")
def test_live_gnani_tts_synthesis(tmp_path):
    """Live verification of Gnani TTS API."""
    tts = GnaniTTSWrapper()
    out_file = tmp_path / "live_test.wav"
    res = tts.synthesize("Hello from ZentroMart support.", output_file=out_file)
    assert res["success"] is True
    assert out_file.exists()
    assert out_file.stat().st_size > 1000
