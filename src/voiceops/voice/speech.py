"""High-level voice turn pipeline (Audio In -> Gnani STT -> SupportAgent -> Normalization -> Gnani TTS -> Audio Out)."""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from voiceops.config import OUTPUTS_DIR, STT_LANG
from voiceops.voice.stt import get_stt_client
from voiceops.voice.tts import get_tts_client
from voiceops.agent.orchestrator import get_support_agent
from voiceops.voice.speech_text import normalize_for_speech

def process_voice_turn(
    audio_path: Path,
    conversation_id: str = "voice_session",
    stt_language: Optional[str] = None,
    tts_voice: Optional[str] = None,
    tts_model: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete end-to-end voice turn using Gnani STT and TTS."""
    out_dir = Path(output_dir) if output_dir else OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    stt = get_stt_client()
    tts = get_tts_client()

    # 1. Speech-to-Text via Gnani
    stt_res = stt.transcribe(Path(audio_path), language_code=stt_language or STT_LANG)
    heard_text = stt_res.get("transcript", "").strip()
    is_empty = stt_res.get("is_empty", False)
    stt_error = stt_res.get("error")

    # 2. Handle empty or unintelligible audio gracefully (Voice experience criteria)
    if is_empty or not heard_text:
        recovery_text = "Sorry, I didn't catch that. Could you please say it again?"
        spoken_recovery = recovery_text
        reply_wav_path = out_dir / f"{conversation_id}_recovery.wav"
        
        # Synthesize recovery reply audio
        tts_res = tts.synthesize(
            text=spoken_recovery,
            output_file=reply_wav_path,
            voice=tts_voice,
            model=tts_model,
            language=stt_language or STT_LANG
        )

        return {
            "success": False,
            "heard_transcript": "",
            "reply_text": recovery_text,
            "spoken_text": spoken_recovery,
            "reply_audio_path": str(reply_wav_path) if tts_res.get("success") else None,
            "retrieved_section_ids": [],
            "decision": "clarify",
            "tool_calls": [],
            "ticket_id": None,
            "error": stt_error or "Empty audio transcript",
            "conversation_id": conversation_id,
        }

    # 3. Agent Response (RAG + tools + state + verification)
    agent = get_support_agent()
    agent_res = agent.respond(
        message=heard_text,
        conversation_id=conversation_id,
        language_hint=stt_res.get("language_code"),
        mode="full",
    )

    # 4. Text-to-Speech via Gnani
    reply_filename = f"{conversation_id}_turn{agent_res.turn_id}_reply.wav"
    reply_wav_path = out_dir / reply_filename

    tts_res = tts.synthesize(
        text=agent_res.spoken_reply,
        output_file=reply_wav_path,
        voice=tts_voice,
        model=tts_model,
        language=agent_res.language,
    )

    return {
        "success": True,
        "heard_transcript": heard_text,
        "reply_text": agent_res.reply,
        "spoken_text": agent_res.spoken_reply,
        "reply_audio_path": str(reply_wav_path) if tts_res.get("success") else None,
        "retrieved_section_ids": [c["section_id"] for c in agent_res.retrieved_chunks],
        "decision": agent_res.decision,
        "tool_calls": agent_res.tool_calls,
        "ticket_id": agent_res.ticket_id,
        "error": tts_res.get("error"),
        "conversation_id": conversation_id,
        "turn_id": agent_res.turn_id,
    }
