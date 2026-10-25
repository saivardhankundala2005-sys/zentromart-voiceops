#!/usr/bin/env python3
"""VoiceOps 2026 - Batch Audio Evaluation CLI Runner.

Processes a folder of customer audio clips using Gnani STT, the ZentroMart Grounded
Customer Support Agent, and Gnani TTS. Generates per-clip structured JSON outputs,
spoken reply WAV files, and consolidated summary reports (CSV and JSON).

Usage:
    python run.py --input-dir <audio_folder> --output-dir <out_folder> [options]
"""

import os
import sys
import re
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd

# Ensure package on path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Safe UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from voiceops.config import DATA_DIR, STT_LANG, TTS_VOICE, TTS_MODEL
from voiceops.voice.stt import get_stt_client
from voiceops.voice.tts import get_tts_client
from voiceops.voice.audio_utils import validate_audio_file
from voiceops.agent.orchestrator import get_support_agent


def parse_args():
    parser = argparse.ArgumentParser(
        description="VoiceOps 2026 Batch Audio Runner for ZentroMart Voice Support Agent"
    )
    # Support both kebab-case and snake_case flags
    parser.add_argument(
        "--input-dir", "--input_dir", "-i",
        dest="input_dir",
        required=True,
        type=str,
        help="Path to folder containing customer audio clips (.wav, .mp3, etc.)",
    )
    parser.add_argument(
        "--output-dir", "--output_dir", "-o",
        dest="output_dir",
        required=True,
        type=str,
        help="Path to folder where replies and JSON metadata will be written",
    )
    parser.add_argument(
        "--data-dir", "--data_dir", "-d",
        dest="data_dir",
        default=str(DATA_DIR),
        type=str,
        help="Override path to ZentroMart dataset folder",
    )
    parser.add_argument(
        "--stt-lang", "--stt_lang",
        dest="stt_lang",
        default=STT_LANG,
        type=str,
        help="Language code for Gnani STT (default: en-IN)",
    )
    parser.add_argument(
        "--voice",
        dest="voice",
        default=TTS_VOICE,
        type=str,
        help="Gnani TTS voice name (default: Kaveri)",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=TTS_MODEL,
        type=str,
        help="Gnani TTS model identifier (default: timbre-v2.5)",
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        default=1,
        type=int,
        help="Concurrency limit (default: 1 sequential for rate safety)",
    )
    return parser.parse_args()


def discover_audio_files(input_dir: Path) -> List[Tuple[str, int, Path]]:
    """Scan directory for audio files and group multi-turn clips while preserving order.
    
    Returns list of tuples: (conversation_id, turn_id, file_path)
    """
    valid_exts = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
    audio_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
    
    items = []
    # Pattern e.g.: conv01_turn1.wav, c01_t2.wav, conversation_12-turn-3.wav
    conv_pattern = re.compile(r"^(?P<conv>.+?)[_-](?:turn|t)(?P<turn>\d+)(?:\..+)?$", re.IGNORECASE)

    for f in audio_files:
        match = conv_pattern.match(f.stem)
        if match:
            conv_id = match.group("conv")
            turn_id = int(match.group("turn"))
        else:
            conv_id = f"single_{f.stem}"
            turn_id = 1
        items.append((conv_id, turn_id, f))

    # Sort primarily by conversation_id, then by turn_id
    items.sort(key=lambda x: (x[0], x[1], x[2].name))
    return items


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[Error] Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VoiceOps 2026 - ZentroMart Voice Support Agent Batch Runner")
    print(f"Input Directory  : {input_dir}")
    print(f"Output Directory : {output_dir}")
    print(f"Gnani STT Lang   : {args.stt_lang}")
    print(f"Gnani TTS Model  : {args.model} (Voice: {args.voice})")
    print("=" * 70)

    audio_items = discover_audio_files(input_dir)
    if not audio_items:
        print(f"[Warning] No supported audio files (.wav, .mp3, etc.) found in {input_dir}.")
        # Create empty results
        summary = {
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(output_dir / "batch_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        pd.DataFrame(columns=[
            "audio_file", "conversation_id", "turn_id", "heard_transcript",
            "reply_text", "decision", "retrieved_section_ids", "ticket_id", "total_ms", "status"
        ]).to_csv(output_dir / "results.csv", index=False)
        return

    print(f"Found {len(audio_items)} audio clip(s) to process.\n")

    stt_client = get_stt_client()
    tts_client = get_tts_client()
    agent = get_support_agent()

    batch_start_time = time.time()
    results_records = []
    success_count = 0
    fail_count = 0

    for idx, (conv_id, turn_id, audio_file) in enumerate(audio_items, 1):
        print(f"[{idx}/{len(audio_items)}] Processing '{audio_file.name}' (Conv: {conv_id}, Turn: {turn_id})...")
        t_start = time.time()
        stt_ms = 0.0
        agent_ms = 0.0
        tts_ms = 0.0

        try:
            # Step 1: Validate audio file format
            is_valid, validation_err = validate_audio_file(audio_file)
            if not is_valid:
                raise ValueError(f"Invalid audio file: {validation_err}")

            # Step 2: Gnani STT Transcription
            t0 = time.time()
            stt_res = stt_client.transcribe(audio_file, language_code=args.stt_lang)
            stt_ms = round((time.time() - t0) * 1000, 1)

            heard_transcript = stt_res.get("transcript", "").strip()
            is_empty = stt_res.get("is_empty", False)
            print(f"   -> STT ({stt_ms} ms): \"{heard_transcript[:80]}{'...' if len(heard_transcript)>80 else ''}\"")

            # Step 3: Support Agent Processing (RAG + Identity Verification + Tools + Policy)
            t0 = time.time()
            if is_empty or not heard_transcript:
                # Fallback for empty audio input
                reply_text = "Sorry, I didn't catch that. Could you please say it again?"
                spoken_text = reply_text
                decision = "clarify"
                retrieved_section_ids = []
                tool_calls = []
                ticket_id = None
                agent_ms = round((time.time() - t0) * 1000, 1)
            else:
                agent_res = agent.respond(
                    message=heard_transcript,
                    conversation_id=conv_id,
                    language_hint=stt_res.get("language_code"),
                    mode="full",
                )
                agent_ms = round((time.time() - t0) * 1000, 1)
                reply_text = agent_res.reply
                spoken_text = agent_res.spoken_reply
                decision = agent_res.decision
                retrieved_section_ids = [c["section_id"] for c in agent_res.retrieved_chunks]
                tool_calls = agent_res.tool_calls
                ticket_id = agent_res.ticket_id

            print(f"   -> Agent ({agent_ms} ms) [{decision.upper()}]: \"{spoken_text[:80]}{'...' if len(spoken_text)>80 else ''}\"")

            # Step 4: Gnani TTS Audio Synthesis
            t0 = time.time()
            reply_wav_filename = f"{audio_file.stem}_reply.wav"
            reply_wav_path = output_dir / reply_wav_filename

            tts_res = tts_client.synthesize(
                text=spoken_text,
                output_file=reply_wav_path,
                voice=args.voice,
                model=args.model,
                language=args.stt_lang,
            )
            tts_ms = round((time.time() - t0) * 1000, 1)
            total_ms = round((time.time() - t_start) * 1000, 1)
            print(f"   -> TTS ({tts_ms} ms): Generated {reply_wav_filename} | Total: {total_ms} ms\n")

            # Write individual turn JSON artifact
            turn_json = {
                "audio_file": audio_file.name,
                "conversation_id": conv_id,
                "turn_id": turn_id,
                "heard_transcript": heard_transcript,
                "reply_text": reply_text,
                "spoken_text": spoken_text,
                "reply_audio_file": reply_wav_filename,
                "reply_audio_path": str(reply_wav_path),
                "retrieved_section_ids": retrieved_section_ids,
                "decision": decision,
                "tool_calls": tool_calls,
                "ticket_id": ticket_id,
                "timings": {
                    "stt_ms": stt_ms,
                    "agent_ms": agent_ms,
                    "tts_ms": tts_ms,
                    "total_ms": total_ms,
                },
                "success": True,
                "error": None,
            }
            with open(output_dir / f"{audio_file.stem}.json", "w", encoding="utf-8") as f:
                json.dump(turn_json, f, indent=2)

            results_records.append({
                "audio_file": audio_file.name,
                "conversation_id": conv_id,
                "turn_id": turn_id,
                "heard_transcript": heard_transcript,
                "reply_text": reply_text,
                "decision": decision,
                "retrieved_section_ids": ";".join(retrieved_section_ids),
                "ticket_id": ticket_id or "",
                "total_ms": total_ms,
                "status": "success",
            })
            success_count += 1

        except Exception as e:
            fail_count += 1
            total_ms = round((time.time() - t_start) * 1000, 1)
            print(f"   [!] Error processing {audio_file.name}: {e}\n")
            err_json = {
                "audio_file": audio_file.name,
                "conversation_id": conv_id,
                "turn_id": turn_id,
                "heard_transcript": "",
                "reply_text": "An internal error occurred while processing this request.",
                "spoken_text": "An internal error occurred.",
                "reply_audio_file": None,
                "reply_audio_path": None,
                "retrieved_section_ids": [],
                "decision": "error",
                "tool_calls": [],
                "ticket_id": None,
                "timings": {"total_ms": total_ms},
                "success": False,
                "error": str(e),
            }
            with open(output_dir / f"{audio_file.stem}.json", "w", encoding="utf-8") as f:
                json.dump(err_json, f, indent=2)

            results_records.append({
                "audio_file": audio_file.name,
                "conversation_id": conv_id,
                "turn_id": turn_id,
                "heard_transcript": "",
                "reply_text": f"Error: {e}",
                "decision": "error",
                "retrieved_section_ids": "",
                "ticket_id": "",
                "total_ms": total_ms,
                "status": "error",
            })

    # Consolidated outputs
    df_results = pd.DataFrame(results_records)
    results_csv_path = output_dir / "results.csv"
    df_results.to_csv(results_csv_path, index=False)

    total_batch_sec = round(time.time() - batch_start_time, 2)
    avg_turn_ms = round(df_results["total_ms"].mean(), 1) if not df_results.empty else 0.0

    decisions_breakdown = {}
    if not df_results.empty:
        decisions_breakdown = df_results["decision"].value_counts().to_dict()

    batch_summary = {
        "total_files": len(audio_items),
        "successful_files": success_count,
        "failed_files": fail_count,
        "total_time_seconds": total_batch_sec,
        "avg_total_ms_per_turn": avg_turn_ms,
        "decisions_breakdown": decisions_breakdown,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2)

    print("=" * 70)
    print(f"Batch Processing Completed in {total_batch_sec} seconds")
    print(f"Successful: {success_count} | Failed: {fail_count}")
    print(f"Results CSV     : {results_csv_path}")
    print(f"Summary JSON    : {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
