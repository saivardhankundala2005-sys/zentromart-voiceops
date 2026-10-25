#!/usr/bin/env python3
"""VoiceOps 2026 - Comprehensive Evaluation Harness.

Runs benchmarking across:
- Step 1: Baseline (Unaugmented LLM)
- Step 2: Grounded RAG (Hybrid dense/sparse retrieval + KB documents, clean & ASR text)
- Step 3: Multi-turn Conversations (Stateful session, Identity verification, Escalations, Tool execution)

Calculates must_include, must_not_include, decision accuracy, tool accuracy,
retrieval recall@6, and response latencies.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voiceops.config import DATA_DIR
from voiceops.agent.orchestrator import get_support_agent
from voiceops.evaluation.metrics import (
    evaluate_must_include,
    evaluate_must_not_include,
    evaluate_decision,
    evaluate_tool_calls,
    evaluate_retrieval,
)


def parse_args():
    parser = argparse.ArgumentParser(description="VoiceOps 2026 Evaluation Harness")
    parser.add_argument("--step", type=str, default="all", choices=["1", "2", "3", "all"],
                        help="Step to evaluate: 1 (Baseline), 2 (RAG), 3 (Conversations), or all")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on number of items to test per step")
    parser.add_argument("--output-dir", type=str, default="reports",
                        help="Directory to save CSV and JSON evaluation reports")
    parser.add_argument("--pause-sec", type=float, default=0.2,
                        help="Pause duration between calls to respect API pacing")
    parser.add_argument("--include-asr", action="store_true", default=True,
                        help="Include simulated ASR text evaluation in Step 2")
    return parser.parse_args()


def run_step1_baseline(agent, questions_df: pd.DataFrame, limit: int = None, pause: float = 0.2) -> pd.DataFrame:
    """Evaluate Step 1: Baseline (Direct LLM generation without KB or order facts)."""
    print("\n" + "=" * 60)
    print("RUNNING STEP 1: BASELINE EVALUATION (Prompting Only)")
    print("=" * 60)

    rows = []
    df_eval = questions_df.head(limit) if limit else questions_df

    for idx, (_, row) in enumerate(df_eval.iterrows(), 1):
        q_id = row["id"]
        question = row["question"]
        expected_dec = row["expected_decision"]
        acceptable_dec = row["acceptable_decisions"]
        must_inc = row["must_include"]
        must_not_inc = row["must_not_include"]

        t0 = time.time()
        res = agent.respond(
            message=question,
            conversation_id=f"eval_step1_{q_id}_{int(time.time()*1000)}",
            mode="baseline"
        )
        latency_ms = round((time.time() - t0) * 1000, 1)

        pass_inc, miss_inc = evaluate_must_include(res.reply, must_inc)
        pass_not_inc, viol_not_inc = evaluate_must_not_include(res.reply, must_not_inc)
        dec_match = evaluate_decision(res.decision, acceptable_dec, expected_dec)

        print(f"[{idx}/{len(df_eval)}] {q_id} | Inc: {'PASS' if pass_inc else 'FAIL'} | "
              f"Not-Inc: {'PASS' if pass_not_inc else 'FAIL'} | Dec: {'PASS' if dec_match else 'FAIL'} ({latency_ms} ms)")

        rows.append({
            "id": q_id,
            "question": question,
            "reply": res.reply,
            "decision": res.decision,
            "expected_decision": expected_dec,
            "decision_match": dec_match,
            "must_include": must_inc,
            "must_include_pass": pass_inc,
            "must_include_missing": "; ".join(miss_inc),
            "must_not_include": must_not_inc,
            "must_not_include_pass": pass_not_inc,
            "must_not_include_violations": "; ".join(viol_not_inc),
            "all_rules_pass": (pass_inc and pass_not_inc and dec_match),
            "latency_ms": latency_ms,
        })
        time.sleep(pause)

    return pd.DataFrame(rows)


def run_step2_rag(agent, questions_df: pd.DataFrame, use_asr: bool = False, limit: int = None, pause: float = 0.2) -> pd.DataFrame:
    """Evaluate Step 2: Grounded RAG with Hybrid Retrieval."""
    name = "STEP 2: GROUNDED RAG (ASR Robustness)" if use_asr else "STEP 2: GROUNDED RAG (Clean Text)"
    print("\n" + "=" * 60)
    print(f"RUNNING {name}")
    print("=" * 60)

    rows = []
    df_eval = questions_df.head(limit) if limit else questions_df

    for idx, (_, row) in enumerate(df_eval.iterrows(), 1):
        q_id = row["id"]
        # Use asr_text if testing ASR robustness, else clean question
        question = row["asr_text"] if (use_asr and pd.notna(row["asr_text"])) else row["question"]
        expected_dec = row["expected_decision"]
        acceptable_dec = row["acceptable_decisions"]
        must_inc = row["must_include"]
        must_not_inc = row["must_not_include"]
        gold_ev = row.get("gold_evidence")
        hard_neg = row.get("hard_negatives")

        t0 = time.time()
        res = agent.respond(
            message=question,
            conversation_id=f"eval_step2_{'asr_' if use_asr else ''}{q_id}_{int(time.time()*1000)}",
            mode="rag"
        )
        latency_ms = round((time.time() - t0) * 1000, 1)

        pass_inc, miss_inc = evaluate_must_include(res.reply, must_inc)
        pass_not_inc, viol_not_inc = evaluate_must_not_include(res.reply, must_not_inc)
        dec_match = evaluate_decision(res.decision, acceptable_dec, expected_dec)

        ret_sec_ids = [c["section_id"] for c in res.retrieved_chunks]
        ret_metrics = evaluate_retrieval(ret_sec_ids, gold_ev, hard_neg)

        print(f"[{idx}/{len(df_eval)}] {q_id} | Inc: {'PASS' if pass_inc else 'FAIL'} | "
              f"Not-Inc: {'PASS' if pass_not_inc else 'FAIL'} | Dec: {'PASS' if dec_match else 'FAIL'} | "
              f"Recall: {ret_metrics['recall']*100:.1f}% ({latency_ms} ms)")

        rows.append({
            "id": q_id,
            "input_text": question,
            "is_asr": use_asr,
            "reply": res.reply,
            "decision": res.decision,
            "expected_decision": expected_dec,
            "decision_match": dec_match,
            "must_include": must_inc,
            "must_include_pass": pass_inc,
            "must_include_missing": "; ".join(miss_inc),
            "must_not_include": must_not_inc,
            "must_not_include_pass": pass_not_inc,
            "must_not_include_violations": "; ".join(viol_not_inc),
            "gold_evidence": gold_ev,
            "hard_negatives": hard_neg,
            "retrieved_section_ids": "; ".join(ret_sec_ids),
            "retrieval_recall": ret_metrics["recall"],
            "has_hard_negative": ret_metrics["has_hard_negative"],
            "all_rules_pass": (pass_inc and pass_not_inc and dec_match),
            "latency_ms": latency_ms,
        })
        time.sleep(pause)

    return pd.DataFrame(rows)


def run_step3_conversations(agent, convs_df: pd.DataFrame, limit: int = None, pause: float = 0.2) -> pd.DataFrame:
    """Evaluate Step 3: Multi-turn Conversations with State, Verification, and Escalation."""
    print("\n" + "=" * 60)
    print("RUNNING STEP 3: MULTI-TURN CONVERSATIONS")
    print("=" * 60)

    rows = []
    # Group by conversation_id
    grouped = convs_df.groupby("conversation_id", sort=False)
    conv_list = list(grouped)
    if limit:
        conv_list = conv_list[:limit]

    total_turns = sum(len(df_c) for _, df_c in conv_list)
    turn_counter = 0

    for conv_idx, (cid, df_conv) in enumerate(conv_list, 1):
        print(f"\n--- Conversation [{conv_idx}/{len(conv_list)}]: {cid} ({len(df_conv)} turns) ---")
        # Ensure fresh state per conversation session
        session_id = f"eval_conv_{cid}_{int(time.time()*1000)}"

        # Sort strictly by turn
        df_conv_sorted = df_conv.sort_values("turn")

        for _, row in df_conv_sorted.iterrows():
            turn_counter += 1
            turn_num = int(row["turn"])
            cust_msg = row["customer_message"]
            expected_dec = row["expected_decision"]
            must_inc = row.get("must_include")
            must_not_inc = row.get("must_not_include")
            exp_tools = row.get("expected_tool_calls")

            t0 = time.time()
            res = agent.respond(
                message=cust_msg,
                conversation_id=session_id,
                mode="full"
            )
            latency_ms = round((time.time() - t0) * 1000, 1)

            pass_inc, miss_inc = evaluate_must_include(res.reply, must_inc)
            pass_not_inc, viol_not_inc = evaluate_must_not_include(res.reply, must_not_inc)
            dec_match = evaluate_decision(res.decision, expected_dec, expected_dec)
            tool_match = evaluate_tool_calls(
                [{"name": t.split("(")[0]} for t in res.tool_calls],
                exp_tools
            )

            print(f"[{turn_counter}/{total_turns}] {cid}-T{turn_num} | Inc: {'PASS' if pass_inc else 'FAIL'} | "
                  f"Not-Inc: {'PASS' if pass_not_inc else 'FAIL'} | Dec: {'PASS' if dec_match else 'FAIL'} | "
                  f"Tool: {'PASS' if tool_match else 'FAIL'} ({latency_ms} ms)")

            rows.append({
                "conversation_id": cid,
                "turn": turn_num,
                "customer_message": cust_msg,
                "reply": res.reply,
                "decision": res.decision,
                "expected_decision": expected_dec,
                "decision_match": dec_match,
                "must_include": must_inc,
                "must_include_pass": pass_inc,
                "must_include_missing": "; ".join(miss_inc),
                "must_not_include": must_not_inc,
                "must_not_include_pass": pass_not_inc,
                "must_not_include_violations": "; ".join(viol_not_inc),
                "expected_tool_calls": exp_tools,
                "actual_tool_calls": "; ".join(res.tool_calls),
                "tool_match": tool_match,
                "all_rules_pass": (pass_inc and pass_not_inc and dec_match and tool_match),
                "latency_ms": latency_ms,
            })
            time.sleep(pause)

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions_path = DATA_DIR / "practice" / "questions.csv"
    convs_path = DATA_DIR / "practice" / "conversations.csv"

    if not questions_path.exists() or not convs_path.exists():
        print(f"[Error] Dataset files not found at {DATA_DIR}.")
        sys.exit(1)

    questions_df = pd.read_csv(questions_path, dtype=str)
    convs_df = pd.read_csv(convs_path, dtype=str)

    agent = get_support_agent()
    summary = {}

    # STEP 1
    if args.step in ["1", "all"]:
        df_s1 = run_step1_baseline(agent, questions_df, limit=args.limit, pause=args.pause_sec)
        df_s1.to_csv(out_dir / "step1_baseline.csv", index=False)
        summary["step1"] = {
            "total_questions": len(df_s1),
            "must_include_accuracy": round(df_s1["must_include_pass"].mean() * 100, 2),
            "must_not_include_accuracy": round(df_s1["must_not_include_pass"].mean() * 100, 2),
            "decision_accuracy": round(df_s1["decision_match"].mean() * 100, 2),
            "all_rules_accuracy": round(df_s1["all_rules_pass"].mean() * 100, 2),
            "avg_latency_ms": round(df_s1["latency_ms"].mean(), 1),
        }

    # STEP 2
    if args.step in ["2", "all"]:
        df_s2_clean = run_step2_rag(agent, questions_df, use_asr=False, limit=args.limit, pause=args.pause_sec)
        df_s2_clean.to_csv(out_dir / "step2_rag_clean.csv", index=False)
        summary["step2_clean"] = {
            "total_questions": len(df_s2_clean),
            "must_include_accuracy": round(df_s2_clean["must_include_pass"].mean() * 100, 2),
            "must_not_include_accuracy": round(df_s2_clean["must_not_include_pass"].mean() * 100, 2),
            "decision_accuracy": round(df_s2_clean["decision_match"].mean() * 100, 2),
            "all_rules_accuracy": round(df_s2_clean["all_rules_pass"].mean() * 100, 2),
            "retrieval_recall": round(df_s2_clean["retrieval_recall"].mean() * 100, 2),
            "hard_negative_hit_rate": round(df_s2_clean["has_hard_negative"].mean() * 100, 2),
            "avg_latency_ms": round(df_s2_clean["latency_ms"].mean(), 1),
        }

        if args.include_asr:
            df_s2_asr = run_step2_rag(agent, questions_df, use_asr=True, limit=args.limit, pause=args.pause_sec)
            df_s2_asr.to_csv(out_dir / "step2_rag_asr.csv", index=False)
            summary["step2_asr"] = {
                "total_questions": len(df_s2_asr),
                "must_include_accuracy": round(df_s2_asr["must_include_pass"].mean() * 100, 2),
                "must_not_include_accuracy": round(df_s2_asr["must_not_include_pass"].mean() * 100, 2),
                "decision_accuracy": round(df_s2_asr["decision_match"].mean() * 100, 2),
                "all_rules_accuracy": round(df_s2_asr["all_rules_pass"].mean() * 100, 2),
                "avg_latency_ms": round(df_s2_asr["latency_ms"].mean(), 1),
            }

    # STEP 3
    if args.step in ["3", "all"]:
        df_s3 = run_step3_conversations(agent, convs_df, limit=args.limit, pause=args.pause_sec)
        df_s3.to_csv(out_dir / "step3_conversations.csv", index=False)
        summary["step3"] = {
            "total_turns": len(df_s3),
            "must_include_accuracy": round(df_s3["must_include_pass"].mean() * 100, 2),
            "must_not_include_accuracy": round(df_s3["must_not_include_pass"].mean() * 100, 2),
            "decision_accuracy": round(df_s3["decision_match"].mean() * 100, 2),
            "tool_call_accuracy": round(df_s3["tool_match"].mean() * 100, 2),
            "all_rules_accuracy": round(df_s3["all_rules_pass"].mean() * 100, 2),
            "avg_latency_ms": round(df_s3["latency_ms"].mean(), 1),
        }

    # Save consolidated summary JSON
    with open(out_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Generate Markdown comparison
    md_lines = [
        "# VoiceOps 2026 - Benchmark Evaluation Summary",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n",
        "| Evaluation Phase | Total Samples | Must-Include Acc | Must-Not-Include Acc | Decision Acc | Overall Pass Rate | Avg Latency |",
        "|---|---|---|---|---|---|---|",
    ]
    if "step1" in summary:
        s1 = summary["step1"]
        md_lines.append(f"| **Step 1: Baseline** | {s1['total_questions']} | {s1['must_include_accuracy']}% | {s1['must_not_include_accuracy']}% | {s1['decision_accuracy']}% | {s1['all_rules_accuracy']}% | {s1['avg_latency_ms']} ms |")
    if "step2_clean" in summary:
        s2 = summary["step2_clean"]
        md_lines.append(f"| **Step 2: Grounded RAG** | {s2['total_questions']} | {s2['must_include_accuracy']}% | {s2['must_not_include_accuracy']}% | {s2['decision_accuracy']}% | {s2['all_rules_accuracy']}% | {s2['avg_latency_ms']} ms |")
    if "step2_asr" in summary:
        s2a = summary["step2_asr"]
        md_lines.append(f"| **Step 2: RAG (ASR Robustness)** | {s2a['total_questions']} | {s2a['must_include_accuracy']}% | {s2a['must_not_include_accuracy']}% | {s2a['decision_accuracy']}% | {s2a['all_rules_accuracy']}% | {s2a['avg_latency_ms']} ms |")
    if "step3" in summary:
        s3 = summary["step3"]
        md_lines.append(f"| **Step 3: Multi-turn Convs** | {s3['total_turns']} turns | {s3['must_include_accuracy']}% | {s3['must_not_include_accuracy']}% | {s3['decision_accuracy']}% | {s3['all_rules_accuracy']}% | {s3['avg_latency_ms']} ms |")

    md_content = "\n".join(md_lines)
    with open(out_dir / "benchmark_comparison.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(md_content)
    print(f"\nAll reports saved under: {out_dir}/")


if __name__ == "__main__":
    main()
