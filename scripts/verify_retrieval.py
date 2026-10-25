#!/usr/bin/env python3
"""Phase B Verification: Measure Retrieval Recall@k and Hard-Negative Hit Rate."""
import sys
import csv
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from voiceops.config import PRACTICE_DIR
from voiceops.knowledge.retriever import get_retriever

def main():
    questions_file = PRACTICE_DIR / "questions.csv"
    if not questions_file.exists():
        print(f"Error: {questions_file} not found.")
        sys.exit(1)

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    print(f"Evaluating Retrieval on {len(questions)} Practice Questions...")
    retriever = get_retriever()
    
    total_evaluable = 0
    total_gold_items = 0
    gold_found_top6 = 0
    hard_neg_hits = 0
    hard_neg_total = 0
    
    per_question_results = []

    for q in questions:
        qid = q["id"]
        q_text = q["question"]
        gold_ev = [s.strip() for s in q["gold_evidence"].split(" ; ") if s.strip()]
        hard_negs = [s.strip() for s in q.get("hard_negatives", "").split(" ; ") if s.strip()]
        
        # If no gold evidence (e.g. unanswerable questions), skip recall calculation
        if not gold_ev:
            continue
            
        total_evaluable += 1
        total_gold_items += len(gold_ev)
        hard_neg_total += len(hard_negs)
        
        # Retrieve top 6 chunks
        retrieved = retriever.retrieve(q_text, top_k=6)
        retrieved_sec_ids = [r["section_id"] for r in retrieved]
        
        # Check gold evidence recall
        matched_gold = [g for g in gold_ev if g in retrieved_sec_ids]
        gold_found_top6 += len(matched_gold)
        
        # Check hard negative hits
        hn_hit = [hn for hn in hard_negs if hn in retrieved_sec_ids]
        hard_neg_hits += len(hn_hit)
        
        recall = len(matched_gold) / len(gold_ev) if gold_ev else 1.0
        per_question_results.append({
            "id": qid,
            "gold": gold_ev,
            "retrieved": retrieved_sec_ids,
            "matched": matched_gold,
            "recall": recall,
            "hn_hit": hn_hit,
        })

    overall_recall = (gold_found_top6 / total_gold_items * 100) if total_gold_items else 0
    hn_rate = (hard_neg_hits / hard_neg_total * 100) if hard_neg_total else 0

    print("\n" + "="*60)
    print("PHASE B RETRIEVAL EVALUATION REPORT")
    print("="*60)
    print(f"Total Questions Evaluated: {total_evaluable}")
    print(f"Total Gold Evidence Items: {total_gold_items}")
    print(f"Gold Evidence Items Found: {gold_found_top6}")
    print(f"Recall@6:                  {overall_recall:.2f}%")
    print(f"Hard Negatives Total:      {hard_neg_total}")
    print(f"Hard Negatives Hits:       {hard_neg_hits} ({hn_rate:.2f}%)")
    print("="*60)
    
    # Show sample queries
    print("\nSample Retrieval Cases:")
    for res in per_question_results[:5]:
        print(f"[{res['id']}] Expected: {res['gold']} | Retrieved: {res['retrieved']} | Recall: {res['recall']*100:.0f}%")

if __name__ == "__main__":
    main()
