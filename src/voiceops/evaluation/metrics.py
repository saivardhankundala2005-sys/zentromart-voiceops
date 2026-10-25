"""Evaluation metrics for VoiceOps 2026 benchmarks.

Evaluates:
- must_include: all semicolon-separated clauses must have at least one slash-separated alternative present.
- must_not_include: none of the semicolon-separated forbidden strings must appear.
- decision: match against acceptable_decisions or expected_decision.
- tool calls: match against expected_tool_calls.
- retrieval: recall@k and hard-negative hit rate.
"""

import re
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd


def normalize_text_for_eval(text: str) -> str:
    """Normalize text for robust substring matching."""
    if not text:
        return ""
    # Normalize unicode whitespace, dashes, currency symbols
    t = text.lower()
    t = t.replace("₹", "rs ").replace("rs.", "rs ").replace("rupees", "rs ")
    t = re.sub(r"\s+", " ", t)
    return t


def evaluate_must_include(text: str, pattern: Optional[str]) -> Tuple[bool, List[str]]:
    """Check that all semicolon-separated clauses have at least one slash-separated alternative present.
    
    Returns:
        (passed, list_of_missing_clauses)
    """
    if pattern is None or pd.isna(pattern) or not str(pattern).strip():
        return True, []

    norm_text = normalize_text_for_eval(text)
    clauses = [c.strip() for c in str(pattern).split(";") if c.strip()]
    missing = []

    for clause in clauses:
        options = [opt.strip() for opt in clause.split("/") if opt.strip()]
        matched = False
        for opt in options:
            norm_opt = normalize_text_for_eval(opt)
            if not norm_opt:
                continue

            # Exact substring match
            if norm_opt in norm_text:
                matched = True
                break

            # Date normalization: e.g. "2026-10-27" -> "27 october" or "october 27"
            date_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", norm_opt)
            if date_match:
                y, m, d = date_match.groups()
                day_num = str(int(d))
                months = {
                    "01": "january", "02": "february", "03": "march", "04": "april",
                    "05": "may", "06": "june", "07": "july", "08": "august",
                    "09": "september", "10": "october", "11": "november", "12": "december"
                }
                month_name = months.get(m, "")
                if (f"{day_num} {month_name}" in norm_text or
                    f"{month_name} {day_num}" in norm_text or
                    f"{day_num}th {month_name}" in norm_text):
                    matched = True
                    break

            # Word boundary regex for short numbers/words
            if len(norm_opt) <= 3 and norm_opt.isdigit():
                if re.search(r"\b" + re.escape(norm_opt) + r"\b", norm_text):
                    matched = True
                    break

        if not matched:
            missing.append(clause)

    return (len(missing) == 0), missing


def evaluate_must_not_include(text: str, pattern: Optional[str]) -> Tuple[bool, List[str]]:
    """Check that none of the semicolon-separated forbidden phrases appear in the text.
    
    Returns:
        (passed, list_of_forbidden_violations_found)
    """
    if pattern is None or pd.isna(pattern) or not str(pattern).strip():
        return True, []

    norm_text = normalize_text_for_eval(text)
    forbidden_terms = [t.strip() for t in str(pattern).split(";") if t.strip()]
    violations = []

    for term in forbidden_terms:
        norm_term = normalize_text_for_eval(term)
        if norm_term and norm_term in norm_text:
            violations.append(term)

    return (len(violations) == 0), violations


def evaluate_decision(actual_decision: str, acceptable_decisions: Optional[str], expected_decision: Optional[str]) -> bool:
    """Verify if actual decision matches acceptable or expected decisions."""
    actual = (actual_decision or "").strip().lower()
    
    acceptable_list = []
    if acceptable_decisions and not pd.isna(acceptable_decisions):
        acceptable_list.extend([d.strip().lower() for d in str(acceptable_decisions).split(";") if d.strip()])
    if expected_decision and not pd.isna(expected_decision):
        acceptable_list.append(str(expected_decision).strip().lower())

    if not acceptable_list:
        return True

    return actual in acceptable_list


def evaluate_tool_calls(actual_tools: List[Dict[str, Any]], expected_str: Optional[str]) -> bool:
    """Verify if expected tool calls were invoked."""
    if expected_str is None or pd.isna(expected_str) or not str(expected_str).strip():
        # No specific tool call mandated
        return True

    exp = str(expected_str).strip().lower()
    actual_names = [t.get("name", "").lower() for t in (actual_tools or [])]

    if "create_ticket" in exp:
        return "create_ticket" in actual_names
    if "lookup_order" in exp:
        return "lookup_order" in actual_names

    return True


def evaluate_retrieval(retrieved_ids: List[str], gold_str: Optional[str], hard_neg_str: Optional[str]) -> Dict[str, Any]:
    """Calculate Recall and Hard-Negative Hit rate for retrieval."""
    gold_set = set()
    if gold_str and not pd.isna(gold_str):
        gold_set = {g.strip() for g in str(gold_str).split(";") if g.strip()}

    hard_neg_set = set()
    if hard_neg_str and not pd.isna(hard_neg_str):
        hard_neg_set = {h.strip() for h in str(hard_neg_str).split(";") if h.strip()}

    ret_set = set(retrieved_ids or [])
    
    hits = len(gold_set.intersection(ret_set))
    total_gold = len(gold_set)
    recall = (hits / total_gold) if total_gold > 0 else 1.0

    hn_hits = len(hard_neg_set.intersection(ret_set))
    has_hard_negative = hn_hits > 0

    return {
        "gold_hits": hits,
        "total_gold": total_gold,
        "recall": recall,
        "hard_neg_hits": hn_hits,
        "has_hard_negative": has_hard_negative,
    }
