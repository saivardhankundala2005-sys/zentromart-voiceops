"""Deterministic policy rules, escalation triggers, and refusal logic (per AGT-ESC)."""
import re
from typing import Dict, Any, List, Optional, Tuple

ESCALATION_KEYWORDS = [
    r"\bhuman\b", r"\bagent\b", r"\bperson\b", r"\bcolleague\b", r"\bmanager\b",
    r"\bsenior\b", r"\bsupervisor\b", r"\bexecutive\b", r"\bbaat\s+karao\b",
    r"\bconnect\s+me\b", r"\blegal\b", r"\bcourt\b", r"\bpolice\b", r"\bconsumer\b",
    r"\bcheating\b", r"\bfraud\b", r"\bunauthorized\b", r"\bdispute\b",
    r"\bsparks?\b", r"\bburning\b", r"\bsmoke\b", r"\bbattery\s+swell\b", r"\binjury\b",
]

COMPENSATION_KEYWORDS = [
    r"\bcompensation\b", r"\bhajaana\b", r"\bmuavza\b", r"\bpenalty\b",
    r"\bextra\s+money\b", r"\bcredit\s+₹?500\b", r"\b₹?500\s+compensation\b",
]

def check_policy_triggers(user_message: str, order_facts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Check whether conversation or order facts trigger mandatory escalation or refusal."""
    result = {
        "should_escalate": False,
        "is_compensation_demand": False,
        "escalation_reasons": [],
        "intent_override": None,
    }

    msg_lower = user_message.lower()

    # 1. Check message for compensation demands
    for pat in COMPENSATION_KEYWORDS:
        if re.search(pat, msg_lower):
            result["is_compensation_demand"] = True
            result["should_escalate"] = True
            result["escalation_reasons"].append(
                "Customer requested unauthorized monetary compensation (policy does not permit compensation; requires handover)."
            )
            break

    # 2. Check message for explicit escalation keywords (legal, safety, human request)
    for pat in ESCALATION_KEYWORDS:
        if re.search(pat, msg_lower):
            result["should_escalate"] = True
            result["escalation_reasons"].append(f"Customer triggered escalation phrase: {pat}")
            break

    # 3. Check order facts (derived facts from verified order)
    if order_facts:
        if order_facts.get("escalation_reasons"):
            result["should_escalate"] = True
            result["escalation_reasons"].extend(order_facts["escalation_reasons"])

    return result
