"""Deterministic Caller Identity Verification State Machine (per AGT-IDV)."""
import re
from enum import Enum
from typing import Dict, Any, Optional, Tuple
from voiceops.tools.bridge import get_tools_bridge

# Spoken number normalization mapping
SPOKEN_DIGITS = {
    "zero": "0", "shunya": "0", "sunya": "0",
    "one": "1", "ek": "1",
    "two": "2", "do": "2",
    "three": "3", "teen": "3",
    "four": "4", "char": "4", "chaar": "4",
    "five": "5", "paanch": "5", "panch": "5",
    "six": "6", "chhe": "6", "che": "6",
    "seven": "7", "saat": "7",
    "eight": "8", "aath": "8",
    "nine": "9", "nau": "9",
}

def extract_digits(text: str) -> str:
    """Extract numeric digits from string, including written/spoken number words."""
    if not text:
        return ""
    words = re.findall(r"\b\w+\b", text.lower())
    result = []
    for w in words:
        if w.isdigit():
            result.append(w)
        elif w in SPOKEN_DIGITS:
            result.append(SPOKEN_DIGITS[w])
    joined = "".join(result)
    raw_digits = "".join(ch for ch in text if ch.isdigit())
    return joined if len(joined) >= len(raw_digits) else raw_digits

def extract_verification_proof(text: str) -> Optional[str]:
    """Extract a 4-digit or 6-digit verification value from text."""
    if not text:
        return None
    # 1. First look for standalone 6-digit pincode in numbers
    pins = re.findall(r"\b(\d{6})\b", text)
    if pins:
        return pins[0]
    
    # 2. Exclude "last 4", "4 digits", "char ank", etc. to avoid matching the single 4
    cleaned = re.sub(
        r"\b(last\s*4|4\s*digits?|four\s*digits?|aakhri\s*char|char\s*ank|four\s*ank)\b",
        "", text, flags=re.IGNORECASE
    )
    four_digits = re.findall(r"\b(\d{4})\b", cleaned)
    if four_digits:
        return four_digits[0]

    # 3. Check for spoken digit sequences in cleaned text
    digits = extract_digits(cleaned)
    if len(digits) == 6 or len(digits) == 4:
        return digits
    elif len(digits) > 4:
        return digits[-4:]
        
    return None

class VerificationState(str, Enum):
    IDLE = "idle"
    PENDING_PROOF = "pending_proof"
    VERIFIED = "verified"
    FAILED_LOCKED = "failed_locked"

class VerificationStateMachine:
    """Tracks caller verification lifecycle for an order within a single conversation."""
    
    def __init__(self, tools_bridge=None):
        self.tools = tools_bridge or get_tools_bridge()
        self.order_id: Optional[str] = None
        self.state: VerificationState = VerificationState.IDLE
        self.retry_count: int = 0
        self.verified_order_data: Optional[Dict[str, Any]] = None

    def reset(self):
        """Reset state for a fresh conversation or order."""
        self.order_id = None
        self.state = VerificationState.IDLE
        self.retry_count = 0
        self.verified_order_data = None

    def process_turn(self, user_message: str, extracted_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Process turn input through the identity verification state machine."""
        response = {
            "state": self.state,
            "order_id": self.order_id,
            "verified": (self.state == VerificationState.VERIFIED),
            "tool_call": None,
            "needs_user_input": False,
            "prompt_for_proof": False,
            "verification_failed": False,
            "locked_out": False,
            "order_data": None,
        }

        # 1. New or updated order ID
        if extracted_order_id and extracted_order_id != self.order_id:
            self.order_id = extracted_order_id
            self.state = VerificationState.PENDING_PROOF
            self.retry_count = 0
            self.verified_order_data = None
            
            # Initial unverified lookup (triggers tool call & audit log)
            tool_res = self.tools.lookup_order(self.order_id)
            response["tool_call"] = f"lookup_order(order_id={self.order_id})"
            response["state"] = self.state
            response["order_id"] = self.order_id
            response["prompt_for_proof"] = True
            response["needs_user_input"] = True
            return response

        # 2. If already locked out
        if self.state == VerificationState.FAILED_LOCKED:
            response["locked_out"] = True
            return response

        # 3. If already verified
        if self.state == VerificationState.VERIFIED:
            response["verified"] = True
            response["order_data"] = self.verified_order_data
            return response

        # 4. If waiting for verification proof
        if self.state == VerificationState.PENDING_PROOF and self.order_id:
            proof = extract_verification_proof(user_message)

            if proof:
                tool_res = self.tools.lookup_order(self.order_id, verification_value=proof)
                response["tool_call"] = f"lookup_order(order_id={self.order_id}, verification_value={proof})"
                
                if tool_res.get("verified") is True:
                    self.state = VerificationState.VERIFIED
                    self.verified_order_data = tool_res
                    response["state"] = self.state
                    response["verified"] = True
                    response["order_data"] = tool_res
                    return response
                else:
                    self.retry_count += 1
                    if self.retry_count >= 2:
                        self.state = VerificationState.FAILED_LOCKED
                        response["state"] = self.state
                        response["locked_out"] = True
                        response["verification_failed"] = True
                    else:
                        response["verification_failed"] = True
                        response["prompt_for_proof"] = True
                        response["needs_user_input"] = True
                    return response

        return response
