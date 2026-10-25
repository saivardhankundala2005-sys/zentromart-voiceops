"""Audit-logging and safe redaction wrapper around the dataset mock tools."""
import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from voiceops.config import TOOLS_DIR

# Dynamically import the real dataset tools module
_tools_py = TOOLS_DIR / "tools.py"
if _tools_py.exists():
    spec = importlib.util.spec_from_file_location("dataset_tools", str(_tools_py))
    _dataset_tools = importlib.util.module_from_spec(spec)
    sys.modules["dataset_tools"] = _dataset_tools
    spec.loader.exec_module(_dataset_tools)
    _raw_lookup_order = _dataset_tools.lookup_order
    _raw_create_ticket = _dataset_tools.create_ticket
else:
    # Fallback simulation if tools.py is ever missing
    def _raw_lookup_order(order_id, verification_value=None):
        return {"verification_required": True, "order_id": order_id}
    def _raw_create_ticket(summary, order_id=None, reason=None, intent=None):
        return {"created": True, "ticket_id": "TKT_SIMULATED"}

class ToolsBridge:
    """Safe, auditable wrapper around dataset tools."""
    
    def __init__(self):
        self.audit_log: List[Dict[str, Any]] = []
        self._created_tickets: Dict[str, str] = {} # Deduplicate tickets per (order_id, reason)

    def log_call(self, name: str, args_redacted: Dict[str, Any], outcome: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "tool": name,
            "args": args_redacted,
            "outcome": {k: v for k, v in outcome.items() if k not in ("phone", "email")},
        }
        self.audit_log.append(entry)

    def lookup_order(self, order_id: str, verification_value: Optional[str] = None) -> Dict[str, Any]:
        """Look up order details with redaction of proof in audit logs."""
        clean_oid = "".join(ch for ch in str(order_id) if ch.isdigit())
        clean_proof = "".join(ch for ch in str(verification_value) if ch.isdigit()) if verification_value else None
        
        # Redacted audit representation
        redacted_args = {
            "order_id": clean_oid,
            "has_verification_value": clean_proof is not None,
            "verification_type": f"{len(clean_proof)}-digit proof" if clean_proof else None,
        }
        
        result = _raw_lookup_order(clean_oid, clean_proof)
        self.log_call("lookup_order", redacted_args, result)
        return result

    def create_ticket(
        self,
        summary: str,
        order_id: Optional[str] = None,
        reason: Optional[str] = None,
        intent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create support handover ticket with minimum length guarantee and deduplication."""
        clean_oid = "".join(ch for ch in str(order_id) if ch.isdigit()) if order_id else None
        
        # Deduplication check: if a ticket was already created for this order + reason in this session
        dedup_key = f"{clean_oid or 'no_order'}_{reason or 'no_reason'}"
        if dedup_key in self._created_tickets:
            existing_tid = self._created_tickets[dedup_key]
            return {
                "created": True,
                "ticket_id": existing_tid,
                "message": "Ticket already created for this issue.",
                "deduplicated": True,
            }

        # Ensure summary meets min 20 character requirement
        clean_summary = (summary or "").strip()
        if len(clean_summary) < 20:
            # Auto-enrich summary with context
            enrichments = []
            if order_id:
                enrichments.append(f"Order: {clean_oid}")
            if reason:
                enrichments.append(f"Reason: {reason}")
            if intent:
                enrichments.append(f"Intent: {intent}")
            if not clean_summary:
                clean_summary = "Customer requested human escalation and handover."
            clean_summary = f"{clean_summary} [{', '.join(enrichments)}]"
            if len(clean_summary) < 20:
                clean_summary = clean_summary + " Handover to support team."

        redacted_args = {
            "order_id": clean_oid,
            "reason": reason,
            "intent": intent,
            "summary_length": len(clean_summary),
        }
        
        result = _raw_create_ticket(clean_summary, clean_oid, reason, intent)
        if result.get("created") and result.get("ticket_id"):
            self._created_tickets[dedup_key] = result["ticket_id"]
            
        self.log_call("create_ticket", redacted_args, result)
        return result

# Global singleton
_TOOLS_BRIDGE_INSTANCE: Optional[ToolsBridge] = None

def get_tools_bridge() -> ToolsBridge:
    global _TOOLS_BRIDGE_INSTANCE
    if _TOOLS_BRIDGE_INSTANCE is None:
        _TOOLS_BRIDGE_INSTANCE = ToolsBridge()
    return _TOOLS_BRIDGE_INSTANCE
