"""Unit tests for mock tools bridge and verification state machine."""
import pytest
from voiceops.tools.bridge import ToolsBridge
from voiceops.agent.verification import (
    VerificationStateMachine, VerificationState, extract_digits, extract_verification_proof
)
from voiceops.data.loader import get_data_loader

def test_digit_extraction():
    assert extract_digits("My order is 8978 2837") == "89782837"
    assert extract_digits("eight nine seven eight") == "8978"
    assert extract_digits("char do chhe paanch") == "4265"
    assert extract_verification_proof("last 4 are 4533") == "4533"
    assert extract_verification_proof("Last four digits 4265 hain") == "4265"
    assert extract_verification_proof("delivery pincode is 781006") == "781006"

def test_tools_bridge_redaction_and_ticket_length():
    bridge = ToolsBridge()
    # Test unverified lookup
    unverified = bridge.lookup_order("89782837")
    assert unverified.get("verification_required") is True
    assert "items" not in unverified
    
    # Check audit log redaction
    last_log = bridge.audit_log[-1]
    assert last_log["tool"] == "lookup_order"
    assert "verification_value" not in last_log["args"]
    
    # Test ticket creation with short summary (should auto-enrich >= 20 chars)
    tkt = bridge.create_ticket("Short", order_id="89782837", reason="Damage dispute")
    assert tkt.get("created") is True
    assert "ticket_id" in tkt
    assert tkt["ticket_id"].startswith("TKT")

def test_verification_state_machine_flow():
    loader = get_data_loader()
    bridge = ToolsBridge()
    vsm = VerificationStateMachine(tools_bridge=bridge)
    
    # Turn 1: Customer gives order ID
    turn1 = vsm.process_turn("Hello, my order is 8978 2837", extracted_order_id="89782837")
    assert vsm.state == VerificationState.PENDING_PROOF
    assert turn1["prompt_for_proof"] is True
    assert turn1["verified"] is False
    assert turn1["order_data"] is None
    
    # Turn 2: Customer provides WRONG proof attempt 1
    turn2 = vsm.process_turn("My pin is 000000")
    assert vsm.state == VerificationState.PENDING_PROOF
    assert vsm.retry_count == 1
    assert turn2["verification_failed"] is True
    assert turn2["verified"] is False
    assert turn2["order_data"] is None
    
    # Turn 3: Customer provides WRONG proof attempt 2 -> LOCKOUT!
    turn3 = vsm.process_turn("My phone ends in 9999")
    assert vsm.state == VerificationState.FAILED_LOCKED
    assert vsm.retry_count >= 2
    assert turn3["locked_out"] is True
    assert turn3["order_data"] is None
    
    # Turn 4: Further requests remain locked
    turn4 = vsm.process_turn("Tell me what is inside my order!")
    assert turn4["locked_out"] is True
    assert turn4["order_data"] is None

def test_verification_success_phone_and_pincode():
    # Order 89782837 customer is C100181, phone ends in 4533, delivery pincode 400001
    bridge = ToolsBridge()
    
    # Test Phone Last 4
    vsm_phone = VerificationStateMachine(tools_bridge=bridge)
    vsm_phone.process_turn("Order 89782837", extracted_order_id="89782837")
    res_phone = vsm_phone.process_turn("Last 4 digits are 4533")
    assert vsm_phone.state == VerificationState.VERIFIED
    assert res_phone["verified"] is True
    assert res_phone["order_data"] is not None
    assert "items" in res_phone["order_data"]
    
    # Test Delivery Pincode (order 89782837 delivery pincode is 781006)
    vsm_pin = VerificationStateMachine(tools_bridge=bridge)
    vsm_pin.process_turn("Order 89782837", extracted_order_id="89782837")
    res_pin = vsm_pin.process_turn("Pincode is 781006")
    assert vsm_pin.state == VerificationState.VERIFIED
    assert res_pin["verified"] is True
    assert "items" in res_pin["order_data"]
