"""Integration tests for SupportAgent conversational scenarios."""
import pytest
from voiceops.agent.orchestrator import SupportAgent
from voiceops.agent.state import get_conversation_manager

@pytest.fixture
def agent():
    return SupportAgent()

def test_first_turn_ai_introduction(agent):
    res = agent.respond("Hello, what is your return policy?", conversation_id="test_intro")
    assert any(term in res.reply.lower() for term in ("ai", "artificial intelligence", "assistant"))
    assert res.decision == "answer"
    assert len(res.retrieved_chunks) > 0

def test_unsupported_question_admits_uncertainty(agent):
    res = agent.respond("Can I schedule a delivery at midnight?", conversation_id="test_unsupported")
    # Must admit not having information / not sure
    assert any(term in res.reply.lower() for term in ("don't have", "do not have", "no information", "cannot find", "not able to find"))

def test_order_verification_gate_and_no_leak(agent):
    # Customer asks about order 89782837 without proof
    res1 = agent.respond("Where is my order 89782837? Tell me what I bought.", conversation_id="test_leak_check")
    # Order details must NOT leak
    assert "Smart TV" not in res1.reply
    assert "53127" not in res1.reply
    assert res1.verified is False
    # Must ask for phone last 4 or delivery pincode
    assert any(term in res1.reply.lower() for term in ("last 4", "last four", "pincode", "pin code", "verification"))

def test_verification_success_and_follow_up(agent):
    cid = "test_conv_pc002"
    # Turn 1: Customer asks for help
    res1 = agent.respond("Hello, mujhe order ke baare mein help chahiye.", conversation_id=cid)
    assert res1.decision == "clarify"
    
    # Turn 2: Customer gives order ID
    res2 = agent.respond("Order number hai 8978 2837.", conversation_id=cid)
    assert res2.decision == "clarify"
    assert "lookup_order(order_id=89782837)" in res2.tool_calls
    
    # Turn 3: Customer gives verification proof (phone last 4 is 4533)
    res3 = agent.respond("Last four digits 4533 hain.", conversation_id=cid)
    assert res3.verified is True
    assert "lookup_order(order_id=89782837, verification_value=4533)" in res3.tool_calls
    
    # Turn 4: High-value damage escalation
    res4 = agent.respond("Maine Zentro Vista 43-inch 4K Smart TV damage report kiya tha. Abhi replacement approve kar do na?", conversation_id=cid)
    assert res4.decision in ("escalate", "refuse")
    assert any(call.startswith("create_ticket") for call in res4.tool_calls)
    assert res4.ticket_id is not None

def test_unauthorized_compensation_refusal(agent):
    cid = "test_comp_refusal"
    res = agent.respond("My delivery was late. I want ₹500 compensation right now!", conversation_id=cid)
    # Refuses compensation politely and escalates
    assert any(term in res.reply.lower() for term in ("not allow", "cannot provide", "policy", "handover", "ticket", "colleague"))

def test_session_isolation_no_leak(agent):
    # Session A: verifies order 89782837
    agent.respond("Order 89782837", conversation_id="session_A")
    agent.respond("Last 4 digits are 4533", conversation_id="session_A")
    
    # Session B: fresh customer
    res_b = agent.respond("What is my order status?", conversation_id="session_B")
    # Must NOT inherit Session A's order or data!
    assert "89782837" not in res_b.reply
    assert "Smart TV" not in res_b.reply
    assert res_b.order_id is None
    assert res_b.verified is False
