"""Test evaluation metrics functions."""
from voiceops.evaluation.metrics import (
    evaluate_must_include,
    evaluate_must_not_include,
    evaluate_decision,
    evaluate_tool_calls,
    evaluate_retrieval,
)


def test_must_include_alternatives():
    text = "The refund to your wallet is instant, processed within 2 hours."
    passed, missing = evaluate_must_include(text, "instant / 2 hours / two hours / तुरंत")
    assert passed
    assert len(missing) == 0

    text2 = "It will take 5 to 7 days."
    passed2, missing2 = evaluate_must_include(text2, "instant / 2 hours / two hours / तुरंत")
    assert not passed2
    assert len(missing2) == 1


def test_must_include_multiple_clauses():
    text = "Your warranty is 12 months. You can claim it directly in the app under My Orders."
    passed, missing = evaluate_must_include(text, "12 ; My Orders / app")
    assert passed
    assert len(missing) == 0

    text_no_12 = "You can claim it in the app under My Orders."
    passed_no_12, missing_no_12 = evaluate_must_include(text_no_12, "12 ; My Orders / app")
    assert not passed_no_12
    assert "12" in missing_no_12


def test_must_include_date_normalization():
    text = "Your refund will reach your account by 27 October 2026."
    passed, missing = evaluate_must_include(text, "2026-10-27 / 27 October 2026")
    assert passed


def test_must_not_include():
    text = "Refunds to wallet take 2 hours."
    passed, violations = evaluate_must_not_include(text, "5 to 7 days ; 5-7 days")
    assert passed
    assert len(violations) == 0

    text_violation = "Refunds take 5 to 7 days."
    passed_v, violations_v = evaluate_must_not_include(text_violation, "5 to 7 days ; 5-7 days")
    assert not passed_v
    assert "5 to 7 days" in violations_v


def test_evaluate_decision():
    assert evaluate_decision("answer", "answer ; escalate", "answer")
    assert evaluate_decision("escalate", "answer ; escalate", "answer")
    assert not evaluate_decision("clarify", "answer ; escalate", "answer")


def test_evaluate_tool_calls():
    tools = [{"name": "lookup_order", "arguments": {"order_id": "89782837"}}]
    assert evaluate_tool_calls(tools, "lookup_order(order_id=89782837)")
    assert not evaluate_tool_calls(tools, "create_ticket()")
