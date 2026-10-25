"""Deterministic business facts derived from verified order data.

Computes return windows, refund SLAs, fees, and escalation triggers in pure Python
so the LLM never performs date arithmetic or guesses policy eligibility.
"""
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional
from voiceops.config import REFERENCE_DATE
from voiceops.data.business_calendar import (
    parse_date, add_business_days, business_days_between,
    add_calendar_days, calendar_days_between, is_business_day
)

def compute_order_facts(order: Dict[str, Any], holidays=None) -> Dict[str, Any]:
    """Compute all derived facts for a verified order as of REFERENCE_DATE."""
    facts = {
        "order_id": order.get("order_id"),
        "customer_first_name": order.get("customer_first_name", "Customer"),
        "is_plus": order.get("is_plus", False),
        "status": order.get("status"),
        "placed_at": order.get("placed_at"),
        "delivered_on": order.get("delivered_on"),
        "payment_method": order.get("payment_method"),
        "delivery_type": order.get("delivery_type"),
        "total_inr": order.get("total_inr"),
        "contacts_last_7_days": int(order.get("contacts_last_7_days") or 0),
        "escalation_reasons": [],
        "items_summary": [],
        "item_facts": {},
    }
    
    placed_d = parse_date(order.get("placed_at"))
    delivered_d = parse_date(order.get("delivered_on"))
    
    # 1. Evaluate Items & Return Eligibility
    items = order.get("items", [])
    for it in items:
        sku = it.get("sku", "")
        title = it.get("title", "Item")
        cat = it.get("return_category", "")
        qty = int(it.get("qty") or 1)
        price = int(it.get("unit_price_inr") or 0)
        
        item_info = {
            "sku": sku,
            "title": title,
            "category": cat,
            "qty": qty,
            "unit_price": price,
            "total_price": qty * price,
            "returnable": False,
            "return_window_days": 0,
            "return_deadline": None,
            "return_open": False,
            "pickup_fee_inr": 0,
            "rule_cited": None,
        }
        
        # Determine Return Window (per pol_returns_v4.md)
        if cat in ("grocery", "innerwear", "digital"):
            item_info["returnable"] = False
            item_info["rule_cited"] = "RET-5: Non-returnable category"
        elif delivered_d:
            item_info["returnable"] = True
            if cat == "electronics":
                # RET-3: Transition rule
                if delivered_d < date(2026, 8, 1):
                    window = 10
                    rule = "RET-3: 10 calendar days (delivered before 1 Aug 2026)"
                else:
                    window = 7
                    rule = "RET-2/RET-3: 7 calendar days (electronics)"
            elif cat == "fashion":
                # RET-8: Festive sale rule (placed 01 Oct to 15 Oct 2026)
                if placed_d and date(2026, 10, 1) <= placed_d <= date(2026, 10, 15):
                    window = 30
                    rule = "RET-8: 30 calendar days (Festive sale order placed 01-15 Oct 2026)"
                else:
                    window = 15
                    rule = "RET-2: 15 calendar days (fashion)"
            elif cat in ("home_kitchen", "books_media"):
                window = 10
                rule = f"RET-2: 10 calendar days ({cat})"
            elif cat == "beauty_sealed":
                window = 7
                rule = "RET-2: 7 calendar days (beauty, unopened with seal intact)"
            else:
                window = 7
                rule = "RET-2: 7 calendar days"
                
            deadline = add_calendar_days(delivered_d, window)
            item_info["return_window_days"] = window
            item_info["return_deadline"] = deadline.isoformat()
            item_info["return_deadline_str"] = deadline.strftime("%d %B %Y")
            item_info["return_open"] = (REFERENCE_DATE <= deadline)
            item_info["rule_cited"] = rule
            
            # Pickup fee (RET-6)
            if cat == "fashion":
                if order.get("is_plus"):
                    item_info["pickup_fee_inr"] = 0
                    item_info["fee_note"] = "Waived (ZentroMart Plus member)"
                else:
                    item_info["pickup_fee_inr"] = 49
                    item_info["fee_note"] = "₹49 reverse pickup fee applies"
            else:
                item_info["pickup_fee_inr"] = 0
                item_info["fee_note"] = "No reverse pickup fee"
                
        facts["items_summary"].append(f"{title} ({sku}, {cat}) - Price: ₹{price}")
        facts["item_facts"][sku] = item_info

    # 2. Check Refund SLAs (REF-2)
    # Check if a return refund or cancellation refund is initiated
    refund_initiated_on = (
        order.get("return_refund_initiated_on") or 
        order.get("cancellation_refund_initiated_on") or 
        order.get("rto_refund_initiated_on")
    )
    refund_method = (
        order.get("return_refund_method") or 
        order.get("cancellation_refund_method") or 
        order.get("rto_refund_method") or 
        order.get("payment_method")
    )
    refund_rrn = (
        order.get("return_rrn") or 
        order.get("cancellation_rrn") or 
        order.get("rto_rrn")
    )
    
    if refund_initiated_on:
        ref_d = parse_date(refund_initiated_on)
        facts["refund_initiated_on"] = refund_initiated_on
        facts["refund_method"] = refund_method
        facts["refund_rrn"] = refund_rrn
        
        # SLA calculation
        method_lower = (refund_method or "").lower()
        if "wallet" in method_lower:
            facts["refund_sla_desc"] = "Instant, within 2 hours"
            facts["refund_due_date"] = ref_d.isoformat() if ref_d else None
            facts["refund_due_date_str"] = ref_d.strftime("%d %B %Y") if ref_d else None
            facts["is_overdue"] = (REFERENCE_DATE > ref_d) if ref_d else False
            facts["overdue_business_days"] = business_days_between(ref_d, REFERENCE_DATE, holidays) if (ref_d and REFERENCE_DATE > ref_d) else 0
        elif "upi" in method_lower:
            facts["refund_sla_desc"] = "1 to 3 business days"
            min_d = add_business_days(ref_d, 1, holidays) if ref_d else None
            max_d = add_business_days(ref_d, 3, holidays) if ref_d else None
            facts["refund_min_date"] = min_d.isoformat() if min_d else None
            facts["refund_due_date"] = max_d.isoformat() if max_d else None
            facts["refund_due_date_str"] = max_d.strftime("%d %B %Y") if max_d else None
            facts["is_overdue"] = (REFERENCE_DATE > max_d) if max_d else False
            facts["overdue_business_days"] = business_days_between(max_d, REFERENCE_DATE, holidays) if (max_d and REFERENCE_DATE > max_d) else 0
        elif "netbanking" in method_lower:
            facts["refund_sla_desc"] = "3 to 5 business days"
            min_d = add_business_days(ref_d, 3, holidays) if ref_d else None
            max_d = add_business_days(ref_d, 5, holidays) if ref_d else None
            facts["refund_min_date"] = min_d.isoformat() if min_d else None
            facts["refund_due_date"] = max_d.isoformat() if max_d else None
            facts["refund_due_date_str"] = max_d.strftime("%d %B %Y") if max_d else None
            facts["is_overdue"] = (REFERENCE_DATE > max_d) if max_d else False
            facts["overdue_business_days"] = business_days_between(max_d, REFERENCE_DATE, holidays) if (max_d and REFERENCE_DATE > max_d) else 0
        else:
            # Card or COD Bank: 5 to 7 business days
            facts["refund_sla_desc"] = "5 to 7 business days"
            min_d = add_business_days(ref_d, 5, holidays) if ref_d else None
            max_d = add_business_days(ref_d, 7, holidays) if ref_d else None
            facts["refund_min_date"] = min_d.isoformat() if min_d else None
            facts["refund_min_date_str"] = min_d.strftime("%d %B %Y") if min_d else None
            facts["refund_due_date"] = max_d.isoformat() if max_d else None
            facts["refund_due_date_str"] = max_d.strftime("%d %B %Y") if max_d else None
            facts["is_overdue"] = (REFERENCE_DATE > max_d) if max_d else False
            facts["overdue_business_days"] = business_days_between(max_d, REFERENCE_DATE, holidays) if (max_d and REFERENCE_DATE > max_d) else 0

        # ESC-1 Rule 5: Refund more than 3 business days past maximum timeline
        if facts.get("overdue_business_days", 0) > 3:
            facts["escalation_reasons"].append(
                f"Refund overdue by {facts['overdue_business_days']} business days (>3 days past SLA)"
            )

    # 3. Check Repeat Contacts (ESC-1 Rule 4)
    if facts["contacts_last_7_days"] >= 2:
        facts["escalation_reasons"].append(
            f"Third or later contact ({facts['contacts_last_7_days']} prior contacts in last 7 days)"
        )
        
    # 4. Check High Value Damage (ESC-1 Rule 1)
    if order.get("damage_sku"):
        damage_sku = order.get("damage_sku")
        dmg_item = facts["item_facts"].get(damage_sku)
        dmg_val = dmg_item["total_price"] if dmg_item else 0
        if dmg_val > 10000:
            facts["escalation_reasons"].append(
                f"High-value damage report: SKU {damage_sku} valued at ₹{dmg_val:,} (>₹10,000 threshold)"
            )

    return facts
