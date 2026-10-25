"""Unit tests for data loader, business calendar, and derived order facts."""
import pytest
from datetime import date
from voiceops.data.business_calendar import (
    is_business_day, add_business_days, business_days_between,
    calendar_days_between, days_since, add_calendar_days
)
from voiceops.data.loader import get_data_loader
from voiceops.data.order_facts import compute_order_facts

def test_business_calendar_rules():
    # 2026-10-18 is Sunday -> False
    assert is_business_day(date(2026, 10, 18)) is False
    # 2026-10-19 is Warehouse closure festival holiday -> False
    assert is_business_day(date(2026, 10, 19)) is False
    # 2026-10-20 is Tuesday and not a holiday -> True
    assert is_business_day(date(2026, 10, 20)) is True
    # 2026-10-24 is Saturday -> True (Mon-Sat are business days)
    assert is_business_day(date(2026, 10, 24)) is True

def test_add_business_days_skipping_sunday_and_holiday():
    # 2026-10-17 is Saturday.
    # 1 business day later: Sunday 18 skipped, Monday 19 (holiday) skipped -> Tuesday 20!
    next_bday = add_business_days(date(2026, 10, 17), 1)
    assert next_bday == date(2026, 10, 20)
    
    # 3 business days from Saturday 2026-10-17:
    # Day 1: Tuesday 2026-10-20
    # Day 2: Wednesday 2026-10-21
    # Day 3: Thursday 2026-10-22
    three_bday = add_business_days(date(2026, 10, 17), 3)
    assert three_bday == date(2026, 10, 22)

def test_business_days_between():
    # Between 2026-10-17 (Sat) and 2026-10-20 (Tue):
    # Sun 18 (non-bus), Mon 19 (holiday), Tue 20 (bus) -> exactly 1 business day
    count = business_days_between(date(2026, 10, 17), date(2026, 10, 20))
    assert count == 1

def test_data_loader_integrity():
    loader = get_data_loader()
    assert len(loader.orders) == 1000
    assert len(loader.customers) == 300
    assert len(loader.products) == 70
    assert len(loader.pincodes) == 36
    assert len(loader.holidays) == 4
    
    # Check string preservation
    sample_order_id = list(loader.orders.keys())[0]
    assert isinstance(sample_order_id, str)
    assert len(sample_order_id) == 8
    
    sample_order = loader.get_order(sample_order_id)
    assert sample_order is not None
    assert isinstance(sample_order["delivery_pincode"], str)
    
    # Check enriched order joins
    full_order = loader.get_order_full(sample_order_id)
    assert full_order is not None
    assert "items" in full_order
    assert "customer_name" in full_order
    assert "phone_masked" in full_order
    assert len(full_order["items"]) > 0

def test_order_facts_festive_sale_and_fees():
    # Synthetic order during festive sale (placed Oct 6, delivered Oct 14)
    # Fashion items have 30 days return window
    mock_order = {
        "order_id": "11112222",
        "customer_first_name": "Rohan",
        "is_plus": False,
        "placed_at": "2026-10-06T10:00",
        "delivered_on": "2026-10-14",
        "status": "delivered",
        "payment_method": "upi",
        "contacts_last_7_days": "0",
        "items": [
            {
                "sku": "FAS-0001",
                "title": "UrbanThread Slim Fit Jeans",
                "return_category": "fashion",
                "qty": "1",
                "unit_price_inr": "1299"
            }
        ]
    }
    facts = compute_order_facts(mock_order)
    item = facts["item_facts"]["FAS-0001"]
    assert item["return_window_days"] == 30
    assert item["return_open"] is True
    assert item["pickup_fee_inr"] == 49
    # Deadline: 2026-10-14 + 30 days = 2026-11-13
    assert item["return_deadline"] == "2026-11-13"

def test_order_facts_plus_member_pickup_waiver():
    mock_order = {
        "order_id": "33334444",
        "customer_first_name": "Pooja",
        "is_plus": True,
        "placed_at": "2026-09-30T10:00",
        "delivered_on": "2026-10-12",
        "status": "delivered",
        "payment_method": "wallet",
        "contacts_last_7_days": "0",
        "items": [
            {
                "sku": "FAS-0002",
                "title": "Kaafi Running Sneakers",
                "return_category": "fashion",
                "qty": "1",
                "unit_price_inr": "2499"
            }
        ]
    }
    facts = compute_order_facts(mock_order)
    item = facts["item_facts"]["FAS-0002"]
    # Placed before Oct 1 -> standard 15 days window (Oct 12 + 15 = Oct 27)
    assert item["return_window_days"] == 15
    assert item["return_deadline"] == "2026-10-27"
    assert item["return_open"] is True
    # Plus member -> pickup fee waived!
    assert item["pickup_fee_inr"] == 0
