"""Business calendar and timeline calculations for ZentroMart.

Rules (per KB-00.3):
- Monday to Saturday are business days.
- Sundays are warehouse closures (non-business days).
- Warehouse holidays (from holidays.csv) are non-business days:
    - 2026-08-15: Independence Day
    - 2026-10-02: Gandhi Jayanti
    - 2026-10-19: Warehouse closure (festival)
    - 2026-11-09: Warehouse closure (festival)
- Return windows are counted in calendar days.
- Refund and logistics SLAs are counted in business days.
"""
from datetime import date, datetime, timedelta
from typing import Set, Union, Optional
from pathlib import Path
import csv

# Default holidays fallback in case file is absent
_DEFAULT_HOLIDAYS = {
    date(2026, 8, 15),
    date(2026, 10, 2),
    date(2026, 10, 19),
    date(2026, 11, 9),
}

def parse_date(val: Union[str, date, datetime, None]) -> Optional[date]:
    """Parse string or datetime to date object safely."""
    if val is None:
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    val_str = str(val).strip()
    if not val_str:
        return None
    # Handle YYYY-MM-DD or YYYY-MM-DDTHH:MM
    if "T" in val_str:
        val_str = val_str.split("T")[0]
    elif " " in val_str:
        val_str = val_str.split(" ")[0]
    try:
        return datetime.strptime(val_str, "%Y-%m-%d").date()
    except ValueError:
        return None

def load_holidays(csv_path: Optional[Union[str, Path]] = None) -> Set[date]:
    """Load holiday dates from holidays.csv or return defaults."""
    if not csv_path or not Path(csv_path).exists():
        return set(_DEFAULT_HOLIDAYS)
    holidays = set()
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = parse_date(row.get("date"))
                if d:
                    holidays.add(d)
        return holidays if holidays else set(_DEFAULT_HOLIDAYS)
    except Exception:
        return set(_DEFAULT_HOLIDAYS)

def is_business_day(d: date, holidays: Optional[Set[date]] = None) -> bool:
    """Check if date d is a business day (Mon-Sat, non-Sunday, non-holiday)."""
    if holidays is None:
        holidays = _DEFAULT_HOLIDAYS
    # weekday(): Mon=0, Tue=1, ..., Sat=5, Sun=6
    if d.weekday() == 6:
        return False
    if d in holidays:
        return False
    return True

def add_business_days(start_date: Union[date, str], n: int, holidays: Optional[Set[date]] = None) -> date:
    """Add n business days to start_date.
    
    If n > 0, steps forward counting only business days.
    """
    cur = parse_date(start_date)
    if not cur:
        raise ValueError(f"Invalid start date: {start_date}")
    if holidays is None:
        holidays = _DEFAULT_HOLIDAYS
        
    days_added = 0
    while days_added < n:
        cur += timedelta(days=1)
        if is_business_day(cur, holidays):
            days_added += 1
    return cur

def business_days_between(start_date: Union[date, str], end_date: Union[date, str], holidays: Optional[Set[date]] = None) -> int:
    """Count number of business days between start_date and end_date.
    
    Excludes start_date, includes end_date if end_date > start_date.
    Returns 0 if end_date <= start_date.
    """
    d1 = parse_date(start_date)
    d2 = parse_date(end_date)
    if not d1 or not d2 or d2 <= d1:
        return 0
    if holidays is None:
        holidays = _DEFAULT_HOLIDAYS
        
    count = 0
    cur = d1 + timedelta(days=1)
    while cur <= d2:
        if is_business_day(cur, holidays):
            count += 1
        cur += timedelta(days=1)
    return count

def calendar_days_between(start_date: Union[date, str], end_date: Union[date, str]) -> int:
    """Count calendar days between start_date and end_date."""
    d1 = parse_date(start_date)
    d2 = parse_date(end_date)
    if not d1 or not d2:
        return 0
    return (d2 - d1).days

def days_since(target_date: Union[date, str], reference_date: Union[date, str]) -> int:
    """Days elapsed from target_date to reference_date (positive if target_date is in past)."""
    return calendar_days_between(target_date, reference_date)

def add_calendar_days(start_date: Union[date, str], n: int) -> date:
    """Add n calendar days to start_date."""
    d = parse_date(start_date)
    if not d:
        raise ValueError(f"Invalid date: {start_date}")
    return d + timedelta(days=n)
