"""Speech normalization for voice: writes text for the ear (per STY-3 and Gnani guide)."""
import re
from datetime import datetime

MONTH_MAP = {
    "01": "January", "02": "February", "03": "March", "04": "April",
    "05": "May", "06": "June", "07": "July", "08": "August",
    "09": "September", "10": "October", "11": "November", "12": "December"
}

def normalize_date_for_speech(match: re.Match) -> str:
    """Convert YYYY-MM-DD to 'D Month YYYY'."""
    year, month, day = match.group(1), match.group(2), match.group(3)
    month_name = MONTH_MAP.get(month, month)
    day_clean = str(int(day))
    return f"{day_clean} {month_name} {year}"

def chunk_order_id(match: re.Match) -> str:
    """Chunk 8-digit order ID into two 4-digit groups."""
    digits = match.group(0)
    return f"{digits[:4]} {digits[4:]}"

def normalize_for_speech(text: str) -> str:
    """Transform written response into speech-optimized text for Gnani TTS."""
    if not text:
        return ""

    s = text

    # 1. Remove markdown links [text](url) -> text
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)

    # 2. Remove markdown symbols: headers, bold, italics, code blocks
    s = re.sub(r"```.*?```", "", s, flags=re.DOTALL)
    s = re.sub(r"`[^`]+`", "", s)
    s = re.sub(r"^#{1,6}\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"^[-*•]\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\d+\.\s*", "", s, flags=re.MULTILINE)

    # 3. Date normalization: YYYY-MM-DD -> 'DD Month YYYY'
    s = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", normalize_date_for_speech, s)

    # 4. Currency and numbers: ₹ or Rs. 1,299 -> 1299 rupees
    s = re.sub(r"(?:₹|Rs\.?|INR)\s*([\d,]+)", r"\1 rupees", s)
    s = s.replace(",", "") # Remove commas from numbers: 10000 rupees -> 10000 rupees

    # 5. Symbols & abbreviations
    s = re.sub(r"\bCOD\b", "cash on delivery", s)
    s = re.sub(r"\bRTO\b", "return to origin", s)
    s = re.sub(r"%", " percent", s)
    s = re.sub(r"\s*/\s*", " or ", s)

    # 6. Break 8-digit order IDs into 4-digit groups (e.g. 8978 2837)
    s = re.sub(r"\b\d{8}\b", chunk_order_id, s)

    # 7. Clean emojis and bracketed citations [RET-2]
    s = re.sub(r"\[[A-Z0-9_\-\.:]+\]", "", s)
    s = re.sub(r"[^\w\s\.,\?!'\-₹\u0900-\u097F]", " ", s)

    # 8. Collapse whitespace and trim
    s = re.sub(r"\s+", " ", s).strip()

    return s
