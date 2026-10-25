"""Configuration management for VoiceOps 2026."""
import os
from datetime import date, datetime
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root or parent
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

def _resolve_data_dir() -> Path:
    env_dir = os.environ.get("DATA_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))
        candidates.append(_PROJECT_ROOT / env_dir)
    
    # Auto-discovery fallbacks
    candidates.extend([
        _PROJECT_ROOT / "Capstone_Project_Dataset_FTTT_2026" / "Capstone_Project_Dataset_FTTT_2026",
        _PROJECT_ROOT / "Capstone_Project_Dataset_FTTT_2026",
        _PROJECT_ROOT / "data",
        Path.cwd() / "Capstone_Project_Dataset_FTTT_2026" / "Capstone_Project_Dataset_FTTT_2026",
        Path.cwd() / "data",
    ])
    
    for candidate in candidates:
        if candidate.exists() and (candidate / "data" / "orders.csv").exists():
            return candidate.resolve()
        if candidate.exists() and (candidate / "orders.csv").exists():
            # If candidate is directly the data folder
            return candidate.parent.resolve()
            
    # Default fallback
    return (_PROJECT_ROOT / "Capstone_Project_Dataset_FTTT_2026" / "Capstone_Project_Dataset_FTTT_2026").resolve()

# Root paths
PROJECT_ROOT = _PROJECT_ROOT
DATA_DIR = _resolve_data_dir()
KB_DIR = DATA_DIR / "kb"
CSV_DIR = DATA_DIR / "data"
PRACTICE_DIR = DATA_DIR / "practice"
TOOLS_DIR = DATA_DIR / "tools"

INDEX_DIR = Path(os.environ.get("INDEX_DIR", _PROJECT_ROOT / "data_index")).resolve()
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", _PROJECT_ROOT / "outputs")).resolve()
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", _PROJECT_ROOT / "reports")).resolve()

# Fixed reference date per capstone specification: 2026-10-20
_ref_date_str = os.environ.get("REFERENCE_DATE", "2026-10-20")
try:
    REFERENCE_DATE = datetime.strptime(_ref_date_str, "%Y-%m-%d").date()
except ValueError:
    REFERENCE_DATE = date(2026, 10, 20)

# API Keys
GNANI_API_KEY = os.environ.get("GNANI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

# Model configurations
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "30"))

# Voice settings (Gnani)
STT_LANG = os.environ.get("STT_LANG", "en-IN")
TTS_MODEL = os.environ.get("TTS_MODEL", "timbre-v2.5")
TTS_VOICE = os.environ.get("TTS_VOICE", "Kaveri")
