"""Central settings for the backend.

Every value can be overridden with an LC_* environment variable, so you can tune
the app for your hardware without touching code.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _int(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


APP_NAME = os.getenv("LC_APP_NAME", "Atlas")
# Shown before amounts when a pasted list has no currency symbol ("" = none, e.g. "₦").
CURRENCY = os.getenv("LC_CURRENCY", "")

# --- Ollama / models -------------------------------------------------------
OLLAMA_URL = os.getenv("LC_OLLAMA_URL", "http://localhost:11434").rstrip("/")
CODE_MODEL = os.getenv("LC_CODE_MODEL", "qwen2.5-coder:7b")
CHAT_MODEL = os.getenv("LC_CHAT_MODEL", "qwen2.5:7b")

# Context window. 8192 tokens of a 7B Q4 model costs ~0.5 GB of VRAM, which is
# about what a 6 GB GTX 1660 Ti can spare. Drop to 4096 if `ollama ps` shows
# the model is not 100% on GPU.
NUM_CTX = _int("LC_NUM_CTX", 8192)
NUM_PREDICT = _int("LC_NUM_PREDICT", 3072)       # max tokens per answer
KEEP_ALIVE = os.getenv("LC_KEEP_ALIVE", "10m")   # how long a model stays loaded
TEMPERATURE_CODE = _float("LC_TEMPERATURE_CODE", 0.2)
TEMPERATURE_CHAT = _float("LC_TEMPERATURE_CHAT", 0.5)

# --- Agent -----------------------------------------------------------------
MAX_TOOL_CALLS = _int("LC_MAX_TOOL_CALLS", 8)
CHARS_PER_TOKEN = 3.0                       # conservative estimate for code
OUTPUT_RESERVE_TOKENS = min(NUM_PREDICT, 2048)
PROMPT_BUDGET_TOKENS = max(1024, NUM_CTX - OUTPUT_RESERVE_TOKENS)
MAX_FILE_CHARS = int(PROMPT_BUDGET_TOKENS * CHARS_PER_TOKEN * 0.30)

# --- Storage ---------------------------------------------------------------
# Kept outside backend/ on purpose: `fastapi dev` / `--reload` watches the
# backend folder, and uploading a Python project would restart the server.
DATA_DIR = Path(os.getenv("LC_DATA_DIR", BASE_DIR.parent / "data")).resolve()
WORKSPACE_DIR = DATA_DIR / "workspace"
BACKUP_DIR = DATA_DIR / "backups"
DB_PATH = DATA_DIR / "localcoder.db"

MAX_UPLOAD_MB = _int("LC_MAX_UPLOAD_MB", 200)
MAX_UNZIPPED_MB = _int("LC_MAX_UNZIPPED_MB", 400)
MAX_PROJECT_FILES = _int("LC_MAX_PROJECT_FILES", 20000)
MAX_SINGLE_FILE_MB = _int("LC_MAX_SINGLE_FILE_MB", 25)

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "LC_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if o.strip()
]
