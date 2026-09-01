"""
config.py

Centralized configuration loader.
Import this in every agent instead of calling load_dotenv() directly.
Ensures .env is always loaded from the project root regardless of
which file or module triggers the import.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Always load from project root
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

# Expose keys as module-level constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Validate at import time — fail fast if keys are missing
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found. "
        "Copy .env.example to .env and add your Groq API key."
    )
# --- Pipeline constants ---
MAX_RETRIES = 2