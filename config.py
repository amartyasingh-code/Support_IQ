"""
config.py

Centralized configuration loader.
Works both locally (.env file) and on Streamlit Cloud (st.secrets).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env for local development (does nothing if file doesn't exist)
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

# Try to get from Streamlit secrets first (for cloud deployment),
# fall back to environment variable (for local development)
try:
    import streamlit as st
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
except Exception:
    # streamlit not available or secrets not configured — use .env
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found. "
        "Copy .env.example to .env and add your Groq API key, "
        "or set it in Streamlit Cloud's Secrets settings."
    )

MAX_RETRIES = 2