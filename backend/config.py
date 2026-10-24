import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _get_setting(name: str, default: str = "") -> str:
    # Local development: read from environment variables or .env.
    value = os.getenv(name, "").strip()
    if value:
        return value

    # Cloud deployment: fall back to Streamlit Secrets.
    try:
        import streamlit as st

        value = st.secrets.get(name, default)
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        # Secrets may be unavailable during local tests.
        pass

    return default


def get_gemini_api_key() -> str:
    key = _get_setting("GEMINI_API_KEY")

    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env or Streamlit Secrets.")

    return key


def get_gemini_model() -> str:
    return _get_setting(
        "GEMINI_MODEL",
        "gemini-3.6-flash",
    )
