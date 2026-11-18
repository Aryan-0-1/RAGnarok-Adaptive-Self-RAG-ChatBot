"""
config.py
---------
Central place for environment configuration and constants.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

# No-op if no .env file exists (e.g. on Streamlit Cloud), so this is always safe to call.
load_dotenv()


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve a config value from st.secrets, then os.environ, then default."""
    try:
        import streamlit as st

        if key in st.secrets:
            value = st.secrets[key]
            if value not in (None, ""):
                return str(value)
    except Exception:
        # st.secrets raises if no secrets.toml exists (e.g. local dev
        # without Streamlit Cloud) and Streamlit may not even be
        # importable in non-UI contexts — either way, fall through.
        pass

    return os.getenv(key, default)


# ---------------------------------------------------------------
# Chat LLM (GROQ)
# ---------------------------------------------------------------
GROQ_CHAT_MODEL = get_env("GROQ_CHAT_MODEL")
GROQ_API_KEY= get_env("GROQ_API_KEY")

# ---------------------------------------------------------------
# Embeddings (Google Generative AI)
# ---------------------------------------------------------------
GOOGLE_API_KEY = get_env("GOOGLE_API_KEY")
GOOGLE_EMBED_MODEL = get_env("GOOGLE_EMBED_MODEL", "models/gemini-embedding-001")
GOOGLE_EMBED_BATCH_SIZE = int(get_env("GOOGLE_EMBED_BATCH_SIZE", "50"))
GOOGLE_EMBED_MAX_RETRIES = int(get_env("GOOGLE_EMBED_MAX_RETRIES", "3"))

# ---------------------------------------------------------------
# Web-search fallback (Tavily, optional)
# ---------------------------------------------------------------
TAVILY_API_KEY = get_env("TAVILY_API_KEY")

# ---------------------------------------------------------------
# Chat persistence
# ---------------------------------------------------------------
CHAT_DB_PATH = get_env("CHAT_DB_PATH", "self_rag_chat_history.sqlite3")

# ---------------------------------------------------------------
# LangSmith tracing (optional; langchain reads these from the
# environment itself, we just make sure they're exported)
# ---------------------------------------------------------------
for _key in (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_ENDPOINT",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
):
    _val = get_env(_key)
    if _val is not None:
        os.environ[_key] = _val

# ---------------------------------------------------------------
# Feature-readiness flags — used by the UI to show banners instead
# of crashing when optional keys are missing.
# ---------------------------------------------------------------
EMBEDDINGS_READY = bool(GOOGLE_API_KEY)
WEB_SEARCH_READY = bool(TAVILY_API_KEY)


def embeddings_ready() -> bool:
    return EMBEDDINGS_READY


def web_search_ready() -> bool:
    return WEB_SEARCH_READY

# ---------------------------------------------------------------
# Graph tuning constants
# ---------------------------------------------------------------
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
RETRIEVER_TOP_K = 4

MAX_SUP_RETRIES = 5         # revise_answer <-> is_sup loop cap
MAX_REWRITE_TRIES = 3        # local rewrite_question <-> retrieve loop cap

GRAPH_RECURSION_LIMIT = 80
