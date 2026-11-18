"""
llm.py: Chat LLM setup.
"""

from langchain_groq import ChatGroq
from . import config

llm = ChatGroq(
    model = config.GROQ_CHAT_MODEL,
    temperature=0,
    api_key = config.GROQ_API_KEY
)
