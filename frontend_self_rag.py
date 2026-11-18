import streamlit as st

from ui.chat import handle_user_input, init_session_state, render_history
from ui.sidebar import render_sidebar

st.set_page_config(page_title="Self-RAG Assistant", page_icon="🧠", layout="wide")
st.title("🧠 Self-RAG Assistant")
st.caption(
    "Ask questions over your PDFs or YouTube transcripts using a LangGraph "
    "Self-RAG workflow — with grounding checks, revision, and a web-search fallback."
)

init_session_state()
render_sidebar()
render_history()
handle_user_input()
