# routing.py

from __future__ import annotations

from typing import Literal

from .. import config
from .state import State


def route_after_decide(state: State) -> Literal["generate_direct", "retrieve"]:
    return "retrieve" if state["need_retrieval"] else "generate_direct"


def route_after_relevance(state: State) -> Literal["generate_from_context", "web_fallback", "no_answer_found"]:
    if state.get("relevant_docs"):
        return "generate_from_context"

    if config.WEB_SEARCH_READY and not state.get("web_search_used"):
        return "web_fallback"

    return "no_answer_found"


def route_after_issup(state: State) -> Literal["accept_answer", "revise_answer"]:
    if state.get("issup") == "fully_supported":
        return "accept_answer"

    if state.get("retries", 0) >= config.MAX_SUP_RETRIES:
        return "accept_answer"  # give up revising; let IsUSE have the final word

    return "revise_answer"


def route_after_isuse(state: State) -> Literal["END", "rewrite_question", "web_fallback", "no_answer_found"]:
    if state.get("isuse") == "useful":
        return "END"

    if state.get("rewrite_tries", 0) < config.MAX_REWRITE_TRIES:
        return "rewrite_question"

    if config.WEB_SEARCH_READY and not state.get("web_search_used"):
        return "web_fallback"

    return "no_answer_found"
