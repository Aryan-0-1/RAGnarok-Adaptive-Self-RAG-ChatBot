"""
vectorstore.py
"""

from __future__ import annotations

from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .embeddings import embed_documents_with_backoff, embeddings_error, get_embeddings

_documents: List[Document] = []
_vector_store: Optional[FAISS] = None
_retriever = None
_chunk_count = 0
_built_with_model: Optional[str] = None


def add_documents(new_docs: List[Document]) -> int:
    """Chunk, embed (batched + backoff), and index new documents.

    Returns the total chunk count after indexing, or raises if
    embeddings aren't configured / the embedding calls fail.
    """
    global _documents, _vector_store, _retriever, _chunk_count, _built_with_model

    if get_embeddings() is None:
        raise RuntimeError(embeddings_error() or "Embeddings not configured.")

    if _built_with_model is not None and _built_with_model != config.GOOGLE_EMBED_MODEL:
        clear_documents()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    ).split_documents(new_docs)

    if not chunks:
        return _chunk_count

    vectors = embed_documents_with_backoff(chunks)
    text_embeddings = list(zip((c.page_content for c in chunks), vectors))
    metadatas = [c.metadata for c in chunks]

    embeddings = get_embeddings()

    if _vector_store is None:
        _vector_store = FAISS.from_embeddings(
            text_embeddings, embeddings, metadatas=metadatas
        )
    else:
        _vector_store.add_embeddings(text_embeddings, metadatas=metadatas)

    _built_with_model = config.GOOGLE_EMBED_MODEL
    _documents.extend(new_docs)
    _chunk_count += len(chunks)
    _retriever = _vector_store.as_retriever(
        search_kwargs={"k": config.RETRIEVER_TOP_K}
    )
    return _chunk_count


def clear_documents() -> None:
    global _documents, _vector_store, _retriever, _chunk_count, _built_with_model
    _documents = []
    _vector_store = None
    _retriever = None
    _chunk_count = 0
    _built_with_model = None


def get_chunk_count() -> int:
    return _chunk_count


def get_retriever():
    """Returns the current retriever, or None if nothing is indexed."""
    return _retriever


def get_config_summary() -> dict:
    """Human-readable pipeline configuration, shown in the sidebar."""
    return {
        "Chat LLM": f"{config.GROQ_CHAT_MODEL}",
        "Embedding Model": config.GOOGLE_EMBED_MODEL if config.EMBEDDINGS_READY else "not configured",
        "Chunk Size / Overlap": f"{config.CHUNK_SIZE} / {config.CHUNK_OVERLAP}",
        "Top-K Retrieval": str(config.RETRIEVER_TOP_K),
        "Max IsSUP Revisions": str(config.MAX_SUP_RETRIES),
        "Max Local Rewrite Tries": str(config.MAX_REWRITE_TRIES),
        "Web-Search Fallback": "enabled (Tavily)" if config.WEB_SEARCH_READY else "disabled (no TAVILY_API_KEY)",
    }
