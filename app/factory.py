"""Factory untuk merakit komponen dari `settings` (env-driven).

Semua komponen yang berat (embedding model) dibuat lazily agar import modul
tetap ringan dan test bisa memakai implementasi fake.
"""
from __future__ import annotations

from functools import lru_cache

from .config import settings
from .rag.embedder import build_embedder, BaseEmbedder
from .rag.vectorstore import build_vector_store, BaseVectorStore
from .rag.llm import build_llm, BaseLLM
from .rag.chat import ChatOrchestrator


@lru_cache(maxsize=1)
def get_embedder() -> BaseEmbedder:
    return build_embedder(
        provider=settings.embedder,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
        device=settings.embedding_device,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> BaseVectorStore:
    url = settings.qdrant_url or None
    persist = settings.vector_persist_path or None
    return build_vector_store(
        provider=settings.vector_store,
        dim=settings.embedding_dim,
        collection=settings.qdrant_collection,
        url=url,
        persist_path=persist,
    )


@lru_cache(maxsize=1)
def get_llm() -> BaseLLM:
    return build_llm(
        provider=settings.llm_provider,
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=settings.groq_temperature,
        max_tokens=settings.groq_max_tokens,
    )


@lru_cache(maxsize=1)
def get_orchestrator() -> ChatOrchestrator:
    store = get_vector_store()
    embedder = get_embedder()
    llm = get_llm()
    return ChatOrchestrator(store, embedder, llm,
                            top_k=settings.top_k,
                            score_threshold=settings.score_threshold)
