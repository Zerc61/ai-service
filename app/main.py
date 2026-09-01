"""Aplikasi FastAPI EJT AI Core — titik masuk utama.

Memuat router: chat (streaming + trip plan), index (ingestion), rag (search).
Menyediakan endpoint health + handshake untuk integrasi dengan Laravel.
"""
from __future__ import annotations

from typing import Dict

from fastapi import FastAPI, Depends

from .config import settings
from .factory import get_vector_store
from .rag.vectorstore import BaseVectorStore
from .routers import chat as chat_router
from .routers import index as index_router
from .routers import rag as rag_router

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Microservice AI untuk East Java Traveling (RAG + Groq).",
)

app.include_router(chat_router.router)
app.include_router(index_router.router)
app.include_router(rag_router.router)


@app.get("/", tags=["health"])
def root() -> Dict[str, str]:
    return {"service": settings.app_name, "status": "ok",
            "version": settings.version}


@app.get("/health", tags=["health"])
def health(store: BaseVectorStore = Depends(get_vector_store)) -> Dict[str, object]:
    return {
        "status": "ok",
        "embedder": settings.embedder,
        "vector_store": settings.vector_store,
        "llm": settings.llm_provider,
        "indexed_points": store.count(),
    }


@app.get("/handshake", tags=["health"])
def handshake() -> Dict[str, object]:
    """Handshake sederhana untuk verifikasi konektivitas dari Laravel."""
    return {"service": settings.app_name, "model": settings.embedding_model,
            "status": "ready"}
