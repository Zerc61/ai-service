"""Fixtures pytest: komponen fake deterministik + dependency overrides.

Semua test berjalan OFFLINE (tanpa jaringan, tanpa API key, tanpa memuat
model embedding). Memakai:
- embedder  -> HashEmbedder (deterministik)
- store     -> MemoryVectorStore (persist off)
- llm       -> FakeLLM (deterministik)
"""
from __future__ import annotations

import os

os.environ.setdefault("SHARED_SECRET", "test-secret")
os.environ.setdefault("EMBEDDER", "hash")
os.environ.setdefault("VECTOR_STORE", "memory")
os.environ.setdefault("LLM_PROVIDER", "fake")

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.chat import ChatOrchestrator
from app.rag.embedder import HashEmbedder
from app.rag.llm import FakeLLM
from app.rag.vectorstore import MemoryVectorStore


@pytest.fixture
def embedder():
    return HashEmbedder(dim=384)


@pytest.fixture
def store():
    return MemoryVectorStore(dim=384)


@pytest.fixture
def llm():
    return FakeLLM(model="test")


@pytest.fixture
def orchestrator(store, embedder, llm):
    return ChatOrchestrator(store, embedder, llm, top_k=5)


@pytest.fixture
def client(store, embedder, llm):
    """TestClient dengan dependency override ke fake (hermetic, offline)."""
    from app.deps import get_chat, get_embedder_dep, get_llm_dep, get_store
    from app.factory import get_vector_store as factory_store

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_embedder_dep] = lambda: embedder
    app.dependency_overrides[get_llm_dep] = lambda: llm
    app.dependency_overrides[get_chat] = lambda: ChatOrchestrator(
        store, embedder, llm, top_k=5)
    # /health memakai get_vector_store dari factory
    app.dependency_overrides[factory_store] = lambda: store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
