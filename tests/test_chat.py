"""Test: prompts + LLM fake + orchestrator chat."""
from __future__ import annotations

import pytest

from app.rag.chat import ChatOrchestrator
from app.rag.embedder import HashEmbedder
from app.rag.ingestion import upsert_records
from app.rag.llm import FakeLLM, build_llm
from app.rag.prompts import build_rag_prompt, get_system_prompt
from app.rag.vectorstore import MemoryVectorStore


def test_personas():
    assert "KAVI" in get_system_prompt("tourist")
    assert "RAKA" in get_system_prompt("umkm")
    assert "MAJA" in get_system_prompt("manager")
    assert get_system_prompt(None) == get_system_prompt("tourist")
    assert get_system_prompt("bogus") == get_system_prompt("tourist")


def test_prompt_contains_context_or_honesty():
    with_ctx = build_rag_prompt("halo", "Pantai Balekambang tiket 20rb")
    assert any("Pantai Balekambang" in m["content"] for m in with_ctx if m["role"] == "system")
    no_ctx = build_rag_prompt("halo", "")
    assert any("tidak memiliki konteks" in m["content"] for m in no_ctx
               if m["role"] == "system")


def test_fake_llm_deterministic():
    llm = FakeLLM(model="t")
    m = [{"role": "user", "content": "cek harga tiket"}]
    assert llm.generate(m) == llm.generate(m)
    out = "".join(llm.stream(m))
    assert out == llm.generate(m)


def test_build_llm_unknown():
    with pytest.raises(ValueError):
        build_llm(provider="nope")


def test_orchestrator_embeds_retrieves_and_builds_prompt_with_role():
    embedder = HashEmbedder(dim=384)
    store = MemoryVectorStore(dim=384)
    upsert_records(store, embedder, [
        {"id": 12, "source_type": "destination", "name": "Pantai Balekambang",
         "description": "Pantai indah di Malang, tiket masuk Rp 20.000."},
    ])
    orch = ChatOrchestrator(store, embedder, FakeLLM(), top_k=5)
    reply = orch.generate("berapa tiket pantai balekambang?", role="tourist",
                          history=[{"role": "user", "content": "halo"},
                                   {"role": "assistant", "content": "hai"}])
    assert reply.startswith("[FakeLLM")


def test_orchestrator_stream_yields_tokens():
    embedder = HashEmbedder(dim=384)
    store = MemoryVectorStore(dim=384)
    orch = ChatOrchestrator(store, embedder, FakeLLM(), top_k=5)
    tokens = list(orch.stream("tes", role="tourist"))
    assert tokens
    assert "".join(tokens).startswith("[FakeLLM")
