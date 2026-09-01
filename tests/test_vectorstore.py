"""Test: embedder (Hash deterministik) + vector store (Memory + persistence)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from app.rag.embedder import HashEmbedder, build_embedder
from app.rag.vectorstore import MemoryVectorStore, VectorPoint, build_vector_store


def test_hash_embedder_is_deterministic_and_unit_norm():
    e = HashEmbedder(dim=384)
    a = e.encode("pantai indah")
    b = e.encode("pantai indah")
    assert a == b
    assert len(a) == 384
    norm = float(np.linalg.norm(np.asarray(a)))
    assert abs(norm - 1.0) < 1e-5


def test_build_embedder_throws_on_unknown():
    with pytest.raises(ValueError):
        build_embedder(provider="nope")


def test_memory_store_upsert_search_delete(tmp_path):
    store = MemoryVectorStore(dim=384)

    def vec(seed):
        rng = np.random.RandomState(seed)
        return (rng.rand(384).astype("float32") * 0.1).tolist()

    store.upsert([
        VectorPoint(id="d1", vector=vec(1), payload={"name": "Pantai Balekambang",
                                                     "source_type": "destination",
                                                     "text": "pantai indah malang"}),
        VectorPoint(id="d2", vector=vec(99), payload={"name": "Candi Penataran",
                                                      "source_type": "destination",
                                                      "text": "candi sejarah"}),
    ])
    assert store.count() == 2

    # search mengembalikan semua (score >= 0) dan urut menurun
    hits = store.search(vec(1), top_k=2)
    assert len(hits) == 2
    assert hits[0].id == "d1"
    assert hits[0].score >= hits[1].score

    # filter by source_type
    only_pantai = store.search(vec(1), top_k=5, filter_payload={"name": "Pantai Balekambang"})
    assert len(only_pantai) == 1
    assert only_pantai[0].id == "d1"

    # delete by filter
    store.delete([], filter_payload={"name": "Pantai Balekambang"})
    assert store.count() == 1


def test_memory_store_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "index.json")
    store = MemoryVectorStore(dim=384, persist_path=path)
    rng = np.random.RandomState(7)
    store.upsert([
        VectorPoint(id="x1", vector=(rng.rand(384).astype("float32") * 0.1).tolist(),
                    payload={"name": "A", "text": "desc a"})
    ])
    assert store.count() == 1

    # store baru dengan path sama memuat ulang
    store2 = MemoryVectorStore(dim=384, persist_path=path)
    assert store2.count() == 1
    hits = store2.search(store2._points["x1"].vector, top_k=1)
    assert hits[0].id == "x1"


def test_memory_store_rejects_wrong_dim():
    store = MemoryVectorStore(dim=384)
    with pytest.raises(ValueError):
        store.upsert([VectorPoint(id="bad", vector=[0.1, 0.2])])


def test_build_vector_store_unknown():
    with pytest.raises(ValueError):
        build_vector_store(provider="nope")


def test_seed_projection_stable():
    from app.rag.vectorstore import _stable_id
    assert _stable_id("destination:12:0") == _stable_id("destination:12:0")
