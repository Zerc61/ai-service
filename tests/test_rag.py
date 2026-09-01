"""Test: chunking + ingestion + retrieval."""
from __future__ import annotations

import numpy as np
import pytest

from app.rag.chunking import build_chunks, chunk_text
from app.rag.embedder import HashEmbedder
from app.rag.ingestion import delete_records, upsert_records
from app.rag.retrieval import retrieve
from app.rag.vectorstore import MemoryVectorStore


def test_chunk_short_text_returns_single():
    assert chunk_text("pendek") == ["pendek"]
    assert chunk_text("") == []


def test_chunk_long_text_splits():
    text = "Kalimat satu ini cukup panjang untuk dipecah. " * 60
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    # tidak ada chunk yang melebihi size + overlap
    assert all(len(c) <= 200 + 20 + 40 for c in chunks)


def test_build_chunks_destination():
    dest = {
        "id": 12, "source_type": "destination", "name": "Pantai Balekambang",
        "location": "Malang", "price": "Rp 20.000",
        "description": "Pantai indah di selatan Malang dengan ombak besar.",
    }
    chunks = build_chunks(dest)
    assert chunks
    assert all(c["payload"]["record_id"] == "12" for c in chunks)
    assert chunks[0]["payload"]["source_type"] == "destination"


def test_upsert_then_retrieve_returns_most_relevant():
    embedder = HashEmbedder(dim=384)
    store = MemoryVectorStore(dim=384)
    upsert_records(store, embedder, [
        {"id": 1, "source_type": "destination", "name": "Pantai Balekambang",
         "description": "Pantai indah dengan pasir putih di Malang."},
        {"id": 2, "source_type": "destination", "name": "Candi Penataran",
         "description": "Candi Hindu bersejarah di Blitar."},
        {"id": 3, "source_type": "hotel", "name": "Hotel Tugu",
         "description": "Hotel mewah di pusat kota Malang."},
    ])
    assert store.count() > 0

    result = retrieve(store, embedder, "pantai pasir putih", top_k=3)
    texts = [p.payload.get("text", "") for p in result.points]
    joined = " | ".join(texts).lower()
    assert "pantai" in joined
    # destinasi pantai seharusnya lebih relevan daripada hotel
    assert result.points[0].payload["source_type"] == "destination"

    # filter by source_type
    only_hotel = retrieve(store, embedder, "pantai", top_k=5,
                          source_type="hotel")
    assert all(p.payload["source_type"] == "hotel" for p in only_hotel.points)


def test_delete_records():
    embedder = HashEmbedder(dim=384)
    store = MemoryVectorStore(dim=384)
    upsert_records(store, embedder, [
        {"id": 1, "source_type": "destination", "name": "Pantai Balekambang",
         "description": "Pantai indah di Malang."},
        {"id": 2, "source_type": "destination", "name": "Candi Penataran",
         "description": "Candi sejarah di Blitar."},
    ])
    before = store.count()
    removed = delete_records(store, ["1"], source_type="destination")
    assert removed > 0
    assert store.count() < before
    rest = retrieve(store, embedder, "candi", top_k=5)
    assert rest.points
    assert all(p.payload["record_id"] != "1" for p in rest.points)
