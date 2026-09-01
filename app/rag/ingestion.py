"""Orkestrasi ingestion RAG: upsert / backfill / delete ke vector store.

Support dua sumber masukan:
- payload langsung dari Laravel (upsert),
- daftar record (backfill) dijadikan chunk lalu di-embed satu batch.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .chunking import build_chunks
from .embedder import BaseEmbedder
from .vectorstore import BaseVectorStore, VectorPoint


def upsert_records(store: BaseVectorStore, embedder: BaseEmbedder,
                   records: List[Dict[str, Any]],
                   chunk_size: int = 600, chunk_overlap: int = 80,
                   source_type: Optional[str] = None) -> int:
    """Embed & simpan chunk dari daftar record. Return jumlah chunk tersimpan."""
    all_chunks: List[Dict[str, Any]] = []
    for rec in records:
        item = dict(rec)
        if source_type and "source_type" not in item:
            item["source_type"] = source_type
        all_chunks.extend(build_chunks(item, chunk_size, chunk_overlap))

    if not all_chunks:
        return 0

    texts = [c["text"] for c in all_chunks]
    vectors = embedder.encode_batch(texts)
    points = [
        VectorPoint(
            id=c["id"], vector=vectors[i],
            payload={**c["payload"], "text": c["text"]},
        )
        for i, c in enumerate(all_chunks)
    ]
    store.upsert(points)
    return len(points)


def delete_records(store: BaseVectorStore, record_ids: List[str],
                   source_type: Optional[str] = None) -> int:
    """Hapus chunk seluruh record.

    Strategi: hapus per record via filter payload `{source_type, record_id}`.
    """
    before = store.count()
    for rid in record_ids:
        f = {"record_id": rid}
        if source_type:
            f["source_type"] = source_type
        store.delete([], filter_payload=f)
    return max(0, before - store.count())


def clear_index(store: BaseVectorStore) -> None:
    store.clear()
