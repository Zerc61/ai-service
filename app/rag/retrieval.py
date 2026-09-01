"""Retrieval RAG: query -> embed -> search -> kumpulan chunk + konteks teks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .embedder import BaseEmbedder
from .vectorstore import BaseVectorStore, VectorPoint


@dataclass
class RetrievalResult:
    query: str
    points: List[VectorPoint]

    @property
    def context(self) -> str:
        """Rakit teks konteks untuk prompt Groq, dengan sumber per chunk."""
        parts = []
        for p in self.points:
            name = p.payload.get("name", p.payload.get("record_id", ""))
            st = p.payload.get("source_type", "source")
            rid = p.payload.get("record_id", "")
            # Teks perlu disimpan; kita simpan via payload 'text' pada ingestion.
            text = p.payload.get("text", "")
            parts.append(f"- [{st} {name} (id {rid})] {text}")
        return "\n".join(parts)


def retrieve(store: BaseVectorStore, embedder: BaseEmbedder, query: str,
             top_k: int = 5, score_threshold: float = 0.0,
             source_type: Optional[str] = None) -> RetrievalResult:
    vector = embedder.encode(query)
    f = {"source_type": source_type} if source_type else None
    points = store.search(vector, top_k=top_k, score_threshold=score_threshold,
                          filter_payload=f)
    return RetrievalResult(query=query, points=points)
