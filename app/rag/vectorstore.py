"""Vector store interface + implementasi.

- `QdrantVectorStore`: pembungkus `qdrant-client`. Untuk MVP bisa memakai mode
  memory (`:memory:`) atau URL lokal. Koleksi dibuat dengan metric COSINE.
- `MemoryVectorStore`: murni Python (numpy), dengan persistensi file opsional.
  Dipakai untuk test yang hermetic (tanpa server Qdrant hidup).

Satuan titik (point) memakai dataclass `VectorPoint`.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np


@dataclass
class VectorPoint:
    id: str
    vector: List[float]
    payload: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class BaseVectorStore(Protocol):
    def upsert(self, points: Sequence[VectorPoint]) -> None: ...

    def search(self, vector: Sequence[float], top_k: int = 5,
               score_threshold: float = 0.0, filter_payload: Optional[Dict] = None
               ) -> List[VectorPoint]: ...

    def delete(self, point_ids: Sequence[str], filter_payload: Optional[Dict] = None) -> None: ...

    def count(self) -> int: ...

    def clear(self) -> None: ...


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    va = np.asarray(a, dtype="float32")
    vb = np.asarray(b, dtype="float32")
    denom = (float(np.linalg.norm(va)) * float(np.linalg.norm(vb))) or 1.0
    return float(np.dot(va, vb) / denom)


class MemoryVectorStore:
    """Store berbasis numpy + json. Opsional persist ke file (atomic)."""

    def __init__(self, dim: int = 384, persist_path: Optional[str] = None) -> None:
        self.dim = dim
        self.persist_path = persist_path
        self._points: Dict[str, VectorPoint] = {}
        if persist_path:
            self._load()

    # --- persistensi ---
    def _load(self) -> None:
        if self.persist_path and os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                for item in raw:
                    p = item["point"]
                    self._points[p["id"]] = VectorPoint(
                        id=p["id"], vector=p["vector"], payload=p.get("payload", {}))
            except Exception:
                self._points = {}

    def _save(self) -> None:
        if not self.persist_path:
            return
        data = [
            {"point": asdict(self._points[pid])}
            for pid in self._points
        ]
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.persist_path) or ".",
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            os.replace(tmp, self.persist_path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)

    # --- API ---
    def upsert(self, points: Sequence[VectorPoint]) -> None:
        for p in points:
            if len(p.vector) != self.dim:
                raise ValueError(f"dimensi vektor {len(p.vector)} != {self.dim}")
            self._points[p.id] = VectorPoint(id=p.id, vector=list(p.vector),
                                             payload=dict(p.payload))
        self._save()

    def search(self, vector: Sequence[float], top_k: int = 5,
               score_threshold: float = 0.0,
               filter_payload: Optional[Dict] = None) -> List[VectorPoint]:
        scored: List[VectorPoint] = []
        for p in self._points.values():
            if filter_payload and not _match_filter(p.payload, filter_payload):
                continue
            score = _cos(vector, p.vector)
            if score >= score_threshold:
                scored.append(VectorPoint(id=p.id, vector=p.vector,
                                          payload=p.payload, score=float(score)))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def delete(self, point_ids: Sequence[str],
               filter_payload: Optional[Dict] = None) -> None:
        if filter_payload:
            ids = [pid for pid, p in self._points.items()
                   if _match_filter(p.payload, filter_payload)]
        else:
            ids = list(point_ids)
        for pid in ids:
            self._points.pop(pid, None)
        self._save()

    def count(self) -> int:
        return len(self._points)

    def clear(self) -> None:
        self._points.clear()
        self._save()


def _match_filter(payload: Dict[str, Any], f: Dict[str, Any]) -> bool:
    for key, value in f.items():
        if payload.get(key) != value:
            return False
    return True


class QdrantVectorStore:
    """Pembungkus qdrant-client. URL kosong / ':memory:' -> mode memory MVP."""

    def __init__(self, dim: int = 384, collection: str = "ejt_destinations",
                 url: Optional[str] = None) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.dim = dim
        self.collection = collection
        if url in (None, "", ":memory:"):
            self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(url=url)
        existing = [c.name for c in self._client.get_collections().collections]
        if collection not in existing:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, points: Sequence[VectorPoint]) -> None:
        from qdrant_client.models import PointStruct
        structs = [
            PointStruct(id=int(_stable_id(p.id)), vector=list(p.vector), payload=dict(p.payload))
            for p in points
        ]
        self._client.upsert(collection_name=self.collection, points=structs)

    def search(self, vector: Sequence[float], top_k: int = 5,
               score_threshold: float = 0.0,
               filter_payload: Optional[Dict] = None) -> List[VectorPoint]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        qfilter = None
        if filter_payload:
            qfilter = Filter(must=[
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_payload.items()
            ])
        resp = self._client.query_points(
            collection_name=self.collection,
            query=list(vector),
            limit=top_k,
            query_filter=qfilter,
            score_threshold=float(score_threshold) if score_threshold else None,
            with_vectors=True,
        )
        return [
            VectorPoint(id=str(p.id), vector=list(p.vector),
                        payload=p.payload, score=float(p.score))
            for p in resp.points
        ]

    def delete(self, point_ids: Sequence[str],
               filter_payload: Optional[Dict] = None) -> None:
        if filter_payload:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            qfilter = Filter(must=[
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_payload.items()
            ])
            selector = qfilter
        else:
            selector = [_stable_id(pid) for pid in point_ids]
        if selector:
            self._client.delete(collection_name=self.collection,
                                points_selector=selector)

    def count(self) -> int:
        return self._client.count(collection_name=self.collection).count

    def clear(self) -> None:
        self._client.delete_collection(self.collection)
        from qdrant_client.models import Distance, VectorParams
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )


def _stable_id(pid: str) -> int:
    """Qdrant butuh int id; petakan string ke int deterministik tanpa tabrakan."""
    return abs(hash(pid)) % (10 ** 12)


def build_vector_store(provider: str = "memory", dim: int = 384,
                       collection: str = "ejt_destinations", url: Optional[str] = None,
                       persist_path: Optional[str] = None) -> BaseVectorStore:
    if provider == "memory":
        return MemoryVectorStore(dim=dim, persist_path=persist_path)
    if provider == "qdrant":
        return QdrantVectorStore(dim=dim, collection=collection, url=url)
    raise ValueError(f"Vector store tidak dikenal: {provider}")
