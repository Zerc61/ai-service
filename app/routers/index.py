"""Router ingestion: memelihara index vektor dari data Laravel.

Endpoints (terproteksi shared secret server-to-server):
- POST /v1/index/upsert           : embed & simpan satu/lebih record
- POST /v1/index/backfill         : embed & simpan banyak record (initial)
- DELETE /v1/index/record         : hapus chunk record tertentu
- DELETE /v1/index/clear          : kosongkan koleksi
- GET  /v1/index/stats            : jumlah titik tersimpan
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_embedder_dep, get_store
from ..rag.chunking import build_chunks
from ..rag.embedder import BaseEmbedder
from ..rag.ingestion import clear_index, delete_records, upsert_records
from ..rag.vectorstore import BaseVectorStore
from ..security import require_secret
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/index", tags=["index"])


class RecordPayload(BaseModel):
    record: Dict[str, Any]
    source_type: Optional[str] = None


class BackfillPayload(BaseModel):
    records: List[Dict[str, Any]]
    source_type: Optional[str] = None
    chunk_size: int = Field(default=600, ge=50, le=4000)
    chunk_overlap: int = Field(default=80, ge=0, le=500)


class DeletePayload(BaseModel):
    record_ids: List[str]
    source_type: Optional[str] = None


@router.post("/upsert")
def upsert(payload: RecordPayload,
           store: BaseVectorStore = Depends(get_store),
           embedder: BaseEmbedder = Depends(get_embedder_dep),
           _: None = Depends(require_secret)) -> Dict[str, Any]:
    chunks = upsert_records(store, embedder, [payload.record],
                            source_type=payload.source_type)
    return {"status": "success", "data": {"indexed_chunks": chunks,
                                          "total": store.count()}}


@router.post("/backfill")
def backfill(payload: BackfillPayload,
            store: BaseVectorStore = Depends(get_store),
            embedder: BaseEmbedder = Depends(get_embedder_dep),
            _: None = Depends(require_secret)) -> Dict[str, Any]:
    try:
        chunks = upsert_records(store, embedder, payload.records,
                                chunk_size=payload.chunk_size,
                                chunk_overlap=payload.chunk_overlap,
                                source_type=payload.source_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "success",
            "data": {"indexed_chunks": chunks, "total": store.count()}}


@router.delete("/record")
def delete(payload: DeletePayload,
          store: BaseVectorStore = Depends(get_store),
          _: None = Depends(require_secret)) -> Dict[str, Any]:
    removed = delete_records(store, payload.record_ids,
                             source_type=payload.source_type)
    return {"status": "success", "data": {"removed": removed,
                                          "total": store.count()}}


@router.delete("/clear")
def clear(store: BaseVectorStore = Depends(get_store),
         _: None = Depends(require_secret)) -> Dict[str, Any]:
    clear_index(store)
    return {"status": "success", "data": {"total": store.count()}}


@router.get("/stats")
def stats(store: BaseVectorStore = Depends(get_store),
         _: None = Depends(require_secret)) -> Dict[str, Any]:
    return {"status": "success", "data": {"total_points": store.count()}}
