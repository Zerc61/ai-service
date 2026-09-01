"""Router retrieval: pencarian konteks berbasis vektor untuk RAG."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends

from ..deps import get_embedder_dep, get_store
from ..rag.embedder import BaseEmbedder
from ..rag.retrieval import retrieve
from ..rag.vectorstore import BaseVectorStore
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/rag", tags=["rag"])


class SearchPayload(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    source_type: Optional[str] = None


@router.post("/search")
def search(payload: SearchPayload,
          store: BaseVectorStore = Depends(get_store),
          embedder: BaseEmbedder = Depends(get_embedder_dep)) -> Dict[str, Any]:
    result = retrieve(store, embedder, payload.query,
                      top_k=payload.top_k,
                      score_threshold=payload.score_threshold,
                      source_type=payload.source_type)
    hits = [
        {
            "id": p.id,
            "score": round(p.score, 4),
            "source_type": p.payload.get("source_type"),
            "record_id": p.payload.get("record_id"),
            "name": p.payload.get("name"),
            "text": p.payload.get("text", ""),
        }
        for p in result.points
    ]
    return {"status": "success", "data": {"query": payload.query, "hits": hits}}
