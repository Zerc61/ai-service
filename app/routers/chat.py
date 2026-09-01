"""Router chat: endpoint utama EJT AI Core.

- POST /v1/chat/stream : chat RAG streaming via Server-Sent Events (SSE)
- POST /v1/trip/plan    : Smart Trip Planner (hasil SELALU draft, tdk memotong saldo)

Chat menerima `role` (tourist/umkm/manager) untuk memilih persona
(KAVI/RAKA/MAJA) sesuai token Sanctum yang diteruskan Laravel.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..deps import get_chat, get_embedder_dep, get_store
from ..rag.chat import ChatOrchestrator
from ..rag.embedder import BaseEmbedder
from ..rag.retrieval import retrieve
from ..rag.tripplanner import build_itinerary, validate_plan_request
from ..rag.vectorstore import BaseVectorStore
from ..security import require_secret

router = APIRouter(prefix="/v1", tags=["chat"])

VALID_ROLES = ("tourist", "umkm", "manager")


class ChatPayload(BaseModel):
    user_id: int
    role: str = "tourist"  # tourist | umkm | manager
    message: str = Field(..., min_length=1)
    session_id: str
    history: List[Dict[str, str]] = Field(default_factory=list)


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(payload: ChatPayload,
                      chat: ChatOrchestrator = Depends(get_chat),
                      _: None = Depends(require_secret)):
    if payload.role not in VALID_ROLES:
        raise HTTPException(status_code=400,
                            detail="role harus tourist, umkm, atau manager")

    def event_source():
        yield sse("meta", {"session_id": payload.session_id, "role": payload.role})
        try:
            for delta in chat.stream(payload.message, role=payload.role,
                                     history=payload.history):
                yield sse("text", {"delta": delta})
        except Exception as exc:  # pragma: no cover - guard jaringan
            yield sse("error", {"message": str(exc)})
            return
        yield sse("done", {"status": "complete"})

    return StreamingResponse(event_source(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


class TripPlanPayload(BaseModel):
    user_id: int
    days: int = Field(default=1, ge=1, le=14)
    budget: Optional[float] = 0.0
    start_city: Optional[str] = None
    preferences: List[str] = Field(default_factory=list)
    session_id: str = ""
    # Data kandidat bisa dikirim langsung oleh Laravel (server-to-server).
    destinations: List[Dict[str, Any]] = Field(default_factory=list)
    hotels: List[Dict[str, Any]] = Field(default_factory=list)


@router.post("/trip/plan")
def trip_plan(payload: TripPlanPayload,
             store: BaseVectorStore = Depends(get_store),
             embedder: BaseEmbedder = Depends(get_embedder_dep),
             _: None = Depends(require_secret)) -> Dict[str, Any]:
    err = validate_plan_request(payload.model_dump())
    if err:
        raise HTTPException(status_code=400, detail=err)

    # Preferensi data eksplisit dari Laravel; fallback retrieval vektor.
    destinations = list(payload.destinations)
    hotels = list(payload.hotels)
    if not destinations and payload.preferences:
        query = " ".join(payload.preferences)
        result = retrieve(store, embedder, query, top_k=20)
        destinations = [p.payload for p in result.points
                        if p.payload.get("source_type") == "destination"]
        hotels = [p.payload for p in result.points
                  if p.payload.get("source_type") == "hotel"]

    plan = build_itinerary(destinations, hotels,
                           days=payload.days, budget=payload.budget or 0.0,
                           start_city=payload.start_city,
                           preferences=payload.preferences)
    plan["session_id"] = payload.session_id
    return {"status": "success", "data": plan}
