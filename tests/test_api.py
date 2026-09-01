"""Test endpoint API via TestClient (offline: Hash + Memory + Fake)."""
from __future__ import annotations

import pytest

AUTH = {"X-AI-Secret": "test-secret"}


def _ingest(client):
    r = client.post("/v1/index/upsert", json={
        "record": {"id": 12, "source_type": "destination",
                   "name": "Pantai Balekambang",
                   "description": "Pantai indah di Malang, tiket Rp 20.000."},
    }, headers=AUTH)
    assert r.status_code == 200
    return r.json()


def test_health_and_handshake(client):
    assert client.get("/").json()["status"] == "ok"
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert client.get("/handshake").json()["status"] == "ready"


def test_index_requires_secret(client):
    # tanpa header -> 401
    assert client.post("/v1/index/upsert", json={"record": {}}).status_code == 401
    # header salah -> 401
    assert client.post("/v1/index/upsert", json={"record": {}},
                       headers={"X-AI-Secret": "wrong"}).status_code == 401


def test_upsert_and_stats(client):
    _ingest(client)
    stats = client.get("/v1/index/stats", headers=AUTH).json()
    assert stats["data"]["total_points"] >= 1


def test_backfill(client, store):
    r = client.post("/v1/index/backfill", json={
        "source_type": "destination",
        "records": [
            {"id": 1, "name": "Coban Rondo", "description": "Air terjun di Batu."},
            {"id": 2, "name": "Candi Penataran", "description": "Candi di Blitar."},
        ],
    }, headers=AUTH)
    assert r.status_code == 200
    assert r.json()["data"]["indexed_chunks"] >= 1
    assert store.count() >= 1


def test_rag_search(client):
    _ingest(client)
    r = client.post("/v1/rag/search", json={"query": "pantai balekambang", "top_k": 3})
    assert r.status_code == 200
    hits = r.json()["data"]["hits"]
    assert hits
    assert hits[0]["name"] == "Pantai Balekambang"


def test_chat_stream_sse(client):
    _ingest(client)
    r = client.post("/v1/chat/stream", json={
        "user_id": 1, "role": "tourist",
        "message": "berapa tiket pantai balekambang?",
        "session_id": "sess-1",
    }, headers=AUTH)
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    body = r.text
    assert "event: meta" in body
    assert "event: text" in body
    assert "event: done" in body
    assert "session_id" in body


def test_chat_invalid_role(client):
    r = client.post("/v1/chat/stream", json={
        "user_id": 1, "role": "admin", "message": "halo", "session_id": "s",
    }, headers=AUTH)
    assert r.status_code == 400


def test_trip_plan_from_payload(client):
    r = client.post("/v1/trip/plan", json={
        "user_id": 1, "days": 2, "budget": 50000,
        "destinations": [
            {"id": 1, "name": "Pantai Balekambang", "price": 20000, "location": "Malang"},
            {"id": 2, "name": "Coban Rondo", "price": 15000, "location": "Batu"},
        ],
    }, headers=AUTH)
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["status"] == "draft"
    assert data["days"] == 2
    assert data["estimated_cost"] <= 50000


def test_trip_plan_validation(client):
    # days di luar rentang ditolak skema pydantic -> 422
    assert client.post("/v1/trip/plan", json={"user_id": 1, "days": 0},
                       headers=AUTH).status_code == 422
