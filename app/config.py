"""Konfigurasi EJT AI Core, dibaca dari environment / .env.

Segala pilihan infra (embedder, vector store, LLM) dibuat berbasis env agar
dapat diuji tanpa layanan nyata (test memakai `hash` + `memory` + `fake`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


@dataclass(frozen=True)
class Settings:
    # --- Identitas layanan ---
    app_name: str = _env("APP_NAME", "EJT AI Core")
    version: str = _env("APP_VERSION", "2.0.0")

    # --- Embedding (lokal, gratis, tanpa API key) ---
    embedder: str = _env("EMBEDDER", "local")  # local | hash
    embedding_model: str = _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedding_dim: int = int(_env("EMBEDDING_DIM", "384"))
    embedding_device: str = _env("EMBEDDING_DEVICE", "cpu")

    # --- Vector store ---
    vector_store: str = _env("VECTOR_STORE", "memory")  # memory | qdrant
    qdrant_url: str = _env("QDRANT_URL", "")
    qdrant_collection: str = _env("QDRANT_COLLECTION", "ejt_destinations")
    vector_persist_path: str = _env("VECTOR_PERSIST_PATH", "")

    # --- LLM (Groq) ---
    llm_provider: str = _env("LLM_PROVIDER", "groq")  # groq | fake
    groq_api_key: str = _env("GROQ_API_KEY", "")
    groq_model: str = _env("GROQ_MODEL", "qwen/qwen3.8-27b")
    groq_temperature: float = float(_env("GROQ_TEMPERATURE", "0.7"))
    groq_max_tokens: int = int(_env("GROQ_MAX_TOKENS", "1024"))

    # --- Keamanan server-to-server (Laravel <-> FastAPI) ---
    shared_secret: str = _env("SHARED_SECRET", "")

    # --- RAG tuning ---
    chunk_size: int = int(_env("CHUNK_SIZE", "600"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "80"))
    top_k: int = int(_env("RAG_TOP_K", "5"))
    score_threshold: float = float(_env("RAG_SCORE_THRESHOLD", "0.0"))

    # --- Smart Trip Planner ---
    trip_default_budget: float = float(_env("TRIP_DEFAULT_BUDGET", "0.0"))

    # Akses cepat koleksi
    collection_names: tuple = field(default=("destination", "hotel", "umkm"))


settings = Settings()
