"""Embedding interface + implementasi.

- `LocalEmbedder`: `BAAI/bge-small-en-v1.5` via sentence-transformers. Berjalan
  100% lokal & gratis (tanpa API key), model dimuat SATU KALI sebagai singleton
  agar RAM hemat (~300MB). Output 384 dimensi.
- `HashEmbedder`: deterministik, HANYA untuk test / fallback cepat (bukan untuk
  produksi). Tidak memuat model apa pun sehingga bisa diuji tanpa jaringan.
"""
from __future__ import annotations

import hashlib
from typing import List, Protocol, Sequence, Union

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _HAVE_ST = True
except Exception:  # pragma: no cover - dependency optional saat test
    _HAVE_ST = False


class BaseEmbedder(Protocol):
    dim: int

    def encode(self, text: str) -> List[float]: ...

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]: ...


class LocalEmbedder:
    """Embedding lokal via sentence-transformers (BAAI/bge-small-en-v1.5)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 dim: int = 384, device: str = "cpu") -> None:
        if not _HAVE_ST:
            raise RuntimeError(
                "sentence-transformers tidak terpasang; "
                "jalankan: pip install sentence-transformers"
            )
        self.model_name = model_name
        self.device = device
        # Singleton seperti Master Plan: muat sekali, pakai berulang.
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = dim

    def encode(self, text: str) -> List[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype="float32").ravel().tolist()

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        vecs = self._model.encode(list(texts), normalize_embeddings=True)
        return [np.asarray(v, dtype="float32").ravel().tolist() for v in vecs]


class HashEmbedder:
    """Deterministik, tanpa model. Untuk test / fallback (bukan produksi)."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vector(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        seed = int(digest[:16], 16) % (2 ** 32)
        rng = np.random.RandomState(seed)
        v = rng.rand(self.dim).astype("float32")
        norm = float(np.linalg.norm(v)) or 1.0
        return (v / norm).tolist()

    def encode(self, text: str) -> List[float]:
        return self._vector(text)

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._vector(t) for t in texts]


def build_embedder(provider: str = "local", model: str = "BAAI/bge-small-en-v1.5",
                   dim: int = 384, device: str = "cpu") -> BaseEmbedder:
    if provider == "hash":
        return HashEmbedder(dim=dim)
    if provider == "local":
        return LocalEmbedder(model_name=model, dim=dim, device=device)
    raise ValueError(f"Embedder tidak dikenal: {provider}")
