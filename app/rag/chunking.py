"""Pemecahan teks menjadi chunk untuk RAG ingestion.

Strategi sederhana: pecah berdasarkan kalimat/paragraf, lalu rakit menjadi
chunk berukuran `chunk_size` karakter dengan overlap `chunk_overlap`.
"""
from __future__ import annotations

from typing import Any, Dict, List

DEFAULT_SIZE = 600
DEFAULT_OVERLAP = 80


def chunk_text(text: str, chunk_size: int = DEFAULT_SIZE,
               chunk_overlap: int = DEFAULT_OVERLAP) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    chunks: List[str] = []
    current = ""
    for sent in sentences:
        piece = sent if sent.endswith(".") else sent + "."
        if current and len(current) + len(piece) + 1 > chunk_size:
            chunks.append(current.strip())
            current = piece
        else:
            current = f"{current} {piece}".strip() if current else piece
        if len(current) >= chunk_size:
            chunks.append(current.strip())
            current = ""
    if current.strip():
        chunks.append(current.strip())

    # gabungkan chunk terlalu kecil agar tidak terlalu banyak fragment
    merged: List[str] = []
    for c in chunks:
        if merged and len(merged[-1]) + len(c) <= chunk_size + chunk_overlap:
            merged[-1] = f"{merged[-1]} {c}"
        else:
            merged.append(c)

    # overlap sederhana: sisipkan sisa kalimat terakhir chunk prev
    result: List[str] = []
    for i, c in enumerate(merged):
        if i > 0 and chunk_overlap > 0 and len(result) > 0:
            prev_tail = result[-1][-chunk_overlap:]
            c = f"{prev_tail} {c}".strip()
        result.append(c)
    return result


def _source_type(payload: Dict[str, Any]) -> str:
    return payload.get("source_type", payload.get("type", "unknown"))


def _build_blocks(item: Dict[str, Any]) -> List[str]:
    """Bangun blok teks tematik dari satu tipe data (destination/hotel/umkm)."""
    st = _source_type(item)
    blocks = []
    if st == "destination":
        blocks.append(
            f"Nama destinasi: {item.get('name', '')}. "
            f"Lokasi: {item.get('location', item.get('address', ''))}. "
            f"Deskripsi: {item.get('description', '')}"
        )
        if item.get("price"):
            blocks.append(f"Harga tiket {item.get('name')}: {item.get('price')}")
        if item.get("facilities"):
            blocks.append(f"Fasilitas {item.get('name')}: {item.get('facilities')}")
        if item.get("opening_hours"):
            blocks.append(f"Jam buka {item.get('name')}: {item.get('opening_hours')}")
    elif st == "hotel":
        blocks.append(
            f"Nama hotel: {item.get('name', '')}. "
            f"Lokasi: {item.get('location', item.get('address', ''))}. "
            f"Deskripsi: {item.get('description', '')}"
        )
        if item.get("price_per_night"):
            blocks.append(f"Harga per malam {item.get('name')}: {item.get('price_per_night')}")
    elif st == "umkm":
        blocks.append(
            f"Nama UMKM: {item.get('name', '')}. "
            f"Produk: {item.get('product', item.get('description', ''))}."
        )
        if item.get("price"):
            blocks.append(f"Harga produk {item.get('name')}: {item.get('price')}")
    else:
        blocks.append(" ".join(
            f"{k}: {v}" for k, v in item.items()
            if isinstance(v, (str, int, float))
        ))
    return [b for b in blocks if b and b.strip()]


def build_chunks(item: Dict[str, Any], chunk_size: int = DEFAULT_SIZE,
                 chunk_overlap: int = DEFAULT_OVERLAP) -> List[Dict[str, Any]]:
    """Rakit chunk dari satu record + metadata payload untuk Qdrant."""
    record_id = str(item.get("id") or item.get("record_id") or item.get("name") or "unknown")
    st = _source_type(item)
    chunks_out: List[Dict[str, Any]] = []
    global_idx = 0
    for block in _build_blocks(item):
        for text in chunk_text(block, chunk_size, chunk_overlap):
            chunks_out.append({
                "id": f"{st}:{record_id}:{global_idx}",
                "text": text,
                "payload": {
                    "source_type": st,
                    "record_id": record_id,
                    "name": item.get("name", ""),
                    "chunk_index": global_idx,
                },
            })
            global_idx += 1
    if not chunks_out:
        chunks_out.append({
            "id": f"{st}:{record_id}:0",
            "text": item.get("description", "") or str(item),
            "payload": {"source_type": st, "record_id": record_id,
                        "name": item.get("name", ""), "chunk_index": 0},
        })
    return chunks_out
