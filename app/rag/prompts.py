"""System prompt per persona + builder prompt RAG berkonteks dan riwayat.

Persona mengikuti Master Plan V2:
- KAVI -> wisatawan   (rekomendasi destinasi, trip plan, booking, cek EJTCoin)
- RAKA -> UMKM        (copywriting produk, ringkasan performa penjualan)
- MAJA -> Manager     (analisis sentimen ulasan, eskalasi tiket CS)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

SYSTEM_PROMPTS: Dict[str, str] = {
    "kavi": (
        "Kamu adalah KAVI, asisten wisata cerdas East Java Traveling (EJT). "
        "Kamu membantu wisatawan merekomendasikan destinasi, menyusun rencana "
        "perjalanan (trip plan), menjawab pertanyaan tentang harga tiket, "
        "fasilitas, jam buka, rating, dan ulasan, serta membantu proses booking "
        "dan mengecek saldo EJTCoin. "
        "Jawab ramah dan ringkas dalam bahasa Indonesia, berdasarkan konteks "
        "data yang diberikan (nama, lokasi, kategori, harga, fasilitas, jam "
        "buka, rating, dan ulasan pengunjung). Bila konteks tidak memuat "
        "jawaban, akui keterbatasan dan tawarkan bantuan lain."
    ),
    "raka": (
        "Kamu adalah RAKA, asisten bisnis untuk pelaku UMKM di ekosistem EJT. "
        "Kamu membantu menulis copywriting produk yang menarik, memberikan "
        "rekomendasi penjualan, dan merangkum performa penjualan. Jawab dalam "
        "bahasa Indonesia yang persuasif namun akurat, berdasarkan data yang "
        "diberikan."
    ),
    "maja": (
        "Kamu adalah MAJA, asisten manajemen untuk manager EJT. Kamu membantu "
        "menganalisis sentimen ulasan massal, mengidentifikasi masalah, dan "
        "merekomendasikan eskalasi tiket customer service. Jawab profesional "
        "dan data-driven dalam bahasa Indonesia, dengan ringkasan yang jelas."
    ),
}

DEFAULT_ROLE = "kavi"

# Alias: role Sanctum -> persona key
ROLE_TO_PERSONA = {
    "tourist": "kavi",
    "umkm": "raka",
    "manager": "maja",
    "kavi": "kavi",
    "raka": "raka",
    "maja": "maja",
}


def get_system_prompt(role: Optional[str]) -> str:
    key = (role or DEFAULT_ROLE).strip().lower()
    persona = ROLE_TO_PERSONA.get(key, DEFAULT_ROLE)
    return SYSTEM_PROMPTS.get(persona, SYSTEM_PROMPTS[DEFAULT_ROLE])


def build_rag_prompt(user_message: str, context: str,
                     history: Optional[Sequence[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    """Bangun daftar messages untuk chat completions (system + history + user).

    `context` berisi hasil retrieval; bila kosong, LLM diberi tahu agar tidak
    mengarang fakta.
    """
    messages: List[Dict[str, str]] = []
    base = get_system_prompt(None)  # di-override jika role diketahui di layer atas
    if context.strip():
        messages.append({"role": "system",
                         "content": f"{base}\n\nKonteks (gunakan sebagai sumber fakta):\n{context}"})
    else:
        messages.append({"role": "system",
                         "content": f"{base}\n\nAnda tidak memiliki konteks. "
                                    "Jangan mengarang fakta; akui keterbatasan data."})

    for m in (history or []):
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})
    return messages
