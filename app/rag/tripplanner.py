"""Smart Trip Planner: susun itinerary dari daftar destinasi/hotel.

Algoritma greedy sederhana yang menghormati:
- durasi (jumlah hari),
- lokasi awal (start city optional),
- batas budget (0 = tanpa batas),
- preferensi pengguna (opsional).

! Prinsip EJT: hasil rencana SELALU draft, tidak pernah memotong saldo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class DirectorItem:
    id: str
    name: str
    price: float = 0.0
    location: str = ""
    score: float = 0.0


def _to_item(d: Dict[str, Any]) -> DirectorItem:
    price = float(d.get("price") or d.get("price_per_night") or 0.0)
    return DirectorItem(
        id=str(d.get("id") or d.get("record_id") or d.get("name") or "?"),
        name=str(d.get("name", "")),
        price=price,
        location=str(d.get("location", d.get("address", ""))),
        score=float(d.get("score", d.get("popularity", 0)) or 0.0),
    )


def build_itinerary(destinations: Sequence[Dict[str, Any]],
                    hotels: Sequence[Dict[str, Any]] = (),
                    days: int = 1, budget: float = 0.0,
                    start_city: Optional[str] = None,
                    preferences: Optional[List[str]] = None) -> Dict[str, Any]:
    """Susun itinerary berdasarkan data yang tersedia (greedy).

    Return dict draft yang aman untuk disimpan (status='draft').
    """
    days = max(1, int(days))
    dests = sorted((_to_item(d) for d in destinations), key=lambda x: x.score,
                   reverse=True)

    # filter by start_city bila diminta (longgar: cocok substring)
    if start_city:
        city_l = start_city.lower()
        scored_in: List[DirectorItem] = []
        scored_out: List[DirectorItem] = []
        for d in dests:
            (scored_in if city_l in d.location.lower() else scored_out).append(d)
        dests = scored_in + scored_out if (scored_in or not scored_out) else scored_out

    # budget constraint (greedy pakai destination dulu)
    chosen: List[DirectorItem] = []
    running = 0.0
    for d in dests:
        if budget > 0 and running + d.price > budget:
            continue
        chosen.append(d)
        running += d.price

    if budget > 0:
        # sisa budget bisa untuk hotel
        chosen_hotels: List[DirectorItem] = []
        for h in sorted((_to_item(x) for x in hotels), key=lambda x: x.score,
                        reverse=True):
            if running + h.price > budget:
                continue
            chosen_hotels.append(h)
            running += h.price
    else:
        chosen_hotels = [_to_item(x) for x in hotels]

    # alokasi tiap hari (round-robin ke destinasi pilihan)
    daily: List[List[DirectorItem]] = [[] for _ in range(days)]
    for i, d in enumerate(chosen):
        daily[i % days].append(d)

    return {
        "status": "draft",
        "days": days,
        "start_city": start_city,
        "budget": budget,
        "preferences": preferences or [],
        "estimated_cost": round(running, 2),
        "itinerary": [
            {
                "day": i + 1,
                "spots": [{"id": d.id, "name": d.name, "location": d.location,
                           "price": d.price} for d in slot],
                "hotel": [{"id": h.id, "name": h.name, "location": h.location,
                           "price": h.price} for h in chosen_hotels]
                         if i == 0 else [],
            }
            for i, slot in enumerate(daily)
        ],
    }


def validate_plan_request(payload: Dict[str, Any]) -> Optional[str]:
    """Validasi ringan. Return pesan error, atau None bila valid."""
    days = payload.get("days")
    if days is not None and (not isinstance(days, int) or days < 1 or days > 14):
        return "days harus bilangan bulat 1-14"
    budget = payload.get("budget")
    if budget is not None and (not isinstance(budget, (int, float)) or budget < 0):
        return "budget tidak boleh negatif"
    dests = payload.get("destinations")
    if dests is not None and not isinstance(dests, list):
        return "destinations harus berupa list"
    return None
