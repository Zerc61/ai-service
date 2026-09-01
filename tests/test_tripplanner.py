"""Test: Smart Trip Planner."""
from __future__ import annotations

from app.rag.tripplanner import build_itinerary, validate_plan_request


def _d(name, price=0.0, loc="Malang", mid=1, score=5.0):
    return {"id": mid, "name": name, "price": price, "location": loc, "score": score}


def test_itinerary_without_budget():
    dests = [_d("Pantai Balekambang", 20000, mid=1),
             _d("Coban Rondo", 15000, mid=2)]
    plan = build_itinerary(dests, days=2)
    assert plan["status"] == "draft"
    assert plan["days"] == 2
    # dua destinasi terdistribusi ke 2 hari
    total_spots = sum(len(day["spots"]) for day in plan["itinerary"])
    assert total_spots == 2


def test_itinerary_respects_budget():
    dests = [_d("Mahal", 100000, mid=1, score=9),
             _d("Murah", 20000, mid=2, score=3)]
    plan = build_itinerary(dests, budget=50000, days=1)
    names = [s["name"] for s in plan["itinerary"][0]["spots"]]
    assert "Mahal" not in names
    assert "Murah" in names
    assert plan["estimated_cost"] <= 50000


def test_itinerary_hotel_added_within_budget():
    dests = [_d("Pantai", 20000, mid=1)]
    hotels = [{"id": 9, "name": "Hotel Tugu", "price": 30000, "score": 8}]
    plan = build_itinerary(dests, hotels=hotels, budget=50000, days=1)
    assert plan["itinerary"][0]["hotel"][0]["name"] == "Hotel Tugu"
    assert plan["estimated_cost"] <= 50000


def test_validate_plan_request():
    assert validate_plan_request({"days": 0}) is not None
    assert validate_plan_request({"days": 15}) is not None
    assert validate_plan_request({"budget": -1}) is not None
    assert validate_plan_request({"days": 3, "budget": 100, "destinations": []}) is None
