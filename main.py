"""Entry point kompatibel: `uvicorn main:app --reload` atau `python main.py`.

Implementasi sesungguhnya di `app/main.py` (paket `app`).
Port default 5001 supaya tidak bentrok dengan Laravel (8000).
"""
from __future__ import annotations

from app.main import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5001, reload=True)
