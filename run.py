"""Jalankan server: `python run.py` (atau `uvicorn run:app --reload`).

Alasan file terpisah: menghindari import app saat `main.py` di-test/di-env
tertentu; `run` menjadi titik jalur dev yang eksplisit.
"""
from __future__ import annotations

import uvicorn

from app.main import app

app = app  # re-export agar `uvicorn run:app` valid

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=5001, reload=False)
