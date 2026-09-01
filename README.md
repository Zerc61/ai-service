# EJT AI Core — Microservice AI (FastAPI)

Microservice AI untuk ekosistem **East Java Traveling (EJT)**. Bertugas melakukan
**RAG (Retrieval-Augmented Generation)** memakai data yang dikirim dari Main API
(Laravel 13). Berjalan terpisah dari Laravel & Vue agar pemrosesan AI tidak
membebani API utama.

## Tech Stack (sesuai standar)
| Komponen | Pilihan |
|---|---|
| Framework | FastAPI + Uvicorn |
| Embedding | `BAAI/bge-small-en-v1.5` via `sentence-transformers` — 100% lokal, gratis, tanpa API key |
| LLM / Chat | Groq API (model env-configurable, default `qwen/qwen3.8-27b` — model `llama-3.3-70b-versatile` tidak diakses akun ini) |
| Vector DB | Qdrant (mode memory untuk MVP) |

## Struktur
```
ai-service/
├── run.py                  # jalankan: python run.py
├── main.py                 # entry kompatibel: uvicorn main:app
├── app/
│   ├── main.py             # aplikasi FastAPI + health/handshake
│   ├── config.py           # pengaturan dari env/.env
│   ├── factory.py          # rakit komponen (lazy singleton)
│   ├── deps.py             # dependency injection FastAPI
│   ├── security.py         # shared-secret server-to-server
│   ├── routers/
│   │   ├── chat.py         # POST /v1/chat/stream (SSE), POST /v1/trip/plan
│   │   ├── index.py        # POST /v1/index/* (ingestion)
│   │   └── rag.py          # POST /v1/rag/search (retrieval)
│   └── rag/
│       ├── embedder.py     # LocalEmbedder / HashEmbedder
│       ├── vectorstore.py  # QdrantVectorStore / MemoryVectorStore
│       ├── chunking.py     # chunk teks -> chunk dari record
│       ├── ingestion.py    # upsert / backfill / delete
│       ├── retrieval.py    # query -> embed -> search -> konteks
│       ├── llm.py          # BaseLLM / GroqLLM / FakeLLM
│       ├── prompts.py      # persona KAVI / RAKA / MAJA + builder prompt
│       ├── chat.py         # orkestrasi chat RAG
│       └── tripplanner.py  # Smart Trip Planner (draft)
└── tests/                  # pytest (offline, deterministik)
```

## Persona Role
Persona dipilih dari `role` yang diteruskan Laravel (token Sanctum):
| Role Sanctum | Persona | Kemampuan |
|---|---|---|
| `tourist` | KAVI | Rekomendasi destinasi, trip plan, booking, cek EJTCoin |
| `umkm` | RAKA | Copywriting produk, ringkasan performa penjualan |
| `manager` | MAJA | Analisis sentimen ulasan, eskalasi tiket CS |

## Setup & Menjalankan
```bash
cd ai-service
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # fastapi uvicorn sentence-transformers groq qdrant-client pytest python-dotenv
cp .env.example .env              # isi GROQ_API_KEY & SHARED_SECRET
python run.py                     # atau: uvicorn app.main:app --reload
```
Model embedding diunduh otomatis sekali lalu di-cache (berjalan offline setelahnya).

## Endpoint
| Method | Path | Keterangan | Auth |
|---|---|---|---|
| POST | `/v1/chat/stream` | Chat RAG streaming (SSE) | `X-AI-Secret` |
| POST | `/v1/trip/plan` | Smart Trip Planner (hasil **draft**) | `X-AI-Secret` |
| POST | `/v1/index/upsert` | Embed & simpan 1 record | `X-AI-Secret` |
| POST | `/v1/index/backfill` | Embed & simpan banyak record | `X-AI-Secret` |
| DELETE | `/v1/index/record` | Hapus chunk record | `X-AI-Secret` |
| DELETE | `/v1/index/clear` | Kosongkan koleksi | `X-AI-Secret` |
| GET | `/v1/index/stats` | Jumlah titik tersimpan | `X-AI-Secret` |
| POST | `/v1/rag/search` | Pencarian konteks vektor | - |
| GET | `/health` `/handshake` | Health check / integrasi Laravel | - |

Contoh chat streaming:
```bash
curl -N -X POST http://127.0.0.1:5001/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-AI-Secret: $SHARED_SECRET" \
  -d '{"user_id":1,"role":"tourist","message":"berapa tiket pantai balekambang?","session_id":"sess-1"}'
```
SSE mengirim `event: meta`, lalu `event: text` (delta), lalu `event: done`.

## Prinsip
- **AI tidak pernah memotong saldo/charge.** Booking = draft (Fase 4), Trip plan = draft.
- **Keamanan:** endpoint mutasi hanya menerima request internal yang sudah divalidasi
  Sanctum oleh Laravel (verifikasi `X-AI-Secret` secara konstan-waktu).

## Test
Seluruh test berjalan **offline & deterministik** (Hash embedder + Memory store + Fake LLM):
```bash
cd ai-service && venv/bin/python -m pytest
```
E2E produksi (lokal + Qdrant): sudah diverifikasi bahwa `BAAI/bge-small-en-v1.5`
memuat offline, menghasilkan vektor 384-dimensi, dan retrieval memeringkat
destinasi yang relevan paling tinggi.
