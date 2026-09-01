#!/usr/bin/env bash
# Backfill destinasi dari JSON hasil `php artisan destinations:export-ai`
# ke EJT AI Core vector store via POST /v1/index/backfill.
#
# Usage:
#   ./scripts/backfill_destinations.sh [path-ke-json] [base-url]
set -euo pipefail

# Lokasi default export (di luar repo backend) & base URL FastAPI
JSON="${1:-}"
BASE_URL="${2:-http://127.0.0.1:5001}"

# Cari JSON di beberapa lokasi umum bila tidak diberikan
if [[ -z "$JSON" ]]; then
    for cand in \
        ../backend/storage/app/ai/destinations-ai.json \
        ../../backend/storage/app/ai/destinations-ai.json \
        ./storage/app/ai/destinations-ai.json \
        ; do
        if [[ -f "$cand" ]]; then
            JSON="$cand"
            break
        fi
    done
fi

if [[ -z "$JSON" ]] || [[ ! -f "$JSON" ]]; then
    echo "ERROR: File JSON destinasi tidak ditemukan." >&2
    echo "Jalankan dulu: php artisan destinations:export-ai" >&2
    echo "Lalu: ./scripts/backfill_destinations.sh <path-json>" >&2
    exit 1
fi

# Muat shared secret dari file .env ai-service (prioritas dari env SHARED_SECRET)
if [[ -z "${SHARED_SECRET:-}" ]] && [[ -f "$(dirname "$0")/../.env" ]]; then
    SHARED_SECRET="$(grep -E '^SHARED_SECRET=' "$(dirname "$0")/../.env" | cut -d= -f2-)"
fi

if [[ -z "${SHARED_SECRET:-}" ]]; then
    echo "ERROR: SHARED_SECRET kosong. Set env SHARED_SECRET atau isi di ai-service/.env" >&2
    exit 1
fi

echo "INFO: Backfilling $(python3 -c "import json,sys;print(len(json.load(open('$JSON'))['records']))") records dari $JSON"
echo "INFO: Target: ${BASE_URL}/v1/index/backfill"

curl -sS -X POST "${BASE_URL}/v1/index/backfill" \
    -H "Content-Type: application/json" \
    -H "X-AI-Secret: ${SHARED_SECRET}" \
    --data-binary "@${JSON}"

echo
