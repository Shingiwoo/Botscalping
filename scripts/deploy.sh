#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."


# Pastikan folder ada
mkdir -p logs


# Build image dan jalankan (REAL)
docker compose build newrealtrading
docker compose up -d newrealtrading


echo "\n✅ newrealtrading real mode berjalan. Lihat log: ./scripts/logs.sh"