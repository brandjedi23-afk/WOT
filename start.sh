#!/usr/bin/env bash
set -euo pipefail

echo "=== Althalus boot ==="
echo "PWD=$(pwd)"
echo "PORT=${PORT:-<unset>}"
echo "PYTHON=$(which python || true)"
echo "UVICORN=$(which uvicorn || true)"
ls -la

exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-8080}"