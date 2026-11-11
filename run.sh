#!/usr/bin/env bash
# PotSpot start script for macOS / Linux. Needs Python 3.10+.
set -e
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/python -m pip install -q --disable-pip-version-check -r requirements.txt
[ -f .env ] || cp .env.example .env
echo "PotSpot is starting at http://127.0.0.1:8002"
exec .venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8002
