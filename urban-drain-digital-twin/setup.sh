#!/usr/bin/env bash
# Urban Drain Digital Twin - one-time setup for macOS and Linux
set -e
echo "=== URBAN DRAIN DIGITAL TWIN - SETUP ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install --only-binary=:all: -r requirements.txt --quiet
python -m backend.network
python -m backend.dataset
python -m backend.train
echo
echo "Setup complete. Now run: ./run.sh"
