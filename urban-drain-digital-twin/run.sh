#!/usr/bin/env bash
# Urban Drain Digital Twin - start the dashboard
source .venv/bin/activate
echo "Dashboard: http://127.0.0.1:8000    API docs: http://127.0.0.1:8000/docs"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
