#!/bin/bash
set -e

# Port configuration (Render passes $PORT, fallback to 8501 locally)
PORT="${PORT:-8501}"
INTERNAL_API_PORT=8000

echo "Starting FastAPI backend on internal port $INTERNAL_API_PORT..."
uvicorn app.api.main:app --host 127.0.0.1 --port $INTERNAL_API_PORT &

# Wait for FastAPI to initialize
sleep 3

echo "Starting Streamlit dashboard on public port $PORT..."
exec streamlit run ui/streamlit_app.py \
    --server.port="$PORT" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
