#!/bin/bash
# 1. Start FastAPI in the background on internal port 8000
uvicorn app.api.main:app --host 127.0.0.1 --port 8000 &

# 2. Start Streamlit on Render's public port ($PORT)
streamlit run ui/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
