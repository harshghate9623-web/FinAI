"""
findoc-ai/run_app.py
Unified local runner for FinDoc AI (FastAPI backend + Streamlit UI).
Automatically coordinates startup order and handles clean shutdown.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def wait_for_api(url: str = "http://127.0.0.1:8000/health", timeout: int = 30) -> bool:
    print("Waiting for FastAPI backend to initialize...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HealthCheck"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print(" FastAPI backend is healthy and responding.")
                    return True
        except Exception:
            time.sleep(1)
    print("! Timed out waiting for FastAPI to become ready, starting Streamlit anyway...")
    return False


def main() -> None:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)
    sys.path.insert(0, root_dir)

    python_exe = sys.executable

    print("=" * 60)
    print("  Starting FinDoc AI Local Services")
    print("=" * 60)
    print(f"Working Directory: {root_dir}")
    print(f"Python Executable: {python_exe}\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = root_dir

    # 1. Start FastAPI backend
    fastapi_cmd = [
        python_exe,
        "-m",
        "uvicorn",
        "app.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    print("[1/2] Launching FastAPI backend on http://127.0.0.1:8000 ...")
    api_proc = subprocess.Popen(fastapi_cmd, env=env)

    # Wait for API to become ready
    wait_for_api()

    # 2. Start Streamlit UI
    streamlit_cmd = [
        python_exe,
        "-m",
        "streamlit",
        "run",
        os.path.join(root_dir, "ui", "streamlit_app.py"),
        "--server.port=8501",
        "--server.address=127.0.0.1",
        "--browser.gatherUsageStats=false",
    ]
    print("[2/2] Launching Streamlit UI on http://localhost:8501 ...\n")
    print("-" * 60)
    print(" FinDoc AI is running locally:")
    print("  * Streamlit Dashboard: http://localhost:8501")
    print("  * FastAPI API Docs:    http://127.0.0.1:8000/docs")
    print("  * Health Endpoint:     http://127.0.0.1:8000/health")
    print(" Press Ctrl+C to terminate both services.")
    print("-" * 60 + "\n")

    ui_proc = subprocess.Popen(streamlit_cmd, env=env)

    def shutdown(sig=None, frame=None):
        print("\nShutting down FinDoc AI local services...")
        try:
            ui_proc.terminate()
        except Exception:
            pass
        try:
            api_proc.terminate()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            api_poll = api_proc.poll()
            ui_poll = ui_proc.poll()
            if api_poll is not None:
                print(f"! FastAPI backend process exited with code {api_poll}")
                break
            if ui_poll is not None:
                print(f"! Streamlit UI process exited with code {ui_poll}")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()

