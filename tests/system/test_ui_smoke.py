"""S2 UI Streamlit page responds in a clean subprocess."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_streamlit_page_health() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/ui/app.py",
            "--server.headless",
            "true",
            "--server.port",
            "8502",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        last_error = ""
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen("http://localhost:8502", timeout=2) as response:
                    assert response.status == 200
                    return
            except Exception as exc:  # pragma: no cover - timing dependent
                last_error = str(exc)
                time.sleep(0.25)
        raise AssertionError(f"Streamlit page did not become ready: {last_error}")
    finally:
        process.terminate()
        process.wait(timeout=10)
