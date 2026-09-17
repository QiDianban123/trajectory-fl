"""S2 UI Streamlit page responds in a clean subprocess."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from streamlit.testing.v1 import AppTest

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


def test_streamlit_page_renders_authoritative_final_results() -> None:
    page = AppTest.from_file(PROJECT_ROOT / "src/ui/app.py", default_timeout=20).run()
    assert not page.exception
    assert page.title[0].value == "Trajectory-FL · 三模式训练与最终结果"
    metrics = {item.label: item.value for item in page.metric}
    assert metrics["项目阶段"] == "S4 · 最终结果"
    assert metrics["最终归档"] == "可用"
    assert metrics["归档文件"] == "27"
    assert metrics["最佳模式（ADE）"] == "Federated"
    assert metrics["训练样本访问"] == "10726580"
    assert metrics["最终评价样本"] == "115301"
