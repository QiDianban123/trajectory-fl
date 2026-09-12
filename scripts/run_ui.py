"""Launch the local S2 Streamlit centralized-console."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", "src/ui/app.py"],
        cwd=root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
