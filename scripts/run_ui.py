"""Launch the local Trajectory-FL training and final-results console."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if importlib.util.find_spec("streamlit") is None:
        print(
            "Streamlit is not installed. Run: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2
    environment = os.environ.copy()
    matplotlib_cache = root / "outputs" / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    environment.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", "src/ui/app.py"],
        cwd=root,
        env=environment,
    )


if __name__ == "__main__":
    raise SystemExit(main())
