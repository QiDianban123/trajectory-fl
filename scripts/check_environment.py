"""Give actionable D1 environment feedback before functional modules are built."""

from __future__ import annotations

import importlib.util
import sys


REQUIRED = ("numpy", "pandas", "yaml", "matplotlib", "torch", "pytest")


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    if missing:
        print("Missing packages: " + ", ".join(missing))
        print("Run: python -m pip install -r requirements.txt")
        return 1
    print("D1 environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
