"""Verify the retained three-mode final archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_MANIFEST = Path("artifacts/three_mode_training_20260916/FINAL_MANIFEST.json")


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.evaluation.archive import ArchiveIntegrityError, verify_final_archive

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    try:
        count = verify_final_archive(args.manifest)
    except ArchiveIntegrityError as exc:
        print(f"Archive verification error: {exc}", file=sys.stderr)
        return 2
    print(f"Final archive verified: {count} retained files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
