"""The committed final archive is complete for its declared retention scope."""

from pathlib import Path

from src.evaluation.archive import verify_final_archive

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_final_archive_manifest_matches_every_retained_file() -> None:
    manifest = PROJECT_ROOT / "artifacts" / "three_mode_training_20260916" / "FINAL_MANIFEST.json"
    assert verify_final_archive(manifest) == 27
