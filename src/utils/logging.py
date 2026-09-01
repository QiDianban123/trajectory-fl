"""Run-scoped JSON logging with no dependency on a specific training backend."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path


class JsonRunFormatter(logging.Formatter):
    """Serialize standard and run-specific logging fields as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("run_id", "split_id", "round_index", "client_id"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class RunContextFilter(logging.Filter):
    """Attach immutable run context to records that do not provide it explicitly."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "run_id", None) is None:
            record.run_id = self.run_id
        return True


def configure_run_logger(
    run_id: str, log_path: str | Path, *, level: int = logging.INFO
) -> logging.Logger:
    """Return an idempotent dedicated logger writing structured JSON lines to ``log_path``."""

    path = Path(log_path)
    if not path.name or not path.suffix:
        raise ValueError("log_path must include a filename with an extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"trajectory_fl.run.{run_id}")
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        if getattr(handler, "_trajectory_fl_run_handler", False):
            logger.removeHandler(handler)
            handler.close()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler._trajectory_fl_run_handler = True  # type: ignore[attr-defined]
    handler.addFilter(RunContextFilter(run_id))
    handler.setFormatter(JsonRunFormatter())
    logger.addHandler(handler)
    return logger
