import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_LOG_DIR: Path | None = None


def _get_log_dir() -> Path:
    global _LOG_DIR
    if _LOG_DIR is None:
        data_dir = os.environ.get("DATA_DIR", "astar_island/data")
        _LOG_DIR = Path(data_dir) / "logs"
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR


class _JsonlHandler(logging.Handler):
    """Appends one JSON line per log record to a JSONL file."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    def emit(self, record: logging.LogRecord) -> None:
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    try:
        log_file = _get_log_dir() / "astar_island.jsonl"
        fh = _JsonlHandler(log_file)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    except Exception:
        pass

    logger.propagate = False
    return logger
