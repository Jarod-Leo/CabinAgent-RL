"""Small training-phase helpers shared by future scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.trajectory_schema import read_jsonl


def require_file(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Required file does not exist: {resolved}")
    return resolved


def load_jsonl_dataset(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(require_file(path))

