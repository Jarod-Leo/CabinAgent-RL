"""BFCL adapter shell for normalized sample JSONL."""

from __future__ import annotations

from src.data.trajectory_schema import BenchmarkTask, load_tasks


DEFAULT_SAMPLE_PATH = "data/raw/bfcl_sample.jsonl"


def load_bfcl_tasks(path: str = DEFAULT_SAMPLE_PATH, limit: int | None = None) -> list[BenchmarkTask]:
    return load_tasks(path=path, benchmark="bfcl", limit=limit)

