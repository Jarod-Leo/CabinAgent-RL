"""CAR-bench adapter shell.

The current implementation loads normalized sample JSONL. The same return
shape is intended for the official CAR-bench integration.
"""

from __future__ import annotations

from src.data.trajectory_schema import BenchmarkTask, load_tasks


DEFAULT_SAMPLE_PATH = "data/raw/carbench_sample.jsonl"


def load_carbench_tasks(path: str = DEFAULT_SAMPLE_PATH, limit: int | None = None) -> list[BenchmarkTask]:
    return load_tasks(path=path, benchmark="carbench", limit=limit)

