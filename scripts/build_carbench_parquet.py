"""Build leakage-safe veRL parquet files from the complete CAR train split."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.integrations.car_bench_runtime import TASK_FAMILIES, read_task_rows


def _row(task: dict[str, Any], family: str, source_split: str) -> dict[str, Any]:
    task_id = str(task["task_id"])
    return {
        "data_source": "car_bench",
        "prompt": [
            {
                "role": "user",
                "content": "Start the CAR-bench online environment for this task.",
            }
        ],
        "agent_name": "car_bench",
        "reward_model": {"style": "rule", "ground_truth": "1.0"},
        "extra_info": {
            "index": task_id,
            "task_id": task_id,
            "task_family": family,
            "task_type": str(task["task_type"]),
            "source_split": source_split,
        },
    }


def build_splits(dataset_root: Path, seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    for family in TASK_FAMILIES:
        family_rows = read_task_rows(dataset_root, family, "train")
        rng.shuffle(family_rows)
        dev_count = round(len(family_rows) * 0.2)
        dev_rows.extend(_row(item, family, "train") for item in family_rows[:dev_count])
        train_rows.extend(_row(item, family, "train") for item in family_rows[dev_count:])
    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    return train_rows, dev_rows


def select_gate_rows(train_rows: list[dict[str, Any]], count: int = 20) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {family: [] for family in TASK_FAMILIES}
    for row in train_rows:
        buckets[row["extra_info"]["task_family"]].append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < count and any(buckets.values()):
        for family in TASK_FAMILIES:
            if buckets[family] and len(selected) < count:
                selected.append(buckets[family].pop())
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="data/official/car-bench-dataset",
        help="Root containing tasks/*.jsonl and mock_data/.",
    )
    parser.add_argument("--output-dir", default="data/processed/carbench")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-count", type=int, default=20)
    args = parser.parse_args()

    from datasets import Dataset

    dataset_root = (ROOT / args.dataset_root).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, dev_rows = build_splits(dataset_root, args.seed)
    gate_rows = select_gate_rows(train_rows, args.gate_count)
    if len(train_rows) != 103 or len(dev_rows) != 26:
        raise RuntimeError(f"Expected CAR 103/26 split, got {len(train_rows)}/{len(dev_rows)}")

    for name, rows in (("train", train_rows), ("dev", dev_rows), ("gate", gate_rows)):
        Dataset.from_list(rows).to_parquet(str(output_dir / f"{name}.parquet"))

    manifest = {
        "seed": args.seed,
        "source": str(dataset_root),
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "gate_count": len(gate_rows),
        "train_families": dict(Counter(row["extra_info"]["task_family"] for row in train_rows)),
        "dev_families": dict(Counter(row["extra_info"]["task_family"] for row in dev_rows)),
        "gate_task_ids": [row["extra_info"]["task_id"] for row in gate_rows],
        "hidden_fields_excluded": ["persona", "instruction", "actions", "context_init_config"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"CAR_PARQUET_OK train={len(train_rows)} dev={len(dev_rows)} gate={len(gate_rows)}")


if __name__ == "__main__":
    main()
