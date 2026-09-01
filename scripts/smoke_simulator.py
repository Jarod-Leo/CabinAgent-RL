"""Validate structured CAR user generations against a running simulator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.integrations.car_bench_runtime import CarBenchSession, TASK_FAMILIES, read_task_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="data/official/car-bench-dataset")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="cabinagent-user-simulator")
    parser.add_argument("--output", default="reports/simulator_smoke.json")
    args = parser.parse_args()

    dataset_root = (ROOT / args.dataset_root).resolve()
    task_ids = [read_task_rows(dataset_root, family, "train")[0]["task_id"] for family in TASK_FAMILIES]
    rows = []
    for task_id in task_ids:
        with CarBenchSession(dataset_root, task_id, "train", args.base_url, args.model) as session:
            messages, tools = session.start()
            first_user = messages[-1]["content"]
            if not first_user or first_user == "###STOP###":
                raise RuntimeError(f"Invalid first user message for {task_id}: {first_user!r}")
            rows.append({"task_id": task_id, "first_user": first_user, "tool_count": len(tools)})

    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"SIMULATOR_SMOKE_OK tasks={len(rows)} output={output}")


if __name__ == "__main__":
    main()
