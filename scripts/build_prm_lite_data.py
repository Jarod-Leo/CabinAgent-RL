"""Build PRM-Lite debug reward rows from trajectories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.trajectory_schema import read_jsonl, write_jsonl
from src.rewards.prm_lite import build_prm_lite_debug_rows


def main() -> None:
    args = parse_args()
    trajectories = read_jsonl(args.input)
    rows = build_prm_lite_debug_rows(trajectories)
    write_jsonl(args.output, rows)
    write_report(args.report_output, len(rows))
    print(f"Built {len(rows)} PRM-Lite debug rows.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/eval_cache/all_trajectories.jsonl")
    parser.add_argument("--output", default="data/reward/prm_lite_debug.jsonl")
    parser.add_argument("--report-output", default="reports/prm_lite_report.md")
    return parser.parse_args()


def write_report(path: str, num_rows: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "\n".join(
            [
                "# PRM-Lite Report",
                "",
                f"- Debug rows: {num_rows}",
                "",
                "Each row contains deterministic component scores and explanation reasons.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
