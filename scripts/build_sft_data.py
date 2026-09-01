"""Build SFT JSONL from successful baseline trajectories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sft_dataset import build_sft_records, split_train_val
from src.data.trajectory_schema import read_jsonl, write_jsonl


def main() -> None:
    args = parse_args()
    trajectories = []
    for input_path in args.input:
        for row in read_jsonl(input_path):
            row.setdefault("metadata", {})["source_file"] = str(input_path)
            trajectories.append(row)
    records = build_sft_records(trajectories)
    train_records, val_records = split_train_val(records, args.val_ratio, args.seed)
    write_jsonl(args.train_output, train_records)
    write_jsonl(args.val_output, val_records)
    write_report(args.report_output, len(trajectories), len(train_records), len(val_records))
    print(f"Built {len(train_records)} train and {len(val_records)} val SFT records.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        default=["data/eval_cache/all_trajectories.jsonl"],
    )
    parser.add_argument("--train-output", default="data/sft/train.jsonl")
    parser.add_argument("--val-output", default="data/sft/val.jsonl")
    parser.add_argument("--report-output", default="reports/sft_data_report.md")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_report(path: str, num_trajectories: int, num_train: int, num_val: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "\n".join(
            [
                "# SFT Data Report",
                "",
                f"- Input trajectories: {num_trajectories}",
                f"- Train records: {num_train}",
                f"- Validation records: {num_val}",
                "",
                "Records are deduplicated from successful environment trajectories and split by task.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
