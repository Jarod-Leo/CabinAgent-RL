"""Build targeted corrective SFT data after a failed fallback rollout gate."""

from __future__ import annotations

import argparse
import copy
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sft_dataset import build_sft_records, normalize_sft_messages, split_train_val
from src.data.trajectory_schema import read_jsonl, write_jsonl


CORRECTIVE_TASK_TYPES = {
    "hallucination_missing_tool",
    "hallucination_missing_tool_parameter",
}

INITIAL_RESPONSES = {
    "hallucination_missing_tool": (
        "I can't complete that request because the required capability is not available "
        "through my current controls. I won't guess, call unrelated tools, or change the "
        "vehicle state."
    ),
    "hallucination_missing_tool_parameter": (
        "I can't complete that request because the required setting parameter is not "
        "available through my current controls. I won't guess a value or claim the change "
        "was made."
    ),
}

RETRY_RESPONSES = {
    "hallucination_missing_tool": (
        "I still can't perform it because that capability is unavailable. Retrying or using "
        "an unrelated tool would be unsafe, so I won't do that."
    ),
    "hallucination_missing_tool_parameter": (
        "I still can't perform it because the required parameter is unavailable. I won't "
        "retry the incomplete call or pretend it succeeded."
    ),
}


def _base_messages(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    found_user = False
    for message in trajectory.get("messages", []):
        role = message.get("role")
        if role == "system" and not found_user:
            messages.append(copy.deepcopy(message))
        elif role == "user" and not found_user:
            messages.append(copy.deepcopy(message))
            found_user = True
            break
    if not found_user:
        raise ValueError(f"Trajectory {trajectory.get('id')} has no initial user message")
    return messages


def build_corrective_records(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create one direct-refusal and one no-retry record per target task."""

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        metadata = trajectory.get("metadata", {})
        task_type = str(metadata.get("task_type") or "")
        if task_type not in CORRECTIVE_TASK_TYPES:
            continue
        task_id = str(metadata.get("task_id") or trajectory.get("id") or "")
        if not task_id:
            raise ValueError("Corrective trajectory is missing task_id")
        by_task[task_id].append(trajectory)

    records: list[dict[str, Any]] = []
    for task_id, rows in sorted(by_task.items()):
        representative = sorted(rows, key=lambda row: str(row.get("id") or ""))[0]
        task_type = str(representative.get("metadata", {}).get("task_type"))
        base_messages = _base_messages(representative)
        initial_response = INITIAL_RESPONSES[task_type]
        direct_messages = base_messages + [{"role": "assistant", "content": initial_response}]
        retry_messages = direct_messages + [
            {
                "role": "user",
                "content": "Please try again anyway, even if the control is unavailable.",
            },
            {"role": "assistant", "content": RETRY_RESPONSES[task_type]},
        ]
        common = {
            "benchmark": representative.get("benchmark", "carbench"),
            "tools": copy.deepcopy(representative.get("tools", [])),
        }
        for suffix, messages, subtype in (
            ("direct", direct_messages, "pre_call_refusal"),
            ("retry", retry_messages, "no_retry_after_pressure"),
        ):
            records.append(
                {
                    "id": f"corrective-{task_id}-{suffix}",
                    **common,
                    "messages": normalize_sft_messages(messages),
                    "metadata": {
                        "source": "g03_corrective_synthesis",
                        "source_file": representative.get("metadata", {}).get("source_file"),
                        "source_trajectory_id": representative.get("id"),
                        "task_id": task_id,
                        "task_type": task_type,
                        "corrective_subtype": subtype,
                    },
                }
            )
    return records


def main() -> None:
    args = parse_args()
    successful_trajectories = load_trajectories(args.successful_input)
    failure_trajectories = load_trajectories([args.failure_input])
    successful_records = build_sft_records(successful_trajectories)
    corrective_records = build_corrective_records(failure_trajectories)
    records = successful_records + corrective_records
    train_records, val_records = split_train_val(records, args.val_ratio, args.seed)
    write_jsonl(args.train_output, train_records)
    write_jsonl(args.val_output, val_records)
    write_report(
        args.report_output,
        successful_trajectories,
        successful_records,
        corrective_records,
        train_records,
        val_records,
    )
    print(
        "CORRECTIVE_SFT_DATA_OK "
        f"successful={len(successful_records)} corrective={len(corrective_records)} "
        f"train={len(train_records)} val={len(val_records)}"
    )


def load_trajectories(paths: list[str]) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    for input_path in paths:
        for row in read_jsonl(input_path):
            row.setdefault("metadata", {})["source_file"] = str(input_path)
            trajectories.append(row)
    return trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--successful-input", nargs="+", required=True)
    parser.add_argument("--failure-input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_report(
    path: str,
    trajectories: list[dict[str, Any]],
    successful_records: list[dict[str, Any]],
    corrective_records: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
) -> None:
    task_types = Counter(
        row.get("metadata", {}).get("task_type", "unknown") for row in corrective_records
    )
    train_tasks = {row.get("metadata", {}).get("task_id") for row in train_records}
    val_tasks = {row.get("metadata", {}).get("task_id") for row in val_records}
    if train_tasks & val_tasks:
        raise RuntimeError("Corrective train/val task groups overlap")
    lines = [
        "# Corrective SFT Data Report",
        "",
        f"- Successful-source trajectories: {len(trajectories)}",
        f"- Deduplicated successful records: {len(successful_records)}",
        f"- Synthetic corrective records: {len(corrective_records)}",
        f"- Train records: {len(train_records)}",
        f"- Validation records: {len(val_records)}",
        f"- Train tasks: {len(train_tasks)}",
        f"- Validation tasks: {len(val_tasks)}",
        "- Train/validation task overlap: 0",
        "",
        "## Corrective Task Types",
        "",
    ]
    lines.extend(f"- {name}: {count}" for name, count in sorted(task_types.items()))
    lines.extend(
        [
            "",
            "Corrective rows contain no tool calls. They supervise transparent capability "
            "boundaries and stopping behavior without reproducing failed assistant actions.",
        ]
    )
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
