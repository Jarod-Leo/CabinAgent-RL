"""Report generation for baseline evaluation."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from src.eval.failure_taxonomy import FAILURE_TYPES, count_failures


METRIC_COLUMNS = [
    "success",
    "tool_accuracy",
    "tool_name_accuracy",
    "argument_accuracy",
    "executable_tool_rate",
    "hallucination_rate",
    "state_consistency",
    "disambiguation_success",
    "avg_turns",
]


def summarize_trajectories(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    if not trajectories:
        return {"num_samples": 0, **{metric: 0.0 for metric in METRIC_COLUMNS}}

    summary: dict[str, Any] = {"num_samples": len(trajectories)}
    for metric in METRIC_COLUMNS:
        values = [float(t.get("metrics", {}).get(metric, 0.0)) for t in trajectories]
        summary[metric] = mean(values) if values else 0.0
    summary["num_failures"] = sum(1 for t in trajectories if t.get("failures"))
    return summary


def group_by_benchmark(trajectories: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        grouped[str(trajectory.get("benchmark", "unknown"))].append(trajectory)
    return dict(grouped)


def write_benchmark_report(
    benchmark: str,
    trajectories: list[dict[str, Any]],
    output_path: str | Path,
    adapter_name: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_trajectories(trajectories)
    failures = count_failures(trajectories)

    lines = [
        f"# {benchmark.upper()} Baseline Report",
        "",
        "## Run",
        "",
        f"- Adapter: `{adapter_name}`",
        f"- Samples: {summary['num_samples']}",
        f"- Samples with failures: {summary['num_failures']}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for metric in METRIC_COLUMNS:
        lines.append(f"| {metric} | {summary[metric]:.4f} |")

    lines.extend(["", "## Failure Counts", "", "| Code | Count | Meaning |", "|---|---:|---|"])
    if failures:
        for code, count in failures.most_common():
            lines.append(f"| {code} | {count} | {FAILURE_TYPES.get(code, 'Unknown')} |")
    else:
        lines.append("| none | 0 | No failures in this run. |")

    lines.extend(["", "## Notes", ""])
    lines.append("These are local sample smoke-test results, not official benchmark scores.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_eval_summary_csv(
    trajectories: list[dict[str, Any]],
    output_path: str | Path = "reports/eval_summary.csv",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped = group_by_benchmark(trajectories)
    columns = ["benchmark", "model", "num_samples", *METRIC_COLUMNS, "num_failures"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for benchmark, items in sorted(grouped.items()):
            summary = summarize_trajectories(items)
            row = {"benchmark": benchmark, "model": "local_rules", **summary}
            writer.writerow(row)


def write_failure_taxonomy_report(
    trajectories: list[dict[str, Any]],
    output_path: str | Path = "reports/failure_taxonomy.md",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures = count_failures(trajectories)

    examples: dict[str, list[str]] = defaultdict(list)
    for trajectory in trajectories:
        for item in trajectory.get("failures", []):
            code = item.get("code", "UNKNOWN")
            if len(examples[code]) < 3:
                examples[code].append(f"{trajectory.get('benchmark')}:{trajectory.get('id')} - {item.get('detail')}")

    lines = [
        "# Failure Taxonomy",
        "",
        "This file is generated from local sample baseline trajectories.",
        "",
        "| Code | Count | Meaning | Example |",
        "|---|---:|---|---|",
    ]
    for code, meaning in FAILURE_TYPES.items():
        example = "<br>".join(examples.get(code, [])) or "-"
        lines.append(f"| {code} | {failures.get(code, 0)} | {meaning} | {example} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

