"""Backfill numeric veRL console metrics into one W&B run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"\bstep:(\d+)\s+-\s+(.*)")
PAIR_RE = re.compile(
    r"(?:^| - )([A-Za-z0-9_./@-]+):"
    r"(?:np\.(?:float64|int64)\()?"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\)?"
    r"(?= - |$)",
    re.IGNORECASE,
)


def parse_metric_line(line: str) -> tuple[int, dict[str, float]] | None:
    clean = ANSI_RE.sub("", line).strip()
    match = STEP_RE.search(clean)
    if not match:
        return None
    step = int(match.group(1))
    metrics = {name: float(value) for name, value in PAIR_RE.findall(match.group(2))}
    metrics["training/global_step"] = float(step)
    return step, metrics


def collect_metrics(log_paths: list[Path]) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for log_path in log_paths:
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parsed = parse_metric_line(line)
                if parsed is None:
                    continue
                step, metrics = parsed
                rows.setdefault(step, {}).update(metrics)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project", default="CabinAgent-RL")
    parser.add_argument("--name", required=True)
    parser.add_argument("--group", default="fallback-grpo-qwen2.5-7b")
    parser.add_argument("--entity")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log_paths = sorted((args.run_dir / "logs").glob("trainer-*.log"))
    if not log_paths:
        raise FileNotFoundError(f"No trainer logs found under {args.run_dir / 'logs'}")
    rows = collect_metrics(log_paths)
    if not rows:
        raise ValueError("No numeric step metrics were parsed")

    summary = {
        "log_files": [path.name for path in log_paths],
        "first_step": min(rows),
        "last_step": max(rows),
        "step_count": len(rows),
        "metric_count": len({key for metrics in rows.values() for key in metrics}),
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    import wandb

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        group=args.group,
        job_type="grpo-backfill",
        tags=["F10", "vanilla", "fallback", "seed-42", "historical-backfill"],
        config={
            "experiment_id": "F10",
            "method": "Vanilla GRPO",
            "seed": 42,
            "source": "verl-console-backfill",
        },
    )
    run.define_metric("*", step_metric="training/global_step")
    for step in sorted(rows):
        run.log(rows[step])
    for key, value in summary.items():
        run.summary[f"backfill/{key}"] = value
    run.finish()
    print(json.dumps({**summary, "wandb_run_url": run.url}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
