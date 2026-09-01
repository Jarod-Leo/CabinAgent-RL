"""Run the local prompt baseline over sample CAR-bench/BFCL tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.adapters.bfcl_adapter import DEFAULT_SAMPLE_PATH as BFCL_SAMPLE_PATH
from src.adapters.bfcl_adapter import load_bfcl_tasks
from src.adapters.carbench_adapter import DEFAULT_SAMPLE_PATH as CAR_SAMPLE_PATH
from src.adapters.carbench_adapter import load_carbench_tasks
from src.adapters.model_adapter import create_model_adapter
from src.data.trajectory_schema import (
    BenchmarkTask,
    Trajectory,
    write_json,
    write_jsonl,
)
from src.eval.metrics import score_task
from src.eval.report_builder import (
    write_benchmark_report,
    write_eval_summary_csv,
    write_failure_taxonomy_report,
)


DEFAULTS = {
    "carbench": {
        "data": CAR_SAMPLE_PATH,
        "trajectory": "data/eval_cache/carbench_trajectories.jsonl",
        "report": "reports/baseline_carbench.md",
    },
    "bfcl": {
        "data": BFCL_SAMPLE_PATH,
        "trajectory": "data/eval_cache/bfcl_trajectories.jsonl",
        "report": "reports/baseline_bfcl.md",
    },
}


def main() -> None:
    args = parse_args()
    adapter = create_model_adapter(args.adapter)
    all_trajectories: list[dict] = []

    benchmarks = ["carbench", "bfcl"] if args.benchmark == "all" else [args.benchmark]
    for benchmark in benchmarks:
        data_path = args.data_path or DEFAULTS[benchmark]["data"]
        tasks = load_tasks_for_benchmark(benchmark, data_path, args.limit)
        trajectories = run_tasks(tasks, adapter_name=args.adapter, failure_dir=args.failure_dir)
        write_jsonl(DEFAULTS[benchmark]["trajectory"], trajectories)
        write_benchmark_report(
            benchmark=benchmark,
            trajectories=trajectories,
            output_path=DEFAULTS[benchmark]["report"],
            adapter_name=adapter.name,
        )
        all_trajectories.extend(trajectories)

    if args.benchmark == "all":
        write_jsonl("data/eval_cache/all_trajectories.jsonl", all_trajectories)

    write_eval_summary_csv(all_trajectories, args.summary_output)
    write_failure_taxonomy_report(all_trajectories, args.failure_taxonomy_output)
    print(
        f"Completed {len(all_trajectories)} sample trajectories with adapter '{adapter.name}'. "
        f"Reports written under reports/."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["carbench", "bfcl", "all"], default="all")
    parser.add_argument("--adapter", default="local_rules")
    parser.add_argument("--data-path", default=None, help="Override data path for a single benchmark run.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--failure-dir", default="failure_cases/baseline")
    parser.add_argument("--summary-output", default="reports/eval_summary.csv")
    parser.add_argument("--failure-taxonomy-output", default="reports/failure_taxonomy.md")
    return parser.parse_args()


def load_tasks_for_benchmark(benchmark: str, data_path: str, limit: int | None) -> list[BenchmarkTask]:
    if benchmark == "carbench":
        return load_carbench_tasks(path=data_path, limit=limit)
    if benchmark == "bfcl":
        return load_bfcl_tasks(path=data_path, limit=limit)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def run_tasks(tasks: list[BenchmarkTask], adapter_name: str, failure_dir: str) -> list[dict]:
    adapter = create_model_adapter(adapter_name)
    trajectories: list[dict] = []
    for task in tasks:
        response = adapter.generate(task.messages, task.tools, task.metadata)
        metrics, failures, tool_results = score_task(task, response)
        trajectory = Trajectory(
            id=task.id,
            benchmark=task.benchmark,
            split=task.split,
            messages=task.messages,
            tools=task.tools,
            expected_tool_calls=task.expected_tool_calls,
            model_response=response.to_dict(),
            predicted_tool_calls=response.to_dict()["tool_calls"],
            tool_results=tool_results,
            metrics=metrics,
            failures=failures,
            metadata=task.metadata,
        ).to_dict()
        trajectories.append(trajectory)

        if failures:
            output_name = f"{task.benchmark}_{task.id}.json".replace("/", "_")
            write_json(Path(failure_dir) / output_name, trajectory)

    return trajectories


if __name__ == "__main__":
    main()

