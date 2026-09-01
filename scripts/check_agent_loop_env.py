"""Validate CAR agent-loop environment interpolation inside a Ray worker."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import ray

    expected_root = str(Path(os.environ["CAR_BENCH_DATASET_ROOT"]).resolve())
    expected_url = os.environ["SIMULATOR_BASE_URL"]
    config_path = str(Path(args.config).resolve())
    ray.init(num_cpus=1, include_dashboard=False)
    try:
        @ray.remote
        def resolve_agent_loop() -> dict[str, object]:
            from omegaconf import OmegaConf

            raw = OmegaConf.load(config_path)
            resolved = OmegaConf.to_container(raw, resolve=True)
            loop = resolved[0]
            dataset_root = str(Path(loop["dataset_root"]).resolve())
            return {
                "dataset_root": dataset_root,
                "dataset_root_exists": Path(dataset_root).is_dir(),
                "simulator_base_url": loop["simulator_base_url"],
                "target": loop["_target_"],
            }

        worker = ray.get(resolve_agent_loop.remote())
    finally:
        ray.shutdown()

    passed = bool(
        worker["dataset_root"] == expected_root
        and worker["dataset_root_exists"]
        and worker["simulator_base_url"] == expected_url
        and worker["target"] == "src.training.car_bench_agent_loop.CarBenchAgentLoop"
    )
    report = {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "expected_dataset_root": expected_root,
        "expected_simulator_base_url": expected_url,
        "ray_worker": worker,
        "status": "PASS" if passed else "FAIL",
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
