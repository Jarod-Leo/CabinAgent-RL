"""Validate Ray/veRL GPU visibility after clearing AMD variables on NVIDIA."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


VISIBLE_KEYS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


def snapshot() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in VISIBLE_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    driver_env = snapshot()
    if not driver_env["CUDA_VISIBLE_DEVICES"]:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be assigned by Slurm")
    if driver_env["HIP_VISIBLE_DEVICES"] or driver_env["ROCR_VISIBLE_DEVICES"]:
        raise RuntimeError(f"AMD visibility leaked into driver: {driver_env}")

    import ray

    ray.init(num_cpus=1, num_gpus=1, include_dashboard=False)
    try:
        @ray.remote(num_gpus=1)
        def probe_worker() -> dict[str, object]:
            before = snapshot()
            from verl.single_controller.base.worker import Worker

            worker = object.__new__(Worker)
            Worker._setup_env_cuda_visible_devices(worker)
            return {"before_verl_hook": before, "after_verl_hook": snapshot()}

        worker_env = ray.get(probe_worker.remote())
    finally:
        ray.shutdown()

    before_hook = worker_env["before_verl_hook"]
    after_hook = worker_env["after_verl_hook"]
    passed = all(
        env["CUDA_VISIBLE_DEVICES"]
        and not env["HIP_VISIBLE_DEVICES"]
        and not env["ROCR_VISIBLE_DEVICES"]
        for env in (before_hook, after_hook)
    )
    report = {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "driver_env": driver_env,
        "ray_worker_env": worker_env,
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
