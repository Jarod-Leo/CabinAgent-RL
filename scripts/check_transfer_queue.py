"""Validate the TransferQueue dependency required by the veRL V1 trainer."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_VERSION = "0.1.7"
REQUIRED_API = (
    "init",
    "kv_batch_get",
    "kv_batch_put",
    "kv_clear",
    "kv_list",
    "kv_put",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import transfer_queue as tq
    from verl.trainer.main_ppo import TaskRunnerV1

    installed_version = importlib.metadata.version("TransferQueue")
    missing_api = [name for name in REQUIRED_API if not hasattr(tq, name)]
    status = "PASS" if installed_version == EXPECTED_VERSION and not missing_api else "FAIL"
    report = {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "expected_version": EXPECTED_VERSION,
        "installed_version": installed_version,
        "module_path": str(Path(tq.__file__).resolve()),
        "required_api": list(REQUIRED_API),
        "missing_api": missing_api,
        # @ray.remote wraps the class in ActorClass, so ordinary class
        # reflection such as __name__ is intentionally unavailable.
        "verl_task_runner_wrapper": type(TaskRunnerV1).__name__,
        "status": status,
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
