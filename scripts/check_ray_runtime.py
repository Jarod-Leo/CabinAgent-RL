"""Start and stop a minimal local Ray runtime to validate its socket path."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ray_tmpdir = Path(os.environ["RAY_TMPDIR"])
    if not str(ray_tmpdir).startswith("/tmp/cabin-ray-"):
        raise ValueError(f"Expected short Job-ID-scoped Ray path, got {ray_tmpdir}")

    import ray

    context = ray.init(num_cpus=1, include_dashboard=False)
    try:
        @ray.remote
        def identity(value: int) -> int:
            return value

        result = ray.get(identity.remote(42))
        report = {
            "schema_version": 1,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "ray_tmpdir": str(ray_tmpdir),
            "ray_address": context.address_info.get("address"),
            "session_dir": context.address_info.get("session_dir"),
            "remote_result": result,
            "status": "PASS" if result == 42 else "FAIL",
        }
    finally:
        ray.shutdown()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
