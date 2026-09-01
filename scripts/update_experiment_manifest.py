"""Atomically record Slurm lifecycle metadata for an experiment run."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def with_lifecycle_metadata(
    manifest: dict[str, object],
    status: str,
    *,
    slurm_job_id: str | None = None,
    node_list: str | None = None,
) -> dict[str, object]:
    updated = dict(manifest)
    updated["status"] = status
    updated["updated_at"] = datetime.now(timezone.utc).isoformat()
    if slurm_job_id:
        updated["slurm_job_id"] = slurm_job_id
    if node_list:
        updated["slurm_node_list"] = node_list
    return updated


def update_manifest(
    run_id: str,
    status: str,
    *,
    slurm_job_id: str | None = None,
    node_list: str | None = None,
) -> Path:
    run_dir = (ROOT / "experiments" / run_id).resolve()
    experiments_root = (ROOT / "experiments").resolve()
    if experiments_root not in run_dir.parents:
        raise ValueError(f"Run directory escapes experiments root: {run_dir}")

    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("run_id") != run_id:
        raise ValueError(f"Manifest run_id mismatch in {manifest_path}")
    manifest = with_lifecycle_metadata(
        manifest,
        status,
        slurm_job_id=slurm_job_id,
        node_list=node_list,
    )

    temp_path = manifest_path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temp_path, manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", required=True, choices=("submitted", "running", "completed", "failed"))
    parser.add_argument("--slurm-job-id")
    parser.add_argument("--node-list")
    args = parser.parse_args()
    print(
        update_manifest(
            args.run_id,
            args.status,
            slurm_job_id=args.slurm_job_id,
            node_list=args.node_list,
        )
    )


if __name__ == "__main__":
    main()
