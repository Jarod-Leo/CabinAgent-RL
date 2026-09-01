"""Create an immutable-by-convention experiment run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
HASH_ROOTS = ("configs", "scripts", "src", "requirements-gpu.txt", "Project.md")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest() -> str:
    digest = hashlib.sha256()
    files: list[Path] = []
    for item in HASH_ROOTS:
        path = ROOT / item
        files.extend(sorted(path.rglob("*")) if path.is_dir() else [path])
    for path in files:
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", default="experiments")
    args = parser.parse_args()

    config = (ROOT / args.config).resolve()
    if not config.is_file() or ROOT not in config.parents:
        raise FileNotFoundError(f"Config must be inside the project: {config}")
    run_dir = (ROOT / args.run_root / args.run_id).resolve()
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    for child in ("logs", "checkpoints", "eval", "metrics"):
        (run_dir / child).mkdir(parents=True, exist_ok=False)
    shutil.copy2(config, run_dir / "config.yaml")
    config_value = yaml.safe_load(config.read_text(encoding="utf-8"))
    parent_config = config_value.get("parent_config") if isinstance(config_value, dict) else None
    parent_path = (ROOT / str(parent_config)).resolve() if parent_config else None
    if parent_path:
        shutil.copy2(parent_path, run_dir / parent_path.name)

    manifest = {
        "run_id": args.run_id,
        "status": "created",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_source": config.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config),
        "source_sha256": source_digest(),
        "host": platform.node(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if parent_path:
        manifest["parent_config_source"] = parent_path.relative_to(ROOT).as_posix()
        manifest["parent_config_sha256"] = sha256_file(parent_path)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(run_dir)


if __name__ == "__main__":
    main()
