"""Audit and safely prune resumable GRPO checkpoints across Slurm processes."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STEP_PATTERN = re.compile(r"global_step_(\d+)")
REQUIRED_RELATIVE_FILES = (
    "actor/model_world_size_1_rank_0.pt",
    "actor/optim_world_size_1_rank_0.pt",
    "actor/extra_state_world_size_1_rank_0.pt",
    "actor/lora_train_meta.json",
    "data.pt",
)


def checkpoint_step(path: Path) -> int:
    match = STEP_PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError(f"Not a global-step checkpoint directory: {path}")
    return int(match.group(1))


def checkpoint_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def validate_checkpoint(path: Path) -> dict[str, object]:
    if not path.is_dir() or path.is_symlink():
        raise FileNotFoundError(f"Checkpoint directory is missing or unsafe: {path}")
    missing = [relative for relative in REQUIRED_RELATIVE_FILES if not (path / relative).is_file()]
    if missing:
        raise ValueError(f"Checkpoint {path} is incomplete; missing: {missing}")
    files = sorted(item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file())
    return {
        "step": checkpoint_step(path),
        "path": path.as_posix(),
        "file_count": len(files),
        "size_bytes": checkpoint_size(path),
        "required_files_present": True,
    }


def audit_run(run_dir: Path, expected_step: int | None = None) -> dict[str, object]:
    run_dir = run_dir.resolve()
    checkpoints = run_dir / "checkpoints"
    marker = checkpoints / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise FileNotFoundError(f"Latest-checkpoint marker is missing: {marker}")
    marker_step = int(marker.read_text(encoding="utf-8").strip())
    if expected_step is not None and marker_step != expected_step:
        raise ValueError(f"Latest marker {marker_step} does not match expected step {expected_step}")
    entries: list[dict[str, object]] = []
    incomplete_entries: list[dict[str, object]] = []
    for item in checkpoints.iterdir():
        if not STEP_PATTERN.fullmatch(item.name):
            continue
        try:
            entries.append(validate_checkpoint(item))
        except ValueError as error:
            incomplete_entries.append(
                {
                    "step": checkpoint_step(item),
                    "path": item.resolve().as_posix(),
                    "file_count": sum(1 for child in item.rglob("*") if child.is_file()),
                    "size_bytes": checkpoint_size(item),
                    "required_files_present": False,
                    "error": str(error),
                }
            )
    entries.sort(key=lambda value: int(value["step"]))
    incomplete_entries.sort(key=lambda value: int(value["step"]))
    if not entries:
        raise FileNotFoundError(f"No complete checkpoints found under {checkpoints}")
    if marker_step not in {int(item["step"]) for item in entries}:
        raise ValueError(f"Latest marker points to missing checkpoint step {marker_step}")
    return {
        "run_dir": run_dir.as_posix(),
        "marker_step": marker_step,
        "checkpoints": entries,
        "incomplete_checkpoints": incomplete_entries,
    }


def prune_run(run_dir: Path, keep_step: int, apply: bool = False) -> dict[str, object]:
    audit = audit_run(run_dir, expected_step=keep_step)
    candidates = [item for item in audit["checkpoints"] if int(item["step"]) != keep_step]
    candidates.extend(audit.get("incomplete_checkpoints", []))
    if any(int(item["step"]) > keep_step for item in candidates):
        raise ValueError("Refusing to prune a checkpoint newer than the selected keep step")
    removed: list[str] = []
    if apply:
        checkpoint_root = (run_dir.resolve() / "checkpoints")
        for item in candidates:
            target = Path(str(item["path"])).resolve()
            if target.parent != checkpoint_root or not STEP_PATTERN.fullmatch(target.name):
                raise ValueError(f"Refusing unsafe checkpoint deletion target: {target}")
            shutil.rmtree(target)
            removed.append(target.as_posix())
        after = audit_run(run_dir, expected_step=keep_step)
        remaining = after["checkpoints"]
        if (
            len(remaining) != 1
            or int(remaining[0]["step"]) != keep_step
            or after.get("incomplete_checkpoints")
        ):
            raise RuntimeError("Checkpoint pruning postcondition failed")
    return {
        **audit,
        "apply": apply,
        "prune_candidates": [item["path"] for item in candidates],
        "removed": removed,
        "status": "pruned" if apply else "audit_only",
    }


def audit_best_run(run_dir: Path) -> dict[str, object]:
    state_path = run_dir.resolve() / "metrics" / "best_checkpoint.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Best-checkpoint state is missing: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("pending_candidate") is not None:
        raise ValueError("Best-checkpoint state still has a pending candidate")
    best = state.get("best")
    if not isinstance(best, dict):
        raise ValueError("Best-checkpoint state does not contain a selected checkpoint")
    best_step = int(best["step"])
    audit = audit_run(run_dir, expected_step=best_step)
    if len(audit["checkpoints"]) != 1 or audit.get("incomplete_checkpoints"):
        raise ValueError("Best-checkpoint postcondition requires exactly one complete checkpoint")
    if int(audit["checkpoints"][0]["step"]) != best_step:
        raise ValueError("The only complete checkpoint is not the recorded best step")
    return {**audit, "best_checkpoint_state": state, "status": "best_checkpoint_verified"}


def audit_series(run_dir: Path, expected_step: int, interval: int = 50) -> dict[str, object]:
    audit = audit_run(run_dir, expected_step)
    state = json.loads((run_dir / "metrics" / "best_checkpoint.json").read_text(encoding="utf-8"))
    expected = set(range(interval, expected_step + 1, interval)) | {expected_step}
    actual = {int(row["step"]) for row in audit["checkpoints"]}
    history = state.get("history", [])
    if state.get("schema_version") != 2 or actual != expected or audit["incomplete_checkpoints"]:
        raise ValueError("Incomplete checkpoint series")
    if {int(row["step"]) for row in history} != expected:
        raise ValueError("Missing validation scores for checkpoint series")
    best = min(history, key=lambda row: (-float(row["score"]), int(row["step"])))
    if state.get("best") != best:
        raise ValueError("Incorrect best-checkpoint selection")
    return {**audit, "best_checkpoint_state": state, "status": "series_verified"}


def write_json(path: Path | None, value: dict[str, object]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(payload, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "prune", "audit-best", "audit-series"))
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    run_dir = ROOT / "experiments" / args.run_id
    if args.mode == "audit-series":
        if not args.expected_step or args.interval <= 0:
            parser.error("audit-series requires positive --expected-step and --interval")
        result = audit_series(run_dir, args.expected_step, args.interval)
    elif args.mode == "audit":
        result = audit_run(run_dir, expected_step=args.expected_step)
    elif args.mode == "audit-best":
        result = audit_best_run(run_dir)
    else:
        if args.expected_step is None:
            parser.error("prune requires --expected-step")
        result = prune_run(run_dir, args.expected_step, apply=args.apply)
    write_json(Path(args.report) if args.report else None, result)


if __name__ == "__main__":
    main()
