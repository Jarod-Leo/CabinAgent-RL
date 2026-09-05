"""Save at veRL's safe pre-rollout boundary; select best without pruning."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from scripts.checkpoint_policy import audit_run, validate_checkpoint

SCHEMA_VERSION = 2
DEFAULT_METRIC_KEY = "val-core/car_bench/reward/mean@1"


def candidate_is_better(score: float, best: dict | None) -> bool:
    return best is None or score > float(best["score"])


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class BestCheckpointController:
    """Publish complete checkpoints before on_step_end wakes policy rollout.

    Disk I/O remains synchronous. Failed saves leave the previous recovery
    point intact. Validation never invokes save or deletes a checkpoint.
    """

    def __init__(self, trainer: Any, metric_key: str = DEFAULT_METRIC_KEY,
                 state_path: Path | None = None) -> None:
        cfg = trainer.config.trainer
        if int(cfg.save_freq) <= 0 or int(cfg.save_freq) != int(cfg.test_freq):
            raise ValueError("Positive matching save/test frequencies are required")
        if str(cfg.v1.trainer_mode) != "sync":
            raise ValueError("Pre-rollout saving requires the sync trainer")
        self.trainer = trainer
        self.root = Path(str(cfg.default_local_dir)).resolve()
        self.run_dir = self.root.parent
        self.state_path = state_path or self.run_dir / "metrics" / "best_checkpoint.json"
        self.metric_key = metric_key
        self.original_save = trainer._save_checkpoint
        self.original_validate = trainer._validate
        self.state = {
            "schema_version": SCHEMA_VERSION, "metric_key": metric_key,
            "comparison": "strict_greater", "tie_break": "earlier",
            "retention": "all_until_review", "baseline": None,
            "best": None, "history": [],
        }
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Use a new run for the changed checkpoint protocol")
            if self.state.get("metric_key") != metric_key:
                raise ValueError("Selection metric changed")
        if (self.root / "latest_checkpointed_iteration.txt").exists():
            audit_run(self.run_dir)

    def save_checkpoint(self) -> None:
        step = int(self.trainer.global_steps)
        target = self.root / f"global_step_{step}"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {target}")
        staging = self.root / f".saving_step_{step}"
        suffix = 0
        while staging.exists():
            suffix += 1
            staging = self.root / f".saving_step_{step}_{suffix}"
        staging.mkdir(parents=True)
        cfg = self.trainer.config.trainer
        original_root = cfg.default_local_dir
        try:
            cfg.default_local_dir = str(staging)
            self.original_save()
            candidate = staging / target.name
            validate_checkpoint(candidate)
            os.replace(candidate, target)
            marker = self.root / "latest_checkpointed_iteration.txt"
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(str(step) + "\n", encoding="utf-8")
            os.replace(temporary, marker)
            (staging / marker.name).unlink(missing_ok=True)
            staging.rmdir()
        finally:
            cfg.default_local_dir = original_root

    def validate_and_record(self) -> dict[str, float]:
        metrics = self.original_validate()
        score = float(metrics[self.metric_key])
        if not math.isfinite(score):
            raise ValueError(f"Nonfinite validation score: {score}")
        step = int(self.trainer.global_steps)
        if step == 0:
            self.state["baseline"] = {"step": 0, "score": score}
        else:
            validate_checkpoint(self.root / f"global_step_{step}")
            # Resume may validate an already scored checkpoint: retain its
            # original selection score, while still reporting the new metrics.
            history = self.state["history"]
            if not any(int(row["step"]) == step for row in history):
                history.append({"step": step, "score": score})
            best = min(history, key=lambda row: (-float(row["score"]), int(row["step"])))
            self.state["best"] = dict(best)
            metrics.update({
                "checkpoint-selection/candidate_score": score,
                "checkpoint-selection/best_score": float(best["score"]),
                "checkpoint-selection/best_step": float(best["step"]),
                "checkpoint-selection/selected": float(int(best["step"]) == step),
            })
        write_state(self.state_path, self.state)
        return metrics

    def install(self) -> None:
        self.trainer._save_checkpoint = self.save_checkpoint
        self.trainer._validate = self.validate_and_record


def install_best_checkpoint_controller(trainer: Any) -> BestCheckpointController:
    config = trainer.config.trainer.get("cabin_best_checkpoint", {})
    controller = BestCheckpointController(trainer, str(config.get("metric_key", DEFAULT_METRIC_KEY)))
    controller.install()
    return controller
