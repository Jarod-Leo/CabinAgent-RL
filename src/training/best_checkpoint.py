"""Training-time selection of the best resumable GRPO checkpoint."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable

from scripts.checkpoint_policy import audit_run, prune_run, validate_checkpoint


SCHEMA_VERSION = 1
DEFAULT_METRIC_KEY = "val-core/car_bench/reward/mean@1"


def _empty_state(metric_key: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_key": metric_key,
        "comparison": "strict_greater",
        "tie_break": "earlier",
        "baseline": None,
        "best": None,
        "pending_candidate": None,
        "history": [],
    }


def load_state(path: Path, metric_key: str) -> dict[str, Any]:
    if not path.exists():
        return _empty_state(metric_key)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported best-checkpoint state schema: {state.get('schema_version')}")
    if state.get("metric_key") != metric_key:
        raise ValueError(
            f"Best-checkpoint metric changed: {state.get('metric_key')!r} != {metric_key!r}"
        )
    return state


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def candidate_is_better(candidate_score: float, best: dict[str, Any] | None) -> bool:
    return best is None or candidate_score > float(best["score"])


class BestCheckpointController:
    """Defer veRL checkpoint writes until the matching dev result is known."""

    def __init__(
        self,
        trainer: Any,
        metric_key: str = DEFAULT_METRIC_KEY,
        state_path: Path | None = None,
    ) -> None:
        if int(trainer.config.trainer.save_freq) <= 0:
            raise ValueError("Best-checkpoint selection requires save_freq > 0")
        if int(trainer.config.trainer.save_freq) != int(trainer.config.trainer.test_freq):
            raise ValueError("Best-checkpoint selection requires save_freq == test_freq")
        if str(trainer.config.trainer.v1.trainer_mode) != "sync":
            raise ValueError("Cabin best-checkpoint selection currently requires V1 sync trainer mode")

        checkpoint_root = Path(str(trainer.config.trainer.default_local_dir)).resolve()
        self.run_dir = checkpoint_root.parent
        self.state_path = state_path or self.run_dir / "metrics" / "best_checkpoint.json"
        self.metric_key = metric_key
        self.trainer = trainer
        self.original_save: Callable[[], None] = trainer._save_checkpoint
        self.original_validate: Callable[[], dict[str, float]] = trainer._validate
        self.pending_step: int | None = None
        self.state = load_state(self.state_path, metric_key)
        self._reconcile()

    def _persist(self) -> None:
        write_state(self.state_path, self.state)

    def _reconcile(self) -> None:
        pending = self.state.get("pending_candidate")
        best = self.state.get("best")
        if pending is not None:
            candidate_step = int(pending["step"])
            candidate_path = self.run_dir / "checkpoints" / f"global_step_{candidate_step}"
            try:
                audit_run(self.run_dir, expected_step=candidate_step)
                validate_checkpoint(candidate_path)
            except (FileNotFoundError, ValueError):
                self.state["pending_candidate"] = None
                self._persist()
            else:
                self.state["best"] = dict(pending)
                self.state["pending_candidate"] = None
                self._persist()
                prune_run(self.run_dir, candidate_step, apply=True)
                best = self.state["best"]

        if best is not None:
            best_step = int(best["step"])
            audit_run(self.run_dir, expected_step=best_step)
            prune_run(self.run_dir, best_step, apply=True)

    def defer_save(self) -> None:
        step = int(self.trainer.global_steps)
        if self.pending_step is not None:
            raise RuntimeError(
                f"A checkpoint save is already pending for step {self.pending_step}; got step {step}"
            )
        self.pending_step = step

    def validate_and_maybe_save(self) -> dict[str, float]:
        metrics = self.original_validate()
        step = int(self.trainer.global_steps)
        if self.pending_step is None:
            if step == 0:
                score = self._score(metrics)
                self.state["baseline"] = {"step": 0, "score": score}
                self._persist()
            return metrics
        if self.pending_step != step:
            raise RuntimeError(
                f"Deferred checkpoint step {self.pending_step} does not match validation step {step}"
            )

        score = self._score(metrics)
        selected = candidate_is_better(score, self.state.get("best"))
        record = {
            "step": step,
            "score": score,
            "selected": selected,
            "reason": "strict_improvement" if selected else "not_strictly_better",
        }

        if selected:
            pending = {"step": step, "score": score}
            self.state["pending_candidate"] = pending
            self._persist()
            self.original_save()
            checkpoint_path = self.run_dir / "checkpoints" / f"global_step_{step}"
            validate_checkpoint(checkpoint_path)
            self.state["best"] = pending
            self.state["pending_candidate"] = None
            self.state["history"].append(record)
            self._persist()
            prune_run(self.run_dir, step, apply=True)
        else:
            self.state["history"].append(record)
            self._persist()

        self.pending_step = None
        best = self.state["best"]
        metrics.update(
            {
                "checkpoint-selection/candidate_score": score,
                "checkpoint-selection/selected": float(selected),
                "checkpoint-selection/best_score": float(best["score"]),
                "checkpoint-selection/best_step": float(best["step"]),
            }
        )
        return metrics

    def _score(self, metrics: dict[str, float]) -> float:
        if self.metric_key not in metrics:
            raise KeyError(f"Validation metric is missing: {self.metric_key}")
        score = float(metrics[self.metric_key])
        if not math.isfinite(score):
            raise ValueError(f"Validation metric is not finite: {self.metric_key}={score}")
        return score

    def install(self) -> None:
        self.trainer._save_checkpoint = self.defer_save
        self.trainer._validate = self.validate_and_maybe_save


def install_best_checkpoint_controller(trainer: Any) -> BestCheckpointController:
    config = trainer.config.trainer.get("cabin_best_checkpoint", {})
    metric_key = str(config.get("metric_key", DEFAULT_METRIC_KEY))
    controller = BestCheckpointController(trainer, metric_key=metric_key)
    controller.install()
    return controller
