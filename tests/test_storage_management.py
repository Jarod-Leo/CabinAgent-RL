import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.archive_storage import inventory, safe_batch_id, safe_relative
from scripts.checkpoint_policy import audit_best_run, prune_run
from scripts.launch_verl import ROOT, build_overrides


class StorageManagementTests(unittest.TestCase):
    def test_best_checkpoint_audit_requires_unique_recorded_step(self) -> None:
        run_dir = ROOT / "experiments" / "test-best-run"
        state = {
            "best": {"step": 50, "score": 0.25},
            "pending_candidate": None,
        }
        audit = {
            "run_dir": run_dir.as_posix(),
            "marker_step": 50,
            "checkpoints": [{"step": 50, "path": "global_step_50"}],
            "incomplete_checkpoints": [],
        }
        with patch.object(Path, "is_file", return_value=True), patch.object(
            Path, "read_text", return_value=json.dumps(state)
        ), patch("scripts.checkpoint_policy.audit_run", return_value=audit):
            result = audit_best_run(run_dir)
        self.assertEqual(result["status"], "best_checkpoint_verified")
        self.assertEqual(result["marker_step"], 50)

    def test_cross_process_checkpoint_prune_keeps_marker_step(self) -> None:
        run_dir = ROOT / "experiments" / "test-run"
        checkpoint_root = run_dir / "checkpoints"
        old = checkpoint_root / "global_step_50"
        keep = checkpoint_root / "global_step_100"
        before = {
            "run_dir": run_dir.as_posix(),
            "marker_step": 100,
            "checkpoints": [
                {"step": 50, "path": old.as_posix()},
                {"step": 100, "path": keep.as_posix()},
            ],
            "incomplete_checkpoints": [],
        }
        after = {
            "run_dir": run_dir.as_posix(),
            "marker_step": 100,
            "checkpoints": [{"step": 100, "path": keep.as_posix()}],
            "incomplete_checkpoints": [],
        }
        with patch("scripts.checkpoint_policy.audit_run", side_effect=[before, after]), patch(
            "scripts.checkpoint_policy.shutil.rmtree"
        ) as remove:
            result = prune_run(run_dir, 100, apply=True)
        self.assertEqual(result["removed"], [old.resolve().as_posix()])
        remove.assert_called_once_with(old.resolve())

    def test_prune_removes_older_incomplete_checkpoint_tombstone(self) -> None:
        run_dir = ROOT / "experiments" / "test-run"
        checkpoint_root = run_dir / "checkpoints"
        stale = checkpoint_root / "global_step_50"
        keep = checkpoint_root / "global_step_100"
        before = {
            "run_dir": run_dir.as_posix(),
            "marker_step": 100,
            "checkpoints": [{"step": 100, "path": keep.as_posix()}],
            "incomplete_checkpoints": [
                {"step": 50, "path": stale.as_posix(), "required_files_present": False}
            ],
        }
        after = {
            "run_dir": run_dir.as_posix(),
            "marker_step": 100,
            "checkpoints": [{"step": 100, "path": keep.as_posix()}],
            "incomplete_checkpoints": [],
        }
        with patch("scripts.checkpoint_policy.audit_run", side_effect=[before, after]), patch(
            "scripts.checkpoint_policy.shutil.rmtree"
        ) as remove:
            result = prune_run(run_dir, 100, apply=True)
        self.assertEqual(result["removed"], [stale.resolve().as_posix()])
        remove.assert_called_once_with(stale.resolve())

    def test_archive_inventory_and_safe_relative_paths(self) -> None:
        source = Path(__file__)
        manifest = inventory(source)
        self.assertEqual(manifest[0]["path"], source.name)
        self.assertEqual(manifest[0]["size_bytes"], source.stat().st_size)
        self.assertEqual(len(manifest[0]["sha256"]), 64)
        self.assertEqual(safe_relative("models/toy").as_posix(), "models/toy")
        self.assertEqual(safe_batch_id("tiering-20260904-a1"), "tiering-20260904-a1")
        for unsafe in ("../models", "/models"):
            with self.assertRaises(ValueError):
                safe_relative(unsafe)
        for unsafe_batch in ("../batch", "batch/name", ".."):
            with self.assertRaises(ValueError):
                safe_batch_id(unsafe_batch)

    def test_fallback_configs_and_hdd_override(self) -> None:
        expected = {
            "vanilla": "F10",
            "turn_discount": "F11",
            "lata": "F12",
            "prm_lite": "F13",
            "prm_lite_lata": "F14",
        }
        for name, experiment_id in expected.items():
            config = ROOT / "configs" / "train" / "fallback_ablations" / f"{name}.yaml"
            experiment, overrides = build_overrides(
                config,
                f"test-{name}",
                ROOT / "experiments" / f"test-{name}",
                validate_paths=False,
            )
            self.assertEqual(experiment["experiment_id"], experiment_id)
            rendered = "\n".join(overrides)
            self.assertIn("algorithm.turn_discount.alpha=", rendered)

    def test_continuous_job_contract_is_single_node_and_no_successor(self) -> None:
        submitter = (ROOT / "scripts" / "submit_fallback_ablation.sh").read_text(
            encoding="utf-8"
        )
        slurm = (ROOT / "scripts" / "slurm_fallback_grpo.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("target_steps=250", submitter)
        self.assertIn("SAVE_FREQ=50", submitter)
        self.assertIn("EVAL_FREQ=50", submitter)
        self.assertIn("time_limit=24:00:00", submitter)
        self.assertNotIn("NEXT_TRAINING_STAGE", submitter)
        self.assertIn("#SBATCH --nodes=1", slurm)
        self.assertIn("#SBATCH --gres=gpu:pro6000:2", slurm)
        self.assertIn("#SBATCH --requeue", slurm)
        self.assertIn("restart-${restart_count}.done", slurm)
        self.assertIn("--ntasks=2 --gpus-per-task=1 --gpu-bind=single:1", slurm)
        self.assertIn("checkpoint_args=(audit-series", slurm)
        self.assertIn("checkpoint_args=(prune", slurm)


if __name__ == "__main__":
    unittest.main()
