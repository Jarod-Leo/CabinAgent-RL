import unittest
from unittest.mock import patch

from scripts.launch_verl import ROOT, build_overrides, load_yaml


class TrainingConfigTests(unittest.TestCase):
    def test_dual_gpu_jobs_require_one_physical_node(self) -> None:
        for relative in (
            "scripts/slurm_direct_rl_gate.sbatch",
            "scripts/slurm_dual_pro6000.sbatch",
            "scripts/slurm_same_node_dual_gpu_smoke.sbatch",
            "scripts/slurm_f10_pilot.sbatch",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --nodes=1", text)
            self.assertIn("#SBATCH --gres=gpu:pro6000:2", text)
            self.assertNotIn("#SBATCH --nodes=2", text)

        parent_validation = (ROOT / "scripts/slurm_validate_f01_parent.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --nodes=1", parent_validation)
        self.assertIn("#SBATCH --gres=gpu:pro6000:1", parent_validation)

        runtime = load_yaml(ROOT / "configs/runtime/pro6000_dual_node.yaml")
        self.assertEqual(runtime["topology"], "single_node_dual_gpu")
        self.assertEqual(runtime["nodes"], 1)
        self.assertEqual(runtime["gres_per_node"], "gpu:pro6000:2")

        same_node_gate = (ROOT / "scripts/slurm_direct_rl_gate_same_node.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("--ntasks=2 --gpus-per-task=1 --gpu-bind=single:1", same_node_gate)
        same_node_smoke = (ROOT / "scripts/slurm_same_node_dual_gpu_smoke.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("--ntasks=2 --gpus-per-task=1 --gpu-bind=single:1", same_node_smoke)

        f10_pilot = (ROOT / "scripts/slurm_f10_pilot.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("--ntasks=2 --gpus-per-task=1 --gpu-bind=single:1", f10_pilot)
        self.assertNotIn("--exclusive", f10_pilot)
        submitter = (ROOT / "scripts/submit_f10_pilot.sh").read_text(encoding="utf-8")
        self.assertNotIn("NEXT_TRAINING_STAGE", submitter)
        self.assertIn("SIMULATOR_GPU_MEMORY_UTILIZATION:-0.86", submitter)
        task_runner = (ROOT / "scripts/run_f10_pilot_task.sh").read_text(encoding="utf-8")
        self.assertIn("unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES", task_runner)
        self.assertIn('module load CUDA/13.0.0', task_runner)
        self.assertIn('export CAR_BENCH_DATASET_ROOT="$PROJECT_ROOT/data/official/car-bench-dataset"', task_runner)

        post_sft_submitter = (ROOT / "scripts/submit_post_sft_gate.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("scripts/slurm_direct_rl_gate_same_node.sbatch", post_sft_submitter)
        self.assertNotIn("scripts/slurm_direct_rl_gate.sbatch)", post_sft_submitter)

    def test_all_ablations_fit_qwen_context_window(self) -> None:
        common = load_yaml(ROOT / "configs/train/grpo_common.yaml")
        sequence_length = common["max_prompt_length"] + common["max_response_length"]
        self.assertEqual(sequence_length, 32768)
        self.assertLessEqual(sequence_length, common["policy_max_model_len"])

        for name in ("vanilla", "turn_discount", "lata", "prm_lite", "prm_lite_lata"):
            _, overrides = build_overrides(
                ROOT / f"configs/train/ablations/{name}.yaml",
                f"test-{name}",
                ROOT / "experiments" / f"test-{name}",
                validate_paths=False,
            )
            rendered = "\n".join(overrides)
            self.assertIn("data.max_prompt_length=24576", rendered)
            self.assertIn("data.max_response_length=8192", rendered)
            self.assertIn("actor_rollout_ref.rollout.max_model_len=32768", rendered)
            self.assertIn("actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768", rendered)
            self.assertIn("actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768", rendered)
            self.assertNotIn("40960", rendered)

    def test_f10_uses_merged_f01_parent_and_fresh_rank32_lora(self) -> None:
        config = ROOT / "configs/train/fallback_ablations/vanilla.yaml"
        experiment, overrides = build_overrides(
            config,
            "test-f10",
            ROOT / "experiments/test-f10",
            validate_paths=False,
        )
        rendered = "\n".join(overrides)
        self.assertEqual(experiment["experiment_id"], "F10")
        self.assertIn("Qwen2.5-7B-Instruct-F01-merged-20260901", rendered)
        self.assertIn("actor_rollout_ref.model.lora_rank=32", rendered)
        self.assertNotIn("lora_adapter_path", rendered)

    def test_system_knobs_are_environment_overridable(self) -> None:
        config = ROOT / "configs/train/fallback_ablations/vanilla.yaml"
        with patch.dict(
            "os.environ",
            {
                "ROLLOUT_GPU_MEMORY_UTILIZATION": "0.60",
                "ROLLOUT_MAX_NUM_BATCHED_TOKENS": "16384",
            },
            clear=False,
        ):
            _, overrides = build_overrides(
                config,
                "test-f10-knobs",
                ROOT / "experiments/test-f10-knobs",
                validate_paths=False,
            )
        rendered = "\n".join(overrides)
        self.assertIn("actor_rollout_ref.rollout.gpu_memory_utilization=0.60", rendered)
        self.assertIn("actor_rollout_ref.rollout.max_num_batched_tokens=16384", rendered)

    def test_ray_uses_short_job_scoped_socket_path(self) -> None:
        runtime_env = (ROOT / "scripts/cluster_runtime_env.sh").read_text(encoding="utf-8")
        self.assertIn("/tmp/cabin-ray-${SLURM_JOB_ID:-manual}", runtime_env)
        self.assertNotIn('RAY_TMPDIR="$PROJECT_ROOT/cache/ray"', runtime_env)

    def test_gpu_runtime_pins_missing_verl_dependencies(self) -> None:
        requirements = (ROOT / "requirements-gpu.txt").read_text(encoding="utf-8")
        self.assertIn("TransferQueue==0.1.7", requirements)
        self.assertIn("flash_attn==2.8.3", requirements)
        installer = (ROOT / "scripts/slurm_install_flash_attention.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("FLASH_ATTN_CUDA_ARCHS=120", installer)
        self.assertIn("MAX_JOBS=4", installer)
        self.assertIn("module load CUDA/13.0.0", installer)


if __name__ == "__main__":
    unittest.main()
