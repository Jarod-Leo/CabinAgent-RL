"""Translate a CabinAgent-RL ablation manifest to veRL Hydra overrides."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def quote_list(path: Path) -> str:
    return f"['{path.as_posix()}']"


def env_value(name: str, default: Any) -> str:
    """Return a Hydra-friendly environment override value."""

    value = os.environ.get(name)
    if value is None:
        value = str(default)
    if isinstance(value, bool):
        return str(value).lower()
    lowered = str(value).lower()
    if lowered in {"true", "false"}:
        return lowered
    return str(value)


def build_overrides(
    config_path: Path, run_id: str, run_dir: Path, validate_paths: bool = True
) -> tuple[dict[str, Any], list[str]]:
    experiment = load_yaml(config_path)
    common = load_yaml(ROOT / str(experiment["parent_config"]))
    train_data = ROOT / str(common["train_data"])
    val_data = ROOT / str(common["val_data"])
    agent_loop = ROOT / str(common["agent_loop_config"])
    configured_model = Path(str(experiment.get("policy_model", common["policy_model"])))
    policy_model = configured_model if configured_model.is_absolute() else ROOT / configured_model
    max_steps = int(os.environ.get("MAX_TRAINING_STEPS", common["max_steps"]))
    save_freq = int(os.environ.get("SAVE_FREQ", common["save_freq"]))
    eval_freq = int(os.environ.get("EVAL_FREQ", common["eval_freq"]))
    train_max_samples = int(os.environ.get("TRAIN_MAX_SAMPLES", -1))
    val_max_samples = int(os.environ.get("VAL_MAX_SAMPLES", -1))
    sequence_length = int(common["max_prompt_length"]) + int(common["max_response_length"])
    policy_max_model_len = int(common["policy_max_model_len"])
    if sequence_length > policy_max_model_len:
        raise ValueError(
            "Configured prompt/response budget exceeds the policy context window: "
            f"{sequence_length} > {policy_max_model_len}"
        )

    missing = [path for path in (train_data, val_data, agent_loop) if not path.exists()]
    if validate_paths and missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Formal veRL launch prerequisites are missing:\n{formatted}")

    overrides = [
        f"algorithm.adv_estimator={experiment['advantage_estimator']}",
        f"data.train_files={quote_list(train_data)}",
        f"data.val_files={quote_list(val_data)}",
        f"data.train_batch_size={common['train_batch_size']}",
        f"data.train_max_samples={train_max_samples}",
        f"data.val_max_samples={val_max_samples}",
        f"data.max_prompt_length={common['max_prompt_length']}",
        f"data.max_response_length={common['max_response_length']}",
        "data.return_raw_chat=True",
        "data.truncation=error",
        f"data.seed={common['seed']}",
        f"actor_rollout_ref.model.path={policy_model.as_posix()}",
        f"actor_rollout_ref.model.lora_rank={common['lora_rank']}",
        f"actor_rollout_ref.model.lora_alpha={common['lora_alpha']}",
        f"actor_rollout_ref.model.target_modules={common['lora_target_modules']}",
        "actor_rollout_ref.model.enable_activation_offload=True",
        "actor_rollout_ref.model.use_remove_padding="
        + env_value("USE_REMOVE_PADDING", True),
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization="
        + env_value("ROLLOUT_GPU_MEMORY_UTILIZATION", 0.35),
        "actor_rollout_ref.rollout.enforce_eager="
        + env_value("ROLLOUT_ENFORCE_EAGER", True),
        "actor_rollout_ref.rollout.max_num_seqs="
        + env_value("ROLLOUT_MAX_NUM_SEQS", 16),
        "actor_rollout_ref.rollout.max_num_batched_tokens="
        + env_value("ROLLOUT_MAX_NUM_BATCHED_TOKENS", 8192),
        f"actor_rollout_ref.rollout.max_model_len={sequence_length}",
        f"actor_rollout_ref.rollout.n={common['group_size']}",
        "actor_rollout_ref.rollout.agent.num_workers="
        + env_value("ROLLOUT_AGENT_WORKERS", 16),
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={agent_loop.as_posix()}",
        f"actor_rollout_ref.actor.optim.lr={common['learning_rate']}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking="
        + env_value("ENTROPY_FROM_LOGITS_WITH_CHUNKING", True),
        "actor_rollout_ref.actor.entropy_from_logits_chunk_size="
        + env_value("ENTROPY_FROM_LOGITS_CHUNK_SIZE", 2048),
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={sequence_length}",
        "actor_rollout_ref.actor.fsdp_config.param_offload="
        + env_value("ACTOR_PARAM_OFFLOAD", True),
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload="
        + env_value("ACTOR_OPTIMIZER_OFFLOAD", True),
        "actor_rollout_ref.actor.use_torch_compile=False",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="
        + env_value("PPO_MICRO_BATCH_SIZE_PER_GPU", 1),
        f"actor_rollout_ref.actor.ppo_mini_batch_size={common['train_batch_size']}",
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={sequence_length}",
        "actor_rollout_ref.ref.entropy_from_logits_with_chunking="
        + env_value("ENTROPY_FROM_LOGITS_WITH_CHUNKING", True),
        "actor_rollout_ref.ref.entropy_from_logits_chunk_size="
        + env_value("ENTROPY_FROM_LOGITS_CHUNK_SIZE", 2048),
        "actor_rollout_ref.ref.fsdp_config.param_offload="
        + env_value("REF_PARAM_OFFLOAD", True),
        "actor_rollout_ref.ref.use_torch_compile=False",
        "reward.custom_reward_function.path=src/rewards/verl_reward.py",
        "reward.custom_reward_function.name=compute_score",
        "trainer.nnodes=1",
        "trainer.n_gpus_per_node=1",
        f"trainer.total_training_steps={max_steps}",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={eval_freq}",
        "trainer.project_name=CabinAgent-RL",
        f"trainer.experiment_name={run_id}",
        f"trainer.default_local_dir={run_dir.as_posix()}/checkpoints",
        "trainer.logger=['console']",
        "trainer.resume_mode=auto",
    ]
    return experiment, overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    run_dir = (ROOT / "experiments" / args.run_id).resolve()
    experiment, overrides = build_overrides(
        config_path, args.run_id, run_dir, validate_paths=not args.dry_run
    )
    os.environ["CABIN_REWARD_MODE"] = str(experiment["reward_mode"])
    os.environ["CABIN_PROCESS_REWARD_WEIGHT"] = str(experiment["process_reward_weight"])

    if args.dry_run:
        from src.training.verl_entrypoint import main as _validated_entrypoint

        del _validated_entrypoint
        print("python -m src.training.verl_entrypoint " + " ".join(overrides))
        return

    if not os.environ.get("SIMULATOR_BASE_URL"):
        raise RuntimeError("SIMULATOR_BASE_URL is required for CAR online rollouts")
    sys.argv = ["src.training.verl_entrypoint", *overrides]
    from src.training.verl_entrypoint import main as verl_main

    verl_main()


if __name__ == "__main__":
    main()
