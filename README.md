# CabinAgent-RL

CabinAgent-RL is a research engineering project for intelligent-cockpit tool agents. It builds a reproducible path from prompt baselines and real multi-turn CAR rollouts to Minimal-SFT fallback, online GRPO ablations, deterministic process rewards, unified evaluation, and vLLM deployment.

The implemented system includes:

- deterministic local prompt baseline adapter,
- sample CAR-bench-like and BFCL-like tasks,
- unified trajectory schema,
- benchmark adapters,
- metrics and failure taxonomy,
- report generation,
- direct-RL rollout-gate and PRM-Lite utilities,
- official CAR/BFCL data preparation and isolation,
- Qwen2.5-72B-AWQ user-simulator and Qwen2.5-7B policy serving,
- same-node dual-Pro-6000 Slurm orchestration,
- corrected Minimal-SFT data/training and immutable parent merging,
- veRL 0.9 multi-turn AgentLoop integration,
- five GRPO reward/advantage configurations,
- checkpoint, failure-attempt, telemetry, and experiment-tracker contracts.

The local baseline is intentionally dependency-light and does not require GPU access. GPU experiments use a Qwen2.5-72B-AWQ user simulator and a Qwen2.5-7B policy with veRL online multi-turn GRPO. Project-specific SFT remains a separately tracked fallback, not a prerequisite silently mixed into the direct-instruct family.

## Current Experiment Status

- Direct-Instruct gate G02 completed 80 real trajectories but had zero mixed-outcome groups, so E10-E14 remain a documented negative result rather than being relabeled as successful.
- Corrected Minimal-SFT F01 completed and was merged into an immutable Qwen2.5-7B parent. Targeted F02/G04 corrective training was retained as a negative ablation because it did not improve the frozen gate criteria overall.
- The bounded F10 Vanilla-GRPO pilot now reaches model initialization, initial validation, and the first 16-rollout training batch. Attempt 8 confirmed that veRL 0.9's dense-padding FSDP path ignores the resolved chunked-entropy flag and still OOMs before optimizer step 1; the proposed next step is a single-GPU validation of veRL's native packed/remove-padding chunked path before another bounded retry.
- No formal F10-F14 result or final CAR/BFCL score is claimed yet. See `Project.md`, `Progress.md`, and `docs/实验阶段/` for attempt-level evidence and current gates.

## Quick Start

Run the sample baseline:

```bash
python -m src.eval.run_baseline --benchmark carbench
python -m src.eval.run_baseline --benchmark bfcl
python -m src.eval.run_baseline --benchmark all
```

Build derived data from generated trajectories:

```bash
python scripts/build_prm_lite_data.py --input data/eval_cache/all_trajectories.jsonl
```

Outputs are written to:

- `reports/baseline_carbench.md`
- `reports/baseline_bfcl.md`
- `reports/eval_summary.csv`
- `reports/failure_taxonomy.md`
- `failure_cases/baseline/*.json`
- `data/eval_cache/*_trajectories.jsonl`

## Repository Layout

```text
configs/          Model, eval, training, and serving configs
scripts/          CLI wrappers and data-building scripts
src/              Baseline adapters, schema, metrics, rewards, training helpers
data/raw/         Local sample benchmark records
data/eval_cache/  Generated trajectories
reports/          Markdown and CSV reports
failure_cases/    Reproducible failure trajectories
checkpoints/      Future training outputs
experiments/      Immutable run manifests, logs, checkpoints, and evaluations
demo/             Future serving demo notes
```

## Result Boundary

The included sample tasks are smoke tests, not official benchmark scores. Complete official CAR-bench and BFCL data, model weights, environments, checkpoints, and full run directories stay on project SSD and are excluded from Git. Selected machine-readable reports and failure evidence are retained under `reports/cluster/`; all dual-GPU experiments use two Pro 6000 GPUs on one physical node with separate component logs.

Run the regression suite with:

```bash
python -B -m unittest discover -s tests -v
python -m compileall src scripts
```
