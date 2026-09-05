# CabinAgent-RL Task Plan

## Goal

Complete the fallback-GRPO ablation sequence reproducibly. The active objective is to export and validate F10's selected step-50 LoRA adapter, implement training-time best-checkpoint selection, and launch F11 Turn-Discount as one 250-step same-node dual-GPU job with 50-step CAR-dev validation and W&B tracking.

## Scope For This Session

- Preserve F10's honest Slurm history and select step 50 as its earliest tied best CAR-dev checkpoint (`0.269231`).
- Export the F10 step-50 actor LoRA through the installed veRL converter and validate parent+adapter loading before proposing exact full-checkpoint deletion targets.
- F11 revised contract: save every50 before waking rollout; keep five LoRA-only recovery points, resume latest, select dev best with earlier ties, and prune only after final review.
- Keep step 0 as a reported baseline but exclude it from trained-checkpoint selection.
- Run local and remote tests, live cluster/storage pre-flight, W&B verification, and Slurm test-only before submitting F11.
- Submit no F12 successor; F11 completion requires manual review and full stage documentation.

## Phases

| Phase | Status | Acceptance |
|---|---|---|
| 0. Planning files | complete | `task_plan.md`, `findings.md`, and `Progress.md` exist and capture initial decisions. |
| 1. Repo scaffold | complete | Required directories, configs, requirements, and helper scripts exist. |
| 2. Baseline core | complete | Model adapter, benchmark adapters, trajectory schema, metrics, taxonomy, and report builder are implemented. |
| 3. Data and scripts | complete | Sample CAR/BFCL data and runnable scripts for eval/data/reward paths exist. |
| 4. Smoke verification | complete | Baseline eval scripts run locally and generate reports/failure cases. |
| 5. Cluster pre-flight | complete | Confirm live Slurm/account/storage rules and select an idle GPU node without changing cluster state. |
| 6. Remote deployment | complete | Create project directories in the permitted home/project SSD locations and sync the runnable baseline. |
| 7. Cluster smoke test | complete | Run through Slurm for about 30 minutes, observe no errors, then confirm the job is no longer running. |
| 8. Official datasets | complete | Download CAR-bench and BFCL assets through Slurm to the project SSD and verify their paths/checksums or file counts. |
| 9. SSD path migration | complete | Project content is checksum-verified beneath `/projects/jiatian001ssd/cabinagentrl`, the old storage project is removed, the new project owns the full 150 GB quota, and current references are updated. |
| 10. Single-node dual-GPU RL architecture | complete | Every dual-model job requests two Pro 6000 GPUs on one physical node and writes allocation plus per-component logs. |
| 11. Direct-RL transition | complete | Removed project-specific SFT from the default path, initialized every branch from Qwen2.5-7B-Instruct, and added a strict machine-readable CAR rollout gate plus guarded minimal-SFT fallback. |
| 12. Simulator environment and smoke | complete | GPU environment, 7B/72B-AWQ snapshots, CAR parquet, AWQ-Marlin serving, and project-SSD caches are verified. |
| 13. CAR agent loop and Direct-RL gate | complete_fail | G00-G02 ran; valid G02 reached consistency 1.0 but mixed group ratio 0.0, so E10-E14 remain blocked. |
| 14. Minimal-SFT fallback | complete_fail | Corrected F01/G03 and F02/G04 completed; F02/G04 is retained as a negative corrective-SFT result and F03/G05 is paused. |
| 15. Fallback veRL pilot | complete | F10 pilot/save/resume and the formal 250-step F10 run completed with usable outcome-gradient signal. |
| 16. Five GRPO ablations | in_progress | F10 is complete and W&B-backfilled; F11 Turn-Discount is selected and awaiting best-checkpoint tooling plus pre-flight. |
| 17. Frozen evaluation | pending | Select checkpoints on CAR dev/BFCL, then run CAR test once and produce the final comparison. |

## Current F11 Turn-Discount Execution

| Step | Status | Acceptance |
|---|---|---|
| Freeze selection/storage contract | complete | CAR dev mean@1 at 50-step boundaries; strict improvement only; ties keep earlier; step 0 excluded; completed runs retain validated LoRA adapter rather than full optimizer checkpoint. |
| Implement best-checkpoint tooling | complete | Local and remote 51 tests, compile, Ray actor import, Hydra resolution, and scheduler validation PASS. |
| Export F10 selected adapter | complete | Job140039 PASS; both full checkpoints deleted with approval. Adapter retained. |
| Remote pre-flight | complete | Live Slurm/QoS/GPU/storage/W&B state, tests, Bash, Hydra resolved config, imports, and both Slurm test-only checks pass. |
| Submit F11 | complete | Save140549 and resume140696 passed; formal retry140980 submitted for250steps with five recovery checkpoints. |
| Monitor and close F11 | pending | 250/250 or honest failed attempt; best-step history, gradients, GPU telemetry, adapter export, stage docs, and GitHub sync complete. |

## Current F02/G04 Attempt

| Step | Status | Acceptance |
|---|---|---|
| Cancel stale G03 | complete | Job `132946` is `CANCELLED` and no longer occupies the submit queue. |
| Build corrective data | complete | Added audited pre-call refusal and no-retry examples for G03 missing-tool families with task-disjoint splitting. |
| Local and remote validation | complete | Local/remote tests, compile/YAML, cluster `bash -n`, Slurm test-only, and real Qwen tokenizer gate pass. |
| F02 smoke/full | complete | Jobs `133303`/`133306` completed and saved Job-ID-scoped logs, manifests, metrics, and adapters. |
| G04 gate | complete_fail | Job `133308` completed 80 trajectories on one physical node; executable, mixed, and loop thresholds failed. |
| Stage record | complete | F02 attempts and final G04 result are recorded in Stage 04, Project, Progress, tracker, and overview. |

## Current F10 Pilot Attempt

| Step | Status | Acceptance |
|---|---|---|
| Route/document update | complete | Preserve G03/G04 FAIL reports; pause F03/G05; name corrected F01 as the shared fallback-GRPO parent; make numerical gate thresholds diagnostic for the pilot. |
| Frozen F01 parent | complete | Merge `133431` and validation `133447` passed; exact 10-file hash inventory, BF16 model/tokenizer load, and one-token generation are verified. |
| Local trainer audit | complete | Implement fresh rank-32 RL LoRA over the merged F01 parent, same-node two-task launcher, telemetry, 5-step cap, checkpoint save, and resume path without altering scientific semantics. |
| Local tests and remote pre-flight | complete | Relevant unit tests/compile/YAML pass; live QoS/GPU/storage rules and queue are rechecked; remote shell syntax and `sbatch --test-only` pass. |
| F10 pilot launch | in_progress | Packed-path smoke `135977` passed; fresh 5-step same-node dual-GPU attempt 9 `135987` is queued with no successor. |
| Manual acceptance | pending | Five optimizer steps plus one resumed step complete; at least one step has non-zero reward variance/advantage/finite gradient; no NaN/OOM/schema error; KL/clip/grad and GPU telemetry are complete. |
| Branch decision | pending | PASS freezes system settings for F10-F14; zero outcome advantage with healthy infrastructure routes to a separately reviewed F13 pilot. |

## Key Decisions

- Use `Qwen/Qwen2.5-7B-Instruct` as the default model id in configs, but make the local baseline use a deterministic rule-based adapter so CI/smoke tests do not require GPU access.
- Store every evaluation item as a JSON trajectory with input, messages, expected calls, predicted calls, tool execution status, metrics, and failure labels.
- Use `sample` mode for immediate runnable development and reserve `official` mode for future CAR-bench/BFCL integrations.
- Write shell scripts as bash-compatible `.sh` files per `Project.md`; keep Python CLIs usable directly on Windows.
- Treat `/projects/jiatian001ssd/cabinagentrl` as the canonical project SSD root after the user-requested migration.
- Use `Qwen/Qwen2.5-72B-Instruct-AWQ` as the local CAR-bench user simulator and `Qwen/Qwen2.5-7B-Instruct` as the trainable policy.
- Request simulator and trainer nodes atomically in one Slurm allocation; let Slurm keep the job pending until both Pro 6000 nodes are available instead of polling from the login node.
- Use vLLM for the simulator service and veRL for on-policy GRPO training; retain the official CAR-bench evaluator protocol and keep simulator identity fixed across comparable runs.
- Remove DPO from the active pipeline. E10-E14 all initialize new LoRA weights from the same Qwen2.5-7B-Instruct revision and never inherit from one another.
- Project-specific SFT is not a prerequisite. It is a separate minimal fallback only when the outcome-only rollout gate fails.
- Keep the complete five-run matrix: Vanilla, Turn-Discount, LATA, PRM-Lite, and PRM-Lite + LATA, with checkpoint evaluation at steps 50/100/150/200/250.
- Use deterministic environment reward during training; an LLM policy evaluator remains outside the training reward path.
- Treat G03/G04 numerical thresholds as preserved diagnostic results rather than a hard blocker for the explicitly approved, bounded F10 pilot; do not relabel either historical FAIL as PASS.
- Use corrected F01, not F02, as the shared initialization for F10-F14. Preserve F02/G04 as a negative transfer result and pause F03/G05.
- Optimize stable throughput only through semantics-preserving system knobs, with roughly 10-15% dynamic VRAM headroom, then freeze the selected settings across all formal branches.
- Never auto-submit a formal run after the pilot; require manual metric, trajectory, checkpoint, and GPU-telemetry review.
- For F11-F14, evaluate the in-memory actor every 50 steps and materialize a full checkpoint only when CAR dev mean@1 strictly improves over prior trained checkpoints; ties retain the earlier step.
- Step 0 remains a shared initialization baseline and does not compete with step 50/100/150/200/250 for each method's selected trained checkpoint.
- After a completed experiment, export and validate the best actor LoRA adapter, configuration, metrics, and inventory; remove full optimizer/RNG checkpoints only after an exact deletion list is separately approved.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| `apply_patch` rejected a patch with two separate updates to `src/training/verl_entrypoint.py` | F11 tooling attempt 1 | No files were changed; combine the import and body edit into one file operation, then reapply the same scoped implementation. |
| `apply_patch` looked for test assertions inside `scripts/launch_verl.py` | F11 tooling attempt 2 | No files were changed; split the launcher and test edits into their correct files. |
| Adapter inventory unit test referenced nonexistent `tests/fixtures/parent_manifest` | F10 adapter-export tooling test 1 | Production code compiled; point the immutable test at existing `tests/fixtures/merge_output` and rerun the full suite with fail-fast command chaining. |
| `git status` failed because this folder is not a git repository | Initial orientation | Record as an environment fact; continue without git-based change tracking. |
| Default PowerShell output showed mojibake for Chinese markdown | Initial `Get-Content` | Re-read files with UTF-8 output encoding. |
| `ModuleNotFoundError: No module named 'src'` when running scripts by file path | First data-builder smoke test | Added repository root to `sys.path` in standalone data builder scripts. |
| `集群详细使用说明.md` is empty | Cluster pre-flight | Use `AGENTS.md` plus live read-only Slurm/storage queries; do not assume partition, QoS, account, or SSD paths. |
| `ssh` and `scp` resolve to sandbox deny wrappers | Cluster pre-flight | Use the explicit Windows OpenSSH executable if direct network access is available. |
| Official GitHub cluster docs could not be fetched from the local environment | Cluster pre-flight | Continue with live read-only cluster mount/help inspection and record the documentation gap. |
| Storage quota exists but no project namespace is assigned | Cluster pre-flight | Fetch official storage guidance through a short Slurm CPU job; keep only code/job scripts under home until the SSD destination is confirmed. |
| Slurm docs job `129937` exited 1 after fetching `login.md` | Cluster pre-flight | The final `grep` had no matches under `set -e`; retry by listing all documentation paths with an explicitly non-fatal filter. |
| Slurm docs-tree job `129938` completed but produced no path list | Cluster pre-flight | Third approach: shallow-clone the official docs into the compute node's temporary directory and list/read relevant files there. |
| Persistent SSH shell exited while reading docs | Cluster pre-flight | A previous `set -e` remained active and an exact-title `grep` had no match; reconnect and explicitly use `set +e` for exploratory commands. |
| `storagemgr` rejected `CabinAgent-RL` as unsafe | Remote deployment | Provision top-level project `cabinagentrl` and create the exact requested `CabinAgent-RL` directory beneath it. |
| Smoke job `129964` could not find `python` on the compute node | Cluster smoke test | Load the live-confirmed `Miniforge3/24.11.3-1` Lmod module in the smoke harness before invoking Python. |
| Dataset job `129963` rejected `set -o pipefail` under `/bin/sh` | Official datasets | Retry the `sbatch --wrap` command with POSIX-compatible `set -eu`; no partial target directories were created. |
| Data verification job `130051` assumed every BFCL `*.json` was JSONL | Official datasets | Support both complete JSON documents and JSONL; CAR task parsing already passed and no data was modified. |
| Protected cutover job `130534` found timestamp-only differences in regenerated reports | SSD path migration | Preserve the old source, repeat the checksum comparison with timestamp metadata excluded, and delete only after job `130535` reports zero content differences. |
| Local `bash -n` is unavailable because WSL service creation is denied | Dual-node scaffolding | Run `bash -n` on the cluster login node, where all new shell and Slurm scripts passed. |
| The cluster base Miniforge module lacks `yaml` | Dual-node scaffolding | Keep PyYAML in `requirements-gpu.txt`; validate config rendering locally now and inside the named project-SSD GPU environment in Stage 11. |
| G03 `132043` crashed in CAR automatic evaluation | Minimal-SFT fallback | Normalize `function.arguments` to objects before Qwen templating, add a round-trip parser test, and retrain the adapter before retrying G03. |
| Remote combined validation stopped at inline YAML command quoting | F02 pre-flight attempt 1 | Tests, compileall, and `bash -n` had already passed; rerun Slurm test-only separately without the fragile inline Python expression. |
| First formal submission command was intercepted by local PowerShell command substitution | F02 submit attempt 1 | No job was submitted; rerun `sbatch --parsable` without remote shell variable capture, then query the returned Job ID separately. |
| F10 attempt 8 resolved chunking but still called dense unchunked entropy | F10 pilot attempt 8 | Dense FSDP path ignores the switch when `use_remove_padding=False`; user approved enabling the native packed/remove-padding path, gated by a one-GPU integration smoke. |
