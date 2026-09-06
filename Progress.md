# CabinAgent-RL Progress

### 2026-09-06：F11 完成与 F12 准备

- F11最佳adapter导出/真实父模型加载验证Job142938已提交，当前PENDING/Priority；单卡highmem Pro6000、2h，来源step100，输出HDD adapters/f11_turn_discount_best_step_100。未删除任何checkpoint，F12尚未提交，等待该验收。导出脚本2项单测和远端Bash语法检查PASS。

- F11 Job140980 COMPLETED/0:0，250steps、10h49m10s，同节点2xPro6000；series_verified，五份各980,828,805bytes，无保存OOM。dev50/100/150/200/250=0.230769/0.269231/0.269231/0.230769/0.230769；best100，匹配F10最佳，尚无提升证据。
- 单卡adapter导出脚本参数化，复用既有导出和父模型加载验证；待最佳产物验收与非最佳清理后启动F12 LATA。SSD21.2/150GB，HDD72.3/250GB；科学参数保持冻结。

## 2026-06-20

### Stage 0: Orientation Started

- Read `Project.md` and `draft.md` with UTF-8 output after detecting terminal mojibake.
- Confirmed the repository currently has only project documents and no code scaffold.
- Confirmed the folder is not a git repository, so progress will be tracked in markdown files instead of commit history for now.
- Decided the first runnable baseline will use deterministic local sample tasks while preserving interfaces for real CAR-bench, BFCL, HF, and vLLM integration.

### Stage 0: Planning Files Complete

- Created `task_plan.md`, `findings.md`, and `Progress.md`.
- Captured the first-session scope: dependency-light sample baseline first, real benchmark/model integration later.
- Logged initial environment issues: no git repository and terminal mojibake unless UTF-8 output is forced.

### Stage 1: Repo Scaffold Complete

- Created the recommended project directories under `configs/`, `scripts/`, `src/`, `data/`, `reports/`, `failure_cases/`, `checkpoints/`, and `demo/`.
- Added local baseline configs for model, CAR/BFCL eval, training placeholders, and vLLM serving.
- Added `README.md`, `requirements.txt`, and `requirements-gpu.txt`.
- Added script entry points for baseline eval, data building, future training, vLLM serving, and latency measurement.
- Added normalized sample CAR-bench-like and BFCL-like JSONL tasks for local smoke testing.

### Stage 2: First Smoke Issue Fixed

- `python -m src.eval.run_baseline --benchmark all` completed and generated 9 sample trajectories.
- Direct data-builder execution initially failed with `ModuleNotFoundError: No module named 'src'` because Python put `scripts/` on `sys.path`.
- Fixed the standalone data builder scripts by adding the repository root to `sys.path`.

### Stage 2: Baseline Core Complete

- Implemented normalized benchmark/task/trajectory schema in `src/data/trajectory_schema.py`.
- Implemented deterministic `local_rules` model adapter plus CAR/BFCL sample adapters.
- Implemented metric scoring, simulated tool validation, failure taxonomy, benchmark reports, unified CSV summary, and failure case persistence.
- Implemented PRM-Lite deterministic scoring and GRPO reward bridge placeholder.

### Stage 3: Data Chain Complete

- Built SFT records from successful trajectories: 6 train and 1 validation sample.
- Built preference pairs from failed trajectories: 2 pairs.
- Built PRM-Lite debug reward rows: 9 rows.
- Generated reports for SFT data, preference data, PRM-Lite, CAR baseline, BFCL baseline, eval summary, and failure taxonomy.

### Stage 4: Smoke Verification Complete

- Final verification commands:
  - `python -B -m src.eval.run_baseline --benchmark all`
  - `python -B scripts\build_sft_data.py --input data\eval_cache\all_trajectories.jsonl`
  - `python -B scripts\build_preference_data.py --input data\eval_cache\all_trajectories.jsonl`
  - `python -B scripts\build_prm_lite_data.py --input data\eval_cache\all_trajectories.jsonl`
- Results:
  - CAR sample baseline: 5 samples, success 0.8000, 1 failure case.
  - BFCL sample baseline: 4 samples, success 0.7500, 1 failure case.
  - Failure taxonomy currently contains 2 `F1_TOOL_NAME_ERROR` examples from intentionally unhandled sample tasks.
- Checked and removed temporary Python bytecode cache; no `__pycache__` directories remain.

## 2026-08-30

### Stage 5: Cluster Pre-flight Complete

- Connected to `login-3.cluster02.eee.ntu.edu.sg` and rechecked live Slurm, account, QoS, GPU, and storage limits.
- Confirmed account `msc` with default QoS `msc`; selected the fully idle `gpu-5090-2` as the preferred smoke-test node, subject to allocation-time availability.
- Verified official cluster guidance through short CPU Slurm jobs; code may stay under home, while datasets, environments, logs, checkpoints, and results must use project SSD.
- Created the former `cabinagentrl` project through `storagemgr` with 150 GB SSD. This was the initial deployment root before the Stage 9 migration.
- Uploaded and extracted the code-only baseline at `/home/jiatian001/CabinAgent-RL`; local and remote deployment archive SHA-256 values match.

### Stage 6: Cluster Smoke Harness Added

- Added `scripts/cluster_smoke.sh` with configurable duration and heartbeat intervals.
- The harness runs the baseline and all three derived-data builders, verifies required outputs, records GPU health every minute, and exits with `SMOKE_OK` only after the requested duration.
- Updated `Project.md` Phase 0 deliverables and acceptance criteria to include the reproducible Slurm smoke path.
- First Slurm launch `129964` exited after 2 seconds because compute nodes do not expose Python by default; no baseline logic ran and the GPU was released immediately.
- Confirmed `Miniforge3/24.11.3-1` provides Python 3.12.9, then updated the harness to load that fixed module version before running checks.
- Dataset job `129963` also exited immediately because `sbatch --wrap` used `/bin/sh`, where `pipefail` is unsupported; the retry uses POSIX-compatible shell options and the target directories remain clean.

### Stage 6: Remote Deployment Complete

- Created the initial private SSD working tree under the former `cabinagentrl` project and added its data, logs, checkpoints, cache, and environment subdirectories.
- Synced the baseline code and updated smoke harness to the SSD directory; `bash -n` passes on the cluster.
- Started corrected dataset download job `129966` on `cpu-1` and corrected 30-minute smoke job `129967` on `gpu-5090-2`.
- Smoke initialization passed Python 3.12.9, RTX 5090 visibility, 9 baseline trajectories, and SFT/preference/PRM-Lite data generation before entering the heartbeat window.

### Stage 7: Thirty-Minute Cluster Smoke Complete

- Slurm job `129967` completed on `gpu-5090-2` with exit code `0:0` after 30 minutes and 2 seconds.
- The harness emitted `SMOKE_OK` at 1,800 seconds after uninterrupted minute-level Python, output, and GPU health checks.
- Confirmed `squeue --me` is empty and `gpu-5090-2` returned to fully idle state with all four RTX 5090 GPUs free.

### Stage 8: Official Dataset Download In Progress

- Slurm job `129966` cloned CAR-bench commit `54990894241f2c07e9b523928c2a29e9b693d313` and Gorilla/BFCL commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` to project SSD.
- Verified BFCL V4 JSON data is included directly in the Gorilla repository and no dataset files use Git LFS.
- CAR-bench's repository contains only reference copies; its official Hugging Face snapshot has 18 files totaling 720,638,694 bytes and still needs to be downloaded.
- Added `scripts/download_hf_dataset.py` to download public dataset snapshots with path validation, size checks, and LFS SHA-256 verification.
- Slurm job `130045` downloaded and verified all 18 CAR-bench snapshot files (720,638,694 bytes) with exit code `0:0` in 47 seconds.
- Added `scripts/verify_official_data.py` for final JSONL parsing checks across official CAR tasks, CAR mock data, and BFCL V4 data.
- First verification job `130051` parsed all 254 CAR tasks, then found that some nested BFCL `*.json` files are formatted JSON documents rather than JSONL; the job exited without modifying data.
- Updated the verifier to accept both complete JSON documents and line-delimited JSON while preserving file/line error context.

### Stage 8: Official Dataset Download Complete

- Verification job `130054` completed with exit code `0:0` and emitted `OFFICIAL_DATA_OK`.
- CAR-bench validation: 6 task files, 254 task records, and 10 mock-data files; the separate Hugging Face snapshot contains 18 fully downloaded and checksum-verified files.
- BFCL validation: 51 BFCL V4 JSON/JSONL files and 9,673 parsed records, including companion answer files and retained official unused categories.
- Final SSD locations:
  - `/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/data/official/car-bench`
  - `/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/data/official/car-bench-dataset`
  - `/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/data/official/gorilla`
- Final audit: `squeue --me` is empty, no `.part` files remain, reports are non-empty, and the project uses about 1.6 GB of its 150 GB SSD allocation.

### Stage 9: SSD Path Migration Pre-flight Complete

- Confirmed no active Slurm jobs before migration.
- Resolved `/projects/cabinagentrl` to the managed SSD project `/projects/_ssd/cabinagentrl`; current usage is about 1.6 GB.
- Confirmed `/projects/jiatian001ssd` does not yet exist and must be provisioned through `storagemgr` before data can be moved safely.
- Local source/config search found no executable code hardcoding the old path; only historical planning and progress records require path updates.

### Stage 9: Destination SSD Provisioned

- Temporarily reduced the former `cabinagentrl` allocation to 10 GB, safely above its current 1.6 GB usage.
- Created `/projects/jiatian001ssd` through `storagemgr` with 140 GB SSD allocation.
- Source data remains unchanged; deletion is deferred until a Slurm copy and full verification succeed.

### Stage 9: Copy And New-Path Verification Complete

- Slurm job `130531` copied 1,234 files (1,555,401,389 bytes) to `/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL`.
- Checksum verification produced an empty difference file; source and destination file counts and byte totals match exactly.
- Slurm job `130532` completed with exit code `0:0` from the new path, passing baseline evaluation, all three data builders, official dataset parsing, and report checks.
- The old source remained intact at this checkpoint pending the explicitly requested final cutover.

### Stage 9: SSD Path Migration Complete

- The first protected cutover job `130534` stopped before deletion because regenerated reports had timestamp-only differences; the old source remained intact.
- Slurm job `130535` repeated the full checksum comparison while ignoring metadata-only timestamp changes, found no content differences, and removed only the exact verified old working tree.
- Deleted the now-empty former `cabinagentrl` storage project and expanded `jiatian001ssd` from 140 GB to the full 150 GB SSD allocation.
- Moved migration audit logs beneath the canonical project tree and confirmed the former `/projects/cabinagentrl` entry no longer exists.
- Canonical cluster root is now `/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL`; no executable script or configuration required a path change.

### Stage 10: SFT-to-GRPO Experiment Design Complete

- Removed DPO/RLAIF from the active project flow and retired the old DPO and two placeholder GRPO configs/scripts.
- Rewrote `Project.md` around one frozen Qwen2.5-7B SFT parent and five independent GRPO branches: E10 Vanilla, E11 Turn-Discount, E12 LATA, E13 PRM-Lite, and E14 PRM-Lite + LATA.
- Fixed formal training to 250 steps with checkpoint/evaluation points at 50, 100, 150, 200, and 250; the former R05/step-300 expansion is not scheduled.
- Added `refine-logs/EXPERIMENT_PLAN.md` and `refine-logs/EXPERIMENT_TRACKER.md` with claims, controls, run order, stopping rules, result-to-claim requirements, and per-attempt records.

### Stage 10: Reward And Advantage Scaffolding Complete

- Reworked PRM-Lite into CAR-specific deterministic rule events with a clipped `[-0.5, 0.5]` process score and fixed `outcome + 0.3 * process_score` composition for E13/E14.
- Added project-local veRL estimators for Turn-Discount and LATA, including stable log-space weights and external registry wiring so installed veRL source does not need modification.
- Added a strict veRL reward adapter that requires the completed environment trajectory from the future CAR agent loop; it fails instead of silently scoring response text without state.
- Added experiment manifests for all five branches and model/runtime configs for the 72B-AWQ simulator, one-GPU 7B trainer, and two-node Pro 6000 allocation.

### Stage 10: Dual-Node Slurm Scaffolding Validated

- Live cluster inspection confirmed 96 GB Pro 6000 GPUs, `gpu:pro6000` GRES, `highmem` eligibility, and the cluster plugin's automatic 4 CPU / 90 GiB allocation per one-GPU node.
- Added a job-specific vLLM simulator service, run manifest creation, veRL override renderer, atomic two-node Slurm orchestrator, one-shot Pro 6000 status command, and submission wrapper without login-node polling.
- Local validation passed Python compile, all YAML parsing, launch command rendering, 7 unit tests, the 9-sample baseline, and 9-row PRM-Lite regeneration.
- Run manifest creation was exercised in a temporary directory; config/common/source SHA-256 fields and required run subdirectories were generated correctly, then the temporary run was removed.
- Cluster validation passed `bash -n` for all new shell/Slurm scripts and `sbatch --test-only` for a two-node Pro 6000 allocation; no GPU job was submitted and `squeue --me` remained empty.
- The cluster's base Miniforge module does not include PyYAML, so the veRL renderer is intentionally deferred to the named GPU environment where `requirements-gpu.txt` installs it; local rendering for all five configs passed.
- The formal launch deliberately remains blocked until `data/processed/carbench/{train,dev}.parquet`, `checkpoints/sft_lora`, and `configs/agent_loop/carbench.yaml` exist. These are Stage 11-12 integration gates, not completed training claims.

### Stage 11: Direct-RL Main Path Complete

- User confirmed that Qwen2.5-7B-Instruct should enter reinforcement learning directly instead of requiring project-specific SFT first.
- Removed `checkpoints/sft_lora` and `actor_rollout_ref.model.lora_adapter_path` from formal veRL prerequisites; every E10-E14 run now creates fresh LoRA weights from the same Qwen2.5-7B-Instruct revision.
- Replaced the former E01 SFT parent in the experiment plan/tracker with G00, a real CAR outcome-only rollout gate.
- Added `configs/train/direct_rl_gate.yaml`, `src/eval/rollout_gate.py`, `scripts/check_rollout_gate.py`, and unit tests for passing, all-zero, and malformed rollout groups.
- Gate thresholds are fixed at 80 trajectories, parse rate >= 0.95, executable rate >= 0.85, mixed outcome group ratio >= 0.20, loop/max-turn rate <= 0.20, and at least one success.
- The gate CLI loads every threshold from `configs/train/direct_rl_gate.yaml`, preventing command defaults and the documented experiment contract from drifting apart.
- Renamed the old SFT config/launcher to guarded `sft_fallback_lora` artifacts. They are disabled by default and allowed only after a persisted gate failure; fallback runs form a separate experiment family.
- Updated `Project.md`, README, AGENTS, task plan, findings, experiment plan, and tracker to reflect the direct-RL data chain and execution order.
- Local validation passed 10 unit tests, 15 YAML files, five Direct-GRPO dry-run renderings with no SFT adapter path, and the nine-sample baseline.
- The updated tree was synchronized to the canonical cluster root; remote shell syntax and two-node `sbatch --test-only` passed, no GPU task was submitted, and `squeue --me` remained empty.

### Stage 12: Official CAR Online Runtime Implemented

- Rechecked the live cluster rules and the newly documented Lmod workflow. The account remains `msc`; Miniforge3 `24.11.3-1` and CUDA `12.8.0` are available, and no user job was active during pre-flight.
- Live Pro 6000 inspection found one free card on `gpu-pro6000-6`, but no free `highmem` Pro 6000. The free low-host-memory card was not used for the 72B-AWQ simulator.
- Added `src/integrations/car_bench_runtime.py`, which reads the complete downloaded CAR JSONL directly, restores hidden tasks only on the environment side, initializes isolated CAR context variables, executes official tools, calls the local 72B simulator with structured output, and applies automatic-only policy checks without an LLM policy evaluator.
- Added the veRL v0.9 token-in/token-out loop in `src/training/car_bench_agent_loop.py`. It preserves generated-token and observation masks, uses the Hermes/Qwen tool-call format, emits full normalized trajectories, and attaches deterministic CAR/PRM-Lite reward before veRL post-processing.
- Added leakage-safe `103/26` train/dev parquet construction plus a stratified 20-task G00 set. Persona, instruction, action ground truth, and initial context are excluded from parquet rows and policy prompts.
- Added standalone two-vLLM G00 rollout collection, a three-family structured simulator smoke, local model snapshot download, and policy vLLM serving.
- Added CPU Slurm jobs for the Conda environment, model snapshots, and parquet generation; added single-GPU simulator smoke, two-GPU G00, and a dependency-controlled pipeline that runs a 2-step E10 trainer smoke before sequential E10-E14 formal jobs.
- Updated trainer defaults for one 96 GB policy GPU: 4 tasks x 4 rollouts, 32K prompt plus 8K accumulated response, LoRA 32/32, FSDP parameter/optimizer offload, async vLLM, and automatic checkpoint resume.
- Local validation passes 12 unit tests, Python compilation, 16 YAML files, and Direct-GRPO command rendering. Cluster shell syntax, Slurm validation, environment installation, model download, parquet generation, and GPU execution remain the next Stage 12 actions. Two sandbox-created test directories were removed after exact-path verification.

### Stage 12: Initial Pipeline Submission Partially Accepted

- Submitted pipeline `20260830T141431Z`. Slurm accepted environment `131248`, model download `131249`, data preparation `131250`, simulator smoke `131251`, and G00 `131252`.
- The live `msc` QoS rejected the sixth and later submissions with `QOSMaxSubmitJobPerUserLimit`. No training smoke or formal experiment job was created, although their run directories had already been initialized.
- Fixed the submit helper so an `sbatch` failure cannot be recorded as an empty successful job ID.
- Reworked post-gate scheduling into success chaining: G00 submits only the trainer smoke after its gate check passes; the smoke and each formal ablation submit exactly one successor after successful completion. This respects the five-submitted-job limit without an idle polling job.
- With explicit user approval, cancelled only the stale pending G00 job `131252`; environment `131248`, model `131249`, data `131250`, and simulator smoke `131251` were left unchanged.
- Prevented stale pre-created run manifests from being reused: every actual submission now selects a fresh `_rN` attempt directory when needed and records submitted/running/completed/failed Slurm metadata atomically, including signal-driven exits.
- Refactored the manifest lifecycle test to a filesystem-independent unit test after Windows sandbox ACLs rejected a temporary-directory fixture; this keeps the test deterministic and avoids touching run artifacts.
- Local validation after the scheduling fix passed 13 unit tests and Python compilation; remote shell syntax, the focused manifest test, and dependency-free `sbatch --test-only` for G00/trainer also passed.
- Environment job `131248` completed successfully in 20m45s. Model job `131249` then failed after 18s because the cluster's shared Hugging Face egress IP returned HTTP 429 with `Retry-After: 153`; Slurm correctly cancelled dependent data `131250` and simulator smoke `131251`.
- Added bounded model-download retries that honor `Retry-After`, back off on 429/5xx/network timeouts, limit snapshot workers to four, retain partial local snapshots, and never persist a token. The resumed chain starts at model download rather than rebuilding the verified environment.
- Remote shell syntax and two focused download retry tests passed. Submitted resumed models `131280`, data `131281`, 30-minute simulator smoke `131282`, and replacement G00 `131283`; all four were accepted under the five-job QoS limit.
- G00 `131283` carries `PIPELINE_ID=20260830T141431Z`. On PASS it will create a fresh `smoke_e10_20260830T141431Z_r1` manifest and submit the two-step trainer smoke; no stale pre-created run directory is reused.
- Models `131280` completed successfully in 13m42s with both 7B and 72B-AWQ marked ready; CAR parquet job `131281` then completed in 16s with train/dev/gate counts `103/26/20`.
- Simulator smoke `131282` acquired `gpu-pro6000-7` but failed after one second because the Windows-created archive did not retain execute permission on `serve_simulator_vllm.sh`; dependent G00 `131283` was cancelled automatically and no model was loaded on the GPU.
- Replaced every in-job invocation of repository shell entrypoints with explicit `bash` calls and normalized signal handling, removing reliance on Unix executable mode after cross-platform synchronization. The next retry starts at simulator smoke and reuses the verified environment, models, and parquet.
- Remote syntax checks, artifact validation, and simulator `sbatch --test-only` passed. Submitted corrected simulator smoke `131298` and dependent G00 `131299`; all high-memory Pro 6000 nodes were allocated, so `131298` is pending on priority and will start automatically when a compatible card is released.
- Simulator smoke `131298` automatically acquired `gpu-pro6000-7` and completed with exit `0:0` in 4m51s. vLLM loaded all 11 AWQ shards (38.76 GiB), exposed a healthy endpoint, and generated valid first-user messages for base, hallucination, and disambiguation CAR tasks; the result is `reports/simulator_smoke_131298.json`.
- Startup profiling reported 39.7 GiB KV cache and 15.88x concurrency at 8192 tokens, but also revealed that vLLM compiled into `/home/jiatian001/.cache/vllm` and that `awq_marlin` is supported and faster than forced `awq`. G00 `131299` was placed on reversible user hold before smoke completion so it could not launch with those settings.
- With explicit user approval, cancelled only held G00 `131299`. Added a shared compute-node runtime environment that redirects vLLM, TorchInductor, Triton, CUDA, FlashInfer, Hugging Face, Ray, and temporary caches to the project SSD.
- Changed the 72B simulator default from forced generic AWQ to vLLM's detected AWQ-Marlin kernel. This is a serving-kernel optimization only; model weights, simulator prompts, sampling settings, reward, and ablation semantics remain frozen.
- Remote syntax checks confirmed every runtime cache resolves beneath the canonical SSD root; optimized simulator and G00 `sbatch --test-only` passed. Submitted AWQ-Marlin/SSD-cache smoke `131323` and dependent G00 `131324`.
- Optimized smoke `131323` completed on `gpu-pro6000-7` with exit `0:0` in 3m50s, confirming AWQ-Marlin and project-SSD caches. G00 `131324` later acquired `gpu-pro6000-[3,7]` but failed in 33s before rollout because policy vLLM rejected `max_model_len=40960` against Qwen2.5-7B's 32768-token model limit; the simulator process was then cancelled by job cleanup and no training was submitted.
- Corrected the shared contract to `24576` prompt plus `8192` cumulative response tokens, added an explicit `policy_max_model_len=32768`, and derived rollout/actor/reference limits from that sum. Added a five-ablation regression test that rejects any return of the invalid 40960-token budget.

## 2026-08-31

### Stage 13: G00 Result Analysis Complete

- Corrected-context G00 job `131880` completed 80 trajectories on `gpu-pro6000-[1,4]` in 5m54s; both vLLM services and the CAR environment remained healthy, then the fixed gate stopped the pipeline before training.
- Archived the raw artifact locally at `reports/cluster/G00-131880/trajectories.jsonl` and re-evaluated it with the strengthened group contract.

| Metric | G00 value | Threshold | Result |
|---|---:|---:|---|
| tool-call parse rate | 0.996667 | >= 0.95 | pass |
| executable tool rate | 0.989062 | >= 0.85 | pass |
| mixed outcome group ratio | 0.10 | >= 0.20 | fail |
| initial-user consistency | 0.85 | 1.00 | fail |
| loop/max-turn rate | 0.0125 | <= 0.20 | pass |
| successful trajectories | 27 | >= 1 | pass |

- Of 20 complete groups, 12 were all-failure, 6 all-success, and only 2 mixed; failure codes were dominated by argument errors (`22`) and safety-boundary errors (`19`). Three groups did not share one identical first user message, including one hallucination task that emitted `###STOP###` before the policy acted.
- Interpretation: the tool/runtime baseline is viable, but G00 cannot yet establish a valid same-state grouped comparison or the required reward variance. The next run must repair group initialization and increase reproducible policy exploration without changing reward or thresholds.

### Stage 13: G01 Adjustment Implemented

- Added a strict initial simulator contract: the first user turn uses greedy decoding, must return `CONTINUE`, and retries with a corrective first-turn instruction before any invalid response can mutate simulator state. Later user turns remain at `temperature=0.2`.
- Centralized G01 policy sampling in `configs/train/direct_rl_gate.yaml`: `temperature=1.0`, `top_p=0.95`, global seed `42`, deterministic per-task/per-trial request seeds, 20 tasks, group size 4, and unchanged outcome reward.
- Added `consistent_initial_user_group_ratio == 1.0` to the gate, persisted all sampling metadata per trajectory, and made the Slurm gate label configurable so G01 artifacts cannot overwrite G00.
- Local verification passed all 18 unit tests, Python compilation, YAML sampling assertions, and G00 reanalysis. The strengthened report correctly rejects G00 at mixed ratio `0.10` and initial-user consistency `0.85`.
- Next action: synchronize the prepared G01 code to the canonical SSD tree, run remote syntax/config pre-flight with no active jobs, then submit `car-g01`. PASS automatically submits the 2-step E10 trainer smoke; FAIL activates the separately tracked minimal-SFT fallback decision.

### Stage 13: G01 Submitted

- Synchronized the 32 KiB Stage 13 archive to the canonical SSD tree; local and remote SHA-256 both equal `4dcfcf6368f0108ee72c45fddd0aada642b8614de9dddbf9f76b706911b7bc66`.
- Remote pre-flight passed Python compilation, gate-focused unit tests, YAML assertions, `bash -n`, and a two-node Pro6000 `sbatch --test-only`. The cluster's unrelated installed `scripts` package prevents namespace-style discovery of `test_carbench_data.py`; local full discovery already passed all 18 tests.
- Submitted G01 as Slurm job `131911` with `PIPELINE_ID=20260830T141431Z` and isolated `GATE_RUN_LABEL=G01`. It requests two high-memory Pro6000 nodes atomically and is pending on priority because every Pro6000 node was allocated at submission time.
- On gate PASS, job `131911` will submit a fresh 2-step E10 trainer-smoke attempt. On gate FAIL, no GRPO job will be submitted and the persisted report will determine the minimal-SFT fallback.

### Stage 13: G01 Attempt 1 Startup Failure Diagnosed

- Job `131911` acquired `gpu-pro6000-[1,7]` but exited after 3 seconds before loading either model. Both service logs show the base Miniforge Python could not import `vllm`; no trajectory or gate report was produced and both GPUs were released immediately.
- Root cause: the pre-flight shell had the project Conda prefix active, and `sbatch --export=ALL` inherited its Conda state. Inside nested `bash -lc`, `module purge/load` reset PATH while legacy `source activate` incorrectly treated the inherited prefix as already active.
- Replaced nested activation in both the G01 gate and formal dual-node trainer with `conda shell.bash hook` plus explicit `conda activate`. This makes compute-node activation independent of the submit shell and fixes the same latent risk before trainer smoke.
- Attempt `131911` remains archived as an infrastructure failure; it does not count as the controlled G01 rollout result. A new Slurm job ID will be used after remote contaminated-shell activation tests and `sbatch --test-only` pass.

### Stage 13: G01 Rollout Result Analyzed

- G01 attempt 2 (`131930`) completed 80 trajectories on `gpu-pro6000-[1,3]` in 5m53s. Parse `0.994479`, executable `0.864625`, and loop/max-turn `0.10` passed; 15 trajectories succeeded.
- Mixed outcome groups improved from `2/20` (`0.10`) to `3/20` (`0.15`) but remained one group below the frozen `0.20` threshold. Initial-user consistency also remained `17/20` (`0.85`), so no trainer smoke was submitted.
- All three inconsistent groups contained only minor wording variants at greedy temperature, confirming that concurrent vLLM requests are not a sufficient equality guarantee. Per-trial policy seeds and sampling metadata were present and correct.
- G01 is not accepted as the final controlled gate because the same-initial-user contract was not satisfied. Added a thread-safe cache keyed by simulator model and the complete initial prompt hash: each task generates one validated first-turn structured response, all four rollouts reuse it, and different tasks remain parallel.
- G02 will rerun the unchanged G01 task set, reward, thresholds, and policy sampling. If initial-user consistency reaches `1.0` but mixed outcome remains below `0.20`, the next stage is minimal SFT rather than another gate sampling sweep.

### Stage 14: Valid Direct-RL Gate Failure Confirmed

- G02 job `131950` completed all 80 trajectories on `gpu-pro6000-[3,8]` in 5m50s. The repaired grouped contract reached initial-user consistency `1.0`.
- Parse `0.995729`, executable `0.888892`, loop/max-turn `0.075`, and 12 successful trajectories passed their thresholds, but mixed reward groups were `0/20` (`0.0 < 0.20`).
- This is the first valid same-initial-state gate conclusion: successful trajectories exist, but each four-rollout group has a constant outcome and cannot provide a GRPO group advantage. No trainer job was submitted.
- Per the frozen stopping rule, Direct-RL sampling adjustment ends at G02. The main E10-E14 family remains blocked and the project moves to a separately identified minimal-SFT fallback family.

### Stage 14: Executable Minimal-SFT Fallback Implemented

- Replaced the guarded placeholder with a real Transformers/PEFT Qwen2.5-7B LoRA trainer. It masks all context tokens, supervises only assistant language/tool-call tokens, enables bf16 and gradient checkpointing, and persists the final adapter, train/eval metrics, tokenization statistics, and CUDA memory peak.
- The fallback dataset builder now merges G00/G01/G02 environment-success trajectories, preserves complete CAR tool schemas and multi-turn messages, avoids duplicate final responses, deduplicates identical conversations, and splits train/validation by task rather than row.
- Fixed the fallback contract at LoRA rank/alpha `16/32`, max length `32768`, one epoch, learning rate `2e-4`, batch size 1 with gradient accumulation 4, and seed 42. This training family is not comparable as a direct E10-E14 initialization.
- Added CPU Slurm data preparation plus a one-Pro6000 F00 2-step/4-record smoke. A successful smoke submits the full one-epoch F01 job; failed attempts retain separate run directories and lifecycle manifests.
- Local validation passed 22 unit tests, Python compilation, CLI loading, and all YAML parsing. Real Qwen tokenizer length statistics, remote shell syntax, data counts, and GPU trainer compatibility remain the Stage 14 cluster pre-flight.
- Added a CPU-side tokenizer pre-flight that rejects insufficient or fully overlength fallback data before any GPU is allocated. Added an explicit `scripts` package marker so cluster test discovery resolves project entrypoints instead of an unrelated installed package.
- First CPU data attempt `131984` built 38 train and 14 validation records, then correctly failed before GPU submission because the progressive-prefix masking method produced zero labeled examples under the installed Transformers 5.10 Qwen template. Replaced it with direct Qwen assistant-boundary masking over each complete multi-turn conversation; attempt `131984` remains recorded as a preprocessing failure.
- Second CPU attempt `131992` confirmed the record-level masking path but still produced zero examples. The root cause was version compatibility: Transformers 5.10 returns a `BatchEncoding` mapping, while the token helper recognized only a built-in `dict` and treated mapping keys as tokens. The helper now accepts generic mappings and `input_ids` attributes, with a regression test using a non-dict mapping.
- Third CPU attempt `131999` completed in 10 seconds: 38 train and 14 validation conversations produced 38/14 labeled examples, with train lengths 3226-4309 tokens and validation lengths 3221-3298; no examples were overlength or missing assistant targets.
- Updated the fallback stage submitter to call the project Conda Python by absolute prefix, so a successful F00 can submit F01 from a clean batch shell without inheriting login-node activation state.

### Stage 14: F00 Smoke Passed, First F01 Attempt Invalidated

- F00 job `132008` completed on `gpu-pro6000-1` with exit `0:0` in 39 seconds. Two optimizer steps ran without NaN/OOM, train loss was `3.608993`, validation loss was `1.909820`, peak allocated CUDA memory was `21.165 GiB`, and the 161.5 MB final adapter plus completed manifest were saved.
- F00 automatically submitted F01 job `132013`, which also exited `0:0`, but its persisted metrics prove it inherited the smoke limits: only 4 train records and 2 optimizer steps were used. It is therefore an invalid full-run attempt, not a completed F01 result.
- Root cause: `sbatch --export=ALL` propagated `MAX_TRAINING_STEPS=2` and `TRAIN_MAX_RECORDS=4` from the smoke batch shell. The full branch now explicitly exports both values as `-1`, overriding inherited smoke settings.
- The `132013` run directory, adapter, metrics, and manifest remain archived. A fresh `_r1` F01 attempt must report 38 train records before it can be accepted as the full one-epoch fallback.

### Stage 14: Full Minimal-SFT F01 Complete

- Corrected F01 attempt 2 (`132020`) completed on `gpu-pro6000-1` with exit `0:0` in 53 seconds. It used all 38 train conversations for one epoch and 10 optimizer steps, then evaluated all 14 validation conversations.
- Train loss was `0.902976`; validation loss was `0.584203`. Loss decreased from `2.119` at the first step to `0.329` at the last, gradients remained finite, and peak allocated CUDA memory was `22.91 GiB`.
- The accepted adapter is `experiments/sft_fallback_full_20260830T141431Z_r1/checkpoints/final_adapter` (161.5 MB adapter weights), with completed manifest and machine-readable metrics. F01 does not alter or replace the direct E10-E14 initialization.
- Added an adapter-aware vLLM policy serving path for G03. At submission time, G03 was defined as the frozen 20x4 gate using base Qwen2.5-7B plus the F01 adapter, with adapter paths recorded per trajectory; the later Stage 15 entry records why attempt 1 failed and invalidated that adapter.
- Made the post-gate action explicit. Direct G00-style gates retain `direct_smoke`, while G03 is submitted with `POST_GATE_ACTION=none` so a PASS cannot accidentally launch the blocked E10 family before F10 configuration exists.

### Stage 15: G03 Attempt 1 Failed And Invalidated The F01 Adapter

- Live Slurm accounting confirms G03 job `132043` is finished with state `FAILED`, elapsed `00:06:10`, and exit code `1:0`; `squeue --me` is empty.
- The job allocated two Pro 6000 GPUs, one each on `gpu-pro6000-1` and `gpu-pro6000-7`. The policy vLLM successfully loaded the F01 LoRA, so this was not a model-loading or GPU-allocation failure.
- The rollout collector reached the official CAR environment, but tool calls repeatedly arrived with empty kwargs or string-valued arguments. CAR automatic policy evaluation then indexed `arguments["on"]` and raised `TypeError: string indices must be integers, not 'str'`.
- Root cause: fallback SFT records preserved OpenAI `function.arguments` as JSON strings; the Qwen chat template applied `tojson` again, so F01 learned double-encoded tool arguments. Its low train/eval loss therefore measured fitting to an invalid target format.
- No G03 gate report or trainer checkpoint was produced, and `POST_GATE_ACTION=none` prevented any GRPO submission. F01 attempt 2 remains archived as a technically completed Slurm run but is rejected as a downstream policy initializer.
- Required correction: normalize arguments to objects before templating, add a template/parser round-trip regression test, rebuild data, rerun CPU tokenizer validation, F00, full F01, and then G03 attempt 2.

### Stage 15: Experiment Documentation Split By Stage

- Added `docs/实验阶段/实验阶段总览.md` plus six independent stage documents covering settings, GPU use, execution results, implemented improvements, remaining blockers, and next actions.
- Updated `Project.md`, the experiment plan, tracker, and task plan so G03 is no longer shown as pending and downstream F10-F14 remain blocked until a corrected G03 PASS.

### Stage 15: Mandatory Stage-Document Workflow Added

- Added an explicit experiment-to-document Map in `AGENTS.md` for stages 01-06, covering infrastructure tasks, G00-G03, F00/F01, E10-E14, F10-F14, and frozen CAR/BFCL evaluation.
- Every experiment implementation, retry, improvement, evaluation, or analysis must first load `Project.md`, `Progress.md`, the stage overview, and every mapped stage document involved in the operation.
- Every attempt, including failed, interrupted, and invalid runs, must be appended to its stage document before the next submission with four separate sections: experiment settings, execution result, improvement reason, and improvement measures.
- The same update must synchronize `Progress.md`, `refine-logs/EXPERIMENT_TRACKER.md`, and, when stage status or route changes, `Project.md` plus the stage overview. Existing failed attempts may not be overwritten or removed.

## 2026-09-01

### Stage 16: Corrected Fallback And Single-Node Dual-GPU Submission Prepared

- Loaded the stage overview plus Stage 02 and Stage 04 documents before implementation, as required by the experiment-document Map.
- Rechecked live cluster limits: account/QoS remains `msc`, the QoS permits two Pro 6000 GPUs per user, the user queue was empty, and Pro 6000 nodes expose 4-10 GPUs each; all highmem cards were allocated at inspection time.
- Replaced both gate and GRPO `--nodes=2` requests with `--nodes=1 --gres=gpu:pro6000:2`. The initial independent-step implementation was later superseded by the single-step/two-task binding recorded below.
- Added Job-ID-scoped allocation, simulator, policy, rollout, gate-check, and trainer logs in addition to Slurm stdout/stderr.
- Fixed fallback SFT messages to use object-valued function arguments, reject double encoding, and validate the real Qwen chat-template tool-call round trip during the CPU tokenizer gate.
- Added automatic corrected fallback chaining: CPU data/tokenizer gate -> F00 smoke -> F01 full -> G03, while preserving `POST_GATE_ACTION=none` after G03.
- Local verification passed 26 unit tests, Python compilation, YAML parsing, and static single-node dual-GPU assertions. Remote shell/Slurm pre-flight and submission remain next.
- Added a short same-node dual-GPU binding smoke that logs allocation and each step, rejects identical/empty `SLURM_STEP_GPUS`, and becomes an `afterok` dependency of G03 while CPU data and one-GPU F00/F01 may proceed independently.

### Stage 16: Same-Node Dual-GPU Smoke Attempt 1 Diagnosed

- Smoke job `132933` acquired two Pro 6000 GPUs on the same physical node `gpu-pro6000-7`; Slurm allocation recorded physical GPU indices `0,4` and both one-GPU steps exited successfully.
- The batch check failed because each near-instant step saw GPU 0: the first step released its GPU before the second allocation was stably concurrent, so the smoke measured sequential reuse rather than simultaneous binding.
- The smoke now keeps both steps alive for 10 seconds before comparison. Attempt 2 will receive a fresh Job ID, and G03 must depend on the replacement PASS rather than failed job `132933`.

### Stage 16: Corrected Fallback Training Completed, Step Model Revised

- CPU job `132934` completed in 1m02s. The real Qwen template gate validated 38/14 train/val records, 45/12 tool calls, zero overlength rows, and zero missing targets.
- Corrected F00 `132935` completed in 45s on one Pro 6000; corrected full F01 `132942` completed in 1m17s on one Pro 6000 and produced a new final adapter.
- Smoke attempt 2 `132950` confirmed that two independent `srun --exclusive` steps are serialized by this cluster's step-level resource allocation, even when the job owns two GPUs. This launch model is rejected for G03 and GRPO.
- Added a single-step model: one `srun` launches two tasks with `--gpus-per-task=1 --gpu-bind=single:1`; task 0 owns the simulator and task 1 owns policy/rollout. Shared sentinel cleanup stops the simulator after gate completion, and all role/service logs remain separate.

### Stage 16: Single-Step Binding Passed And G03 Attempt 3 Started

- Single-step smoke `132966` completed on `gpu-pro6000-3` in 10 seconds. Its two tasks reported distinct GPU UUIDs and emitted `SAME_NODE_DUAL_GPU_OK`.
- Submitted corrected G03 attempt 3 as job `132967`, depending on successful smoke `132966`. Slurm started it on `gpu-pro6000-10` with one node, two tasks, two Pro 6000 GPUs, eight CPUs, and 180 GiB node memory.
- The 72B-AWQ simulator and 7B+corrected-LoRA policy both loaded successfully and returned healthy endpoints. Allocation, per-role, simulator, policy, rollout, gate-check, and Slurm logs are present and Job-ID scoped.
- Old auto-submitted G03 `132946` remains pending with `DependencyNeverSatisfied` on failed smoke `132933`. It references the old launcher and will not be cancelled without explicit approval.

### Stage 16: G03 Attempt 3 Completed With A Valid Gate Failure

- G03 `132967` completed all 80 trajectories on `gpu-pro6000-10` using one physical node and two Pro 6000 GPUs. Both services were healthy; the single-step/two-task launcher and corrected tool-argument format worked end to end.
- The gate failed three frozen thresholds: executable `0.844693 < 0.85`, mixed group `0.15 < 0.20`, and loop/max-turn `0.2625 > 0.20`. Parse `0.999375`, initial-user consistency `1.0`, and 14 successful trajectories passed.
- Group outcomes were 15 all-fail, 2 all-success, and 3 mixed. Failure counts were argument error 32, loop/verbose 21, safety boundary 16, capability hallucination 7, and tool name 1.
- All 12 `hallucination_missing_tool_parameter` trajectories failed with mean executable rate about 0.42, identifying the clearest corrective-data target. F10-F14 remain blocked; thresholds will not be relaxed and G03 will not be repeated unchanged.
- Archived the 80 trajectories and machine-readable report locally under `reports/cluster/G03-132967/`.

### Stage 17: F02/G04 Corrective Attempt Prepared

- Reloaded `Project.md`, `Progress.md`, the experiment overview, Stage 04, and Stage 05 before changing the experiment route.
- Rechecked the live `msc` association and Pro6000 inventory, then cancelled explicitly authorized stale job `132946`; Slurm now records it as `CANCELLED` and the user queue became empty.
- Added a targeted F02 data builder. It merges environment-success trajectories from G00-G03 with two no-tool corrective dialogues for each G03 missing-tool family task while preserving task-disjoint train/validation groups.
- Corrective records never include the failed assistant tool call because all assistant spans are supervised by the current Qwen SFT encoder. They teach direct capability-boundary refusal and stopping after user retry pressure.
- Added a separate F02 config, CPU data/tokenizer job, two-step smoke -> full one-epoch submission chain, and automatic G04 submission through the proven single-node/two-task launcher.
- F02 retains the same 7B base, LoRA rank/alpha `16/32`, learning rate `2e-4`, one epoch, and seed 42. G04 changes only the adapter and freezes the G03 20x4 tasks, sampling, outcome reward, and thresholds.
- Local validation passed 28 unit tests, Python compilation, and all YAML parsing. Remote shell syntax, real Qwen tokenizer checks, and Slurm Job IDs remain pending.

### Stage 17: F02 Completed And G04 Started

- CPU job `133301` completed in 54 seconds and built 66 deduplicated successful records plus 14 corrective records. The 60/20 train/val split covers 9/2 tasks with zero overlap.
- Real Qwen tokenization produced train lengths 3226-10468 and val lengths 3221-3297, zero overlength/no-target rows, and 80/16 round-tripped tool calls.
- F02 smoke `133303` completed two steps on `gpu-pro6000-8` with train/eval loss `3.312624/2.140017` and peak memory `21.16 GiB`.
- Full F02 `133306` completed 15 steps on `gpu-pro6000-8` using all 60/20 records. Train/eval loss was `1.088965/0.739128`, peak memory was `34.304 GiB`, and the final adapter was saved under `experiments/sft_corrective_full_f02_20260901_stage17/checkpoints/final_adapter`.
- G04 `133308` was automatically submitted only after F02 completion. Slurm allocated one physical node `gpu-pro6000-7`, two Pro6000 GPUs, two tasks, eight CPUs, and 180 GiB node memory; Job-ID-scoped Slurm and component logs are configured.
- G04 preserves the G03 20x4 task set, policy/simulator sampling, outcome reward, and all frozen thresholds. `POST_GATE_ACTION=none` prevents any unreviewed F10 submission.

### Stage 17: G04 Completed With A Valid Scientific Failure

- G04 `133308` completed all 80 trajectories on one physical node `gpu-pro6000-7` with two Pro6000 GPUs in 5m26s. Both vLLM services loaded and the single-step/two-task topology remained healthy.
- The Slurm state is `FAILED` because the gate checker returned nonzero for a scientific FAIL, not because rollout crashed. Parse `0.995833`, initial-user consistency `1.0`, and 12 successful trajectories passed.
- Executable `0.846829`, mixed group ratio `0.10`, and loop/max-turn `0.275` failed. Groups were 16 all-fail, 2 all-success, and 2 mixed.
- Failure counts were argument error 33, loop/verbose 22, safety boundary 13, capability hallucination 8, and tool name 1.
- F02 improved missing-tool executable behavior from `0.8175` to `0.8673`, but missing-tool-parameter fell from `0.4161` to `0.2681`. Relative to G03, total executable rose only `0.002136`, mixed fell `0.05`, loop rose `0.0125`, and success fell from 14 to 12.
- F10-F14 remain blocked. The project will not relax thresholds, repeat G04 unchanged, or add epochs without an explicit new corrective-data hypothesis.
- G04 trajectories and report are archived locally under `reports/cluster/G04-133308/` for paired failure analysis.

### Stage 18: Corrected-F01 F10 Bounded Pilot Implemented Locally

- 用户确认暂停 F03/G05；G03/G04 的 FAIL 报告和阈值保持不变，但阈值只作为正式放大诊断，不再禁止有明确风险边界的短 RL pilot。
- 冻结初始化方案：把 corrected F01 `132942` 的 rank-16 LoRA safe-merge 回相同 Qwen2.5-7B 基座，生成不可覆盖且带文件哈希 manifest 的父模型；F10-F14 各自从该父模型创建 fresh rank-32 RL LoRA，不直接续训 F01 adapter。
- 新增 F10 fallback vanilla config、merge 工具与单 GPU Slurm job、单节点双 GPU/单 `srun` pilot launcher、人工分离的 step-5 start 与 step-6 resume 提交器、每角色 GPU telemetry 和逐 trajectory reward audit。移除了 pilot 的自动 successor 路径。
- `launch_verl.py` 现在允许实验级 policy parent 和仅系统语义的环境参数覆盖；group size、每步 task 数、sampling、长度、reward/advantage、LoRA、optimizer/LR、数据和 simulator 默认不变。
- Pilot 验收固定为：5 optimizer steps；至少一步 reward variance、advantage、finite nonzero gradient 均有效；KL/clip/grad norm finite；无 NaN/OOM/schema error；保存并恢复至少再走一步；记录显存/利用率/step time/wait。五步内不要求任务性能提高。
- 本地 32 项 unit tests、18 个 YAML 解析、F10 veRL dry-run rendering 与 Python compilation 全部通过。Windows 本地 WSL service 无权限，`bash -n` 按既有规则留给集群登录节点验证。Slurm 尚未提交；下一动作是实时集群 preflight、同步代码并提交 corrected-F01 parent merge，验证后才允许 F10 start。

### Stage 18: Corrected-F01 Parent Merge Submitted

- 实时 preflight 确认 `cluster02` / `msc`、用户队列为空、SSD `77.3/150 GB`、Qwen2.5-7B base 约 15 GB、corrected F01 adapter 约 165 MB 且 rank 16；目标 derived parent 不存在。
- 本地/远端部署包 SHA-256 均为 `cc1196ca28aba2cdb57d22116d1de5ca3174b3babe736555cefd181e08de7565`。远端 Bash syntax、32 tests、18 YAML、F10 dry-run、merge/F10 `sbatch --test-only` 全部通过。
- Corrected F01 parent merge 已提交为 job `133431`：1 node、1 task、1x Pro 6000、4 CPU、90 GiB memory、2 小时，当前 `PENDING (Priority)`。目标使用 PEFT safe merge、BF16 safetensors 和完整文件哈希 manifest。
- merge job 排队/运行期间冻结远端脚本；完成并验证 parent 前不提交 F10。

### Stage 18: Corrected-F01 Parent Merge Completed

- Job `133431` 于 2026-09-01 04:26 UTC 在 `gpu-pro6000-4` 启动并用 64 秒完成，Slurm state `COMPLETED`、exit `0:0`；PEFT safe merge 和四个 safetensor shards 写出无错误。
- 新父模型 `models/derived/Qwen2.5-7B-Instruct-F01-merged-20260901` 约 15 GB。manifest 记录 corrected F01 rank 16、正确 adapter digest、base/adapter 路径、BF16 merge 方法、Job ID，以及 10 个模型/tokenizer文件的 size/SHA-256。
- 下一步仍是独立 GPU load validation（全量 hash、model/tokenizer load）；通过前不提交 F10。
- 已实现独立 validation 脚本与 1x Pro 6000 Slurm job：先严格比对 manifest 的文件集合/size/SHA-256，再以 BF16 实际加载 merged model/tokenizer 并执行 one-token generation，输出 machine-readable PASS report。

### Stage 18: Parent Validation Attempt 1 Failed Before Artifact Read

- Validation job `133439` 在 `gpu-pro6000-4` 启动后 1 秒以 exit `1:0` 失败，未生成 report；错误是 file-path entrypoint 无法 import `scripts.merge_lora_parent`。
- 该失败发生在 manifest read/hash/model load 之前，没有修改 parent，也没有产生有效 GPU 指标；分类为代码封装缺陷，不是 artifact 或资源失败。
- 已修复 validation entrypoint 的项目根目录 `sys.path` 初始化；其余 hash/load/generation 协议保持冻结。完成本地与远端回归后以新 Job ID 重试，PASS 前不提交 F10。

### Stage 18: Parent Validation Attempt 2 Passed

- Retry job `133447` 在 `gpu-pro6000-2` 用 53 秒完成，Slurm state `COMPLETED`、exit `0:0`、stderr 为空。
- Report `reports/f01_parent_validation_133447.json` 为 `PASS`：10 个 inventory 文件、`15,242,726,337` bytes 的 exact set/size/SHA-256 全匹配；BF16 model 加载 `7,615,616,512` 参数，tokenizer size `151,665`，4-token input 成功生成 1 token。
- Parent merge/load dependency 已闭合。下一动作是创建全新 F10 run 并提交恰好 5 optimizer steps；作业不得自动提交 resume 或正式 successor。

### Stage 18: F10 Five-Step Pilot Submitted

- 新 run `f10_pilot_20260901_stage18` 已提交为 job `133456`；manifest source/config SHA-256 分别为 `c35a937e...46935` / `7fd65bc8...9dc05`，target 5 optimizer steps。
- 请求同一节点 2x Pro 6000、2 tasks、8 CPU、180 GiB；一个 `srun` 内 simulator/trainer 各绑 1 GPU。初始吞吐设置为 memory util `0.60`、16 seq、16384 batched tokens、16 workers、microbatch 1 和 offload enabled。
- 科学设置保持 outcome-only、4 tasks x 4 rollouts、fresh rank-32 LoRA、LR `1e-6`、seed 42、32K/20-turn、相同 CAR 数据与 72B simulator。当前 job `PENDING`，无 automatic successor。

### Stage 18: F10 Pilot Attempt 1 Failed Before veRL Import

- Job `133456` 在 `gpu-pro6000-7` 运行 2m11s 后失败：trainer direct-file entrypoint 无法 import `src`；trainer task exit 1，simulator task 按 cleanup 收到 TERM，聚合 exit 15。
- 同节点两个 task 的不同 GPU UUID、72B simulator load/health 与 telemetry 均成立；simulator 抽样峰值约 `93.3/97.9 GiB`，无 OOM。失败发生在 veRL/model/rollout/optimizer 之前，0/5 steps、无 checkpoint/reward audit。
- Run manifest 已正确标记 failed，目录完整保留。已在 `launch_verl.py` 加入 ROOT `sys.path`，并让 dry-run 实际 import 项目 entrypoint以覆盖该缺陷；回归通过后用新 run/Job ID 重试。

### Stage 18: F10 Pilot Attempt 2 Reached Ray And Failed On Socket Path

- Job `133478` 在 `gpu-pro6000-11` 运行 3m56s；launcher/veRL/Hydra config validation 均通过，证明 attempt 1 修复有效。
- Ray 初始化因 SSD `RAY_TMPDIR` 展开后的 plasma AF_UNIX socket path 超过 Linux 107-byte 上限而失败。0/5 steps、无 checkpoint/reward audit；simulator health 正常，manifest 标记 failed。
- 修复限定为把 Ray ephemeral socket/session metadata 放到 `/tmp/cabin-ray-$SLURM_JOB_ID`；持久日志/checkpoints 继续留在 SSD，对象存储继续使用共享内存。新增静态回归后以新 run/Job ID 重试，科学参数不变。
- 为避免第三次双 GPU 作业仅用于发现 Ray 启动问题，新增 10 分钟 CPU Slurm smoke：实际 `ray.init`、执行一个 remote task、`ray.shutdown` 并输出 machine-readable report。该 smoke PASS 后才提交 F10 attempt 3。

### Stage 18: Ray Short-Path CPU Smoke Passed

- CPU job `133503` 在 `cpu-1` 用 29 秒完成，Slurm `COMPLETED`、exit `0:0`。
- Machine-readable report 为 PASS：tmpdir `/tmp/cabin-ray-133503`、session dir 保持短路径、Ray address 正常、remote task 返回 42 并完成 shutdown。
- AF_UNIX 修复已用真实 Ray runtime 验证；允许创建新 run/Job ID 提交 F10 attempt 3，科学和吞吐参数不变。
- 两个先前 attempt 的 simulator telemetry 均显示默认 memory util `0.92` 会达到约 `93.3/97.9 GiB`（95.4%，余量仅 4.6%）。按已确认的系统调优边界，attempt 3 将 simulator cap 降到 `0.86`，预计保留约 10% 动态余量；max seq 仍为 16，policy/trainer cap `0.60` 暂不改。

### Stage 18: F10 Pilot Attempt 3 Reached Remote Task And Found Missing Dependency

- Job `133512` 在 `gpu-pro6000-11` 运行 2m15s；Ray short path、veRL import/config validation 和 remote `TaskRunnerV1.run()` 均成功，证明前两次 infrastructure fixes 已闭合。
- Ray worker 随后因 `ModuleNotFoundError: transfer_queue` 失败。命名环境为 veRL 0.9.0，但 installed metadata 未声明该依赖，`find_spec`/`pip show` 均确认未安装。
- 仍为 0/5 steps、无 checkpoint/reward audit，manifest 标记 failed。下一步先审计官方 package/version，再通过独立 Slurm 环境修复与 import smoke；PASS 前不再占用双 GPU。

### Stage 18: TransferQueue Repair Prepared

- 官方 veRL Python requirements 明确固定 `TransferQueue==0.1.7`；稳定容器也采用显式版本与 `--no-deps` 安装，说明它是 veRL V1 trainer 的独立运行时依赖，而当前 wheel metadata 漏装。
- 已把该版本加入 `requirements-gpu.txt`，新增 CPU Slurm 环境修复作业和 machine-readable import/API smoke；smoke 同时导入 `verl.trainer.main_ppo.TaskRunnerV1`，避免只验证孤立包。
- 安装限定在既有项目 SSD Conda 环境，使用 `--no-deps` 避免改动冻结的 Torch/CUDA/vLLM 栈。该作业 PASS 并完成 attempt 记录前，不提交新的双 GPU F10。
- CPU job `133532` 已成功安装 `TransferQueue-0.1.7` 且完成 veRL runner 联合导入，但 checker 把 Ray `ActorClass` wrapper 当作普通 class 读取 `__name__`，在写报告前失败；Slurm `FAILED`、exit `1:0`、耗时 1m19s。
- 已修正该反射字段为 wrapper type；版本/API/联合导入标准不放宽。必须用新 CPU Job ID 产生 machine-readable PASS，才能解锁 F10 attempt 4。

### Stage 18: TransferQueue CPU Smoke Passed

- Retry job `133541` 在 `cpu-1` 用 1m04s 完成，Slurm `COMPLETED`、exit `0:0`；报告 `reports/transfer_queue_smoke_133541.json` 为 `PASS` 并已归档本地。
- Installed/expected version 均为 `0.1.7`，六个 veRL 所需 KV API 全部存在，`verl.trainer.main_ppo.TaskRunnerV1` 以 `ActorClass(TaskRunnerV1)` 成功联合导入。stderr 仅为 CPU node 预期 accelerator warning。
- 缺依赖阻塞已闭合，允许创建全新 run/Job ID 提交 F10 attempt 4；科学参数及 attempt 3 的 simulator `0.86`、policy/trainer `0.60` caps 保持不变，无自动 successor。

### Stage 18: F10 Attempt 4 Found Slurm GPU Visibility Conflict

- Job `133549` 在 `gpu-pro6000-11` 运行 2m40s；已越过 TransferQueue、103/26 数据加载与 worker-group 创建，但 worker actor 因环境同时包含 `CUDA_VISIBLE_DEVICES` 和 `ROCR_VISIBLE_DEVICES` 被 veRL 主动拒绝。
- 仍为 0/5 steps、无 checkpoint/reward audit。Simulator 0.86 cap 峰值约 87.6/97.9 GiB（89.5%，10.5% headroom），双 UUID 正常且无 OOM；policy 尚未加载。
- 修复限定为在 NVIDIA task 中保留 CUDA、清除 AMD 的 ROCR/HIP visibility。新增 1x Pro 6000 Ray/veRL hook smoke；该 JSON smoke PASS 前不提交 attempt 5。

### Stage 18: NVIDIA Ray/veRL Visibility Smoke Passed

- 单卡 job `133567` 在 `gpu-pro6000-2` 用 29 秒完成，Slurm `COMPLETED`、exit `0:0`，JSON report 为 `PASS` 并已归档本地。
- 原始 Slurm 环境同时给出 CUDA `0` 与 ROCR `0`；清除 AMD vars 后，driver、Ray GPU actor 和 veRL visibility hook 后均只有 CUDA `0`，验证 Ray 不会重新注入 ROCR。
- 允许创建全新 F10 attempt 5；模型/数据/reward/优化器/LoRA 与 simulator `0.86`、policy `0.60` 保持冻结，无自动 successor。

### Stage 18: F10 Attempt 5 Found Missing FlashAttention2

- Job `133581` 在 `gpu-pro6000-8` 运行 2m36s；visibility 修复有效，首次进入 exact 7B policy module build，但 Transformers 因 veRL 默认启用 FlashAttention2 而环境无 `flash_attn`，在权重加载前失败。
- 仍为 0/5 steps、无 checkpoint/reward audit；policy GPU 约 3 MiB，simulator/topology 正常，无 OOM。当前栈为 Torch 2.11.0+cu130、CUDA 13.0、Transformers 5.10.4。
- 官方 veRL stable vLLM image 固定并 force-build `flash_attn==2.8.3`；该 tag 支持 CUDA>=12.8 的 sm_120。已准备单卡 sm_120 本地编译、FA2 前后向 finite 与 exact parent load/generate smoke；PASS 前不提交 attempt 6。
- 安装 attempt `133600` 在 42 秒内失败且未改动环境：外部 nvcc 12.8 与 Torch cu130 的 extension guard 不匹配。集群实时确认 `CUDA/13.0.0` 可直接加载；retry 仅对齐 trainer/编译 toolchain，simulator 继续使用已验证 CUDA 12.8。

### Stage 18: FlashAttention2 Exact-Parent Smoke Passed

- Retry job `133615` 在 `gpu-pro6000-2` 用 20m25s 完成，Slurm `COMPLETED`、exit `0:0`；sm_120-only 63.2 MB wheel 已安装为 `flash_attn-2.8.3`。
- JSON `PASS`：Pro 6000 capability 12.0、BF16 FA2 forward/backward finite、Torch CUDA 13.0；exact merged F01 parent 以 FlashAttention2 加载并生成 1 token，peak allocated 15.29 GB。
- FA2 阻塞已闭合，允许全新 F10 attempt 6。Trainer 使用 CUDA 13.0.0；simulator 保留 CUDA 12.8.0；科学参数和两侧 caps 不变，无 automatic successor。

### Stage 18: F10 Attempt 6 Reached First Rollout Dispatch

- Job `133674` 在 `gpu-pro6000-7` 运行 4m21s，首次完整初始化 actor/ref、policy vLLM、checkpoint/reward loop 并进入 fit；首批 AgentLoop worker 因缺 `CAR_BENCH_DATASET_ROOT` 无法 resolve Hydra config，0/5 steps。
- Policy 初始化瞬时峰值约 90.2/97.9 GiB（92.2%，7.8% headroom），稳定约 87.8 GiB，无 OOM。当前不与 env 修复同时改 policy cap。
- 已显式 export canonical CAR root，新增 CPU Slurm/Ray OmegaConf resolve smoke；路径/target/simulator URL 全部 PASS 前不提交 attempt 7。

### Stage 18: AgentLoop Ray Environment Smoke Passed

- CPU job `133700` 在 32 秒内 `COMPLETED`、exit `0:0`；真实 Ray worker 完整 resolve AgentLoop OmegaConf，JSON 为 `PASS` 并已归档。
- Canonical CAR root 在 worker 中存在，CAR loop target 与 simulator URL 均正确。允许提交全新 F10 attempt 7；其他设置冻结，policy cap 暂不因初始化瞬时峰值单独调整。

### Stage 18: F10 Attempt 7 Reached Training Batch And OOMed

- Job `133709` 在 `gpu-pro6000-3` 使用同节点 2x Pro 6000 运行 `7m07s`，首次完成完整初始化、26-task 初始验证和首批 16 条训练 rollout；初始验证 reward mean@1 为 `0.230769`。
- Step 1 old-log-prob 的未分块 entropy softmax 尝试额外分配 `20.44 GiB`，当时 trainer 卡仅余 `18.86 GiB`，因此 CUDA OOM；仍为 `0/5` optimizer steps，无 checkpoint/reward audit。
- Simulator/trainer 抽样峰值分别为 `87,576/97,887 MiB`（89.47%）和 `96,055/97,887 MiB`（98.13%），两卡最大利用率 100%。完整 run 与 Slurm/component/telemetry 日志已归档到 `reports/cluster/F10-PILOT-133709/`。
- 下一步先验证并启用 veRL 的 chunked entropy 路径，避免构造全量 softmax 中间张量；除这一数学等价的显存系统开关外，冻结科学设置与现有 memory caps。新 Job ID 提交前先完成配置回归与 attempt 记录。

### Repository Hygiene And Public-Git Preparation

- 用户批准清理本地部署传输包、旧 `.sync` 包和两个测试临时目录，并要求后续阶段成功时更新公开 GitHub 仓库。
- 新增 `.gitignore`，排除根目录传输包、同步/临时目录、Python 缓存、本地环境、模型、checkpoints、experiments、官方/派生数据、运行缓存与常见凭据文件；保留源码、配置、阶段文档和精选实验报告。
- 公开发布安全审计发现 `AGENTS.md` 含实际集群登录凭据；已替换为不含用户名、主机和密码的安全占位说明。首次提交前仍需执行全仓库敏感信息与大文件复核。
- 初次清理审计时目录尚未初始化 Git；公开仓库名称确认后已进入本地初始化与首次发布检查。
- 用户确认公开仓库名为 `CabinAgent-RL`、默认分支 `main`；本地 Git 已初始化。首次提交候选为 184 个文件、约 6.97 MiB，最大单文件约 2.17 MiB。
- 全仓库发布扫描未发现私钥或 GitHub token；两处 `api_key="local-vllm"` 为本地 OpenAI-compatible vLLM 占位值。36 项 unit tests 与 `compileall` 全部通过。
- README 已从早期脚手架描述更新到 Stage 18 的真实状态，明确记录 G02/F02 负结果、F10 当前 OOM 阻塞和“尚无正式 F10-F14/最终 benchmark 结论”的边界。
- 新增 `.gitattributes`，公开仓库中的 Markdown、Python、shell/Slurm、YAML、JSON/JSONL 与表格文本统一为 LF，避免 Windows CRLF 造成全文件尾随空格误报或 Linux 脚本执行问题；safetensors fixture 明确按二进制处理。
- GitHub 官方授权确认账户为 `Jarod-Leo`；创建公开仓库 `https://github.com/Jarod-Leo/CabinAgent-RL`，默认分支 `main`，并设置本地 `origin`。
- 首次公开 commit `ad0d12c6cd7b155472bd9ae0c12b50977241ff94` 已推送；`git ls-remote` 验证远端 `main` 与本地 HEAD 完全一致，工作区为 `main...origin/main` 且无未提交文件。
- 后续每个阶段只有在实验记录、必要验证和成功判定全部完成后才创建并推送阶段 commit；失败 attempt 仍及时写入实验文档，但不会标记为阶段成功。

### Stage 18: F10 Chunked-Entropy Retry Prepared Locally

- 重新加载 Project、Progress、实验阶段总览与 Stage 05 文档，并实时核对 cluster02 account/QoS、GPU inventory、quota、SSD 和用户队列；当前用户队列为空，SSD 使用约 93/150 GB。
- 核对远端 veRL 0.9 源码与 Hydra config，确认 FSDP actor/ref 分别读取 `actor_rollout_ref.actor/ref.entropy_from_logits_with_chunking` 与 `entropy_from_logits_chunk_size`，内置默认 chunk size 为 2048。
- Launcher 现对 actor/ref 两侧显式启用 chunked entropy，F10 submitter 固定导出 `true/2048`；model、数据、4x4 group、sampling、长度、reward/advantage、LoRA、LR、offload、两侧 memory caps 与资源拓扑均未改变。
- 新增双侧配置渲染回归；本地 36 项 unit tests 和 `compileall src scripts` 全部通过。下一步为远端同步、真实 Hydra 解析与 Slurm test-only，通过后提交全新 5-step/no-successor attempt。

### Stage 18: F10 Chunked-Entropy Attempt 8 Submitted

- 远端 Bash syntax、veRL dry-run、36 tests 与 `sbatch --test-only` 全部通过；真实渲染确认 actor/ref 两侧 chunked entropy 均为 `true`、chunk size 均为 `2048`。
- 新 run `f10_pilot_20260901_stage18_r7` 已提交为 Slurm job `134671`：同节点 2x Pro 6000、2 tasks、8 CPU、180 GiB node memory、target 5 steps、无 successor。
- Job 当前因集群 Pro 6000 满载处于 `PENDING (Priority)`；用户队列中仅此作业。排队/运行期间冻结执行代码，启动后按既有 telemetry 与 optimizer 验收契约监测。

### Stage 18: F10 Attempt 8 Exposed Dense-Path Chunking Gap

- Job `134671` 在 `gpu-pro6000-11` 使用同节点 2x Pro 6000 运行 `10m21s` 后失败；双侧配置确实解析为 chunking `True/2048`，初始 validation 与首批 16 条 rollout 均完成。
- 当前 veRL FSDP 的 packed/remove-padding 分支实现 chunked entropy，但项目冻结的 dense-padding 分支无条件调用未分块 entropy；step 1 因额外申请 `20.80 GiB`、仅余 `18.15 GiB` 再次 OOM，仍为 `0/5`，无 checkpoint/gradient/reward audit。
- Simulator/trainer telemetry 峰值为 `87,576/97,887 MiB`（89.47%）与 `90,565/97,887 MiB`（92.52%）；完整日志已归档 `reports/cluster/F10-PILOT-134671/`，用户队列已清空。
- 推荐下一步是启用 veRL 原生 `use_remove_padding=True` packed 路径，并先做单 GPU exact-parent/FA2/LoRA/log-prob/entropy smoke；该系统路径变化需用户确认后实施，科学设置和 memory caps 保持冻结。

### 2026-09-02 Packed-Path Repair Confirmed

- User confirmed `use_remove_padding=True` as the next semantics-preserving system-path fix.
- Execution contract: update the launcher and regression tests, build a one-GPU exact-parent/FA2/LoRA finite log-prob/entropy integration smoke, and submit a fresh five-step same-node two-GPU F10 only if the smoke produces a machine-readable PASS.
- Frozen controls: data, model parent, fresh rank-32 LoRA, 4x4 effective batch, sampling, 32K/20-turn limits, reward/advantage, optimizer/LR, simulator, memory caps, and no-successor boundary.

### Stage 18: Packed-Path Repair Prepared Locally

- F10 launcher 和 submitter 已默认启用 `USE_REMOVE_PADDING=true`，actor/ref 的 chunked entropy 仍冻结为 `true/2048`；模型、数据、4x4 rollout、seed、LR、reward、LoRA、长度、offload、memory caps 和双卡拓扑均未改变。
- 新增单卡 `slurm_packed_entropy_smoke.sbatch` 与机器可读 checker，覆盖 exact corrected-F01 parent、FlashAttention2、fresh rank-32/alpha-32 LoRA、packed valid-token entropy/log-prob、有限非零 LoRA gradient，并检查 veRL FSDP packed 分支源码契约。
- 本地 training-config tests 与 Python compile 已通过。下一步是在集群重新核对实时规则、同步代码并运行该 smoke；JSON PASS 前不提交新的双卡 F10。
- 2026-09-02 远端 API preflight 确认 chunk helper 签名为 `(logits, chunk_size=2048)`，真实 `prepare_model_outputs` 所属类为 `FSDPEngineWithLMHead`；在占用 GPU 前修正 checker 反射目标并增加静态回归，未产生无效 Slurm attempt。

### Stage 18: Packed-Path GPU Smoke Submitted

- 代码-only bundle SHA-256 `179d8b6c6ee2e4005ef9041ea33475102abc55788a1723317946b4570afd90c3` 已同步至集群；远端 Bash、36 tests、compileall、veRL dry-run 与 Slurm test-only 全部通过。
- 单卡 packed-path smoke job `135977` 已提交，当前 `PENDING (Priority)`；资源为 1x Pro 6000、4 CPU、33 GiB、30 分钟，无依赖与 successor。
- 队列等待期间冻结执行代码；machine-readable JSON PASS 前不提交双卡 F10。

### Stage 18: Packed-Path GPU Smoke Passed

- Job `135977` 于 `gpu-pro6000-3` 运行 44 秒后 `COMPLETED`、exit `0:0`，machine-readable report 为 `PASS`。
- Exact corrected-F01 parent/FA2、fresh rank32/alpha32 LoRA、veRL packed branch、chunked entropy/log-prob 与 backward 全部通过；LoRA gradient norm `2.50868`，峰值 allocated/reserved 约 `14.93/15.05 GiB`。
- 四份原始产物已归档到 `reports/cluster/PACKED-ENTROPY-SMOKE-135977/`。Packed-path 前置门禁已闭合，允许在文档同步后创建新的双卡 5-step F10 run；无 successor。

### Stage 18: F10 Packed-Path Attempt 9 Submitted

- 新 run `f10_pilot_20260902_stage18_r8` 已提交为 Slurm job `135987`；manifest source/config digest 为 `9f5aa580...699b14c` / `7fd65bc8...819dc05`。
- Job 请求同节点 2x Pro 6000、2 tasks、8 CPU、180 GiB highmem，target 5 steps，当前 `PENDING (Priority)`；scheduler 暂估 2026-09-03 02:59:05 UTC 在 `gpu-pro6000-11` 启动。
- `use_remove_padding=true` 为相对 attempt 8 的唯一执行路径修复；其余科学与吞吐设置冻结，无 successor。运行期间不得修改执行代码。

### Stage 18: F10 Packed-Path Five-Step Pilot Passed

- Job `135987` 在 `gpu-pro6000-11` 使用同节点 2x Pro 6000 于 2026-09-02 12:57:57--13:26:40 UTC 完成，Slurm `COMPLETED`、exit `0:0`、运行 `28m43s`；5/5 optimizer steps、final validation 和 `global_step_5` checkpoint 均已落盘。
- Step 1/2/3/5 的 reward 范围均为 `[0,1]`，advantage 同时含正负值，grad norm 分别为 `0.071629/0.025236/0.017444/0.035955`；无 NaN、OOM 或 reward-schema 错误，五步 pilot 核心门禁 PASS。Step 4 outcome advantage 全零但仅该组同分，不影响其他四步的有效信号。
- Simulator/trainer 显存峰值为 `87,576/97,887 MiB`（89.47%）和 `94,529/97,887 MiB`（96.57%），两卡最大利用率均 100%；trainer 已接近安全上限，不再提高显存占用参数。Initial/final CAR dev mean@1 为 `0.230769/0.269231`，不作五步性能提升声明。
- 完成进度 100% 后出现 Ray DataLoader worker shutdown traceback，但 Slurm exit 0，且 step-5 metrics/checkpoint/final validation 已保存，因此定性为成功伴随非致命 shutdown warning。精选日志、telemetry、manifest 与小型 checkpoint 元数据已归档到 `reports/cluster/F10-PILOT-135987/`；公开仓库副本已将一处 CAR 样例中的 50 字符 RapidAPI 参数值替换为 `[REDACTED]`，远端原始日志不改动。

### Stage 18: SSD Cleanup And Step-6 Resume Retention Prepared

- 经用户确认，精确删除远端 `cache/pip` 和 5 个已作废或被正式结果替代的 SFT 目录，以及旧传输包和 Job `135987` 临时目录；删除前已将 114 项 manifest/config/metrics/小日志归档并校验到 `reports/cluster/SSD-CLEANUP-20260903/`。corrected F01、F02 正式负结果、模型、数据和当前 step-5 checkpoint 均保留。
- SSD 占用从约 `124.5` 降至 `117.1 GiB`，释放约 `7.4 GiB`；全项目当前只有 `f10_pilot_20260902_stage18_r8/checkpoints/global_step_5` 一个 GRPO checkpoint。
- Launcher 新增 actor/critic checkpoint retention=`1`；F10 start/resume 分别使用 save frequency `5/-1`。Step-6 resume 将保留已验收 step-5 恢复点但不新建第二份约 30 GiB checkpoint。相关 training-config tests 与 Python compilation 已通过，下一步是远端回归、Slurm test-only 和独立 resume 提交。

### Stage 18: F10 Step-6 Resume Submitted

- Git `c6800f8` 已推送公开 GitHub；远端 36 tests、compileall、Bash syntax、veRL rendering、retention 字段源码核对和 Slurm test-only 全部通过。
- 同一 run 的 resume 已提交为 Job `136347`，2026-09-02 16:43:12 UTC 起处于 `PENDING (Priority)`；目标从 `global_step_5` 自动恢复到 total step 6，同节点 2x Pro 6000、无依赖、无 successor。
- Resume 冻结 attempt 9 的模型/数据/采样/reward/advantage/优化器/显存参数，使用 `SAVE_FREQ=-1` 和 retention `1/1`；验收时必须证明 step 6 完成且仍只存在原 step-5 checkpoint。

### Stage 18: F10 Step-6 Resume Passed

- Job `136347` 在 `gpu-pro6000-10` 使用同节点 2x Pro 6000 于 2026-09-02 16:43:38--16:54:48 UTC 完成，Slurm `COMPLETED`、exit `0:0`、运行 `11m10s`；日志确认从 `global_step_5` 恢复 model/optimizer/extra 状态并完成 step 6 与 final validation。
- Step 6 reward/advantage 全零、`pg_loss=0`，仅 KL loss `5.53149e-4` 产生 `grad_norm=1.17578e-5`；该结果不单独证明 outcome 学习，但恢复基础设施 PASS，且 attempt 9 的 step 1/2/3/5 已提供有效 outcome-gradient 证据。
- Simulator/trainer 峰值显存为 `87,575/97,887 MiB`（89.47%）与 `90,932/97,887 MiB`（92.89%），最大利用率均 100%；没有 NaN、OOM 或 reward-schema error。结束时的 DataLoader worker traceback 发生在 100% 进度后，Slurm/manifest/final metrics 均正常，记为非致命 shutdown warning。
- `SAVE_FREQ=-1` 生效：全项目仍只有约 30 GiB 的 `global_step_5`，没有新增 `global_step_6`。精选日志、telemetry、manifest 和小型恢复元数据已归档到 `reports/cluster/F10-RESUME-136347/`。
- F10 五步信号门禁与独立恢复门禁均已闭合。正式 fallback F10 解锁；下一步从 corrected-F01 merged parent 新建 fresh rank-32 LoRA，先处理 pilot checkpoint 存储，再完成 formal 250-step config 验证与独立提交，不自动启动其他消融。

### Stage 19: Formal F10 Step-50 Preparation

- 正式 F10 总目标仍为 250 steps，但按 `50/100/150/200/250` 评测点拆为独立可恢复 segment；每段完成、记录、归档后才人工提交下一段，优化器/model/extra 状态连续且无自动 successor。
- 新增 formal F10 submitter：start 只允许 fresh run 到 step 50，resume 只允许后续四个冻结边界；保存/评测 frequency `50/50`、checkpoint retention `1/1`，其余模型、数据、4x4 rollout、sampling、reward/advantage、LoRA、LR、packed/chunked 路径、offload、memory caps 与双卡拓扑继承已通过的 pilot。
- 下一步先完成本地测试与代码推送，再同步集群；校验删除 pilot checkpoint 后实时检查集群和 SSD，最后仅提交正式 F10 的 step-50 segment。

### Stage 19: Formal F10 Step-50 Submitted

- Git `9096f74` 已推送并同步集群；远端 Bash、36 tests、compileall、Hydra rendering 与 Slurm test-only 全部通过，正式 override 为 target/save/eval `50/50/50`、retention `1/1`、caps `0.86/0.60`。
- 已在个人队列为空时删除完成恢复验证使命的 pilot `global_step_5` 和 stale marker；`storagemgr` 显示 SSD 从 `117.2` 降至 `85.8/150 GB`，当前完整 GRPO checkpoint 为 0。
- 新 run `f10_formal_20260903_stage19` / Job `136868` 于 2026-09-03 05:11:13 UTC 提交，当前 `PENDING (Priority)`；请求同节点 2x Pro 6000、2 tasks、8 CPU、180 GiB、12 小时，无 dependency/successor。
- Run 从 validated corrected-F01 merged parent 初始化 fresh rank-32 LoRA，不继承 pilot checkpoint。排队/运行期间冻结执行代码；50/50 完成并记录前不提交下一 segment 或其他消融。

### Stage 19: Formal F10 Step-50 Passed

- Job `136868` 在 `gpu-pro6000-3` 使用同节点 2x Pro 6000 于 2026-09-03 05:12:05--07:43:33 UTC 完成，Slurm `COMPLETED`、exit `0:0`、运行 `2h31m28s`；50/50 optimizer steps、final validation 和 `global_step_50` checkpoint 均已落盘。
- `21/50` 步（42%）产生非零 group-normalized outcome advantage 与有限非零 gradient，`29/50` 步为组内同分；800 条在线训练 trajectory 中 94 条 reward=1，mean reward `0.1175`。无 NaN、OOM、reward-schema error 或 aborted trajectory。
- 有效 outcome 步 grad norm 为 `0.00935--0.12299`，rollout/actor correlation 平均 `0.999110`，rollout-correction KL 平均 `0.001020`，clip fraction 全程 0。Initial/final CAR dev mean@1 均为 `0.269231`，当前不作性能提升声明。
- 平均/中位 step time 为 `172.06/170.60s`，平均吞吐 `1231.53 token/s`。Simulator/trainer 峰值显存为 `87,576/97,887 MiB`（89.47%）和 `94,529/97,887 MiB`（96.57%），不再提高显存占用参数。
- 唯一完整 checkpoint 含 11 文件、`31,443,788,637` bytes；SSD 当前 `117.3/150 GB`。100% 进度后的 DataLoader worker traceback 与此前一致，因 checkpoint/final metrics/manifest/Slurm 均成功，记为非致命 shutdown warning。
- Step-50 segment PASS；公开安全摘要已归档到 `reports/cluster/F10-FORMAL-136868/`。下一动作是在记录与 GitHub 推送后，从同一 checkpoint 独立恢复到 step 100；F11-F14 仍不启动。

### Stage 19: Step-100 Submission Attempt 1 Cancelled Before Allocation

- Target=100 的 Slurm test-only 已通过，但正式提交器在 `sbatch` 返回 Job `137581` 后因新 SSH shell 无裸 `python` 命令而无法更新 manifest；Job 在 pending、0 秒、无节点分配时立即精确取消，未消耗 GPU，step-50 checkpoint 与 run manifest 均未修改。
- 根因是 formal submitter 与 batch lifecycle update 隐式依赖交互 shell 的 Conda PATH。两处现统一使用并验证项目绝对解释器 `$GPU_ENV/bin/python`，新增静态回归覆盖 init、submitted、running 和 final 状态更新。
- veRL retention=1 的源码确认旧 checkpoint 会在新 checkpoint 成功写完后删除。为增加保存峰值余量，仅删除 69 MiB 可再生 pip 下载缓存；训练编译缓存、模型、数据、F01/F02 结果、失败日志和唯一 step-50 checkpoint 全部保留。
- 完成本地/远端回归并推送修复后，以新 Job ID 重新提交 step-100 resume；科学设置、caps、retention 与 checkpoint 均不变。

### Stage 19: Formal F10 Step-100 Resume Running

- 提交器修复 Git `400794b` 已推送并同步集群；本地/远端 36 tests、compile、Bash syntax、文件 SHA-256、项目绝对 Python 和 Slurm test-only 全部通过。
- Step-100 resume 已以新 Job `137588` 提交，并于 2026-09-03 11:49:49 UTC 在 `gpu-pro6000-7` 启动；同节点 2x Pro 6000、2 tasks、8 CPU、180 GiB、12 小时，manifest 已为 running。
- Allocation 显示物理 GPU indices `0,5` 和两个不同 GPU UUID；trainer 已从 `global_step_50` 恢复并完成 step 51。该步 reward mean `0.0625`、advantage range `[-0.5,1.5]`、grad norm `0.0300013`、KL loss `0.0006214`、clip fraction 0，无 OOM/NaN/schema error。
- 当前进度 `51/100`。训练设置、caps `0.86/0.60`、retention `1/1` 与 step-50 完全冻结，无 successor。
- 下一验收是确认从 `global_step_50` 恢复并完成 step 51--100；保存成功后必须只保留 `global_step_100`。完成并记录前不提交 step 150 或 F11-F14。

### 2026-09-04 Stage 19: Formal F10 Step-100 Training Complete, Storage Remediation Required

- Slurm `137588` completed with exit `0:0` after `2:15:49`; the trainer restored model/optimizer/extra state from step 50 and reached 100/100 without NaN, OOM, reward-schema error or aborted trajectories.
- Steps 51--100 contained `17/50` effective outcome-gradient steps and `33/50` zero-advantage steps; cumulative steps 1--100 contain `38/100` effective outcome-gradient steps. The mean of the 50 reported batch reward means was `0.111667`, mean rollout/actor correlation `0.999066`, mean rollout-correction KL `0.000764`, and clip fraction remained zero.
- Mean/median step time improved to `154.06/150.95s`; mean throughput was `1394.67 token/s`. Simulator/trainer peak memory was `87,575/95,054 MiB` of `97,887 MiB` (`89.47%/97.11%`), so no further VRAM-filling adjustment is safe.
- Final CAR dev mean@1 was `0.230769`; this does not establish performance improvement. The previously observed post-completion DataLoader shutdown warning recurred after all required artifacts were written and remains non-fatal.
- Both `global_step_50` and `global_step_100` remain, each exactly `31,443,788,637` bytes with the same 11-file schema. Live storage reached `148.8/150 GB`. Installed veRL source confirms retention tracks only checkpoints saved in the current process and does not register a checkpoint loaded by a new resume process, so retention=`1` could not see step 50.
- Provisioned the approved 250 GB HDD project at `/projects/cabinagentrlarchive`; the storage manager rejected the requested hyphenated name as unsafe, so the equivalent alphanumeric permanent name was used. No SSD source has been deleted.
- Successor training remains blocked until the selected step-50 baseline checkpoint is copied to HDD by Slurm, verified by exact file set/size/SHA-256, the user confirms deletion of the exact SSD source, and project-level checkpoint pruning is regression-tested.

### 2026-09-04 Stage 19: Continuous-Run And Storage Tooling Validated Locally

- Added generic F10--F14 launchers for one continuous 250-step Slurm allocation, 50-step save/eval boundaries, same-node dual-Pro6000 topology, restart-count guard of two infrastructure retries, and no automatic successor experiment.
- Added cross-process checkpoint audit/pruning that validates the latest marker and resumable five-file schema before removing an explicitly older global-step directory. Successful continuous jobs require the final keep-step postcondition.
- Added a CPU-Slurm HDD archive path using `.incoming`, exact size/SHA-256 inventories, atomic cutover, unsafe/overlapping-path and parent/internal-symlink rejection, a 180 GB soft cap, and no SSD source deletion.
- Added F11--F14 fallback configs and model-path overrides so immutable parents can load directly from HDD after a compute-node smoke. Existing F10 scientific settings and VRAM caps remain frozen.
- Requeue attempts use restart-specific simulator/trainer completion sentinels, so a stale sentinel from an interrupted allocation cannot terminate the replacement simulator. Local validation passed: 40 unit tests, Python compileall, 22 YAML files, and `git diff --check`.
- Next is remote source/storage audit, code-only sync, remote dry-run, and a verified copy-only archive attempt; no training successor has been submitted. The cluster SSH path is currently timing out before banner exchange, so no remote state was mutated during this preparation.

### 2026-09-04 Stage 19: HDD Copy-Only Archive Passed

- VPN/SSH recovered. Live checks confirmed an empty user queue, SSD `148.8/150 GB`, HDD `0/250 GB`, and four source trees totaling `103,536,774,364` bytes with no symlinks.
- The synchronized bundle hash matched locally and remotely; remote 40 tests, compileall, Bash syntax, Hydra dry-run, checkpoint audit, and both Slurm test-only paths passed.
- CPU archive Job `138014` completed in `25m24s` with exit `0:0`. The report verified 133 files and `103,536,774,364` bytes by exact path/size/SHA-256, removed the temporary incoming batch, and confirms `source_deleted=false`; all four SSD sources and HDD targets coexist.
- Simulator smoke now accepts an explicit immutable model path instead of forcing the SSD default, with a static regression test; local suite is now 41 tests. Next is the 7B/72B HDD load smoke without changing serving semantics.
- HDD corrected-F01 parent Job `138060` passed in `1m46s` with exit `0:0`: 10-file/15.24 GB manifest hash, BF16 load of 7.616B parameters, tokenizer, and one-token generation all succeeded directly from HDD. This is well below the ~20-minute threshold; 72B smoke is next and no SSD source has been deleted.
- HDD 72B-AWQ simulator Job `138064` passed in `4m37s` with exit `0:0`. Eleven shards loaded in `122.16s`, full model load took `128.29s`, and unchanged AWQ-Marlin/FlashAttention2 health plus three request classes succeeded.
- Both immutable models now pass direct HDD compute-node loading well below the ~20-minute threshold. No SSD source has been deleted and no F10 successor is active; the exact four-item SSD deletion list now requires explicit user confirmation.

### 2026-09-04 Stage 19: HDD Acceptance Passed And F10 Continuous Run Submitted

- 通过免密 SSH 重新建立集群会话后完成上一阶段只读验收：用户队列为空；归档/加载三作业 `138014/138060/138064` 全部 `COMPLETED/0:0`；`archive-138014.json` 为 `status=verified`、`source_deleted=false`、133 files/`103,536,774,364` bytes。
- 四项 SSD 源与 HDD 副本逐项 `du -sb` 字节完全一致（step-50 checkpoint 31,443,788,637 B；72B-AWQ 41,607,445,835 B；7B base 15,242,811,314 B；F01-merged 15,242,728,578 B）；step-100 checkpoint 完好且 manifest 为 completed；`138060/138064` 报告状态均为 `PASS`。
- 远端与本地 6 个关键执行文件（连续 launcher/sbatch/submitter/checkpoint_policy/launch_verl/vanilla.yaml）SHA-256 全部一致，远端 41 unit tests `OK`；确认 `submit_fallback_ablation.sh` 默认的 policy/simulator 路径均指向已验证 HDD，且 sbatch 具备 `--requeue`、restart sentinel 与 `prune --expected-step 250` 后置条件。
- 用户确认精确清单后仅删除四个已归档 SSD 目录（`global_step_50`、7B、72B-AWQ、F01-merged，合计 103,536,774,364 bytes）；复核 `checkpoints/` 仅剩 `global_step_100` + latest marker，模型目录清空，SSD 按字节数核算由约 `148.8` 降至约 `52.4 GiB`。
- `checkpoint_policy.py audit` PASS（唯一 checkpoint step 100、11 文件 schema 完整、marker=100）后，以 `submit_fallback_ablation.sh f10 resume f10_formal_20260903_stage19` 提交连续作业 Job `138821`（`car-f10-full`，24h、同节点 2x Pro 6000、requeue≤2），manifest 已更新为 submitted；集群 Pro 6000 满载，当前 `PENDING (Priority)`。
- 遗留问题：Job `138821` 需要完成 step 101--250 并通过 `prune --expected-step 250` 后置条件（最终仅保留 `global_step_250`）才算 PASS；F11-F14 仍等待 F10 结果与人工门禁。

### 2026-09-04 Stage 19: F10 Reached Step 250; Automatic Prune Failed

- Job `138821` 在 `gpu-pro6000-11` 使用同节点 2x Pro 6000，于 2026-09-04 04:56:28--13:37:47 UTC 运行 `08:41:19`。Trainer 从 `global_step_100` 恢复 model/optimizer/RNG/LR scheduler 并完成 step 101--250、step-150/200/250 validation 和完整 `global_step_250` 保存。
- Step 101--250 有 `76/150` 个有效 outcome-gradient step，平均 batch reward mean `0.117139`；有效 step grad norm 最小/均值/最大为 `0.010334/0.040361/0.164800`。累计 step 1--250 为 `114/250` 个有效 outcome-gradient step。无 NaN、OOM、reward-schema error 或 aborted trajectory。
- CAR dev mean@1 在 step 100/150/200/250 分别为 `0.230769/0.269231/0.230769/0.230769`，最终没有相对 step-100 提升。Simulator/trainer 峰值显存为 `87,576/95,055 MiB`，不再提高显存参数。
- Slurm 最终为 `FAILED/1:0`，不是训练失败：veRL 在 step 150/200 轮换后留下各含一个 `data.pt` 的 7,316-byte 残留目录；旧 `checkpoint_policy.py` 在验证目标 step-250 前严格验证所有 step 目录，因此自动 prune 被不完整旧目录阻断。step-250 本身为 11 文件、`31,443,788,637` bytes，marker=250。

### 2026-09-04 Stage 19: F10 Checkpoint Postcondition Recovered

- 修复 `checkpoint_policy.py`：audit 分离完整与不完整 checkpoint；prune 仍严格要求 marker/保留点完整、拒绝删除任何更新 step 或越界/软链接目标，但允许把更旧的不完整 step 目录作为显式候选。新增 tombstone 回归；本地和远端各 5 项 storage tests 与 compileall 通过。
- 用户确认精确清单后删除 `global_step_100`（11 文件、`31,443,788,637` bytes）、`global_step_150`（1 文件、7,316 bytes）和 `global_step_200`（1 文件、7,316 bytes）。远端 post-audit PASS：仅剩完整 `global_step_250`，11 文件、`31,443,788,637` bytes，marker=250。
- F10 训练与存储后置条件至此闭环，但保留 Job `138821` 的真实 Slurm FAILED 历史。后续实验不自动提交；先确定 W&B 历史回填/实时记录契约，再由 post-F10 人工门禁决定 F11--F14 的首个实验。

### 2026-09-04 Stage 19: W&B F10 Backfill Passed; F11 Selected

- 用户确认 W&B project `CabinAgent-RL`：F10 回填历史数值曲线，F11--F14 原生实时记录；凭据仅通过交互式登录写入服务器用户 `~/.netrc`，未进入仓库、命令参数、Slurm 日志或实验配置。
- 新增 `scripts/backfill_wandb_from_verl.py`，只解析三个 F10 trainer console 日志中的数值字段；本地 2 项 parser/segment-merge 测试与远端真实 dry-run 通过。正式回填上传 step 0--250 共 251 个 step、97 个数值指标，W&B run ID `2ut4t5d4`。
- `launch_verl.py` 默认将未来 GRPO logger 设置为 `['console','wandb']`，仍可用 `WANDB_ENABLED=0` 显式关闭；本地/远端 W&B 与 training-config 定向测试均通过。原始对话、工具输出、模型和 checkpoint 不上传。
- 用户选择 F11 Turn-Discount 为下一实验，并要求每 50 steps 用 dev 验证、每个实验最终只保留效果最佳 checkpoint。由于 CAR dev 分数存在并列，提交前尚需冻结 tie-break；在此之前不提交 F11。

### 2026-09-05 Stage 20: F11 Best-Checkpoint Tooling Validated Locally

- 用户确认完整选择契约：F11 Turn-Discount 单次 250 steps；训练作业内每 50 steps 用 CAR dev mean@1 验证；step 0 只作 baseline；仅严格提升时保存，同分保留较早 checkpoint；中断从当前最佳恢复并允许重复少量训练区间。
- 新增 Ray actor 内的 best-checkpoint controller：延迟 veRL 边界保存，先使用内存模型验证，再调用原生保存；选择状态原子持久化并带 pending 恢复，保存完整后复用严格 audit/prune。W&B 同步记录 candidate/best score、best step 和是否入选。
- 启用 veRL 原生 LoRA-only 可恢复 checkpoint：模型 shard 只保存 LoRA，同时继续保存 optimizer 与 RNG/LR scheduler；上游 retention 暂停，由项目选择器按最佳 step 管理。新增 `audit-best` 最终后置条件。
- 新增 F10 step-50 adapter 导出/验证 Slurm 入口：使用 veRL FSDP merger 只导出 PEFT LoRA，校验 rank/alpha、safetensors keys，并以 HDD corrected-F01 parent 完成 one-token GPU generation；目标已存在时拒绝覆盖。
- 本地 51/51 unit tests、Python compilation 与 diff check 通过；未提交或修改任何 Slurm job。下一步同步远端、检查 Hydra/Ray/veRL 实际路径，并先提交单卡 F10 adapter export/validation。

### 2026-09-05 Stage 20: F10 Best Adapter Export Submitted

- 实时 pre-flight：账户/QoS=`msc/msc`，用户队列原为空；SSD `45.4/150.0 GB`、HDD `103.5/250.0 GB`。HDD F10 step-50 checkpoint（约 30 GB）与 corrected-F01 merged parent（约 15 GB）存在，目标 adapter 路径不存在；当前 Pro 6000 满载。
- commit `41e9dcd` 已推送 GitHub 并同步执行文件到远端。远端 Bash、compileall、51/51 tests、Ray actor class import、W&B credential verify、Hydra resolved config 与 F10/F11 Slurm test-only 均 PASS；F11 解析为 Turn-Discount `alpha=1.05`、250 steps、50-step dev、sync、LoRA-only、strict-best metric。
- 单卡 highmem Pro 6000 adapter export/validation Job `140039` 已提交，当前 `PENDING (Priority)`；0 GPU time、0 文件变更。完成并记录其结果前不提交 F11。
- Job `140039` 随后在 `gpu-pro6000-3` 运行 `00:03:30` 并 `COMPLETED/0:0`。导出 rank/alpha `32/32`、392 tensors 的 F10 step-50 actor LoRA；adapter 本体 `161,533,560` bytes，目录合计 `161,535,915` bytes，逐文件 SHA-256 与 parent+adapter CUDA one-token generation 均 PASS（峰值 `15,786,684,416` bytes）。
- 两个完整 F10 checkpoint 仍原样保留：HDD step 50 与 SSD step 250 均为 11 files / `31,443,788,637` bytes。下一步按契约给出 exact paths 并等待用户单独确认删除；未经确认不清理、不提交 F11。

### 2026-09-05: F10 approved cleanup

- 用户明确授权后删除 HDD step 50 与 SSD step 250，各 11 files / 31,443,788,637 bytes；重新核对路径与 adapter PASS manifest 后执行，两个目标均已不存在。完整训练状态永久删除，最佳 adapter 与父模型保留。
- 实时配额复核：SSD 14.0/150 GB，HDD 72.3/250 GB。F11 已提交 Job `140302`，run `f11_formal_20260905_stage20`，250 steps、每 50 steps dev、同节点 2x Pro6000、24h、W&B 实时记录。执行代码冻结至作业结束。

### 2026-09-05: F11 save OOM remediation

- Job140980已在gpu-pro6000-9 RUNNING，同节点两块Pro6000，初始模型加载阶段；正式训练结果待产出。

- save/resume验收记录完成后提交正式F11 attempt2 Job140980，run f11_formal_20260905_stage20_r2；250steps/save50/eval50、同节点2xhighmem Pro6000、24h、caps0.86/0.60；从F01父模型新建LoRA，不继承smoke。提交前SSD16.2/150GB、HDD72.3/250GB。执行代码冻结，等待训练结果。

- 恢复Job140696已COMPLETED/0:0（gpu-pro6000-9，15m03s）：model/optimizer/RNG/LR scheduler加载成功，step2完成并保存2.42s，两个checkpoint完整各0.981GB，latest2/best1，series_verified。save/resume门禁闭合，下一步独立新run重跑正式F11 250steps。

- GPU保存Job140549通过，W&B API确认step1与save_seconds=1.9279已同步；独立恢复Job140696已提交，从step1恢复至step2。真实checkpoint模型state为323.1MB、optimizer646.3MB，总0.981GB；五份约4.90GB。

- Job140302 FAILED/15:0，gpu-pro6000-7，2h19m18s；执行到 step50 并完成 dev=0.269231，与 baseline 持平。保存前 load_fsdp_model_to_gpu 申请890MiB、仅338MiB空闲；rollout进程占58.36GiB。无完整checkpoint，常规metrics到49。
- 根因：自定义延迟save移到validate内、rollout已唤醒；恢复原生sync顺序（on_sample_end sleep -> update actor -> save -> on_step_end wake -> validate）。无需新增sleep或修改FSDP底层。
- 远端52tests和Bash检查PASS，SSD14.1/150GB；修复commit38b255c。独立save smoke Job140549已COMPLETED/0:0，11m12s；step1保存1.93s，11文件980,828,869bytes，series_verified，无OOM。下一步独立resume到step2；正式run尚未重提。
- 新契约：每50steps全存、保留五个，最新恢复，最佳分数独立记录；原子staging发布和latest marker，写入失败保留旧恢复点，结束后人工验收最佳再清理。52 tests PASS（首次Windows临时目录权限失败，正常权限重跑通过）。GPU save/resume smoke待执行，正式run尚未重提。
