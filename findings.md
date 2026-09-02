# CabinAgent-RL Findings

## Initial Repository State

- Workspace root: `E:\CabinAgent-RL`.
- Existing files before development: `Project.md`, `draft.md`.
- No `.git` repository is present.
- `Project.md` defines a full pipeline from prompt baseline to SFT, DPO, GRPO, PRM-Lite, unified eval, and vLLM latency.

## Baseline Interpretation

- The user asked to start the foundation baseline runnable part for the whole project.
- The current practical target is Phase 0 plus the executable portion of Phase 1:
  - repo scaffold,
  - configs,
  - local sample benchmark records,
  - model adapter interface,
  - deterministic local baseline adapter,
  - CAR/BFCL adapter shells,
  - unified trajectory and metrics,
  - reports and failure taxonomy.
- Real Qwen/vLLM/CAR-bench/BFCL execution requires external packages, datasets, models, and likely GPU servers. This session should create the integration-ready baseline and local smoke path first.

## Initial Project Constraints From Project.md

- Main benchmarks: CAR-bench and BFCL V4 subset.
- Default policy: `Qwen2.5-7B-Instruct`.
- Teacher/judge should not be trained. This initial note is superseded by the fixed local 72B simulator design below.
- PRM-Lite starts as deterministic rules, not a learned reward model.
- Required final metrics include success, tool accuracy, hallucination, state consistency, and latency.

## Cluster Smoke Run (2026-08-30)

- The requested acceptance window is about 30 minutes with no anomaly, followed by explicit job shutdown/completion verification.
- Cluster operations must use Slurm; the login node is limited to lightweight setup and monitoring.
- Code/config may live under home, while datasets, environments, weights, caches, logs, and results must use the project SSD.
- `集群详细使用说明.md` exists but is currently empty, so all mutable paths and resource flags must be derived from live read-only cluster commands before use.
- The intended official datasets are CAR-bench and BFCL, based on `Project.md`; their final destination must be a live-confirmed project SSD path.
- SSH login to `login-3.cluster02.eee.ntu.edu.sg` succeeded on 2026-08-30; Slurm reports version 23.11.4.
- Live association: cluster `cluster02`, account `msc`, default QoS `msc`; the account allows one RTX 5090 GPU (or up to two of several other listed GPU types).
- `gpu-5090-2` was fully idle at pre-flight time (4/4 RTX 5090 GPUs free); it is the preferred smoke-test node subject to allocation-time availability.
- `storagemgr` is not installed on the login node. The home filesystem is on the small root volume, so datasets/results must not be placed there.
- Login banner official docs point to `github.com/NTUEEECluster/docs`; browser retrieval returned no content and local HTTP access was blocked by the environment proxy, so live cluster help/mount inspection is required.
- Interactive login reveals Ceph mounts: `/cluster`, `/projects`, and `/home` are on the SSD filesystem; `/projects/_hdd` is the HDD tier.
- `storagemgr -user jiatian001` reports 150 GB SSD and 250 GB HDD allocated, but 0 B assigned on both tiers.
- Neither `/projects/jiatian001` nor `/cluster/jiatian001` exists, and the account cannot create a top-level `/projects` entry directly. Large data cannot be placed safely until a project namespace is assigned or another documented path is confirmed.
- CPU Slurm job `129937` ran on `cpu-1` and successfully fetched the official login guide. Its final state was `FAILED` only because a no-match `grep` returned exit code 1 after the useful fetch; Slurm execution and outbound access from compute nodes are working.
- The local code-only deployment archive is 27,248 bytes and excludes checkpoints, generated reports, caches, and large data.
- Official `cluster.md` confirms regular `msc` users must provision `/projects/<name>` through `storagemgr`; active datasets, environments, checkpoints, logs, and results belong on SSD.
- The code archive was uploaded and extracted under `/home/jiatian001/CabinAgent-RL`; local and remote SHA-256 both equal `d51af3f489d92500b45d787c042474a29b4116101a40313c70bd2bf2beba3b8f`.
- Official docs also confirm shared storage is visible from login and compute nodes, CPU/GPU resources are assigned by Slurm, and code/configuration may remain under home.
- `storagemgr` initially created the former `cabinagentrl` project with the account's full 150 GB SSD allocation. The TUI rejected hyphens despite the docs saying they are permitted.
- Compute nodes require the Lmod module `Miniforge3/24.11.3-1`; it exposes Python 3.12.9. The login shell's Python availability cannot be assumed inside Slurm jobs.
- Smoke job `129967` completed with exit code `0:0` after 30:02 and emitted `SMOKE_OK` at 1,800 seconds; the selected GPU node returned to idle afterward.
- BFCL V4 data ships inside the Gorilla repository. CAR-bench's complete official data is a separate Hugging Face snapshot with 18 files totaling 720,638,694 bytes, so cloning the GitHub repository alone is insufficient.
- CAR-bench official snapshot validation passed at `/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/data/official/car-bench-dataset`: 18 files, 720,638,694 verified bytes, 254 tasks, and 10 mock-data JSONL files.
- BFCL validation passed under `/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/data/official/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data`: 51 BFCL V4 JSON/JSONL files and 9,673 records including companion answers and retained unused categories.
- Final cluster audit found no active jobs, no partial download files, and about 1.6 GB used from the 150 GB SSD project allocation.

## SSD Path Migration (2026-08-30)

- Requested destination: `/projects/jiatian001ssd/cabinagentrl`.
- Local search found no executable script or config hardcoding `/projects/cabinagentrl`; only historical planning/progress records contain the old path.
- The migration must preserve the existing `CabinAgent-RL` working tree, official datasets, logs, reports, and permissions, then confirm the old path no longer owns live data.
- Live pre-flight found no active Slurm jobs. `/projects/cabinagentrl` is a managed symlink to `/projects/_ssd/cabinagentrl`, using about 1.6 GB of its 150 GB allocation.
- `/projects/jiatian001ssd` and `/projects/jiatian001ssd/cabinagentrl` do not exist. The destination parent must be provisioned through `storagemgr`; `/projects` cannot be created in directly.
- Safe migration order: split quota to provision `jiatian001ssd`, copy and verify through a CPU Slurm job, remove the verified old project, then return the full SSD quota to `jiatian001ssd`.
- `storagemgr` successfully reduced the former `cabinagentrl` project to 10 GB and provisioned `/projects/jiatian001ssd` with 140 GB. The source still contains its full 1.6 GB working tree.
- Slurm migration job `130531` copied 1,234 files totaling 1,555,401,389 bytes. A checksum dry-run reported zero differences and source/destination file counts and bytes matched exactly.
- Slurm validation job `130532` ran the local baseline, all derived-data builders, and official CAR/BFCL parsing from the new path and emitted `MIGRATION_PATH_OK`.
- Protected cutover job `130534` correctly stopped before deletion on timestamp-only report differences created by validation; source data remained available.
- Cutover job `130535` ignored metadata-only timestamps but retained full checksum comparison, reported zero content differences, removed the exact old working tree, and completed with exit code `0:0`.
- The empty former storage project was deleted, `jiatian001ssd` was expanded to the full 150 GB SSD quota, and the old `/projects/cabinagentrl` entry no longer exists.
- Migration logs now live inside `/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL/logs`; the canonical project root is `/projects/jiatian001ssd/cabinagentrl/CabinAgent-RL`.

## Dual-Node Local Simulator Design (2026-08-30)

- The selected topology is a local `Qwen/Qwen2.5-72B-Instruct-AWQ` CAR-bench user simulator served by vLLM plus a trainable `Qwen/Qwen2.5-7B-Instruct` policy optimized with veRL.
- The two roles should receive two Pro 6000 nodes in one atomic Slurm allocation. A custom login-node polling loop is unnecessary and would create a race between independent allocations.
- CAR-bench also has a policy-evaluator role. It is excluded from the training reward and the 72B simulator must not silently serve as both user and judge.
- Live inventory found 11 Pro 6000 nodes with 96 GB GPUs. The legal request is `--gres=gpu:pro6000:1`; `--constraint=highmem` raises the one-GPU node memory allowance to 90 GiB on eligible nodes.
- Pro 6000 nodes have heterogeneous GPU counts, but this design deliberately requests one GPU on each of two nodes. The simulator and trainer communicate through the allocation's internal network.
- The cluster submit plugin enforces 4 CPUs and 92,160 MiB per one-Pro-6000 node. Scripts therefore omit explicit memory and CPU requests, matching cluster policy.
- A two-node `sbatch --test-only` passed and estimated `gpu-pro6000-[7,10]`; no GPU job was submitted. The correct availability strategy is one pending atomic allocation, not login-node polling.
- `Miniforge3/24.11.3-1` and CUDA 12.8+ modules are available. No supported container runtime was found, so the GPU environment will be a named Conda prefix on project SSD.
- The base Miniforge module does not provide PyYAML. The project renderer passed locally, and its remote execution must wait for the Stage 11 GPU environment installation declared in `requirements-gpu.txt`.

## Approved Ablation Matrix (2026-08-30)

- DPO and the former R05 expansion are removed from the active plan.
- This decision was superseded on 2026-08-30: E10-E14 now start directly from the same Qwen2.5-7B-Instruct revision with fresh LoRA weights.
- The five retained experiments are Vanilla GRPO, Turn-Discount, LATA, PRM-Lite, and PRM-Lite + LATA.
- Turn-Discount follows the reference implementation's token-position weighting `alpha^(L-1-t)` with mean-one normalization and `alpha=1.05`.
- LATA applies the same weighting and then scales by `1/sqrt(L)`.
- CAR PRM-Lite uses deterministic rule events clipped to `[-0.5, 0.5]`; E13/E14 use `outcome + 0.3 * process_score`.
- Formal runs stop at 250 steps and evaluate checkpoints 50/100/150/200/250 on the same CAR dev and BFCL manifests. CAR test remains frozen until model selection is complete.

## Direct-RL Initialization Decision (2026-08-30)

- Project-specific SFT was removed from the default training path because Qwen2.5-7B-Instruct is already instruction-tuned and can be tested directly in the real CAR environment.
- The direct-RL gate must use at least 20 CAR train tasks with four same-initialization rollouts per task.
- PASS requires parse rate >= 0.95, executable rate >= 0.85, mixed outcome group ratio >= 0.20, loop/max-turn rate <= 0.20, and at least one successful trajectory.
- Outcome-only group variance is the decisive signal because E10-E12 cannot learn when almost all groups are all-zero; PRM-Lite must not be used to make this gate pass.
- CAR official task files provide hidden goals and ground-truth actions rather than complete expert assistant trajectories. Removing mandatory SFT also removes the unresolved need for a 72B assistant-teacher role.
- A failed gate permits only a separately tracked minimal SFT fallback for formatting/basic interaction. Fallback-initialized runs cannot replace or be mixed with E10-E14.

## F02 Corrective Route (2026-09-01)

- The user explicitly authorized cleaning stale job `132946` and proceeding to the next experiment attempt. Job `132946` was cancelled and the queue became empty.
- G03 has seven unique hallucination tasks: three `hallucination_missing_tool_parameter` and four `hallucination_missing_tool` tasks. They account for the clearest invalid-call and max-turn concentration.
- Corrective records must not retain failed assistant tool calls because the current SFT encoder labels every assistant span. The safe construction is a fresh system/user prefix followed by a correct no-tool response, plus a second no-retry dialogue for the same task.
- F02 will train from the same Qwen2.5-7B base on the union of successful gate trajectories and corrective records. This avoids adapter stacking while retaining F01's successful-behavior coverage.
- G04 must keep the G03 task manifest, sampling, reward, and thresholds unchanged. It may only change the policy adapter produced by F02.
# 2026-09-01 F10 pilot route decision

- F02 was not merely paused: data job `133301`, smoke `133303`, full SFT `133306`, and G04 `133308` all completed. G04 produced a valid scientific FAIL.
- G04 (`F02`) was slightly better than G03 (`F01`) on executable rate (`0.846829` vs `0.844693`) but worse on mixed groups (`0.10` vs `0.15`), loop/max-turn (`0.275` vs `0.2625`), and successes (`12/80` vs `14/80`). Treat F02/G04 as a negative corrective-SFT ablation and do not proceed to F03/G05 now.
- User-confirmed route: initialize all fallback GRPO branches from corrected F01 adapter `experiments/sft_fallback_full_20260901_stage16/checkpoints/final_adapter`; preserve original gate FAIL reports but demote the numerical gate cutoffs to diagnostics rather than hard GRPO blockers.
- First action is a manually reviewed 5-optimizer-step F10 Vanilla GRPO pilot. It must not auto-submit formal runs.
- Pilot PASS criteria: at least 5 optimizer steps; at least one step with non-zero reward variance, non-zero advantage, and finite non-zero gradient; no NaN/OOM/reward-schema failure; finite KL/clip fraction/grad norm; GPU memory/utilization/step-time/wait metrics; checkpoint save and resume followed by at least one additional step. Five-step task performance improvement is not required.
- Throughput tuning may adjust only semantics-preserving system parameters. Preserve group size 4, four tasks per optimizer step, effective batch, sampling, sequence/turn limits, reward/advantage definitions, LoRA initialization, optimizer/LR, data, and simulator protocol. Target stable throughput with roughly 10-15% dynamic VRAM headroom rather than 100% allocation.
- If F10 has zero outcome advantage across the pilot while infrastructure is healthy, run a separately reviewed 5-step F13 PRM-Lite pilot. Only if F13 also has no effective gradients should a Qwen3-8B migration be considered; migration planning is explicitly deferred.
- The repository's `Progress.md` is the authoritative progress log and collides case-insensitively with the planning skill's `progress.md` name on Windows, so this session will use `task_plan.md`, `findings.md`, and the existing `Progress.md` rather than creating a duplicate-case file.
- Initial F10 implementation audit found that `scripts/slurm_dual_pro6000.sbatch` still launches simulator and trainer as two independent exclusive `srun` steps. This topology was experimentally rejected by job `132950` and now violates AGENTS rule 16; F10 must use one `srun --ntasks=2 --gpus-per-task=1 --gpu-bind=single:1` with explicit role dispatch.
- `scripts/launch_verl.py` still points `actor_rollout_ref.model.path` at the base Qwen2.5-7B snapshot and has no corrected-F01 initialization path. A verified veRL-v0.9-compatible adapter initialization mechanism is required before submission.
- Existing `submit_next_training_stage.sh smoke` is unsuitable: it runs only two steps, truncates train/val samples, sets an automatic vanilla successor, uses the obsolete launcher topology, and does not implement the agreed manual review boundary or resumed step.
- Existing GRPO scripts lack per-role GPU telemetry, a pilot acceptance report, and an explicit no-successor pilot submitter. These are implementation prerequisites rather than scientific-parameter changes.
- Official veRL LoRA documentation confirms `actor_rollout_ref.model.lora_adapter_path` loads an existing PEFT adapter *instead of creating a new one* for multi-stage training. Therefore directly pointing veRL at corrected F01 would continue its rank-16 LoRA, not realize the documented `F01 policy + new rank-32 RL LoRA` design.
- The scientifically clean implementation of the documented design is to materialize/merge corrected F01 into an immutable parent policy snapshot, use that same merged snapshot as the frozen reference, then attach a fresh rank-32 RL LoRA for each F10-F14 branch. The cheaper alternative is to continue training the F01 rank-16 adapter directly; this changes the planned parameterization and makes the SFT/RL adapter a single mutable object.
- This adapter-initialization choice is a genuine research tradeoff and must be confirmed before code/config implementation or Slurm submission.
- User confirmed the clean initialization design: merge corrected F01 into an immutable Qwen2.5-7B parent snapshot, use that parent as the shared actor/reference initialization, and attach a fresh rank-32 RL LoRA for each F10-F14 branch. Do not continue training the rank-16 SFT adapter directly.

## Frozen-parent implementation details

- F01 is a PEFT LoRA artifact produced by `scripts/train_sft_fallback.py`; the corrected adapter is stored remotely at `experiments/sft_fallback_full_20260901_stage16/checkpoints/final_adapter`.
- The GPU environment already includes `transformers`, `peft`, `torch`, veRL 0.9.0, and vLLM 0.20.2, so the merge can use PEFT `merge_and_unload(safe_merge=True)` inside a Slurm GPU job without a new dependency.
- The merged artifact must be written to a new immutable parent directory on project SSD. Metadata must record the base and adapter paths, adapter-config digest, output inventory, and creation time; an existing non-empty target must be rejected.
- F10 must point `actor_rollout_ref.model.path` at that merged parent. veRL then creates a fresh rank-32 LoRA from the common GRPO config, keeping the SFT parent frozen and the trainable RL delta independent.
- The initial pilot will stop at optimizer step 5 and save a checkpoint. A separately submitted resume job will target step 6. Both use one same-node two-task `srun`, per-role telemetry, and no automatic successor.
- `launch_verl.py` currently reads the policy only from the common manifest and hardcodes rollout/offload knobs. The smallest compatible change is to allow an experiment-level `policy_model` override and environment-controlled systems knobs while leaving scientific defaults untouched.
- Existing lifecycle metadata cannot represent a resumed job separately. The resume submitter should retain the same run/checkpoint directory and append job IDs/phases to a dedicated submission log while updating the manifest's current Slurm fields.
- The veRL console logger is the authoritative source for optimizer metrics (reward variance/advantages/KL/clip fraction/grad norm). The pilot wrapper must preserve complete trainer stdout/stderr and sample GPU telemetry independently so the acceptance report can be produced after the job.
- The first focused local test run passed all launch/config/manifest checks but the two merge-helper tests failed because Windows sandbox permissions deny writes to the process-wide `%TEMP%`. This is a test-environment issue, not a merge-code failure; temporary test directories must be rooted inside the writable repository.
- Repository-rooted `TemporaryDirectory` creation is also ACL-restricted in this desktop sandbox. The merge-helper unit tests therefore need immutable checked-in fixtures and must avoid filesystem mutation entirely; the other 30 tests and Python compilation passed.
- After switching to immutable fixtures, the full local suite passes all 32 tests. The new launch contract statically proves one same-node two-task `srun`, no `--exclusive`, no automatic successor, F10 merged-parent selection, and a fresh rank-32 LoRA.
- The authoritative project and Stage 05 documents are still on the superseded route (`F10-F14 blocked`, next `F03/G05`). Before cluster work they must be updated to distinguish: historical gate FAIL remains valid; F02 is a negative result; F03 is paused; only a bounded F10 pilot is authorized; formal runs remain locked pending manual pilot review.
- Project, Progress, Stage 04, Stage 05, the experiment overview, and the file-based plan now encode the approved bounded-pilot route. The formal family remains locked and the next executable dependency is the immutable-parent merge.
- Final local verification passes 32 unit tests, all 18 YAML manifests, the F10 veRL dry-run rendering, and earlier Python compilation. Local `bash -n` remains unavailable because Windows denies WSL service creation; shell syntax must be checked remotely before submission.

## 2026-09-01 live cluster preflight for F10

- Live association remains `cluster02` / account `msc` / default QoS `msc`; `override-limits-but-killable` is available but is not needed or selected. Partition max time is three days.
- Storage is healthy: project `jiatian001ssd` has 150 GB assigned and 77.3 GB used. The repository currently uses about 73 GB.
- The user queue was empty at preflight. Pro6000 capacity was busy overall, with only isolated single cards free at that instant; the one-GPU parent merge may start sooner than the later same-node two-GPU pilot.
- The corrected F01 adapter exists, is about 165 MB, declares PEFT LoRA rank 16 / alpha 32, and its `adapter_config.json` SHA-256 is `0cf0be17ac42687850315d4530701b5e72e51164af97f65b4c04baaf5dd50789`.
- The Qwen2.5-7B base snapshot exists and is about 15 GB. The intended derived-parent target is absent, satisfying the no-overwrite precondition.
- Remote deployment archive SHA-256 matched locally/remotely (`cc1196ca28aba2cdb57d22116d1de5ca3174b3babe736555cefd181e08de7565`). Remote Bash syntax, 32 tests, 18 YAML manifests, F10 dry-run, and both Slurm test-only checks passed.
- Immutable-parent merge job `133431` was submitted under normal `msc` QoS with 1x Pro6000, 4 CPUs, 90 GiB node memory, and a two-hour limit. It is pending for priority; no F10 job has been submitted.
- Merge job `133431` backfilled almost immediately on `gpu-pro6000-4` and completed in 64 seconds with exit `0:0`. It produced four safetensor shards plus tokenizer/config files (about 15 GB total) and a 10-file SHA-256 inventory; stderr is empty.
- The artifact-level merge contract is satisfied, but F10 remains blocked until a separate compute-node validation recomputes every manifest hash and actually loads the merged model and tokenizer.
- The first validation-helper test run exposed only a test-classification mistake: the new one-GPU parent validator had been added to the existing list that asserts all entries are dual-GPU jobs. The production Slurm request is intentionally 1x Pro6000; the test is split into explicit single- and dual-GPU assertions.
- Parent validation attempt 1 job `133439` failed in one second before artifact access because direct script execution omitted the repository root from `sys.path`. This is the same standalone-entrypoint packaging class seen in earlier builders; add ROOT explicitly, rerun all preflight checks, and preserve the failed job as a separate attempt.
- Retry job `133447` passed in 53 seconds on `gpu-pro6000-2`: exact 10-file/15,242,726,337-byte inventory hashes matched, the merged BF16 model loaded 7,615,616,512 parameters, tokenizer size was 151,665, and one-token generation succeeded. F10 may now be submitted under the bounded-pilot contract.
- A remote command that tried `python -m unittest tests.test_*` failed because `tests/` has no package initializer; this was a preflight invocation error. The preceding full remote `unittest discover` (34 tests after the added validator tests locally; 32 before those files) and production direct-file entrypoint both pass.
- F10 start job `133456` was submitted for run `f10_pilot_20260901_stage18` with target step 5, same-node 2x Pro6000, a single two-task `srun`, vLLM memory utilization 0.60, 16 seqs, 16384 batched tokens, 16 workers, microbatch 1, and offload enabled. The run has no automatic successor.
- F10 attempt 1 job `133456` failed after 2m11s before veRL import because direct-file `launch_verl.py` lacked ROOT on `sys.path`. It produced 0 optimizer steps and no checkpoint. Both GPU bindings were distinct and the 72B simulator reached health, isolating the failure to launcher packaging.
- The original dry-run returned before importing `src.training.verl_entrypoint`, so it could not catch this class. The fix both inserts ROOT and makes dry-run import the project entrypoint before rendering; retry must use a new run directory and Job ID.
- F10 attempt 2 job `133478` proved launcher, veRL import, Hydra overrides, and config validation work, then failed at `ray.init()` because `$PROJECT_ROOT/cache/ray/.../plasma_store` exceeds Linux's 107-byte AF_UNIX path limit. This is a runtime path defect, not an optimizer or model failure.
- The scoped fix is a short `/tmp/cabin-ray-$SLURM_JOB_ID` for ephemeral Ray sockets/session metadata only. Persistent experiment logs/checkpoints stay on SSD and Ray's object store stays in shared memory.
- CPU Slurm job `133503` validated the exact runtime fix in 29 seconds: `/tmp/cabin-ray-133503`, a short Ray session dir, a successful remote result of 42, and clean shutdown. It is now safe to retry the dual-GPU pilot without spending GPUs merely to test Ray socket creation.
- Simulator telemetry at memory util 0.92 peaked around 93.3/97.9 GiB (95.4% used). For attempt 3, lower only the simulator system cap to 0.86, retaining 16 max seqs and expecting roughly 10% total headroom; keep policy/trainer 0.60 until its first real load profile exists.
- F10 attempt 3 job `133512` successfully started Ray and executed remote `TaskRunnerV1`, then failed because `transfer_queue` is absent. The installed veRL is 0.9.0, its distribution metadata does not list this module, and both `find_spec` and `pip show` confirm it is missing. Audit official source/package versions before any installation.
- Official veRL requirements pin `TransferQueue==0.1.7`; job `133532` installed it with `--no-deps` and successfully imported veRL's remote TaskRunner. The smoke itself failed only because `@ray.remote` exposes an `ActorClass` wrapper without ordinary `__name__`; record the wrapper type and rerun for a formal JSON PASS.
- Retry `133541` produced the formal PASS report: exact TransferQueue 0.1.7, all six required KV APIs, and veRL's `ActorClass(TaskRunnerV1)` import are compatible. The F10 dependency blocker is closed without changing Torch/CUDA/vLLM packages.
- F10 attempt 4 (`133549`) reached veRL actor creation, where cluster02's simultaneous CUDA and ROCR visibility variables triggered a deliberate conflict guard. Preserve CUDA and clear AMD-only HIP/ROCR variables on NVIDIA tasks, then verify the exact Ray actor + veRL hook path on one GPU before another dual-GPU attempt.
- GPU smoke `133567` confirmed cluster02 starts with CUDA=0 and ROCR=0, and that clearing AMD variables yields CUDA-only state in the driver, Ray GPU actor, and after veRL's visibility hook. It is safe to retry the dual-GPU pilot without changing training semantics.
- F10 attempt 5 reached exact policy module construction and exposed the next missing runtime layer: veRL's automodel engine defaults to FlashAttention2. The official veRL stable image pins/force-builds flash_attn 2.8.3, whose tagged setup emits sm_120 kernels on CUDA >=12.8; compile only sm_120 and validate forward/backward plus exact parent loading before retrying.
- FA2 install `133600` failed at PyTorch's extension guard because the loaded nvcc was 12.8 while Torch is cu130; no wheel was built or installed. Cluster CUDA/13.0.0 is available, so align only the trainer/build toolchain to 13.0 and leave the validated simulator process on 12.8.
- FA2 retry `133615` built a 63.2 MB sm_120 wheel and passed BF16 forward/backward plus exact merged-parent FlashAttention2 loading/generation on Pro 6000 (15.29 GB peak). The trainer attention runtime is now validated end to end.
- F10 attempt 6 completed all model/runtime initialization and reached the first 24 AgentLoop requests. Every request failed only because CAR_BENCH_DATASET_ROOT was not exported into Ray; validate both required OmegaConf env values inside a real Ray worker before retrying.
- CPU smoke `133700` confirmed the exact AgentLoop config resolves canonical CAR root, target, and simulator URL inside a real Ray worker. Environment propagation is closed without changing any task or reward semantics.
- F10 attempt 7 `133709` was the first run to complete initialization, initial validation, and a 16-rollout training batch. It failed before the first optimizer step because unchunked `entropy_from_logits` tried to allocate another 20.44 GiB with 18.86 GiB free; trainer telemetry peaked at 96,055/97,887 MiB (98.13%).
- The resolved veRL config exposes `entropy_from_logits_with_chunking=False` and a 2048-token chunk size. Enabling the built-in chunked entropy path is the narrowest semantics-preserving fix because it reduces the full-vocabulary softmax intermediate without changing group size, effective batch, sampling, sequence limits, reward, advantage, optimizer, or model.
- F10 attempt 8 `134671` resolved actor/ref and actor FSDP chunking as `True/2048`, but `prepare_model_outputs` still called unchunked entropy because `actor_rollout_ref.model.use_remove_padding=False` selects veRL's dense branch. The installed veRL 0.9 source only honors chunking in the packed/remove-padding branch.
- Attempt 8 ran 10m21s on `gpu-pro6000-11`, completed initial validation and 16 rollouts, then OOMed at 0/5 while requesting 20.80 GiB with 18.15 GiB free. Simulator/trainer telemetry peaked at 89.47%/92.52%.
- User approved the recommended next route: enable `use_remove_padding=True`, validate exact parent + FA2 + LoRA + finite log-prob/entropy through a one-GPU packed-path smoke, and submit a new five-step two-GPU F10 only after smoke PASS. All scientific settings and memory caps remain frozen.
