# CabinAgent-RL Experiment Tracker

| ID | Attempt | Status | Initialization | Slurm job | Best step | CAR dev | BFCL | CAR test | Run directory | Notes |
|---|---:|---|---|---|---:|---:|---:|---:|---|---|
| E00 | 1 | pending | Qwen2.5-7B-Instruct | - | 0 | - | - | - | - | Official-model prompt baseline |
| G00 | 1 | failed | Qwen2.5-7B-Instruct | 131880 | 0 | - | - | - | experiments/G00 | Mixed ratio 0.10; first-user consistency 0.85 |
| G01 | 1 | failed | Qwen2.5-7B-Instruct | 131911 | 0 | - | - | - | experiments/G01 | Conda activation failed before model load; no rollout |
| G01 | 2 | failed | Qwen2.5-7B-Instruct | 131930 | 0 | - | - | - | experiments/G01 | Mixed 0.15; initial-user consistency 0.85; invalid grouped contract |
| G02 | 1 | failed | Qwen2.5-7B-Instruct | 131950 | 0 | - | - | - | experiments/G02 | Valid gate: consistency 1.0, mixed reward ratio 0.0 |
| F00 | 1 | complete | Qwen2.5-7B-Instruct + LoRA | 132008 | 2 | - | - | - | experiments/sft_fallback_smoke_20260830T141431Z | loss 3.609; eval loss 1.910; 21.165 GiB peak |
| F00 | 2 | complete | Qwen2.5-7B-Instruct + LoRA | 132935 | 2 | - | - | - | experiments/sft_fallback_smoke_20260901_stage16 | corrected arguments; train/eval loss 3.313/1.798 |
| F01 | 1 | failed | Qwen2.5-7B-Instruct + LoRA | 132013 | 2 | - | - | - | experiments/sft_fallback_full_20260830T141431Z | Invalid: inherited 2-step/4-record smoke limits |
| F01 | 2 | failed | Qwen2.5-7B-Instruct + LoRA | 132020 | 10 | - | - | - | experiments/sft_fallback_full_20260830T141431Z_r1 | Slurm complete; invalidated by G03 because tool arguments were double-encoded |
| F01 | 3 | complete | Qwen2.5-7B-Instruct + LoRA | 132942 | 10 | - | - | - | experiments/sft_fallback_full_20260901_stage16 | corrected arguments; train/eval loss 0.873/0.603 |
| G03 | 1 | failed | F01 adapter | 132043 | 0 | - | - | - | experiments/G03 | 2 Pro6000; 6m10s; CAR evaluator TypeError on string arguments; no gate report |
| G03 | 2 | cancelled | corrected F01 adapter | 132946 | 0 | - | - | - | experiments/G03 | DependencyNeverSatisfied on failed smoke 132933; cancelled with explicit authorization |
| G03 | 3 | failed | corrected F01 adapter | 132967 | 0 | - | - | - | experiments/G03 | 80 trajectories; parse 0.9994; executable 0.8447; mixed 0.15; loop 0.2625; FAIL |
| F02-DATA | 1 | complete | G00-G03 success + corrective | 133301 | 0 | - | - | - | data/sft/attempts/f02_20260901_stage17 | 60/20 train/val; 14 corrective; tokenizer gate PASS |
| F02-S | 1 | complete | Qwen2.5-7B-Instruct + corrective LoRA | 133303 | 2 | - | - | - | experiments/sft_corrective_smoke_f02_20260901_stage17 | train/eval loss 3.313/2.140; 21.16 GiB |
| F02 | 1 | complete | Qwen2.5-7B-Instruct + corrective LoRA | 133306 | 15 | - | - | - | experiments/sft_corrective_full_f02_20260901_stage17 | 60/20 records; train/eval loss 1.089/0.739; 34.304 GiB |
| G04 | 1 | failed | F02 adapter | 133308 | 0 | - | - | - | experiments/G04 | 80 trajectories; parse 0.9958; executable 0.8468; mixed 0.10; loop 0.275; FAIL |
| E10 | 1 | blocked_on_G02 | Qwen2.5-7B-Instruct + new LoRA | - | - | - | - | - | - | Vanilla GRPO; direct family remains blocked |
| E11 | 1 | blocked_on_G02 | Qwen2.5-7B-Instruct + new LoRA | - | - | - | - | - | - | Turn-Discount; direct family remains blocked |
| E12 | 1 | blocked_on_G02 | Qwen2.5-7B-Instruct + new LoRA | - | - | - | - | - | - | LATA; direct family remains blocked |
| E13 | 1 | blocked_on_G02 | Qwen2.5-7B-Instruct + new LoRA | - | - | - | - | - | - | PRM-Lite; direct family remains blocked |
| E14 | 1 | blocked_on_G02 | Qwen2.5-7B-Instruct + new LoRA | - | - | - | - | - | - | PRM-Lite + LATA; direct family remains blocked |
| F10 | 1 | blocked_on_G03 | corrected F01 adapter + new RL LoRA | - | - | - | - | - | - | Vanilla GRPO |
| F01-MERGE | 1 | complete | corrected F01 adapter -> immutable merged parent | 133431 | 0 | - | - | - | models/derived/Qwen2.5-7B-Instruct-F01-merged-20260901 | gpu-pro6000-4; 64s; 15GB; 10-file hash manifest; load validation pending |
| F01-VALIDATE | 1 | failed | immutable merged parent | 133439 | 0 | - | - | - | - | gpu-pro6000-4; 1s; standalone import path failure before artifact read |
| F01-VALIDATE | 2 | complete | immutable merged parent | 133447 | 0 | - | - | - | reports/f01_parent_validation_133447.json | gpu-pro6000-2; 53s; 10-file hash + BF16 load + tokenizer + generation PASS |
| F10 | 2 | failed | validated corrected-F01 parent + fresh rank32 RL LoRA | 133456 | 0/5 | - | - | - | experiments/f10_pilot_20260901_stage18 | gpu-pro6000-7; 2m11s; launcher standalone import failure before veRL; no checkpoint |
| F10 | 3 | failed | validated corrected-F01 parent + fresh rank32 RL LoRA | 133478 | 0/5 | - | - | - | experiments/f10_pilot_20260901_stage18_r1 | gpu-pro6000-11; 3m56s; Ray AF_UNIX socket path too long; no checkpoint |
| RAY-SMOKE | 1 | complete | short Job-ID-scoped Ray runtime | 133503 | 0 | - | - | - | reports/ray_runtime_smoke_133503.json | cpu-1; 29s; ray.init/remote/shutdown PASS |
| F10 | 4 | failed | validated corrected-F01 parent + fresh rank32 RL LoRA | 133512 | 0/5 | - | - | - | experiments/f10_pilot_20260901_stage18_r2 | gpu-pro6000-11; 2m15s; missing transfer_queue in Ray worker; no checkpoint |
| TQ-SMOKE | 1 | failed | TransferQueue 0.1.7 + veRL TaskRunner import | 133532 | 0 | - | - | - | - | cpu-1; 1m19s; package/install and joint import succeeded; checker assumed ActorClass had __name__; no JSON report |
| TQ-SMOKE | 2 | complete | TransferQueue 0.1.7 + veRL TaskRunner import | 133541 | 0 | - | - | - | reports/transfer_queue_smoke_133541.json | cpu-1; 1m04s; exact version + 6 APIs + ActorClass TaskRunner joint import PASS |
| F10 | 5 | failed | validated corrected-F01 parent + fresh rank32 RL LoRA | 133549 | 0/5 | - | - | - | experiments/f10_pilot_20260901_stage18_r3 | gpu-pro6000-11; 2m40s; CUDA+ROCR visibility conflict in veRL worker; no checkpoint |
| GPU-ENV-SMOKE | 1 | complete | Ray GPU actor + veRL visibility hook | 133567 | 0 | - | - | - | reports/gpu_visible_env_smoke_133567.json | gpu-pro6000-2; 29s; CUDA-only driver/actor/hook PASS after clearing HIP/ROCR |
| F10 | 6 | failed | validated corrected-F01 parent + fresh rank32 RL LoRA | 133581 | 0/5 | - | - | - | experiments/f10_pilot_20260901_stage18_r4 | gpu-pro6000-8; 2m36s; veRL default FlashAttention2 missing before policy weight load; no checkpoint |
| FA2-SMOKE | 1 | failed | flash_attn 2.8.3 sm_120 + exact parent | 133600 | 0 | - | - | - | - | gpu-pro6000-2; 42s; nvcc 12.8 mismatched Torch cu130 before compile; package not installed |
| FA2-SMOKE | 2 | complete | flash_attn 2.8.3 sm_120 + exact parent | 133615 | 0 | - | - | - | reports/flash_attention_smoke_133615.json | gpu-pro6000-2; 20m25s; BF16 fwd/bwd + exact parent FA2 load/generate PASS; 15.29GB peak |
| F10 | 7 | failed | validated corrected-F01 parent + fresh rank32 RL LoRA | 133674 | 0/5 | - | - | - | experiments/f10_pilot_20260901_stage18_r5 | gpu-pro6000-7; 4m21s; full init then Ray AgentLoop missing CAR_BENCH_DATASET_ROOT; no checkpoint |
| AGENT-LOOP-ENV | 1 | complete | Ray OmegaConf environment propagation | 133700 | 0 | - | - | - | reports/agent_loop_env_smoke_133700.json | cpu-1; 32s; canonical CAR root exists and target/URL resolve PASS |
| F10 | 8 | failed | validated corrected-F01 parent + fresh rank32 RL LoRA | 133709 | 0/5 | - | - | - | experiments/f10_pilot_20260901_stage18_r6 | gpu-pro6000-3; 7m07s; init/validation/16 rollouts passed; old-log-prob entropy softmax CUDA OOM; no checkpoint |
| F10 | 9 | failed | validated corrected-F01 parent + fresh rank32 RL LoRA + chunked entropy | 134671 | 0/5 | - | - | - | experiments/f10_pilot_20260901_stage18_r7 | gpu-pro6000-11; 10m21s; dense-padding FSDP path ignored resolved chunking and OOMed in old-log-prob entropy; no checkpoint |
| F11 | 1 | blocked_on_G03 | corrected F01 adapter + new RL LoRA | - | - | - | - | - | - | Turn-Discount |
| F12 | 1 | blocked_on_G03 | corrected F01 adapter + new RL LoRA | - | - | - | - | - | - | LATA |
| F13 | 1 | blocked_on_G03 | corrected F01 adapter + new RL LoRA | - | - | - | - | - | - | PRM-Lite |
| F14 | 1 | blocked_on_G03 | corrected F01 adapter + new RL LoRA | - | - | - | - | - | - | PRM-Lite + LATA |

Statuses: `pending`, `prepared`, `blocked_on_F00`, `blocked_on_F01`, `blocked_on_G02`, `blocked_on_G03`, `blocked_on_failed_dependency`, `queued`, `running`, `failed`, `stopped`, `cancelled`, `complete`. Never replace a failed row; add a new attempt.
