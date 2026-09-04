# CabinAgent-RL Direct-RL Experiment Plan

## Claims

1. Qwen2.5-7B-Instruct has sufficient initial CAR tool behavior and outcome variance to enter GRPO without project-specific SFT.
2. Long-horizon advantage shaping changes training stability and retained reasoning length relative to vanilla GRPO.
3. CAR-specific PRM-Lite reduces invalid, redundant, hallucinated and policy-violating behavior.
4. PRM-Lite + LATA combines a dense signal source with less length-diluted propagation and should outperform either component alone.

Current evidence status: claim 1 is not supported by the valid G02 gate because mixed outcome group ratio was `0.0`. Claims 2-4 are untested because no veRL training run has started.

## Fixed Conditions

- E10-E14 initialize new LoRA weights from the same immutable Qwen2.5-7B-Instruct model revision.
- CAR train/dev manifests, BFCL subset, seeds, simulator model/prompt, tool environment, rollout limits and evaluation protocol are identical.
- No GRPO branch starts from another branch.
- Project-specific SFT is absent from the main matrix and may run only after a recorded direct-RL gate failure.
- CAR test remains hidden from model selection.
- Formal runs stop at 250 steps; step 300 and the former R05 expansion are outside the approved matrix.

## Direct-RL Gate

Run the direct-RL gate on at least 20 CAR train tasks with four rollouts per task. All rollouts in a group share initial state and an identical first user message. PASS requires parse rate >=0.95, executable rate >=0.85, mixed outcome group ratio >=0.20, initial-user consistency =1.00, loop/max-turn rate <=0.20 and at least one success.

The gate uses outcome reward only. A FAIL produces `reports/direct_rl_gate.json` and permits a separate minimal-SFT fallback family; it does not modify E10-E14 silently.

G00 job `131880` passed functional metrics but failed mixed reward variance (`0.10`) and initial-user consistency (`0.85`). G01 is one controlled rerun with the same tasks, models, reward and thresholds: initial simulator turns are greedy and must return `CONTINUE`; policy sampling is reproducible at temperature `1.0`, top-p `0.95`, seed `42`. A second failure routes to minimal SFT rather than another sampling sweep.

G01 job `131930` improved mixed variance to `0.15` but retained `0.85` first-user consistency because concurrent greedy vLLM requests produced wording variants. G02 repairs only this unmet grouped-rollout contract by generating and caching one validated initial response per full task prompt. If G02 reaches consistency `1.0` but still fails reward variance, it routes directly to minimal SFT.

G02 job `131950` reached first-user consistency `1.0` but produced zero mixed groups, so the direct-RL claim is not supported. The direct E10-E14 matrix remains frozen and blocked. F00/F01 form a separate fallback family: a two-step LoRA SFT smoke followed by one full epoch over deduplicated successful CAR trajectories. Any later GRPO run initialized from F01 must use F10-F14 identifiers.

F01 job `132020` technically completed the 38-record fallback, but G03 attempt 1 (`132043`) failed after 6m10s on two Pro 6000 GPUs. The SFT targets had retained string-valued OpenAI `function.arguments`; Qwen templating double-encoded them, and CAR automatic evaluation crashed on string arguments. The old adapter is rejected for downstream use even though its training loss was low.

The corrected fallback gate remained a valid FAIL and is not rewritten. A separately approved bounded F10 pilot directly established optimizer-signal and resume viability, after which the formal fallback family was manually unlocked. F10 is now complete; F11-F14 remain subject to a post-F10 human gate and never auto-chain.

## Main Run Matrix

| ID | Initialization | Reward | Advantage | Steps | Checkpoint eval |
|---|---|---|---|---:|---|
| E00 | Qwen2.5-7B-Instruct | none | none | 0 | CAR dev/test, BFCL |
| G00 | Qwen2.5-7B-Instruct | outcome | none | rollout only | >=20 tasks x 4 |
| G01 | Qwen2.5-7B-Instruct | outcome | none | rollout only | controlled gate rerun |
| G02 | Qwen2.5-7B-Instruct | outcome | none | rollout only | grouped-initialization repair |
| F00 | Qwen2.5-7B-Instruct + LoRA | SFT | assistant-token CE | 2 steps | infrastructure smoke |
| F01 | Qwen2.5-7B-Instruct + LoRA | SFT | assistant-token CE | 1 epoch | fallback parent only |
| G03 | F01 adapter | outcome | none | rollout only | post-SFT variance gate |
| E10 | Qwen2.5-7B-Instruct + new LoRA | outcome | GRPO | 250 | 50/100/150/200/250 |
| E11 | Qwen2.5-7B-Instruct + new LoRA | outcome | Turn-Discount | 250 | 50/100/150/200/250 |
| E12 | Qwen2.5-7B-Instruct + new LoRA | outcome | LATA | 250 | 50/100/150/200/250 |
| E13 | Qwen2.5-7B-Instruct + new LoRA | outcome + 0.3 PRM-Lite | GRPO | 250 | 50/100/150/200/250 |
| E14 | Qwen2.5-7B-Instruct + new LoRA | outcome + 0.3 PRM-Lite | LATA | 250 | 50/100/150/200/250 |
| F10 | corrected F01 adapter + new RL LoRA | outcome | GRPO | 250 | 50/100/150/200/250 |
| F11 | corrected F01 adapter + new RL LoRA | outcome | Turn-Discount | 250 | 50/100/150/200/250 |
| F12 | corrected F01 adapter + new RL LoRA | outcome | LATA | 250 | 50/100/150/200/250 |
| F13 | corrected F01 adapter + new RL LoRA | outcome + 0.3 PRM-Lite | GRPO | 250 | 50/100/150/200/250 |
| F14 | corrected F01 adapter + new RL LoRA | outcome + 0.3 PRM-Lite | LATA | 250 | 50/100/150/200/250 |

## Run Order

1. Preserve the completed G00-G02 direct gate artifacts and the blocked E10-E14 conclusion.
2. Normalize fallback tool-call arguments, rebuild data, and pass the tokenizer plus round-trip gates.
3. Rerun F00 and full F01 in fresh attempt directories; do not reuse the invalid adapter.
4. Run G03 attempt 2 and archive every trajectory plus the machine-readable report.
5. If G03 passes, run a 1-5 step F10 single-node dual-GPU smoke including save/resume.
6. Run F10, F11/F12, then F13/F14 to isolate infrastructure, advantage, reward and joint effects.
7. Evaluate every saved checkpoint on CAR dev and BFCL, select by the declared dev rule, then run frozen CAR test once.

## Checkpoint Selection

Primary key: highest CAR dev task success. Ties are broken by lower automatic policy violation, lower invalid tool rate, then fewer average turns. This rule applies identically to every branch.

## Required Measurements

- Task: CAR success/consistent pass/final state and BFCL tool/argument accuracy.
- Behavior: parse/executable rate, hallucination, clarification, policy violation, invalid/redundant calls, recovery and loops.
- Horizon: turns and response-token p50/p95.
- RL: group reward distribution, zero-variance ratio, KL, clip fraction, grad norm and checkpoint health.
- System: simulator latency/throughput, trainer wait fraction, step time, GPU memory and failures.

## Stopping Rules

- Stop on NaN, repeated OOM, corrupted checkpoints, wrong base revision, or reward/trajectory schema mismatch.
- Pause when train reward rises while two consecutive CAR dev evaluations regress.
- Do not change simulator identity, prompt or sampling semantics for only a subset of branches.
- Failed attempts remain in the tracker and retries receive a new attempt number.

## Result-to-Claim Contract

The final report must show all five branches, all evaluated checkpoints and negative results. Claims require frozen evaluation rather than train reward. E14 is jointly beneficial only if it exceeds both E12 and E13 under the same protocol, with uncertainty reported where feasible.
