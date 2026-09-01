# 03 Direct-RL 门禁阶段

## 状态

已完成，实验结论为 **FAIL**。Direct-Instruct 初始化的 E10-E14 不允许启动。

## 目标

验证 Qwen2.5-7B-Instruct 在不经过项目 SFT 的情况下，是否能为 outcome-only GRPO 产生合法且具有组内方差的 CAR rollout。

## 固定设置

| 项目 | 设置 |
|---|---|
| Task | 固定 20 个 CAR train task |
| Group | 每 task 4 rollout，共 80 trajectory |
| Policy sampling | temperature `1.0`，top-p `0.95`，seed `42` 派生 trial seed |
| Simulator | 72B-AWQ；首轮同组共享，后续 temperature `0.2` |
| Reward | CAR deterministic outcome，不使用 PRM-Lite 或 LLM evaluator reward |
| Parse threshold | `>= 0.95` |
| Executable threshold | `>= 0.85` |
| Mixed group threshold | `>= 0.20` |
| Initial-user consistency | `= 1.00` |
| Loop/max-turn threshold | `<= 0.20` |
| Success threshold | `>= 1` |

## 执行结果

| Run | Job / GPU | Parse | Executable | Mixed | Consistency | Loop | Success | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| G00 | `131880`, 2 GPU | 0.996667 | 0.989062 | 0.10 | 0.85 | 0.0125 | 27 | FAIL |
| G01 attempt 1 | `131911`, 2 GPU | - | - | - | - | - | - | Conda 启动失败 |
| G01 attempt 2 | `131930`, 2 GPU | 0.994479 | 0.864625 | 0.15 | 0.85 | 0.10 | 15 | FAIL |
| G02 | `131950`, 2 GPU | 0.995729 | 0.888892 | 0.00 | 1.00 | 0.075 | 12 | 合法 FAIL |

G02 是首个满足同组初始状态一致性的有效结论。20 个 group 中没有 mixed outcome group，因此标准 group-normalized GRPO 没有可用的组内优势信号。

## 已完成改进

- G00 后增加首轮 simulator 的 `CONTINUE` 契约和可复现 policy sampling。
- G01 后增加按完整初始 prompt 哈希的一次生成/同组复用缓存，消除并发 greedy 文本差异。
- 修复 Conda 激活、7B context 上限和跨平台 shell executable 问题。
- 冻结 stopping rule：G02 若 consistency 达到 1.0 但 mixed ratio 仍失败，不继续调 gate，不放宽阈值。

## 遗留问题与下一步

- Direct-Instruct E10-E14 保持阻塞，不能用 fallback 结果改写该结论。
- 按预先约定进入独立 Minimal-SFT fallback 家族 F00/F01/G03/F10-F14。

## 主要产物

- `configs/train/direct_rl_gate.yaml`
- `src/eval/rollout_gate.py`
- `scripts/check_rollout_gate.py`
- `reports/cluster/G00-131880/trajectories.jsonl`
