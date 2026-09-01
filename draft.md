# CabinAgent-RL: 短周期智能座舱 Agentic RL 项目草案

**日期**: 2026-06-19  
**约束**: 周期较短，不从零构建 benchmark；计算资源为 2 台 A100 80GB 服务器，按总计约 2×A100 80GB 规划。

## 1. 项目定位

本项目借鉴 `agentic-grpo-longhorizon` 的闭环思路，但做短周期可落地版本：不自建完整 benchmark，而是基于成熟公开 benchmark 搭建座舱 Agent 的训练、评测和优化闭环。

核心目标:

> 使用成熟 benchmark 验证智能座舱多轮工具 Agent 的工具调用、状态跟踪、能力边界感知和模糊请求处理能力，并在 2×A100 80GB 资源内完成 Prompt baseline、SFT/DPO 和小规模 GRPO 改进。

项目不是追求从零做一个完整车厂级数据闭环，而是做一个能展示 JD 关键能力的研究工程:

- 大模型工具调用落地
- 多轮对话状态跟踪
- Agentic RL 小闭环
- Reward / preference 数据构造
- 离线评测与失败分析
- vLLM 推理和低资源训练部署

## 2. 推荐主线

### 2.1 Benchmark 选择

短周期下建议采用“三层 benchmark”，不要铺太大。

| 层级 | Benchmark | 是否 must-run | 作用 |
|---|---|---:|---|
| 座舱主 benchmark | **CAR-bench** | 是 | 直接评测 in-car voice assistant，多轮工具使用、能力边界、歧义处理 |
| 工具调用基础 benchmark | **BFCL V4** | 是 | 评测函数调用准确性、可执行性、多轮/agentic tool use、幻觉 |
| 多轮 Agent benchmark | **tau2/3-bench** | 可选 | 复用成熟 tool-agent-user pipeline，作为技术路线参考或外部泛化评测 |
| DST 辅助 benchmark | **MultiWOZ 2.4** | 可选 | 专门评测 dialogue state tracking，辅助证明 DST 能力 |

推荐取舍:

1. **必须做 CAR-bench**: 它和智能座舱 JD 最贴近，包含车载语音助手、多轮工具、能力限制、歧义等元素。
2. **必须做 BFCL V4 子集**: 它是成熟函数调用评测，能证明工具调用基础能力，不依赖座舱自建环境。
3. **不建议短周期内自建 Cockpit-Bench**: 最多只做 20-50 条内部 smoke test，不作为核心结果。
4. **tau-bench 原仓库只作参考**: 原 repo 已提示任务过旧，若要跑应优先看 tau2/3-bench。
5. **MultiWOZ 2.4 放到辅助**: 有时间再做 DST fine-tuning 或 DST eval，不让它拖慢主线。

### 2.2 为什么 CAR-bench 适合作为主 benchmark

CAR-bench 是现成的汽车座舱方向 benchmark，目标是评估多轮工具型 LLM agent 在真实用户场景中的可靠性，覆盖:

- base multi-turn task completion
- hallucination / limit-awareness under missing capabilities
- disambiguation / uncertainty resolution
- navigation、productivity、charging、vehicle control 等工具域

这正好对应 JD 里的:

- 智能座舱对话系统
- 多轮规划
- Agentic Tool-Use
- 工具幻觉
- 座舱环境下复杂交互

## 3. 模型选型

资源限制是 2×A100 80GB，因此主模型不要上 32B/72B 训练。推荐以 7B/8B 为 policy，32B 只做 teacher/judge。

### 3.1 推荐配置

| 角色 | 模型 | 部署/训练方式 | 理由 |
|---|---|---|---|
| Policy 主模型 | Qwen2.5-7B-Instruct 或 Qwen3-8B | LoRA / QLoRA SFT、DPO、GRPO | 2×A100 可承受，中文和工具调用能力较强 |
| Teacher / Judge | Qwen3-32B 或闭源 API | 推理，不训练 | 用于生成偏好数据、失败标签、RLAIF judge |
| Reward / PRM | 规则版 Cabin-PRM-Lite 起步；后续 Qwen3-4B/8B classifier | 初期不训练 RM | 周期短，先用可解释规则奖励，避免 RM 数据成本 |
| 推理服务 | vLLM | BF16/FP16，LoRA merge 后服务化 | 对齐 JD 中推理加速和工程部署 |
| 端侧示意模型 | Qwen2.5-1.5B/3B 或 Qwen3-1.7B/4B | 蒸馏/量化作为 optional | 只做部署延迟 demo，不作为主训练对象 |

### 3.2 推荐默认模型

第一版建议:

- **Policy**: Qwen2.5-7B-Instruct
- **Teacher/Judge**: Qwen3-32B-Instruct 或 API 强模型
- **训练框架**: LLaMA-Factory / TRL 做 SFT + DPO，veRL 做小规模 GRPO
- **推理框架**: vLLM

原因:

- Qwen2.5-7B 工具链更成熟，踩坑少。
- Qwen3-8B 可以作为升级版，但如果周期很短，优先稳定复现。
- 2×A100 80GB 足够跑 7B LoRA SFT/DPO/GRPO，不适合做 32B RL。

## 4. 技术路线

短周期项目应采用成熟路线，不做太多自研算法。推荐路线:

```text
Prompt Baseline
  -> LoRA SFT
  -> DPO/RLAIF
  -> 小规模 GRPO + PRM-Lite
  -> 独立 benchmark eval
  -> failure analysis
  -> vLLM latency demo
```

### 4.1 阶段 0: 环境和基线

目标:

- 跑通 CAR-bench 和 BFCL V4 子集。
- 建立 Qwen policy 的 prompt-only baseline。
- 收集失败轨迹。

系统:

- Qwen2.5-7B-Instruct + vLLM
- tool schema 使用 benchmark 原生 schema
- 不改 benchmark

输出:

- `baseline_carbench_report.json`
- `baseline_bfcl_report.json`
- `failure_cases/*.json`
- `failure_taxonomy.md`

关键指标:

- task success / pass rate
- consistent pass rate
- tool call accuracy
- hallucination rate
- disambiguation success
- limit-awareness accuracy

### 4.2 阶段 1: LoRA SFT

目标:

- 让模型学会 benchmark 的工具调用格式。
- 提升多轮任务中工具选择和参数生成。
- 学会座舱场景中的能力边界表达。

数据来源:

- benchmark 提供的成功轨迹或可生成轨迹。
- teacher model 对失败任务生成修正轨迹。
- 少量人工/规则过滤后的高质量样本。

训练建议:

| 配置 | 建议值 |
|---|---|
| 方法 | LoRA SFT |
| 模型 | Qwen2.5-7B-Instruct |
| GPU | 1×A100 80GB |
| max length | 8k 起步，必要时 16k |
| batch | micro batch 1-2，gradient accumulation |
| epoch | 1-3 |
| 框架 | LLaMA-Factory / TRL / Axolotl |

训练重点:

- assistant tool call 部分计算 loss。
- user/tool/system turn mask 掉。
- 保持 benchmark 原始工具 schema，不引入自定义工具。

### 4.3 阶段 2: DPO / RLAIF

目标:

- 优先解决工具幻觉、能力边界、歧义处理这三类坏行为。
- 在 RL 前先用偏好优化降低明显错误。

偏好数据构造:

| chosen | rejected |
|---|---|
| 不确定时追问 | 直接执行模糊指令 |
| 能力缺失时明确说明限制 | 编造工具或声称已执行 |
| 使用合法工具和参数 | 生成不存在工具或非法参数 |
| 保持上一轮状态一致 | 忘记目标、位置、乘员或偏好 |
| 简洁自然回复 | 冗长、重复、打断驾驶体验 |

训练建议:

| 配置 | 建议值 |
|---|---|
| 方法 | DPO 或 SimPO |
| 模型 | SFT checkpoint |
| GPU | 1×A100 80GB |
| 数据量 | 2k-10k preference pairs |
| 目的 | 稳定降低 bad behavior |

为什么 DPO 放在 GRPO 前:

- 成本低，工程成熟。
- 对工具幻觉和安全边界很有效。
- 能给后续 GRPO 一个更稳定的 policy 起点。

### 4.4 阶段 3: 小规模 GRPO + PRM-Lite

目标:

- 借鉴 `agentic-grpo-longhorizon`，做一个轻量 Agentic RL 闭环。
- 不追求大规模训练，而是证明过程奖励能改善多轮工具决策。

推荐先做 outcome-only GRPO，再做 PRM-Lite GRPO 对比。

#### PRM-Lite 设计

不训练 reward model，先用规则奖励:

```text
reward = outcome
       + 0.2 * tool_validity
       + 0.2 * state_consistency
       + 0.2 * limit_awareness
       + 0.2 * disambiguation
       + 0.2 * response_efficiency
```

规则示例:

- 非法工具名: -0.1
- 工具参数 schema 不合法: -0.1
- 缺失能力时仍执行: -0.2
- 模糊请求未追问: -0.1
- 重复调用同一失败工具: -0.05
- 合理追问: +0.05
- 正确引用上一轮状态: +0.05
- 成功完成任务: +1.0

#### GRPO 配置建议

| 配置 | 建议值 |
|---|---|
| 模型 | Qwen2.5-7B SFT/DPO checkpoint |
| 框架 | veRL |
| GPU | 2×A100 80GB |
| rollout backend | vLLM |
| group size | 4 起步，稳定后 8 |
| max response length | 4096-8192 |
| max turns | 使用 benchmark 默认，不额外放大 |
| step | 100-300 step |
| save/eval freq | 50 step |
| 训练方式 | LoRA RL，避免全参 |

短周期不要做:

- 不训 32B/72B policy。
- 不自建复杂 user simulator。
- 不从零训练 reward model。
- 不大改 veRL 内核，除非必要。

### 4.5 可选: LATA / Turn-Discount 小消融

如果时间允许，可做一个小消融:

| 方法 | 目的 |
|---|---|
| GRPO outcome-only | baseline |
| GRPO + PRM-Lite | 验证过程奖励 |
| GRPO + PRM-Lite + LATA | 验证长链路信号传播 |

但如果周期很短，主线只做 **DPO vs GRPO + PRM-Lite** 即可。

## 5. 计算资源规划

假设资源总计 2×A100 80GB。

### 5.1 服务器分工

| 服务器 | 用途 |
|---|---|
| Server A | 训练 policy: SFT / DPO / GRPO |
| Server B | teacher/judge 推理、vLLM eval、数据生成、benchmark eval |

如果两台服务器不能高速互联:

- SFT/DPO 单卡完成。
- GRPO 时尽量把 policy rollout 和训练放同一台。
- teacher/judge 离线生成数据，不参与在线 rollout。

### 5.2 训练可行性

| 任务 | 资源 | 预计可行性 |
|---|---|---|
| Qwen2.5-7B LoRA SFT | 1×A100 | 高 |
| Qwen2.5-7B DPO | 1×A100 | 高 |
| Qwen2.5-7B GRPO LoRA | 2×A100 | 中高 |
| Qwen3-8B LoRA SFT/DPO | 1×A100 | 高 |
| Qwen3-14B QLoRA | 1-2×A100 | 中 |
| 32B policy RL | 2×A100 | 不建议 |
| 72B user simulator 本地常驻 | 2×A100 | 不建议 |

推荐保守路线:

- 主实验全用 7B/8B。
- 32B 只做离线 judge 或数据生成。
- GRPO 只跑 100-300 step，用于展示闭环和趋势。

## 6. 实验设计

### 6.1 Claim Map

| Claim | 需要证明什么 | 对应实验 |
|---|---|---|
| C1: 成熟 benchmark 上可稳定提升座舱工具 Agent | SFT/DPO/GRPO 能提高 CAR-bench 和 BFCL 指标 | Prompt vs SFT vs DPO vs GRPO |
| C2: 过程奖励能降低工具幻觉和能力边界错误 | PRM-Lite 比 outcome-only GRPO 更少 hallucination | GRPO outcome-only vs GRPO + PRM-Lite |
| C3: 低资源配置可完成可复现闭环 | 2×A100 能完成训练和评测 | 训练日志、显存、吞吐、延迟报告 |

### 6.2 Must-run 实验

| ID | 实验 | Benchmark | 指标 | 优先级 |
|---|---|---|---|---|
| E0 | Prompt baseline | CAR-bench + BFCL V4 subset | success, tool accuracy, hallucination | MUST |
| E1 | LoRA SFT | CAR-bench + BFCL V4 subset | 相对 E0 的提升 | MUST |
| E2 | DPO/RLAIF | CAR-bench | limit-awareness, disambiguation, hallucination | MUST |
| E3 | GRPO outcome-only | CAR-bench train/dev subset | success, reward curve, failure modes | MUST |
| E4 | GRPO + PRM-Lite | CAR-bench train/dev subset | hallucination drop, success gain | MUST |
| E5 | vLLM latency eval | held-out eval subset | latency P50/P95 | MUST |

### 6.3 Nice-to-have 实验

| ID | 实验 | 价值 |
|---|---|---|
| A1 | MultiWOZ 2.4 DST eval | 补充 DST 能力证明 |
| A2 | tau2/3-bench eval | 证明多轮工具 Agent 泛化 |
| A3 | LATA 消融 | 对齐参考项目，展示长链路 credit assignment |
| A4 | 4B 蒸馏/量化 | 对齐车端部署 |
| A5 | 人工抽检 100 条失败样本 | 增强可信度 |

## 7. 指标体系

### 7.1 主指标

| 指标 | 对应 JD |
|---|---|
| task success / pass rate | 决策成功率、任务执行成功率 |
| consistent pass rate | 多次采样稳定性 |
| tool call accuracy | Agentic Tool-Use |
| executable tool rate | 工程可执行性 |
| hallucination rate | 工具幻觉 |
| limit-awareness accuracy | 能力边界和安全 |
| disambiguation success | 模糊指令处理 |
| state consistency | 多轮状态跟踪 |
| avg turns / tool calls | 响应流畅度 |
| latency P50/P95 | 工程部署 |

### 7.2 失败分类

每次 eval 都要输出 failure taxonomy:

- F1: 工具名错误
- F2: 参数 schema 错误
- F3: 状态跟踪错误
- F4: 模糊请求未澄清
- F5: 缺失能力时幻觉执行
- F6: 安全边界错误
- F7: 多轮规划顺序错误
- F8: 回复冗长或循环

这部分是项目展示价值的关键，和参考项目的 failure diagnosis 对齐。

## 8. 里程碑计划

### Week 1: Benchmark 和 baseline

目标:

- 跑通 CAR-bench。
- 跑通 BFCL V4 子集。
- Qwen2.5-7B prompt baseline。
- 自动保存失败轨迹。

产出:

- `reports/baseline_carbench.md`
- `reports/baseline_bfcl.md`
- `failure_cases/baseline/*.json`

### Week 2: SFT + DPO

目标:

- 构造 SFT 数据和 DPO preference pairs。
- 跑 LoRA SFT。
- 跑 DPO/RLAIF。
- 做 E0/E1/E2 对比。

产出:

- `checkpoints/sft_lora`
- `checkpoints/dpo_lora`
- `reports/sft_dpo_ablation.md`

### Week 3: 小规模 GRPO

目标:

- 基于 veRL 跑 outcome-only GRPO。
- 加入 PRM-Lite reward。
- 做 E3/E4 对比。

产出:

- `checkpoints/grpo_outcome`
- `checkpoints/grpo_prm_lite`
- `reports/grpo_ablation.md`

### Week 4: 整理和部署 demo

目标:

- 统一 eval 所有模型。
- 生成失败分析。
- vLLM 部署最佳 checkpoint。
- 测 latency。
- 输出项目 README 和图表。

产出:

- `reports/final_report.md`
- `reports/latency_report.md`
- `README.md`
- `demo/`

如果周期只有 2 周:

- 保留 Week 1 + Week 2。
- Week 3 的 GRPO 做最小 50-100 step sanity。
- 不做 LATA、不做蒸馏、不做 MultiWOZ。

## 9. 推荐 repo 结构

```text
cabina-agent-rl/
  configs/
    model/
    train/
    eval/
  scripts/
    run_carbench_eval.sh
    run_bfcl_eval.sh
    build_sft_data.py
    build_preference_data.py
    train_sft.sh
    train_dpo.sh
    train_grpo.sh
    serve_vllm.sh
  src/
    rewards/
      prm_lite.py
    data/
      sft_dataset.py
      preference_dataset.py
    eval/
      metrics.py
      failure_taxonomy.py
    training/
      grpo_reward_fn.py
  reports/
    baseline_carbench.md
    sft_dpo_ablation.md
    grpo_ablation.md
    final_report.md
  failure_cases/
  checkpoints/
  demo/
```

## 10. 最小可交付版本

短周期最小可交付不需要完整 RL 大实验，只要闭环完整:

1. 使用 CAR-bench 和 BFCL V4，不自建 benchmark。
2. Qwen2.5-7B prompt baseline。
3. LoRA SFT。
4. DPO/RLAIF。
5. 小规模 GRPO + PRM-Lite，哪怕只跑 50-100 step。
6. 统一离线评测表。
7. 失败案例分析。
8. vLLM 部署和 latency 报告。

最终表格建议:

| Model | CAR success | CAR consistent pass | BFCL tool acc | hallucination rate | avg turns | latency P95 |
|---|---:|---:|---:|---:|---:|---:|
| Prompt | - | - | - | - | - | - |
| SFT | - | - | - | - | - | - |
| SFT + DPO | - | - | - | - | - | - |
| SFT + DPO + GRPO | - | - | - | - | - | - |
| SFT + DPO + GRPO + PRM-Lite | - | - | - | - | - | - |

不要预填结果，实际跑完再写。

## 11. 参考资料

- CAR-bench GitHub: https://github.com/CAR-bench/car-bench
- CAR-bench Challenge: https://car-bench.github.io/car-bench/
- CAR-bench paper: https://arxiv.org/html/2601.22027v1
- BFCL V4 leaderboard: https://gorilla.cs.berkeley.edu/leaderboard.html
- BFCL V4 web search / score composition: https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html
- BFCL V3 multi-turn function calling: https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html
- tau-bench GitHub: https://github.com/sierra-research/tau-bench
- tau2-bench GitHub: https://github.com/sierra-research/tau2-bench
- tau-bench paper: https://arxiv.org/abs/2406.12045
- MultiWOZ 2.4 paper: https://aclanthology.org/2022.sigdial-1.34/
- Qwen3 official blog: https://qwenlm.github.io/blog/qwen3/
- Qwen function calling docs: https://qwen.readthedocs.io/en/latest/framework/function_call.html
- vLLM tool calling docs: https://docs.vllm.ai/en/latest/features/tool_calling/
