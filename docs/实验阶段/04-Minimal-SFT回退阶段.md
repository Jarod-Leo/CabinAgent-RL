# 04 Minimal-SFT 回退阶段

## 状态

进行中。F01 的训练作业已完成，但 G03 证明其工具调用参数格式无效；旧 adapter 不再允许作为后续 policy 初始化。

## 目标

- 只修复基础 CAR 交互和工具调用格式，不把 SFT 重新定义为主实验前置。
- 使用修复后的 adapter 重跑冻结的 20x4 gate，验证能否产生可学习的组内 outcome 方差。

## 固定设置

| 项目 | 设置 |
|---|---|
| 数据来源 | G00/G01/G02 中 environment-success 的完整多轮 trajectory |
| 数据切分 | 按 task 互斥；38 train / 14 validation |
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| LoRA | rank `16`，alpha `32` |
| Loss | assistant-token-only cross entropy |
| Epoch / LR | 1 epoch，`2e-4` |
| Batch | micro batch 1，gradient accumulation 4 |
| Max length | `32768` |
| Seed | `42` |
| G03 | 与 G02 相同的 20 task x 4 rollout、采样、reward 和全部阈值 |

## 执行结果

| Run | Job / GPU | 结果 |
|---|---|---|
| CPU data attempt 1 | `131984`, 0 GPU | assistant mask 为空，失败 |
| CPU data attempt 2 | `131992`, 0 GPU | `BatchEncoding` 兼容问题，失败 |
| CPU tokenizer gate | `131999`, 0 GPU | 38/14 全部可训练，最大 4309 tokens |
| F00 | `132008`, 1 Pro 6000 | 2 steps；train/eval loss `3.608993/1.909820`；通过 |
| F01 attempt 1 | `132013`, 1 Pro 6000 | 误继承 4 records/2 steps；实验无效 |
| F01 attempt 2 | `132020`, 1 Pro 6000 | 38 train、14 val、10 steps；loss `0.902976/0.584203`；Slurm 完成 |
| G03 attempt 1 | `132043`, 2 Pro 6000 | `FAILED`，运行 `6:10`，无 gate report |

G03 实际使用 `gpu-pro6000-1` 和 `gpu-pro6000-7`。7B vLLM 已成功加载 F01 LoRA，说明失败发生在真实 rollout/环境评估阶段，而不是 adapter 加载阶段。

## G03 失败原因

训练数据保留了 OpenAI 格式的 `tool_calls[].function.arguments` JSON 字符串。Qwen chat template 对其再次 `tojson`，使模型学习到双重编码的参数。G03 中 CAR runtime 因而解析出空 kwargs 或字符串 arguments；最终官方自动规则 evaluator 在访问 `arguments["on"]` 时触发：

```text
TypeError: string indices must be integers, not 'str'
```

该问题同时解释了日志中的大量 missing positional arguments。F01 的低 loss 只表示模型拟合了错误序列化目标，不能证明工具调用可执行。

## 已完成改进

- 数据按 task 去重和互斥切分，避免 train/validation 泄漏。
- assistant mask 改为 Qwen token boundary，并兼容 Transformers 5.10 `BatchEncoding`。
- F00/F01 隔离 attempt 目录；full run 显式覆盖 smoke 环境变量。
- G03 使用 adapter-aware vLLM 并固定 `POST_GATE_ACTION=none`，失败或 PASS 都不会误启动 E10。

## 必须完成的改进

1. 在构建 SFT 消息时将 `function.arguments` 解析成字典，拒绝双重编码字符串。
2. 增加 round-trip 测试：消息 -> Qwen chat template -> tool parser 后，arguments 必须仍是对象且参数完整。
3. 重新生成数据并重跑 CPU tokenizer gate、F00、F01；旧 F01 adapter 仅保留审计。
4. 使用新 adapter 提交 G03 attempt 2；只有 machine-readable gate report 全部 PASS 才解锁 F10-F14。

## 主要产物

- `configs/train/sft_fallback_lora.yaml`
- `scripts/build_sft_fallback_data.py`
- `scripts/train_sft_fallback.py`
- 集群 `experiments/sft_fallback_full_20260830T141431Z_r1`
- 集群 `logs/slurm/car-g03-132043.err`

### 2026-09-01 arguments 修复与 G03 attempt 2 准备

#### 实验设置

- 数据来源、38/14 task-disjoint split、LoRA rank/alpha `16/32`、1 epoch、seed 42 和 G03 全部阈值保持冻结。
- 数据准备在 CPU Slurm job 中完成，并在申请 F00 GPU 前使用真实 Qwen tokenizer 执行 tool-call round-trip gate。
- F00/F01 各使用 1 块 Pro 6000；G03 使用同一物理节点上的 2 块 Pro 6000。

#### 执行结果

- `function.arguments` 规范化、双重编码拒绝和模板 round-trip 单元测试已实现。
- 本地 26 项单元测试、Python compilation 和全部 YAML 解析通过。
- 本条记录时尚未提交新 Slurm attempt；远端预检和 Job ID 将在提交后追加。

#### 改进原因

- G03 `132043` 证明旧 F01 学习了字符串化 arguments，低训练 loss 不能代表工具调用格式正确。
- 旧 G03 使用跨节点双 GPU，与当前集群单任务同节点规则冲突。

#### 改进措施

- SFT builder 深拷贝消息，将合法 JSON arguments 解析为字典，拒绝无效、非对象或双重编码值。
- CPU tokenizer gate 将 Qwen 模板渲染出的 `<tool_call>` 再解析，并与原始 name/arguments 逐项比较。
- 数据任务保存 pipeline-specific 数据副本；F00 -> F01 -> G03 使用新的 attempt 目录自动串行提交。
- G03 分别保存 allocation、simulator、policy、rollout、gate-check 和 Slurm 日志，且固定 `POST_GATE_ACTION=none`。
- G03 attempt 2 必须依赖同节点双 GPU 绑定 smoke 成功；若 smoke 失败，G03 保持 dependency 阻塞且不加载模型。

### Corrected fallback / Slurm 132934, 132935, 132942

#### 实验设置

- 保持 38/14 split、LoRA rank/alpha `16/32`、1 epoch、seed 42；先 CPU tokenizer/round-trip gate，再 F00 两步 smoke 和完整 F01。

#### 执行结果

- CPU `132934` 完成：train/val 分别 38/14 examples，round-trip tool calls 45/12，零 overlength、零 missing target。
- F00 `132935` 在单块 Pro 6000 上完成；F01 `132942` 在单块 Pro 6000 上完成并保存 `experiments/sft_fallback_full_20260901_stage16/checkpoints/final_adapter`。
- 自动提交的旧 G03 `132946` 依赖失败 smoke `132933`，当前为 `DependencyNeverSatisfied`，不得作为有效 attempt 运行。

#### 改进原因

- 数据格式修复已通过真实 tokenizer gate，需要用新 adapter 重跑 G03；同时旧双独立-step 启动模型已被 132950 证明会串行化。

#### 改进措施

- 新建单一 srun/双 task 的 G03 脚本，不修改待定 132946 引用的旧脚本。
- 新 G03 必须依赖单一-step 双 task smoke PASS，并继续固定 `POST_GATE_ACTION=none`。

### G03 attempt 3 / Slurm 132967

#### 实验设置

- Policy adapter：`experiments/sft_fallback_full_20260901_stage16/checkpoints/final_adapter`。
- 保持冻结的 20 task x 4 rollout、policy/simulator sampling、outcome reward 和全部 gate 阈值。
- 资源：`--nodes=1 --gres=gpu:pro6000:2`；一个 srun step 内 task 0/1 各绑定 1 GPU；依赖已通过的 smoke `132966`。

#### 执行结果

- Job `132967` 在单一物理节点 `gpu-pro6000-10` 运行 `5:40`，Slurm 显示 `NumNodes=1`、`NumTasks=2`、`gres/gpu:pro6000=2`；两个 vLLM health 均通过并完成 80 条 trajectory。
- Gate 结果为 FAIL：parse `0.999375` 通过；executable `0.844693 < 0.85`、mixed group `0.15 < 0.20`、loop/max-turn `0.2625 > 0.20`；success `14/80`、initial-user consistency `1.0`。
- 20 个 group 中 15 个 all-fail、2 个 all-success、3 个 mixed。失败标签为 argument error `32`、verbose/loop `21`、safety boundary `16`、capability hallucination `7`、tool name `1`。
- `hallucination_missing_tool_parameter` 的 12 条 trajectory 全部失败，平均 executable 约 `0.42`；它是参数错误和循环的主要高风险任务族。
- 日志：`logs/slurm/car-g03-v2-132967.{out,err}` 及 `experiments/G03/logs/{allocation,role-0,role-1,simulator,policy,rollout,gate-check}-132967.*`。

#### 改进原因

- Arguments 双重编码和同节点双卡启动问题已经排除；剩余失败是模型对缺失工具参数/能力边界的处理不足，以及重复工具调用导致的 max-turn。
- 当前 successful trajectory 与 mixed group 都存在，但仍不足以支持稳定的 fallback GRPO 初始化。

#### 改进措施

- 不放宽 gate 阈值、不启动 F10-F14、不无改动重复 G03。
- 下一候选改进是独立 F02：针对 `hallucination_missing_tool_parameter`、argument error 和 loop 构建边界拒绝/澄清与停止重复调用的 corrective SFT 数据；保持模型、split、gate tasks、sampling 和 reward 不变，再由 G04 验证。
- F02 数据来源和构造方式属于新的研究取舍，实施前需确认；本次不自动提交。
- 保留旧 `132946` 为无效依赖 attempt；在获得明确取消授权前不执行 `scancel`。

### 无效队列清理 / Slurm 132946

#### 实验设置

- 只取消已确认依赖永久失败、引用旧 launcher 的 G03 attempt 2；不批量取消其他任务。

#### 执行结果

- 用户明确授权后执行 `scancel 132946`；`sacct` 状态为 `CANCELLED`，随后 `squeue --me` 为空。

#### 改进原因

- `132946` 依赖失败的 `132933`，状态为 `DependencyNeverSatisfied`，不可能形成有效实验且占用提交队列槽位。

#### 改进措施

- 后续 gate 统一由 `scripts/slurm_direct_rl_gate_same_node.sbatch` 提交；不再引用旧的双 independent-step launcher。

### F02 corrective-SFT 与 G04 实施准备

#### 实验设置

- F02 从相同 `Qwen2.5-7B-Instruct` 基座初始化 LoRA，保持 rank/alpha `16/32`、1 epoch、学习率 `2e-4`、seed 42 和 task-disjoint 80/20 split。
- 数据为 G00-G03 environment-success 轨迹，加上 G03 七个 missing-tool family task 的两类纠正记录：首次直接能力边界拒绝，以及用户施压后不重试。
- G04 仅替换 policy adapter；20 task x 4 rollout、首轮一致性、policy/simulator sampling、outcome reward 和全部 gate 阈值与 G03 相同。
- F02 使用 1 块 Pro 6000；G04 使用同一物理节点 2 块 Pro 6000，一个 `srun` step 内两个 task 各绑定 1 GPU。

#### 执行结果

- 已实现 corrective builder、独立配置、CPU tokenizer 任务、F02 smoke/full 自动链和 G04 同节点提交路径。
- 本地 28 项单元测试、Python compileall 和全部 YAML 解析通过；尚未提交新的 Slurm Job，真实 tokenizer 结果和 Job ID 待追加。

#### 改进原因

- G03 的 arguments 序列化与运行拓扑已排除；剩余主要问题是缺失工具参数导致的 32 次 argument error，以及 missing-tool family 中的重复调用和 max-turn。
- 当前 encoder 会监督记录中的所有 assistant span，因此 corrective 数据不能保留失败 tool call；否则会把错误动作重新训练进去。

#### 改进措施

- 每条 corrective record 从原 system 与首条 user 重建，不包含失败 tool call；同一 task 的直接拒绝和 no-retry 记录固定进入同一 split。
- 先执行 CPU 数据/tokenizer gate，再执行 F02 两步 smoke 和完整 1 epoch；任一步失败即停止自动链。
- F02 完成后自动提交 G04，但 G04 固定 `POST_GATE_ACTION=none`；只有人工核验 machine-readable PASS 后才能解锁 F10-F14。

### F02 数据/tokenizer gate / Slurm 133301

#### 实验设置

- 输入 G00-G03 共 320 条 gate trajectory；成功轨迹去重后与 G03 missing-tool family corrective records 合并，按 task、seed 42 做 80/20 split。
- CPU 作业使用真实 Qwen2.5-7B tokenizer，最大长度保持 32768，并执行已有 tool-call object round-trip gate。

#### 执行结果

- Job `133301` 在 `cpu-1` 运行 `00:00:54`，状态 `COMPLETED`、exit `0:0`。
- 得到 66 条成功记录和 14 条纠正记录，split 为 60 train / 20 val、9/2 个 task、task overlap 0。
- train token 为 3226-10468，val 为 3221-3297；零 overlength、零 missing target；train/val 分别 round-trip 80/16 个 tool call。
- 日志：`logs/slurm/car-f02-data-133301.{out,err}`、`logs/stages/f02-data-133301.{out,err}`。

#### 改进原因

- F02 在申请 GPU 前必须证明 synthetic no-tool records 与原多轮工具轨迹可被同一 Qwen 模板稳定编码，且不会破坏原工具格式。

#### 改进措施

- 数据/tokenizer gate 已满足，无需修改；允许进入两步 F02 smoke。

### F02 smoke / Slurm 133303

#### 实验设置

- 相同 F02 配置，但仅取前 4 条 train record、运行 2 optimizer step；1 块 Pro 6000，完整 20 条 val 用于兼容性检查。

#### 执行结果

- Job `133303` 在 `gpu-pro6000-8` 运行 `00:00:49`，状态 `COMPLETED`、exit `0:0`。
- train/eval loss 为 `3.312624/2.140017`，峰值显存 `21.16 GiB`；4/20 个 train/val example 均未跳过并保存 final adapter。
- 日志：`logs/slurm/car-f02-smoke-133303.{out,err}` 和 `experiments/sft_corrective_smoke_f02_20260901_stage17/logs/`。

#### 改进原因

- Corrective 数据含新的 no-tool 多轮形态，需先排除 trainer、collator、显存和保存路径故障。

#### 改进措施

- 两步训练无 NaN/OOM，adapter 与 manifest 完整；无需修改，允许完整 F02。

### F02 full / Slurm 133306

#### 实验设置

- 从相同 Qwen2.5-7B 基座初始化 LoRA rank/alpha `16/32`；使用全部 60 train / 20 val，1 epoch、15 optimizer steps、学习率 `2e-4`、seed 42。
- 单块 Pro 6000；不从 F01 或 smoke adapter 继续训练。

#### 执行结果

- Job `133306` 在 `gpu-pro6000-8` 运行 `00:01:08`，状态 `COMPLETED`、exit `0:0`。
- train/eval loss 为 `1.088965/0.739128`，峰值显存 `34.304 GiB`；60/20 records 全部产生训练样本，零跳过。
- adapter：`experiments/sft_corrective_full_f02_20260901_stage17/checkpoints/final_adapter`；日志：`logs/slurm/car-f02-133306.{out,err}` 及 run 内 allocation/trainer 日志。

#### 改进原因

- G03 对 missing-tool parameter、能力边界和停止重复调用的行为不足，需要在不改变 gate 协议的前提下产生新 adapter。

#### 改进措施

- F02 已按冻结配置完成；不根据训练 loss 宣称门禁改善，交由 G04 的真实 20x4 rollout 判定。

### G04 / Slurm 133308

#### 实验设置

- Policy 使用 F02 full adapter；72B-AWQ simulator、20 task x 4 rollout、seed/sampling、outcome reward 和门槛与 G03 完全相同。
- Slurm：`NumNodes=1`、`NumTasks=2`、`gres/gpu:pro6000=2`；一个 `srun` step 内 simulator/policy 各绑定 1 GPU。

#### 执行结果

- Job `133308` 在单一物理节点 `gpu-pro6000-7` 使用 2 块 Pro 6000、2 tasks、8 CPU、180 GiB node memory，运行 `00:05:26` 并完成 80 条 trajectory。
- Slurm 最终状态为 `FAILED`、exit `15:0`，原因是 machine-readable gate checker 对科学门禁 FAIL 返回非零；simulator task 随后按设计收到 TERM，非基础设施崩溃。
- Gate：parse `0.995833` PASS，initial-user consistency `1.0` PASS，success `12/80` PASS；executable `0.846829 < 0.85`、mixed group `0.10 < 0.20`、loop/max-turn `0.275 > 0.20` FAIL。
- 20 个 group 为 16 all-fail、2 all-success、2 mixed。失败标签：argument error 33、verbose/loop 22、safety boundary 13、capability hallucination 8、tool name 1。
- 相比 G03，missing-tool executable 从 `0.8175` 升到 `0.8673`，但 missing-tool-parameter 从 `0.4161` 降到 `0.2681`；总 executable 仅增加 `0.002136`，mixed 减少 `0.05`，loop 增加 `0.0125`，success 减少 2。
- 日志：`logs/slurm/car-g04-133308.{out,err}`、`experiments/G04/logs/{allocation,role-0,role-1,simulator,policy,rollout,gate-check}-133308.*`；报告：`reports/direct_rl_gate_G04_133308.json`。

#### 改进原因

- G04 用来测量 F02 是否同时修复 executable 与 loop，并保留至少 20% mixed group；不得用训练 loss 替代该验证。

#### 改进措施

- 保持 F10-F14 阻塞，不放宽阈值，不无改动重跑 G04，也不凭训练 loss 宣称 F02 有效。
- F02 对 missing-tool 有局部收益，但对 missing-tool-parameter 和 mixed/loop 有负迁移；下一 attempt 必须改变纠正目标或采样权重并建立对照，而不能简单增加 epoch。
- 候选 F03 需要先审计 14 条 synthetic records 在 train/val 的分布，并对 missing-tool-parameter 使用 task-specific 能力边界目标与显式 loss weighting/oversampling；这属于新的研究取舍，实施前需用户确认。

### 阶段收束与 F10 交叉引用（非实验 attempt）

- G03 与 G04 的 machine-readable FAIL 均保持有效，不修改阈值、不重标 PASS。F02/G04 被归档为有局部收益但总体负迁移的 corrective-SFT 消融。
- 用户决定暂停 F03/G05，改为用真实 optimizer 直接测量“小于 gate 阈值是否仍能产生可学习信号”。该决定只授权 Stage 05 的有界 F10 pilot，不解锁正式消融。
- F10 不使用 F02 adapter。corrected F01 job `132942` 的 rank-16 adapter 将先合并进相同 Qwen2.5-7B 基座并冻结为新父模型；各 RL 分支再从该父模型独立创建 fresh rank-32 LoRA。
- 后续结果、GPU 调参与 checkpoint/resume 记录全部写入 Stage 05；本阶段不再自动提交 F03/G05。
