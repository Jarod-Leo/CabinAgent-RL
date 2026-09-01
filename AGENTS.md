# AGENTS.md

本文件用于指导在本仓库中工作的编码代理。

## 项目目标

CabinAgent-RL 面向智能座舱工具调用 Agent，路线包括 Prompt Baseline、Direct-RL Rollout Gate、五组 GRPO 消融、PRM-Lite、统一评测与 vLLM 部署。项目特定 SFT 仅作为 gate 失败后的兜底。开始工作前先阅读 `Project.md`，当前状态以 `Progress.md` 为准。

## 开发约定

- 使用 Python，优先沿用 `src/` 中已有接口和数据结构，避免无关重构。
- 配置放入 `configs/`，命令入口放入 `scripts/`，核心逻辑放入 `src/`。
- 原始数据、派生数据、报告和失败样例分别写入 `data/`、`reports/`、`failure_cases/`，不要提交大型模型权重。
- 保持 CPU 本地基线轻量；GPU 训练依赖仅在目标 CUDA 环境中安装。
- 样例 CAR-bench/BFCL 数据只用于冒烟测试，不得表述为官方评测结果。
- 完成一个阶段后，在 `Progress.md` 记录日期、改动、验证结果和遗留问题。

## 实验阶段文档加载与记录规则

实施任何实验、实验修复、参数调整、重跑、评测或结果分析前，必须按顺序加载并核对：

1. `Project.md`：确认总路线、冻结条件和当前门槛。
2. `Progress.md`：确认最新实际状态、失败记录和未完成事项。
3. `docs/实验阶段/实验阶段总览.md`：确认实验所属阶段及上下游依赖。
4. 下表映射的对应阶段文档：确认该阶段设置、已执行 attempt、当前阻塞和下一动作。

在完成上述加载前，不得修改该实验的代码/配置，不得提交或重跑 Slurm 作业。若一个操作跨越多个阶段，必须加载所有涉及阶段的文档；结果写入实际产出所属阶段，并在其他阶段增加交叉引用。

### 阶段映射 Map

| 阶段 | 必须加载的文档 | 实验/任务映射 |
|---|---|---|
| 01 基础与数据准备 | `docs/实验阶段/01-基础与数据准备阶段.md` | 本地 baseline、sample smoke、CAR/BFCL 下载与完整性验证、数据目录和 SSD 迁移 |
| 02 集群运行时与双模型部署 | `docs/实验阶段/02-集群运行时与双模型部署阶段.md` | Conda/CUDA/veRL/vLLM 环境、7B/72B-AWQ 权重、CAR parquet、simulator/policy 服务、单卡或双节点基础设施 smoke |
| 03 Direct-RL 门禁 | `docs/实验阶段/03-Direct-RL门禁阶段.md` | G00、G01、G02 及其重试/复盘；Direct-Instruct E10-E14 的解锁或阻塞结论 |
| 04 Minimal-SFT 回退 | `docs/实验阶段/04-Minimal-SFT回退阶段.md` | fallback 数据构建/tokenizer gate、F00、F01、G03 及其修复和重试 |
| 05 GRPO 消融训练 | `docs/实验阶段/05-GRPO消融训练阶段.md` | 双节点 veRL trainer smoke、E10-E14、F10-F14、checkpoint/resume 和训练期消融比较 |
| 06 统一评测与报告 | `docs/实验阶段/06-统一评测与报告阶段.md` | E00 正式模型 baseline、CAR dev/test、BFCL、checkpoint 选择、最终表格、失败分析和系统效率报告 |

无法按表确定所属阶段时，先加载总览并更新本 Map；若需要新增阶段，必须先创建 `docs/实验阶段/NN-名称阶段.md`，再开始实施。

### 每次实验完成后的强制记录

每个 attempt 无论成功、失败、中断或被判无效，都必须在继续下一次提交前更新对应的 `xxx阶段.md`。不得覆盖旧 attempt，必须按时间追加，并至少包含以下四个独立小节：

```markdown
### <实验 ID / attempt / Slurm Job>

#### 实验设置
- 代码/配置版本、模型与 adapter、数据 split、seed、关键超参数、GPU/节点、输出目录。

#### 执行结果
- Slurm 最终状态、运行时间、实际 GPU 数、核心指标、产物路径；失败时记录错误和最后有效进度。

#### 改进原因
- 用日志、指标或对照结果说明为什么需要改进；区分代码缺陷、数据缺陷、配置问题、资源问题和科学假设失败。

#### 改进措施
- 写明实际修改、保持冻结的条件、验证方法、下一 attempt 的通过标准；无改进时明确写“无”及原因。
```

完成记录后还必须同步：

- `Progress.md`：追加本次过程、验证和遗留问题。
- `Project.md`：更新当前阶段、门槛或路线变化；纯历史细节不重复堆入主文档。
- `refine-logs/EXPERIMENT_TRACKER.md`：每个 attempt 单独一行，记录状态、Job ID、结果和产物，不得删除失败记录。
- `docs/实验阶段/实验阶段总览.md`：仅在阶段状态、关键结论、GPU 摘要或下一执行顺序变化时更新。

## 常用验证

```bash
python -B -m src.eval.run_baseline --benchmark all
python -B scripts/build_prm_lite_data.py --input data/eval_cache/all_trajectories.jsonl
python -B -m unittest discover -s tests -v
python -m compileall src scripts
```

提交改动前至少运行与改动范围对应的命令，并检查 `reports/`、`failure_cases/` 和生成的 JSONL 是否合理。若因缺少 GPU、模型或官方数据无法验证，应在 `Progress.md` 明确说明。

## NTU EEE GPU 集群实验规则
连接命令: `ssh <cluster-user>@<cluster-host>`；认证凭据必须通过安全渠道提供，不得写入仓库。使用的是 Slurm 管理系统。
本文件适用于所有集群操作；遇到例外、故障或不确定事项时，必须先查阅"集群详细使用说明.md"和其中链接的官方文档，记住只有在不确定时才查手册，**不需要**每次都查询。
1. 每次集群会话开始前重新核对官方规则；GPU、QoS、账户和存储限额必须用 `sacctmgr`、`scontrol`、`sinfo`、`sshare`、`storagemgr` 实时确认。
2. 登录节点仅用于编辑、轻量配置、提交和监控；训练、推理、预处理、评测、编译、下载及其他重负载必须通过 Slurm。
3. AI 代理、索引器和文件监视器应在本地运行；登录节点有 16 GB 用户内存上限，断开连接还会清理 `tmux`、`nohup` 等进程。
4. 正式实验优先 `sbatch`；`srun`/`salloc` 仅用于不超过 2 小时、1 块 GPU 的短时调试，批处理任务最长 3 天。
5. GPU 任务必须指定型号或合法约束；不要同时设置 `--mem` 或 `--cpus-per-task`，且单个任务不能跨节点拼接 GPU。
6. 多卡、长任务和参数扫描前必须先通过短时间单卡冒烟测试，只申请实验确实需要的资源并及时释放空闲任务。
7. 数据集、Conda 环境、权重、检查点、缓存、日志和结果必须放项目 SSD；`/home` 只放代码配置，`/tmp` 只作小型临时空间，HDD 只存冷数据。
8. 软件使用 Lmod，Python 依赖安装在命名的 Conda 环境；禁止 `sudo`、修改系统包或向只读基础环境安装软件。
9. 长任务必须定期保存检查点并支持自动恢复；使用 `override-limits-but-killable` 前必须确认账户有权限且用户接受抢占。
10. 提交前记录代码版本、配置、随机种子、环境版本和输出位置；日志路径应包含 Slurm 任务编号且不得覆盖既有结果。
11. Slurm 不会快照代码；任务排队后不得修改其脚本或代码，需要变更时先 `scancel`，建立新版本后重新提交。
12. 使用 `squeue --me`、`scontrol show job <任务编号>` 和 `sacct -j <任务编号>` 监控；失败、废弃或空闲任务应及时取消。
13. 密码、密钥和令牌不得进入仓库、脚本、参数、历史或日志；禁止访问他人目录、共享账户或使用 `chmod -R 777`。
14. 删除数据、批量取消任务、递归改权限或重建环境前，必须只读核对准确目标并取得用户明确同意。
15. 可以通过sinfo -N -O "NodeList:18,StateComplete:10,CPUsState:15,Gres:25,GresUsed:25"命令查看集群中的卡的情况包括型号和使用情况。
16. 本项目所有双 GPU 作业必须使用同一物理节点：`--nodes=1 --gres=gpu:pro6000:2`。Simulator 与 policy/trainer 必须在同一个 `srun` step 中作为两个 task 运行，使用 `--ntasks=2 --gpus-per-task=1 --gpu-bind=single:1` 各绑定 1 GPU，并分别保存服务、训练/rollout、资源分配和 Slurm stdout/stderr 日志；禁止恢复跨节点双卡或两个 independent-exclusive-step 拓扑。

## 契约
- 在执行任务之前，先对我的需求进行彻底的澄清访谈，直到我们对目标、范围、约束、取舍、边界情况和成功标准形成共同理解。沿决策树逐层解决问题，每次只问一个问题，并为每个问题提供你的推荐答案和理由。
- 能通过检查代码库、文件、配置、日志或已有上下文确认的事实，请自行调查；但涉及意图和取舍的决策必须等待我确认。主动指出隐含假设、冲突、依赖关系及失败场景。
- 在我明确确认已经达成共识之前，不要开始实施。最后总结所有已确认决策、未决问题与验收标准。
- 每次修改完代码都要同步在Progress.md和Project.md进行更新
- 每次实施实验前必须按“实验阶段文档加载与记录规则”加载对应阶段文档；每次实验完成或改进后，必须在对应 `xxx阶段.md` 中完整记录实验设置、执行结果、改进原因和改进措施。
