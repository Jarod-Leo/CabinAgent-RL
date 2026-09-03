# 05 GRPO 消融训练阶段

## 状态

进行中：F10 pilot、step-6 resume 与正式 step-50 segment 均已 PASS；Job `137588` 正在从唯一的 `global_step_50` checkpoint 恢复训练到 step 100。

## 目标

在同一初始化、数据、simulator、seed 和评测协议下，比较五组 agentic GRPO 改进。

## 实验家族

- E10-E14：Direct-Instruct 初始化。G02 已给出合法 gate FAIL，因此该家族保持阻塞并作为失败结论保留。
- F10-F14：corrected F01 初始化的独立 fallback 家族。G03/G04 的合法 FAIL 结论保留；经人工批准，仅先运行 F10 5-step 诊断 pilot，不将其解释为 gate PASS，也不能与 E10-E14 混为同一家族。

## 固定设置

| ID | 实验 | Reward | Advantage |
|---|---|---|---|
| 10 | Vanilla GRPO | outcome | 标准 group-normalized GRPO |
| 11 | Turn-Discount | outcome | `alpha^(L-1-t)`，均值归一为 1 |
| 12 | LATA | outcome | Turn-Discount 后除以 `sqrt(L)` |
| 13 | PRM-Lite | `outcome + 0.3 * process` | 标准 GRPO |
| 14 | PRM-Lite + LATA | `outcome + 0.3 * process` | LATA |

共同训练设置：

| 项目 | 设置 |
|---|---|
| Framework | veRL 0.9，online multi-turn GRPO |
| Group | 4 task x 4 rollout / step |
| Policy LoRA | rank/alpha `32/32` |
| Simulator | 固定 72B-AWQ vLLM，不训练 |
| 训练长度 | 最多 250 steps |
| Checkpoint/eval | 50、100、150、200、250 评测；SSD 仅保留最新 1 个完整可恢复 checkpoint |
| Turn discount alpha | `1.05` |
| PRM-Lite range | clip 到 `[-0.5, 0.5]` |
| Process weight | `0.3` |
| 节点 | 同一台 highmem Pro 6000 物理节点，共 2 GPU；simulator/trainer 各 1 GPU |
| F 家族父模型 | corrected F01 rank-16 adapter safe-merge 到 Qwen2.5-7B 后冻结 |
| F 家族 RL delta | 每个分支 fresh rank/alpha `32/32`，互不继承 |

## 执行结果

- E10-E14：未启动；阻塞原因是 G02 mixed outcome group ratio `0.0`。
- F10-F14 正式 run：F10 首个 step-50 segment Job `136868` 已 PASS，step-100 resume Job `137588` 正在运行；F11-F14 未启动且不会自动串联。
- F10 pilot 已产生 `global_step_5` 可恢复 checkpoint；初始/最终 CAR dev mean@1 为 `0.230769/0.269231`，仅作五步集成诊断，不解释为性能提升。

## 已完成改进

- 五组 reward/advantage 配置和项目本地 veRL estimator 已实现。
- 每组独立初始化，不允许实验之间继承 checkpoint。
- 训练总步数固定 250，删除 DPO 和 R05/step-300 计划。
- checkpoint、manifest、失败 attempt 和自动恢复契约已建立。

## F10 pilot 契约

1. corrected F01 adapter 必须先 safe-merge 为不可变父模型，记录 base/adapter 路径、adapter config digest 和全部输出文件哈希；已存在目标不可覆盖。
2. 初始 F10 恰好运行到 optimizer step 5，保存 checkpoint；停止后人工核验，再用独立 Slurm 作业从同一 run 恢复并至少完成 step 6。
3. 至少一个 step 同时具有非零 reward variance、非零 advantage 和有限非零 gradient；KL、clip fraction、grad norm 有限；不得有 NaN、OOM 或 reward-schema 错误。
4. 记录 simulator/trainer 两块 GPU 的显存、利用率、step time 和等待开销。只允许调整 rollout batching、worker 数、offload 等系统参数，保持 group size 4、4 tasks/step、有效 batch、sampling、32K/20-turn 限制、reward/advantage、rank-32 LoRA、optimizer/LR、数据和 simulator 不变。
5. 初始 pilot、恢复和任何正式 run 均不得自动提交 successor。五步内不要求性能提高；pilot 通过后仍须人工决定正式顺序。

### F10 pilot preparation / local

#### 实验设置

- 配置：`configs/train/fallback_ablations/vanilla.yaml`，父模型目标为 `models/derived/Qwen2.5-7B-Instruct-F01-merged-20260901`，GRPO/LoRA/数据/simulator 语义继承冻结 common config。
- 计划资源：父模型 merge 使用 1 块 Pro 6000；F10 使用同一物理节点 2 块 Pro 6000，一个 `srun --ntasks=2 --gpus-per-task=1 --gpu-bind=single:1`。

#### 执行结果

- 尚未提交 Slurm job，optimizer steps 为 0。
- 本地全部 32 项 unit tests 通过；新 launcher 静态验证无 independent `--exclusive` step、无自动 successor，并保存 allocation、role、simulator、trainer、reward audit 与逐 GPU telemetry。

#### 改进原因

- 旧 trainer launcher 使用两个独立 `srun --exclusive`，已被 job `132950` 证明会串行化；旧 config 也不能实现“F01 父策略 + fresh rank-32 RL delta”。
- Gate 阈值只能预估 outcome-only GRPO 风险，不能替代真实 optimizer 梯度测量，因此需要严格限步且人工审查的 pilot。

#### 改进措施

- 新增 PEFT safe-merge、不可变 parent manifest、F10 fallback config、单-step双-task launcher、独立 start/resume 提交器、GPU telemetry 和 reward audit。
- 初始吞吐参数为 vLLM memory utilization `0.60`、16 seq、16384 batched tokens、16 agent workers、microbatch 1 与 actor/ref offload；根据 pilot telemetry 只在人工边界调整，正式分支再统一冻结。
- 下一动作是同步集群、实时核对 Slurm/配额规则并提交 parent merge；merge 验证通过前不得提交 F10。

### Corrected F01 parent merge / Slurm 133431

#### 实验设置

- 代码包 SHA-256：`cc1196ca28aba2cdb57d22116d1de5ca3174b3babe736555cefd181e08de7565`；远端 32 tests、18 YAML、veRL dry-run、Bash syntax 和两个 `sbatch --test-only` 均通过。
- Base：`models/Qwen/Qwen2.5-7B-Instruct`（约 15 GB）；adapter：corrected F01 `experiments/sft_fallback_full_20260901_stage16/checkpoints/final_adapter`（rank/alpha `16/32`，adapter config SHA-256 `0cf0be17ac42687850315d4530701b5e72e51164af97f65b4c04baaf5dd50789`）。
- PEFT `merge_and_unload(safe_merge=True)`、BF16、5 GB safetensor shards；目标 `models/derived/Qwen2.5-7B-Instruct-F01-merged-20260901` 必须事先不存在。
- Slurm：job `133431`，`cluster02` / `msc`，1 node、1 task、1x Pro 6000、highmem、2 小时；输出与 allocation 日志均按 Job ID 保存。

#### 执行结果

- 2026-09-01 04:25 UTC 已提交，当前 `PENDING (Priority)`；Slurm 实际请求为 4 CPU、90 GiB memory、1x Pro 6000。尚未加载模型或产生父模型文件。
- Job 随后于 `04:26:18` 在 `gpu-pro6000-4` 启动，`04:27:22` 结束；Slurm `COMPLETED`、exit `0:0`、elapsed `00:01:04`。
- PEFT safe merge 与四个 safetensor shard 写出成功，目标约 15 GB。`parent_manifest.json` 记录正确的 base/adapter、rank 16、adapter config digest、Slurm job、BF16 merge 方法，以及 10 个模型/tokenizer文件的 size 与 SHA-256。
- Slurm stdout 仅有 Transformers `torch_dtype` deprecated warning，无 stderr、OOM 或 merge error。

#### 改进原因

- 直接把 F01 adapter 作为 veRL `lora_adapter_path` 会继续训练原 rank-16 SFT delta，不能实现“冻结 SFT 父策略 + fresh rank-32 RL delta”，且分支间科学边界不清晰。

#### 改进措施

- Slurm state、parent manifest、文件清单和目标大小已通过；下一步用独立轻量 GPU Slurm 验证全量 manifest 哈希和 model/tokenizer 实际加载。该验证通过前仍不提交 F10。
- 作业排队/运行期间冻结其脚本和 merge 代码，不以新归档覆盖远端执行快照。

### Corrected F01 parent validation attempt 1 / Slurm 133439

#### 实验设置

- 验证 merge job `133431` 生成的父模型；严格重算 manifest 文件集合、size、SHA-256，然后 BF16 加载 model/tokenizer 并执行 one-token generation。
- Slurm：1 node、1 task、1x Pro 6000、highmem、1 小时；job `133439` 在 `gpu-pro6000-4` 获得 4 CPU 与 90 GiB memory。

#### 执行结果

- Job `133439` 于 `04:32:18` 启动，1 秒后 `FAILED`、exit `1:0`；未生成 validation report。
- 错误为 `ModuleNotFoundError: No module named 'scripts.merge_lora_parent'`。验证脚本以文件路径执行时 `sys.path[0]` 为 `scripts/`，项目根目录没有加入 import path。
- 失败发生在读取 manifest、计算 hash、加载模型之前；父模型没有被修改，GPU 计算与显存结果无效。

#### 改进原因

- 这是 standalone Python entrypoint 的代码封装缺陷，不是 parent artifact、依赖、资源或科学假设失败。本地 unit test 以 package import 方式运行，未覆盖 file-path entrypoint。

#### 改进措施

- 在 validation entrypoint 中像其他 standalone builder 一样显式把项目根目录加入 `sys.path`，保留其余验证协议不变。
- 重新执行本地/远端 tests、compile、direct file-path smoke 与 Bash syntax，再以新 Job ID 重试；在 retry PASS 前不提交 F10。

### Corrected F01 parent validation attempt 2 / Slurm 133447

#### 实验设置

- 与 attempt 1 相同 parent、manifest、hash/load/generation 协议和 1x Pro 6000 资源；唯一代码变化是 standalone entrypoint 显式加入项目根目录。
- 本地 34 tests、compile 和 direct file-path `--help` 通过；远端 direct file-path 与 Bash syntax 通过。远端以 `tests.*` 模块名点名测试失败是因为 `tests/` 不是 package，完整 `discover` 已在前一轮通过，不影响生产入口。

#### 执行结果

- Job `133447` 于 `04:35:18` 在 `gpu-pro6000-2` 启动，运行 `00:00:53` 后 `COMPLETED`、exit `0:0`，stderr 为空。
- Machine-readable report `reports/f01_parent_validation_133447.json` 为 `PASS`：10 files、`15,242,726,337` bytes 的文件集合/size/SHA-256 全部匹配。
- BF16 model 实际加载成功：`7,615,616,512` parameters；tokenizer size `151,665`；4-token input 成功生成 1 token。report 正确交叉引用 source merge job `133431`。

#### 改进原因

- Attempt 2 用于证明 attempt 1 的失败只来自入口路径，而不是 parent corruption、Transformers/PEFT 兼容性、GPU 架构或内存问题。

#### 改进措施

- Parent dependency 现已 PASS，可按冻结配置创建新的 F10 run 并提交 5-step pilot。
- 保持 no-successor 边界；F10 完成后先记录和人工验收 optimizer/GPU/checkpoint 指标，不能直接自动提交 resume 或正式 runs。

### F10 Vanilla pilot start / Slurm 133456

#### 实验设置

- Run：`experiments/f10_pilot_20260901_stage18`；config SHA-256 `7fd65bc8b3f7b99866ac500e8bf05d9dbf524068b4e41054c9c05d9718199dc05`；source SHA-256 `c35a937e0fbfca63741185f60d598d3ad83843780a24256e4cb06ea3bc4446935`。
- Policy/reference parent：validated corrected-F01 merged snapshot（merge `133431`、validation `133447`）；actor 创建 fresh rank/alpha `32/32` LoRA。Reward 为 outcome-only GRPO，group `4 tasks x 4 rollouts`，seed 42，LR `1e-6`，32K/20-turn 上限与 CAR train/dev 数据冻结。
- 初始系统参数：rollout GPU memory utilization `0.60`、max seqs `16`、max batched tokens `16384`、agent workers `16`、microbatch `1`、actor/ref offload enabled；这些只影响吞吐，不改变科学语义。
- Slurm：job `133456`，同一物理节点 2x Pro 6000、2 tasks、8 CPU、180 GiB memory、6 小时；单个 `srun --ntasks=2 --gpus-per-task=1 --gpu-bind=single:1`，role 0 simulator、role 1 trainer。`MAX_TRAINING_STEPS=5`、save/eval freq 5，无 successor。

#### 执行结果

- 2026-09-01 04:40 UTC 已提交，当前 `PENDING`；run manifest 为 `submitted`，submission ledger 记录 start/job `133456`/target step 5。尚无 optimizer step 或 GPU telemetry。
- Job `133456` 于 `04:41:18` 在 `gpu-pro6000-7` 启动，运行 `00:02:11` 后 Slurm `FAILED`；trainer task exit `1:0`，simulator task 按 sentinel/step cleanup 收到 TERM，job 聚合 exit `15:0`。
- 单-step双-task绑定成功，两角色看到不同 GPU UUID。72B simulator 正常加载并达到 health；峰值抽样约 `93,335/97,887 MiB`（95.4%），未 OOM。Trainer 在任何 veRL import/model load/rollout/optimizer 之前报 `ModuleNotFoundError: No module named 'src'`。
- Run manifest 正确更新为 `failed`，optimizer steps `0/5`、checkpoint 为空、reward audit 未产生；无科学训练结果。

#### 改进原因

- 这是第一个真实 veRL optimizer pilot，用来直接检验 G03 未过 mixed/loop 阈值时是否仍可在部分 batch 产生非零 outcome advantage 与有效梯度，而不是要求五步性能提升。
- 失败根因是 `scripts/launch_verl.py` 作为 file-path entrypoint 时未加入项目根目录。此前 dry-run 在 import `src` 前提前返回，因此本地/远端 dry-run 未覆盖该缺陷；parent、simulator、双 GPU topology 与资源均非根因。

#### 改进措施

- 保留本 run 和全部日志，不复用失败目录。修复 launcher ROOT `sys.path`，并让 `--dry-run` 也 import 项目 entrypoint，确保 direct-file regression 能覆盖同类错误。
- 重新执行本地/远端 tests、compile、dry-run/import 与 Bash syntax，然后创建新 run / 新 Job ID retry；科学和系统参数保持不变。仅在有效 5-step checkpoint 与指标人工验收后才提交 step-6 resume。

### F10 Vanilla pilot start attempt 2 / Slurm 133478

#### 实验设置

- 新 run `experiments/f10_pilot_20260901_stage18_r1`；validated F01 parent、fresh rank-32 LoRA、outcome GRPO、数据、seed、LR、batch/group、长度/轮数、simulator 和吞吐参数与 attempt 1 相同。
- 唯一修复是 launcher standalone ROOT path，并由本地/远端 34 tests、direct-file import dry-run、compile/Bash syntax 验证。Slurm 同节点 2x Pro 6000，target 5 steps，无 successor。

#### 执行结果

- Job `133478` 于 `04:48:18` 在 `gpu-pro6000-11` 启动，运行 `00:03:56` 后 Slurm `FAILED`；trainer task exit `1:0`，simulator cleanup TERM，聚合 exit `15:0`。
- Launcher 修复有效：veRL 成功 import、Hydra overrides 与 config validation 全通过，随后在 `ray.init()` 失败。错误为 `AF_UNIX path length cannot exceed 107 bytes`，plasma socket 位于长 SSD 路径 `.../cache/ray/ray/session_.../sockets/plasma_store`。
- 72B simulator 正常 health；optimizer steps `0/5`、无 checkpoint/reward audit。Manifest 正确标记 failed；parent 与数据未修改。

#### 改进原因

- 这是 Ray 对 Unix-domain socket 路径长度的运行时约束。项目 SSD 根路径本身较长，把 `RAY_TMPDIR` 设为 `$PROJECT_ROOT/cache/ray` 会必然超限；不是 GPU、显存、veRL config 或科学假设失败。

#### 改进措施

- 仅将短生命周期 Ray socket/session metadata 放在 Job-ID-scoped `/tmp/cabin-ray-$SLURM_JOB_ID`；persistent run/trainer/simulator/GPU/reward/checkpoint 日志仍保存到 SSD，Ray object store 仍使用共享内存。
- 增加静态回归，验证 RAY_TMPDIR 不再指向长项目路径。完成本地/远端 tests 与 Bash syntax 后创建全新 run/Job ID attempt 3；不复用失败 run。

### Ray short-path CPU smoke / Slurm 133503

#### 实验设置

- 在 CPU Slurm compute node 复用正式 GPU env 和 `cluster_runtime_env.sh`，实际执行 `ray.init(num_cpus=1) -> remote identity task -> ray.shutdown`。
- 期望 `RAY_TMPDIR=/tmp/cabin-ray-$SLURM_JOB_ID`，输出 `reports/ray_runtime_smoke_133503.json`；1 node/1 task/1 CPU/11 GiB、10 分钟，无 GPU。

#### 执行结果

- Job `133503` 于 `04:57:49` 在 `cpu-1` 启动，运行 29 秒后 `COMPLETED`、exit `0:0`。
- Report `PASS`：Ray address `11.11.11.165:32899`，tmpdir `/tmp/cabin-ray-133503`，session dir 位于相同短路径，remote result `42`；stderr 仅正常的 local Ray startup info。

#### 改进原因

- 在再次占用两块 Pro 6000 和加载 72B simulator 前，需要用最小资源证明 AF_UNIX 修复真实生效，而不只依赖静态字符串检查。

#### 改进措施

- Ray short-path dependency 已闭合；允许使用全新 run/Job ID 提交 F10 attempt 3，保持 attempt 2 的所有科学/吞吐参数不变。
- 例外是基于已记录 telemetry 的语义不变显存调优：simulator cap 从 `0.92` 降到 `0.86`，把约 95.4% 占用降至预期约 89%，恢复 10% 左右 headroom；max seq 仍为 16，policy/trainer cap 仍为 `0.60`。

### F10 Vanilla pilot start attempt 3 / Slurm 133512

#### 实验设置

- 新 run `experiments/f10_pilot_20260901_stage18_r2`；validated F01 parent、fresh rank-32 LoRA、outcome GRPO、数据/seed/LR/batch/group/长度/轮数与前两次相同。
- Runtime 使用已通过 `133503` 的短 Ray tmpdir；simulator memory cap 根据 telemetry 从 `0.92` 调为 `0.86`，policy/trainer `0.60`，其他吞吐参数不变。Slurm 同节点 2x Pro 6000，target 5 steps，无 successor。

#### 执行结果

- Job `133512` 于 `05:02:49` 在 `gpu-pro6000-11` 启动，运行 `00:02:15` 后 Slurm `FAILED`；trainer task exit `1:0`，simulator cleanup TERM，聚合 exit `15:0`。
- Ray short path 生效：local Ray instance 成功启动，veRL config validation 通过，远程 `TaskRunnerV1.run()` 已执行。随后 worker import `transfer_queue` 时报 `ModuleNotFoundError`。
- 环境确认为 veRL `0.9.0`，其已安装 distribution metadata 未声明 `transfer_queue`，当前环境 `find_spec`/`pip show` 均为空。0/5 optimizer steps、无 checkpoint/reward audit，manifest 标记 failed。

#### 改进原因

- 这是 veRL 0.9 运行时依赖缺失或 package/version mismatch；不是 Ray socket、模型、数据、GPU topology、显存或 GRPO 科学假设失败。Attempt 3 证明前两个 infrastructure fixes 都已生效。

#### 改进措施

- 先核对已安装 veRL 源码中的 import 位置、官方 veRL 0.9 依赖/安装说明和 `transfer_queue` 的正式包来源/版本；不得猜包名直接安装。
- 依赖方案确认后通过独立 Slurm 环境修复/导入 smoke 安装到命名 Conda env，并记录版本；smoke PASS 前不再提交双 GPU F10。

### TransferQueue environment repair attempt 1 / Slurm 133532

#### 实验设置

- 根据官方 veRL Python requirements 固定 `TransferQueue==0.1.7`；在 `cpu-1` 的 Slurm compute job 中激活项目 SSD 命名环境，以 `pip install --no-deps` 避免依赖解析改动冻结的 Torch/CUDA/vLLM 栈。
- Smoke 检查 installed version、TransferQueue 高层 KV API，并联合导入 `verl.trainer.main_ppo.TaskRunnerV1`；输出计划为 `reports/transfer_queue_smoke_133532.json`。资源为 1 node/1 task/CPU、15 分钟，无 GPU。

#### 执行结果

- Job `133532` 在 `cpu-1` 运行 `00:01:19` 后 `FAILED`、exit `1:0`。`TransferQueue-0.1.7` wheel 已成功安装，pip 阶段无错误；veRL/TaskRunner 联合导入也已执行。
- 报告构造时访问 `TaskRunnerV1.__name__` 抛出 `AttributeError`：该符号经 `@ray.remote` 后是 `ActorClass(TaskRunnerV1)` wrapper，不是普通 Python class。因异常发生在 JSON 写入前，本 attempt 无 machine-readable report。
- CPU-only accelerator warning与本次失败无关；没有 GPU、训练、模型、数据或 checkpoint 产物。

#### 改进原因

- 失败属于 smoke checker 的反射假设错误，不是 TransferQueue 安装或 veRL 联合导入失败。必须修复 checker 并生成正式 PASS report，不能仅凭 pip success 解锁双 GPU retry。

#### 改进措施

- 将 veRL runner 记录改为稳定的 wrapper type（预期 `ActorClass`），保留版本、关键 API 和联合导入检查不变。
- 完成本地编译与远端 Bash syntax 后，以新 Job ID 重跑 CPU smoke；PASS 并完成记录前不提交 F10 attempt 4。

### TransferQueue environment repair attempt 2 / Slurm 133541

#### 实验设置

- 复用 attempt 1 已安装的 `TransferQueue==0.1.7`，仍在 `cpu-1` Slurm compute node 激活相同项目 SSD Conda 环境；`pip install --no-deps` 只确认 exact requirement already satisfied。
- 唯一 checker 修复是将 runner 标识改为 Ray wrapper type；expected version、六个 KV API、`TaskRunnerV1` 联合导入和 JSON report 标准全部保持不变。

#### 执行结果

- Job `133541` 运行 `00:01:04` 后 `COMPLETED`、exit `0:0`。Report `reports/transfer_queue_smoke_133541.json` 为 `PASS`，已归档到 `reports/cluster/TQ-SMOKE-133541/`。
- Installed/expected version 均为 `0.1.7`；`init`、`kv_put`、`kv_batch_put`、`kv_batch_get`、`kv_list`、`kv_clear` 六个 API 无缺失；veRL runner wrapper 为 `ActorClass(TaskRunnerV1)`。
- stderr 只有 CPU node 无 accelerator 的预期 warning；无 dependency resolution、import 或 API error。

#### 改进原因

- Attempt 2 用 machine-readable evidence 证明 attempt 1 的失败仅是 checker reflection，不是 TransferQueue/veRL compatibility 问题，并闭合 F10 attempt 3 的缺依赖阻塞。

#### 改进措施

- TransferQueue runtime dependency 现已 PASS；允许创建新的 F10 attempt 4 run/Job ID。
- F10 的 parent、fresh rank-32 LoRA、科学参数和 attempt 3 的 simulator/policy memory caps 保持冻结；仍无自动 successor，必须在有效 step-5 结果后人工验收。

### F10 Vanilla pilot start attempt 4 / Slurm 133549

#### 实验设置

- 新 run `experiments/f10_pilot_20260901_stage18_r3`；使用通过 `133541` smoke 的 TransferQueue 0.1.7 环境。Validated F01 parent、fresh rank-32 LoRA、outcome GRPO、数据/seed/LR/batch/group/长度/轮数全部冻结。
- 系统参数保持 attempt 3 的 simulator memory cap `0.86`、policy/trainer `0.60`、max seqs 16、batched tokens 16384、workers 16、microbatch 1/offload。Slurm 同节点 2x Pro 6000、target 5 steps、无 successor。

#### 执行结果

- Job `133549` 于 `05:23:49` 在 `gpu-pro6000-11` 启动，运行 `00:02:40` 后 Slurm `FAILED`、聚合 exit `15:0`；trainer task exit 1，simulator按 cleanup 结束。
- TransferQueue 初始化、CAR train/dev 数据加载（103/26）、总步数 5 config 和 actor-rollout worker group 创建均已越过。Worker actor 构造时 veRL 报 `ValueError: Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set.`
- 两角色 GPU UUID 不同。Simulator cap 0.86 实测峰值约 `87,576/97,887 MiB`（约 89.5%，约 10.5% headroom），无 OOM。Policy model 尚未加载，optimizer `0/5`、无 checkpoint/reward audit，manifest 正确标记 failed。

#### 改进原因

- cluster02 NVIDIA task 环境同时暴露 `CUDA_VISIBLE_DEVICES` 与 AMD compatibility 变量 `ROCR_VISIBLE_DEVICES`；veRL 0.9 为避免 GPU 编号语义冲突而主动拒绝。该问题与 TransferQueue、模型、数据、显存和 GRPO 假设无关。
- Simulator telemetry 证明 0.86 cap 达到目标余量，无需继续降低；policy cap 尚无真实 load 数据，保持不变。

#### 改进措施

- 在 NVIDIA F10 task source runtime 后显式清除 `ROCR_VISIBLE_DEVICES` 与 `HIP_VISIBLE_DEVICES`，保留 Slurm 分配的 `CUDA_VISIBLE_DEVICES`；记录清除前环境用于审计。
- 再次占用双卡前，先用 1x Pro 6000 Slurm smoke 启动 Ray GPU worker并直接调用 veRL 的 visibility hook，要求 driver/actor 均只有 CUDA namespace 且输出 JSON PASS。

### NVIDIA GPU visibility smoke / Slurm 133567

#### 实验设置

- 1 node/1 task/1x Pro 6000、10 分钟，在 `gpu-pro6000-2` 复用正式 Conda/CUDA/Ray/veRL 环境。Shell 先记录 Slurm 原始 visibility，再清除 HIP/ROCR。
- Python smoke 要求 Slurm driver 有 CUDA 且无 HIP/ROCR；实际启动 `ray.init(num_gpus=1)` 和单 GPU remote worker，并在 actor 内直接调用 `Worker._setup_env_cuda_visible_devices`，hook 前后均检查三类变量。

#### 执行结果

- Job `133567` 运行 29 秒后 `COMPLETED`、exit `0:0`；report `reports/gpu_visible_env_smoke_133567.json` 为 `PASS`，已归档到 `reports/cluster/GPU-ENV-SMOKE-133567/`。
- 原始 Slurm shell 为 `CUDA_VISIBLE_DEVICES=0`、`ROCR_VISIBLE_DEVICES=0`、HIP unset。清理后 driver、Ray GPU actor、veRL hook 后均为 CUDA `0`，HIP/ROCR 均不存在。
- stderr 仅 Ray local instance startup info；无 visibility、actor creation 或 GPU access error。

#### 改进原因

- 该 smoke 复现并覆盖了 attempt 4 的准确 guard path，证明显式清除 AMD namespace 后 Ray 不会重新注入 ROCR，且 veRL worker hook 可通过。

#### 改进措施

- GPU visibility 阻塞已闭合；允许使用全新 run/Job ID 提交 F10 attempt 5。
- 保持全部科学设置和 simulator/policy caps 不变，继续执行 5-step/no-successor 协议。

### F10 Vanilla pilot start attempt 5 / Slurm 133581

#### 实验设置

- 新 run `experiments/f10_pilot_20260901_stage18_r4`；生产 task 使用 `133567` 验证过的 CUDA-only normalization。Validated F01 parent、fresh rank-32 LoRA、outcome GRPO、数据/seed/LR/batch/group/长度/轮数和系统 caps 均与 attempt 4 相同。
- Slurm 同节点 2x Pro 6000，target 5 steps、无 successor。

#### 执行结果

- Job `133581` 在 `gpu-pro6000-8` 运行 `00:02:36` 后 `FAILED`、聚合 exit `15:0`；trainer exit 1，simulator cleanup。Visibility guard 已通过，actor worker 首次进入 exact 7B policy module build。
- Transformers 在权重加载前报 `ImportError: FlashAttention2 has been toggled on ... flash_attn ... isn't installed`。环境为 Torch `2.11.0+cu130`、CUDA runtime 13.0、Transformers 5.10.4，确认无 `flash_attn` distribution。
- Policy GPU 在失败前仅约 3 MiB，optimizer `0/5`、无 checkpoint/reward audit。Simulator 与双 GPU topology 正常，无 OOM；manifest 标记 failed。

#### 改进原因

- veRL FSDP automodel engine 默认 `attn_implementation=flash_attention_2`，当前自建环境遗漏其训练依赖。官方 veRL stable vLLM container 固定 `flash_attn==2.8.3` 并 force build；该 tag 的 setup 明确在 CUDA >=12.8 支持 `sm_120`，与 Pro 6000 Blackwell、当前 Torch/CUDA 栈匹配。
- 相比切换到 SDPA，安装官方 pin 保持 veRL 默认高效 attention 路径，更符合长上下文训练的速度/显存目标；不改变模型、数据或优化算法。

#### 改进措施

- 将 `flash_attn==2.8.3` 加入 GPU requirements；在单卡 Slurm job 用 CUDA 12.8 toolchain、`FLASH_ATTN_CUDA_ARCHS=120`、`MAX_JOBS=4`、no-build-isolation 强制本地编译，避免不匹配的通用 wheel。
- 安装后必须实际执行 BF16 FA2 forward/backward finite check，并用 exact merged F01 parent 以 `flash_attention_2` 加载和生成 1 token；JSON PASS 前不提交 F10 attempt 6。

### FlashAttention2 install/load smoke attempt 1 / Slurm 133600

#### 实验设置

- 1 node/1 task/1x Pro 6000、2 小时；固定 `flash_attn==2.8.3`、force local build、`FLASH_ATTN_CUDA_ARCHS=120`、`MAX_JOBS=4`，随后计划执行 kernel 前后向与 exact parent load/generate。
- Attempt 1 加载仓库此前统一使用的外部 CUDA 12.8 module，而命名环境 PyTorch 为 2.11.0+cu130。

#### 执行结果

- Job `133600` 在 `gpu-pro6000-2` 运行 42 秒后 `FAILED`、exit `1:0`；源码下载/metadata 正常，在 C++/CUDA build extension 开始时终止。
- PyTorch extension guard 报 detected CUDA `12.8` 与 PyTorch compile CUDA `13.0` mismatch。未生成 wheel、未安装 package、未进入 GPU kernel/model smoke，环境保持无 `flash_attn`。

#### 改进原因

- 失败是外部 nvcc toolchain 与 cu130 PyTorch 不匹配，不是 flash-attn 2.8.3、sm_120、GPU、内存或源码编译错误。实时 `module spider/avail` 确认集群提供可直接加载的 `CUDA/13.0.0`。

#### 改进措施

- 安装/验证 job 改用 CUDA 13.0.0，与 Torch cu130 对齐；版本、sm_120-only、4 jobs 和全部 PASS 标准保持不变，以新 Job ID 重试。
- 正式双角色 job 仅把 policy/trainer role 切到 CUDA 13.0.0；72B simulator role 保留已验证的 CUDA 12.8.0，避免无关改变。

### FlashAttention2 install/load smoke attempt 2 / Slurm 133615

#### 实验设置

- 与 attempt 1 保持相同 official `flash_attn==2.8.3`、force build、sm_120-only、MAX_JOBS=4 和 exact parent smoke；唯一修复是外部 module 改为 CUDA 13.0.0，与 Torch 2.11.0+cu130 对齐。
- 单节点 1x Pro 6000、2 小时；安装后必须通过独立 BF16 FA2 forward/backward finite，以及 exact merged F01 parent 的 FlashAttention2 load/1-token generation。

#### 执行结果

- Job `133615` 在 `gpu-pro6000-2` 运行 `00:20:25` 后 `COMPLETED`、exit `0:0`。63,235,995-byte wheel 成功编译并安装为 `flash_attn-2.8.3`。
- Report `reports/flash_attention_smoke_133615.json` 为 `PASS`，已归档 `reports/cluster/FA2-SMOKE-133615/`：device capability `[12,0]`、BF16 kernel forward/backward finite、Torch CUDA 13.0。
- Exact merged F01 7B parent 以 `model_attn_implementation=flash_attention_2` 成功加载并生成 1 token；peak allocated `15,287,706,112` bytes。stderr 仅 dtype deprecation 和正常 weight progress。

#### 改进原因

- Attempt 2 证明官方 pin、sm_120 extension、当前 Torch/CUDA/Transformers 与 exact parent 在目标 Pro 6000 上端到端兼容，闭合 attempt 5 的缺依赖问题。

#### 改进措施

- FlashAttention2 runtime dependency 已闭合；允许创建全新 F10 attempt 6 run/Job ID。
- Trainer role 使用 CUDA 13.0.0，simulator role 继续 CUDA 12.8.0；其他科学和系统参数冻结，仍无 automatic successor。

### F10 Vanilla pilot start attempt 6 / Slurm 133674

#### 实验设置

- 新 run `experiments/f10_pilot_20260901_stage18_r5`；trainer 使用通过 `133615` 的 CUDA 13/FA2 2.8.3，simulator 保持 CUDA 12.8。Validated F01 parent、fresh rank-32 LoRA、outcome GRPO、数据/seed/LR/batch/group/长度/轮数和 caps 冻结。
- 同节点 2x Pro 6000、target 5 steps、无 successor。

#### 执行结果

- Job `133674` 在 `gpu-pro6000-7` 运行 `00:04:21` 后 `FAILED`、聚合 exit `15:0`。首次完整完成 actor/ref、policy vLLM、checkpoint engine 和 reward loop 初始化，并打印 `ready to fit`。
- 首批 24 个 AgentLoop worker 在 Hydra resolve `configs/agent_loop/carbench.yaml` 时均缺 `CAR_BENCH_DATASET_ROOT`；随后 TransferQueue 收集空 keys 报错。根因是 F10 task 未显式 export 该 env 给 Ray worker。
- Policy GPU 初始化瞬时峰值约 `90,200/97,887 MiB`（92.2%，7.8% headroom），稳定段约 87.8 GiB；simulator 无 OOM。仍为 `0/5` optimizer steps、无 checkpoint/reward audit，manifest failed。

#### 改进原因

- 此问题发生在首个真实 rollout 的 agent-loop config instantiate，模型/FA2/policy vLLM/TransferQueue infrastructure 已通过。`SIMULATOR_BASE_URL` 已导出，但同一 config 需要的 CAR dataset root 在新 launcher 中遗漏。
- Policy 瞬时 headroom 低于目标 10%，但没有 OOM；在获得首个有效 step 前不基于初始化瞬时值改变 cap，避免同时混入两项修复。

#### 改进措施

- 在 trainer 启动前显式 export canonical `$PROJECT_ROOT/data/official/car-bench-dataset`；路径不改变既有训练数据或任务语义。
- 先用 CPU Slurm/Ray worker 实际 resolve agent-loop OmegaConf，要求 dataset path 存在、target 与 simulator URL 均正确并输出 JSON PASS；PASS 前不提交 attempt 7。

### AgentLoop Ray environment smoke / Slurm 133700

#### 实验设置

- CPU Slurm compute job 复用正式 Conda/Ray 和短 tmpdir；显式 export canonical CAR root 与 dummy local simulator URL。
- 在真实 Ray remote worker 中加载并完全 resolve `configs/agent_loop/carbench.yaml`，检查 dataset path 等于预期且存在、target 为 CAR loop、simulator URL 保持一致；输出 JSON。

#### 执行结果

- Job `133700` 在 `cpu-1` 运行 32 秒后 `COMPLETED`、exit `0:0`。Report `reports/agent_loop_env_smoke_133700.json` 为 `PASS`，已归档 `reports/cluster/AGENT-LOOP-ENV-133700/`。
- Ray worker 成功看到 `/projects/_ssd/jiatian001ssd/cabinagentrl/CabinAgent-RL/data/official/car-bench-dataset`，目录存在；target 与 simulator URL 均正确。stderr 仅 Ray startup info。

#### 改进原因

- Smoke 覆盖了 attempt 6 精确失败点，证明 driver export 会传入 Ray actor并完成 OmegaConf interpolation，而不是只在 shell 中存在。

#### 改进措施

- AgentLoop environment dependency 已闭合；允许全新 F10 attempt 7。
- 维持全部科学参数、trainer/simulator CUDA 分工和 memory caps；policy cap 的调整推迟到至少一个有效 step 或 OOM evidence 后。

### F10 Vanilla pilot start attempt 7 / Slurm 133709

#### 实验设置

- 新 run `experiments/f10_pilot_20260901_stage18_r6`；source/config SHA-256 为 `6676280d3ed24b1b4cf85a5ec95a23874fda3538518bbdbf58fdf1d5ee14c5a0` / `7fd65bc8b3f7b99866ac500e8bf05d9dbf524068b4e41054c9c05d971819dc05`。
- 使用 validated corrected-F01 merged parent、fresh rank/alpha `32/32` RL LoRA、outcome-only GRPO、4 tasks x 4 rollouts、seed 42、LR `1e-6`、相同 CAR train/dev 与 72B simulator；trainer CUDA 13.0/FA2 2.8.3，simulator CUDA 12.8。
- 系统参数保持 simulator cap `0.86`、rollout cap `0.60`、max seqs 16、batched tokens 16384、AgentLoop workers 16、actor/ref offload、dynamic batch 和 microbatch 1；target 5 steps，无 successor。
- Slurm job `133709`：`cluster02` / `msc`，`gpu-pro6000-3`，同节点 2x Pro 6000、2 tasks、8 CPU、180 GiB，单一 two-task `srun`；产物位于 `experiments/f10_pilot_20260901_stage18_r6`，本地归档位于 `reports/cluster/F10-PILOT-133709/`。

#### 执行结果

- Job 于 2026-09-01 06:31:51 UTC 启动，运行 `00:07:07` 后 `FAILED`，trainer task exit 1，simulator task 随 step cleanup 终止；manifest 正确标记 `failed`。
- Attempt 7 首次跑通完整 actor/ref/policy-vLLM 初始化、26-task 初始验证和首批 16 条训练 rollout。初始验证 reward mean@1 为 `0.230769`，turns min/max/mean 为 `5/42/15.8462`。
- 在 step 1 的 `_compute_old_log_prob` 中，`entropy_from_logits` 对完整 logits 执行 softmax 时尝试再分配 `20.44 GiB`，当时仅剩 `18.86 GiB`，触发 `torch.OutOfMemoryError`。训练进度仍为 `0/5`，无 checkpoint、reward-audit 或 optimizer/gradient 指标。
- 84 个 5 秒 telemetry 样本显示：simulator 卡峰值 `87,576/97,887 MiB`（89.47%），trainer 卡峰值 `96,055/97,887 MiB`（98.13%）；两卡最大利用率均为 100%，未出现 NaN 或 reward-schema error。

#### 改进原因

- 这是首次由真实 rollout 长度触发的 trainer 显存证据。环境、依赖、双卡绑定、模型加载、AgentLoop 与 simulator 均已越过，当前根因是 old-log-prob entropy 的未分块全词表 softmax 中间张量，而不是 scientific reward/advantage 假设失败。
- Effective batch、group size、上下文上限和 sampling 均属于冻结科学设置，不应为规避 OOM 而先行缩小；降低 rollout cap 也不能直接解决 FSDP old-log-prob softmax 的瞬时张量。

#### 改进措施

- 保留本 attempt 与全部日志，不创建 checkpoint 假象。先核对当前 veRL 已解析配置中的 `entropy_from_logits_with_chunking` / `entropy_from_logits_chunk_size` 路径，再启用数学等价的分块 entropy 计算，以压低 softmax 峰值；actor/ref 两侧配置一致。
- 下一 attempt 使用新 run/Job ID，除该显存系统开关外保持模型、数据、reward/advantage、sampling、有效 batch、长度、LoRA、优化器/LR、两侧 memory cap 和资源拓扑冻结；仍为 5 steps、无 successor，并继续采集相同 telemetry。
- 新 attempt 提交前须完成配置渲染/测试并更新本记录；step 5 checkpoint 与 step-6 resume 未通过前，正式 F10-F14 继续阻塞。

### F10 Vanilla pilot start attempt 8 / Slurm 134671

#### 实验设置

- 复用 attempt 7 的 validated corrected-F01 parent、fresh rank-32 LoRA、outcome-only GRPO、4 tasks x 4 rollouts、seed 42、LR `1e-6`、CAR 数据、32K/20-turn、simulator `0.86` 与 rollout `0.60` caps、offload 和同节点 2x Pro 6000 拓扑。
- 唯一计划变更为 actor/ref 两侧 `entropy_from_logits_with_chunking=true`、`entropy_from_logits_chunk_size=2048`；target 仍为 5 optimizer steps，无 successor。

#### 执行结果

- 已核对远端 veRL 0.9 的 `dp_actor.yaml`、`dp_ref.yaml` 与 FSDP engine 实现，确认上述两个 Hydra 路径分别传播到 actor/ref engine，并调用内置 chunked entropy 路径。
- 本地配置渲染回归、36 项 unit tests 与 `compileall src scripts` 均通过；远端 Bash syntax、真实 veRL dry-run、36 tests 与 `sbatch --test-only` 也通过，双侧配置解析为 `true/2048`。
- 新 run `experiments/f10_pilot_20260901_stage18_r7` 已提交为 job `134671`。Slurm 确认 1 node、2 tasks、8 CPU、2x `gpu:pro6000`。
- Job 于 2026-09-01 16:08:59 UTC 在 `gpu-pro6000-11` 启动，运行 `00:10:21` 后 `FAILED`、exit `15:0`；trainer task exit 1，simulator task 按 cleanup 以 143 退出，manifest 标记 `failed`。
- 双侧 resolved config 明确显示 actor/ref 与 actor FSDP engine 的 chunking 为 `True/2048`；初始 26-task validation reward mean@1 为 `0.230769`，turns min/max/mean 为 `5/44/15.8462`，随后完成首批 16 条训练 rollout。
- Step 1 `_compute_old_log_prob` 最终仍从 dense-padding `prepare_model_outputs` 直接调用未分块 `verl_F.entropy_from_logits(logits)`，尝试分配 `20.80 GiB`，当时仅余 `18.15 GiB`，CUDA OOM。仍为 `0/5`，无 checkpoint、reward audit、advantage 或 gradient 指标。
- 123/121 个 telemetry 样本显示 simulator/trainer 峰值分别为 `87,576/97,887 MiB`（89.47%）和 `90,565/97,887 MiB`（92.52%），两卡最大利用率均 100%。产物已归档到 `reports/cluster/F10-PILOT-134671/`。

#### 改进原因

- Attempt 7 的 OOM 位于未分块全词表 entropy softmax，原计划用内置 chunked 路径避免该瞬时中间张量。
- Attempt 8 证明配置路径本身无误，但当前 veRL 0.9 只有 `use_remove_padding=True` 的 packed 分支根据 `engine_config.entropy_from_logits_with_chunking` 调用分块实现；项目显式冻结为 `actor_rollout_ref.model.use_remove_padding=False` 后进入 dense 分支，该分支无条件调用未分块 entropy。失败属于 runtime/config-path 覆盖缺口，不是 chunk size、模型、数据、reward/advantage 或科学假设失败。

#### 改进措施

- Attempt 8 保留且不产生 checkpoint 假象；当前不重复提交相同配置。
- 下一候选修复有两条：推荐启用 veRL 已支持的 `use_remove_padding=True` packed 路径，使现有 chunking 真正生效并同时去除 padding；备选是在项目运行时为 dense 分支补齐 chunked 调用。前者改动更小且不修改安装环境，但必须先确认用户接受这一数学等价的表示/系统路径变化，并用单 GPU packed-path smoke 验证 exact parent、FA2、LoRA 和 finite log-prob/entropy 后才允许新的双 GPU attempt。
- 模型、数据、4x4 effective batch、sampling、长度/轮数、reward/advantage、LoRA、optimizer/LR、simulator 和现有 memory caps 继续冻结；未获确认前不修改代码或提交下一作业。

### F10 packed-path preflight / local preparation

#### 实验设置

- 代码路径将 `actor_rollout_ref.model.use_remove_padding` 改为环境可控且默认 `true`；F10 submitter 显式导出该值。actor/ref chunked entropy 保持 `true/2048`，其余 attempt 8 科学设置与资源上限全部冻结。
- 新增单节点单卡 Pro 6000、30 分钟的 packed-path integration smoke；模型固定为 corrected F01 merged parent，并挂载 fresh rank-32/alpha-32、all-linear LoRA。

#### 执行结果

- 本地 training-config regression 与 Python compilation 通过；尚未提交 Slurm，因此暂无 Job ID 或 GPU 结果。
- 远端登录节点 API preflight 确认 chunk helper 签名正确，但输出方法实际归属 `FSDPEngineWithLMHead`；已在 GPU 提交前修正 checker 的反射目标，未产生无效 Slurm attempt。

#### 改进原因

- Attempt 8 已证明 dense-padding 分支不消费 chunked-entropy 配置。必须在再次占用双卡前验证 veRL 原生 packed 分支、FA2、exact parent、LoRA backward 和 entropy/log-prob 数值健康。

#### 改进措施

- Smoke 输出 machine-readable JSON，并要求 exact parent 以 FA2 加载、packed 分支源码契约存在、entropy/log-prob/loss 有限、LoRA gradient 有限且非零；仅 PASS 后允许创建新的 5-step F10 run/Job ID。

### F10 packed-path preflight / Slurm 135977

#### 实验设置

- Git commit `ba12b7ad9135b11d9579fed37a10951cea4ce765`；单节点 1 task、1x Pro 6000、30 分钟，CUDA 13.0 与既有项目 Conda 环境。模型为 exact corrected-F01 merged parent，临时挂载 fresh rank/alpha `32/32` all-linear LoRA。
- Checker 使用两条不等长输入形成 packed valid-token logits，调用 veRL `entropy_from_logits_with_chunking(..., chunk_size=2048)`，计算 token log-prob/NLL backward，并核对 FA2、packed 分支源码契约、有限数值与有限非零 LoRA gradient。

#### 执行结果

- 本地与远端 36 项 tests、`compileall`、Bash syntax、真实 veRL dry-run 和 `sbatch --test-only` 全部通过；远端渲染明确为 remove-padding `true` 且 actor/ref chunking `true/2048`。
- Job `135977` 于 2026-09-02 12:13:01 UTC 提交，当前 `PENDING (Priority)`；请求 1 node、1 GPU、4 CPU、33 GiB node memory，无依赖且无 successor。集群 test-only 当时预测资源约在 2026-09-03 20:20 可用。
- Job 实际于 12:13:55 UTC 在 `gpu-pro6000-3` 启动，运行 44 秒后 `COMPLETED`、exit `0:0`。`reports/packed_entropy_smoke_135977.json` 为 `PASS`。
- Exact parent 使用 FlashAttention2；两条序列长度为 10/17，shift 后 packed valid tokens 为 25。Entropy、log-prob、loss 与 LoRA gradients 全部有限，fresh LoRA gradient norm 为 `2.508679700952618`，trainable parameters 为 `80,740,352`。
- CUDA peak allocated/reserved 为 `16,028,334,592/16,162,750,464` bytes（约 14.93/15.05 GiB）。stderr 只有未安装的可选 engine warning、dtype deprecation 和正常权重进度，无 CUDA/FA2/OOM 错误；完整 JSON、allocation 与 Slurm 日志已归档到 `reports/cluster/PACKED-ENTROPY-SMOKE-135977/`。

#### 改进原因

- 该 smoke 是 attempt 8 dense-path OOM 后的强制单卡门禁，避免在 packed/LoRA/entropy 组合未经验证时直接消耗双卡作业。

#### 改进措施

- 排队和运行期间冻结 checker、Slurm 脚本与训练 launcher。只有 `reports/packed_entropy_smoke_135977.json` 为 PASS 且 Slurm `COMPLETED` 后，才记录本 attempt 完成并提交新的双卡 5-step F10；否则先记录失败并修复，不跨过门禁。
- 本 attempt 已闭合 packed-path 单卡门禁。完成本记录及 Project/Progress/tracker/总览同步后，允许以新 run/Job ID 提交 5-step F10；科学与既有吞吐参数保持冻结，仍无 automatic successor。

### F10 Vanilla pilot start attempt 9 / Slurm 135987

#### 实验设置

- 新 run `experiments/f10_pilot_20260902_stage18_r8`；source/config SHA-256 为 `9f5aa580d47a4458b9338c043df09a31688d79fc86a3b1fa0540e181ee699b14c` / `7fd65bc8b3f7b99866ac500e8bf05d9dbf524068b4e41054c9c05d971819dc05`。
- 复用 validated corrected-F01 parent，初始化 fresh rank/alpha `32/32` all-linear RL LoRA；outcome GRPO、4 tasks x4 rollouts、seed 42、LR `1e-6`、CAR train/dev、32K/20-turn、72B simulator、simulator/rollout caps `0.86/0.60`、max seqs 16、batched tokens 16384、workers 16、microbatch 1 与 offload 全部冻结。
- 唯一相对 attempt 8 的执行路径变化为 `use_remove_padding=true`，actor/ref chunked entropy 保持 `true/2048`。同一物理节点 2x Pro 6000、2 tasks/单一 `srun`，target 5 steps、save/eval=5，无 successor。

#### 执行结果

- Packed smoke PASS 记录同步后，远端 Bash、真实 veRL config rendering 和 `sbatch --test-only` 再次通过；个人队列提交前为空。
- Job `135987` 于 2026-09-02 12:21:29 UTC 提交，当前 `PENDING (Priority)`；请求 1 node、2 tasks、8 CPU、180 GiB、2x Pro 6000/highmem。当前调度估计为 2026-09-03 02:59:05 UTC、`gpu-pro6000-11`，仅为 scheduler estimate。
- Manifest 状态为 `submitted`，明确记录 Job ID 与 source/config digest；训练尚未启动，当前为 `0/5` 且无 checkpoint。
- Job 实际于 2026-09-02 12:57:57 UTC 在 `gpu-pro6000-11` 启动，13:26:40 UTC 以 `COMPLETED`、exit `0:0` 结束，总时长 `28m43s`；同节点 2x Pro 6000、2 tasks、8 CPU、180 GiB 配置与申请一致。
- 训练完成 `5/5` optimizer steps。逐步核心指标为：step 1/2/3/5 reward mean 均 `0.0625`、范围 `[0,1]`，advantage 范围均含正负值且 grad norm 分别为 `0.071629/0.025236/0.017444/0.035955`；step 4 reward mean `0.25` 但组内 advantage 全零，grad norm `1.723e-05` 为 KL-only。五步均无 NaN/OOM/reward-schema error，clip fraction 为 `0`，KL 项有限。
- Step time 为 `188.06/228.82/184.95/136.75/320.07s`；step 5 包含约 `46.33s` checkpoint 和 `100.68s` final validation。Initial/final CAR dev mean@1 为 `0.230769/0.269231`，五步不足以作性能改进结论。
- Step-5 checkpoint 完整保存到 `experiments/f10_pilot_20260902_stage18_r8/checkpoints/global_step_5`，约 `30 GiB`，`latest_checkpointed_iteration.txt=5`；manifest 已为 completed。训练进度达到 100% 后 Ray DataLoader worker 在 shutdown 阶段被终止并打印 traceback，但它发生在 step-5 指标、checkpoint 和 final validation 落盘之后，Slurm 聚合退出码仍为 0，因此记录为成功伴随 shutdown warning。
- Simulator/trainer telemetry 分别采集 341/339 个样本，显存峰值为 `87,576/97,887 MiB`（89.47%）和 `94,529/97,887 MiB`（96.57%），两卡利用率最大均为 100%；trainer 已接近安全上限，不再提高显存占用参数。精选结果已校验归档到 `reports/cluster/F10-PILOT-135987/`；公开仓库日志副本将一处 CAR 样例中的 RapidAPI 参数值替换为 `[REDACTED]`，集群原始日志保持不变。

#### 改进原因

- Attempt 8 的 dense branch 忽略 chunked entropy；单卡 job `135977` 已证明 native packed path 对 exact parent、FA2、fresh LoRA、entropy/log-prob/backward 数值健康，因此按已确认路线进入真实 5-step 集成验证。
- Attempt 9 证明 packed path 修复了此前的 optimizer 前 OOM，并满足“至少一步 mixed reward + 非零 advantage + 有限非零 gradient”的核心门禁。step 4 的全零 outcome advantage 是采样组同分现象，不否定其他四步的有效学习信号。

#### 改进措施

- 排队/运行期间冻结所有执行代码与脚本。运行后采集 simulator/trainer telemetry、initial validation、逐 step reward variance/advantage、gradient/KL/clip、step time、checkpoint 与 resume 元数据。
- 只有 5 steps、step-5 checkpoint 和至少一步非零 reward variance/advantage/finite nonzero gradient 全部成立且无 NaN/OOM/schema error，才进入人工验收；本作业不会自动提交 step-6 resume 或正式 F10-F14。
- 上述五步标准已满足，允许按既定边界从同一 run 单独提交 step-6 resume；resume 仍无 successor，完成并记录前不启动正式 F10-F14。

### F10 step-6 resume preparation / local

#### 实验设置

- 恢复对象固定为 `f10_pilot_20260902_stage18_r8/global_step_5`，目标仅从 step 5 恢复并完成 step 6；模型、数据、seed、4x4 rollout、sampling、reward/advantage、LoRA、optimizer/LR、长度、simulator、packed/chunked 路径、offload、memory caps 与同节点双 GPU 拓扑全部冻结。
- SSD checkpoint 策略按用户确认改为每个 run 最新 `1` 个完整可恢复 checkpoint；launcher 显式设置 actor/critic retention 为 `1`。本次单步恢复使用 `SAVE_FREQ=-1`，保留已验收的 step-5 checkpoint，避免临时再生成约 30 GiB 的 step-6 checkpoint。

#### 执行结果

- 本地 training-config 回归与 Python compilation 通过；尚未提交 step-6 Slurm 作业，因此本小节当前无 Job ID 或训练指标。
- 提交前 SSD 已完成精确清理：删除 pip cache 和 5 个已作废/被正式结果替代的 SFT smoke/full 目录，保留 corrected F01、F02 正式负结果、模型、数据及 step-5 checkpoint；占用由约 `124.5` 降至 `117.1 GiB`。被删 SFT 的 114 项配置/manifest/metrics/小日志已先做 SHA-256 校验归档到 `reports/cluster/SSD-CLEANUP-20260903/`。

#### 改进原因

- 当前 checkpoint 约 30 GiB，SSD 总额 150 GiB；若 step-6 按旧 `save_freq=5` 的 last-step 行为再次保存，可能在清理旧 checkpoint 前短暂占用第二份约 30 GiB，超出安全余量。
- Step-6 的验收目标是证明 checkpoint 可恢复并继续产生第 6 个 optimizer step，不要求创建新的 checkpoint；保留已验证的 step-5 恢复点即可满足失败恢复需要。

#### 改进措施

- 提交器按 start/resume 分支分别使用 `save_freq=5/-1`，并新增 `MAX_ACTOR_CKPT_TO_KEEP=1`、`MAX_CRITIC_CKPT_TO_KEEP=1` 的显式导出与配置渲染回归。
- 同步集群后必须执行 Bash syntax、unit tests、veRL dry-run 和 `sbatch --test-only`；全部通过才提交 resume。作业完成后验证从 step 5 加载、step 6 指标落盘、原 step-5 checkpoint 仍存在且全项目没有新增第二个 GRPO checkpoint。

### F10 step-6 resume attempt 1 / Slurm 136347

#### 实验设置

- 同一 run `f10_pilot_20260902_stage18_r8` 从 `global_step_5` 自动恢复，目标 total steps=`6`；实际恢复代码版本为 Git `c6800f8`，部署包 SHA-256 `5f6e21e34522b54b14f34970e7b2137aec7926b9180ba10ec4759ae189b9c96e`。Run manifest 保留创建时的 source/config digest，并通过 `submissions.tsv` 单独记录 resume Job。
- 科学与系统训练设置完全继承 attempt 9；唯一恢复期存储设置为 `SAVE_FREQ=-1`、actor/critic retention=`1`。资源仍为 cluster02/msc、同节点 2x Pro 6000、2 tasks、8 CPU、180 GiB、6 小时，单一双 task `srun`，无 successor。

#### 执行结果

- 提交前远端 36 tests、`compileall`、Bash syntax、veRL override rendering 与 `sbatch --test-only` 全部通过；渲染确认 total steps `6`、resume auto、save frequency `-1`、retention `1/1`、packed/chunked 路径不变。Installed veRL 源码与默认配置均确认支持两个 retention 字段。
- Job `136347` 于 2026-09-02 16:43:12 UTC 提交，当前 `PENDING (Priority)`、无 dependency，申请资源为 1 node/2 Pro 6000/2 tasks/8 CPU/180 GiB。`submissions.tsv` 已追加 resume 行；当前仍仅有 `global_step_5` 一个 GRPO checkpoint。
- Job 实际于 2026-09-02 16:43:38 UTC 在 `gpu-pro6000-10` 启动，16:54:48 UTC 以 `COMPLETED`、exit `0:0` 结束，总时长 `11m10s`；资源实际为同节点 2x Pro 6000、2 tasks、8 CPU、180 GiB。
- 日志明确打印从 `global_step_5` 恢复并将 global step 设为 5；checkpoint 内容配置为 `model/optimizer/extra`，随后完成 step 6 和 final validation。Step 6 reward/advantage 全零，`pg_loss=0`，但有限 KL loss `5.53149e-4` 产生有限 `grad_norm=1.17578e-5`；该单步只验证恢复，不作为 outcome 学习信号证据。
- Step 6 rollout/actor 概率相关性为 `0.999194`，rollout-correction KL `0.0007705`，step time `173.23s`，吞吐 `1186.59 token/s`；initial/final CAR dev mean@1 均为 `0.230769`，不作性能声明。
- Simulator/trainer telemetry 各 131 个样本，显存峰值分别为 `87,575/97,887 MiB`（89.47%）和 `90,932/97,887 MiB`（92.89%），最大利用率均 100%。全项目仍只有约 30 GiB 的 `global_step_5`，没有 `global_step_6`，retention/no-save 策略生效。
- 训练进度到 100% 后再次出现 Ray DataLoader worker shutdown traceback，但 final metrics 已输出、Slurm 与全部 steps 均为 `COMPLETED/0:0`、manifest 为 completed；与 attempt 9 一致，记录为非致命 shutdown warning。精选证据已校验归档到 `reports/cluster/F10-RESUME-136347/`。

#### 改进原因

- 五步 attempt 9 已验证从 fresh LoRA 训练和保存 checkpoint；pilot 闭环还缺独立 Slurm 进程实际加载 optimizer/model/data 状态并继续完成 step 6 的证据。
- SSD 仅余约 32.9 GiB，而一个完整 checkpoint 约 30 GiB；恢复验证若创建第二份 checkpoint 会使存储余量接近零，因此该单步仅验证可恢复性与继续训练，不保存新的恢复点。
- 恢复结果证明 model、optimizer、extra/data 状态可由独立 Slurm 进程继续训练；step 6 的 outcome advantage 全零属于该批 rollout 同分，不是恢复失败，也不取代 attempt 9 已获得的四个有效 outcome-gradient steps。

#### 改进措施

- 排队/运行期间冻结执行脚本与代码。完成后核对 Slurm 状态、checkpoint load 日志、step-6 reward/advantage/gradient/KL/step time、final validation、双卡 telemetry 与 checkpoint 目录数量。
- PASS 标准：真实从 step 5 加载且完成第 6 个 optimizer step，无 NaN/OOM/schema error，原 step-5 checkpoint 完整存在且没有新增 `global_step_6`；无论成功或失败，先完成本 attempt 四节结果更新再决定正式 F10-F14。
- 上述恢复 PASS 标准已满足。正式 fallback F10 解锁；按实验家族契约从同一 corrected-F01 merged parent 新建 fresh rank-32 RL LoRA，不从 pilot checkpoint 继续训练，也不自动启动 F11-F14。
- 正式 F10 提交前先归档并删除已完成验证使命的 pilot 完整 checkpoint，以维持 SSD“仅 1 个最新完整 checkpoint”策略；再实时核对集群资源、完成 formal config/test-only，并以新 run/Job ID 提交。

### Formal F10 step-50 preparation / local

#### 实验设置

- 正式 F10 总目标保持 250 optimizer steps，评测/保存边界保持 `50/100/150/200/250`；实际调度拆为五个可恢复的 50-step segment，每段完成和记录后才人工提交下一段，不改变数据顺序、优化器状态或训练目标。
- 首段从 validated corrected-F01 merged parent 初始化 fresh rank/alpha `32/32` all-linear RL LoRA，target=`50`、save/eval frequency=`50`、actor/critic retention=`1`。复用已通过 pilot/resume 的 packed/chunked 路径、`0.86/0.60` memory caps、4x4 rollout 和同节点双卡 runtime，不从 pilot checkpoint 继承。

#### 执行结果

- 新增 `scripts/submit_f10_formal.sh`：start 仅允许 target 50，resume 仅允许 100/150/200/250；每次独立提交、最长 12 小时、无 `NEXT_TRAINING_STAGE` 或 automatic successor。
- 尚未删除 pilot checkpoint、同步集群或提交 formal Slurm job；当前正式 F10 optimizer steps 为 0。
- Git `9096f74` 已推送并以 SHA-256 `fc26f23b410cc85c24126c2dbf533b66c5f33c6425ba4301af16a6a1181e3f79` 的代码包同步。远端 Bash syntax、36 tests、`compileall`、Hydra rendering 和 `sbatch --test-only` 全部通过；渲染确认 target/save/eval `50/50/50`、retention `1/1`、packed/chunked 与 caps 均正确。
- Pilot `global_step_5` 及 1 字节 stale marker 已在无排队作业时精确删除；删除前路径、LoRA metadata 和 `data.pt` 均核对。`storagemgr` 显示 SSD 由 `117.2` 降至 `85.8/150 GB`，当前完整 GRPO checkpoint 数为 0。

#### 改进原因

- 一个完整 FSDP checkpoint 约 30 GiB，而 SSD 配额 150 GiB。一次性 250-step job 不利于在每个 50-step 边界先验收、归档和处理 checkpoint，也会削弱失败定位与恢复边界。
- Step-6 已证明 model/optimizer/extra 可由新 Slurm 作业连续恢复，因此按评测点分段不会重置训练状态；它只改变调度边界，不改变正式实验的科学设置。

#### 改进措施

- 本地与远端必须验证 formal override 恰为 target/save/eval `50/50/50`、fresh run 不含 checkpoint、retention `1/1`、无 successor，且继续使用单一 `srun --ntasks=2 --gpus-per-task=1 --gpu-bind=single:1`。
- 提交前先校验并删除 pilot `global_step_5`，然后实时核对 account/QoS/GPU/队列/SSD；全部通过后仅提交 formal F10 step-50 segment。
- 上述准备已完成；正式 attempt 已独立记录如下。排队/运行期间冻结 Git `9096f74` 对应执行代码。

### Formal F10 step-50 segment attempt 1 / Slurm 136868

#### 实验设置

- Run `experiments/f10_formal_20260903_stage19`；执行代码 Git `9096f74`，manifest source/config SHA-256 为 `90267732151d8cb2648e5f01a7e2b04a81e09a8f62b74b2da886b336b1e886ae3` / `7fd65bc8b3f7b99866ac500e8bf05d9dbf524068b4e41054c9c05d971819dc05`。
- 从 validated corrected-F01 merged parent 新建 fresh rank/alpha `32/32` all-linear LoRA；outcome GRPO、CAR train/dev、seed 42、4 tasks x4 rollouts、LR `1e-6`、32K/20-turn、72B simulator、packed/chunked entropy、offload 与 caps `0.86/0.60` 继承通过的 pilot，不加载任何 pilot RL checkpoint。
- Segment target/save/eval=`50/50/50`，actor/critic retention=`1/1`；cluster02/msc、1 node、2x Pro 6000、2 tasks、8 CPU、180 GiB、12 小时，单一双 task `srun`，无 dependency 和 successor。

#### 执行结果

- Job `136868` 于 2026-09-03 05:11:13 UTC 提交，当前 `PENDING (Priority)`、start time unknown。Slurm 已确认 `ReqTRES=cpu=8,mem=180G,node=1,gres/gpu:pro6000=2`。
- Fresh run manifest 为 submitted，checkpoint 目录为空；冻结的 `config.yaml` 与 `grpo_common.yaml` 已写入 run。当前正式 F10 为 `0/50` optimizer steps。
- Job 实际于 2026-09-03 05:12:05 UTC 在 `gpu-pro6000-3` 启动，07:43:33 UTC 以 `COMPLETED`、exit `0:0` 结束，运行 `02:31:28`；实际资源为同一物理节点 2x Pro 6000、2 tasks、8 CPU、180 GiB。
- 训练完成 `50/50` optimizer steps。`21/50` 步（42%）具有非零 group-normalized outcome advantage 和有限非零 gradient，`29/50` 步 outcome advantage 全零；有效 outcome 步的 grad norm 范围/均值为 `0.00935--0.12299/0.04131`。全部 50 步 grad norm 有限，范围/均值为 `9.83e-6--0.12299/0.01774`。
- 50 个 batch 共 800 条在线训练 trajectory，其中 94 条 reward=1，mean reward `0.1175`。每十步有效 outcome-gradient 数为 `6/2/4/5/4`，没有出现连续退化为全零信号的趋势。
- Rollout/actor probability correlation 平均 `0.999110`，rollout-correction KL 平均 `0.001020`，50 步 clip fraction 均为 0；无 NaN、OOM、reward-schema error 或 aborted trajectory。
- 初始/最终 CAR dev mean@1 均为 `0.269230769`，因此当前只支持“训练信号与运行时健康”，不支持性能已提升的结论。平均/中位 step time 为 `172.06/170.60s`，平均吞吐 `1231.53 token/s`。
- `global_step_50` 已保存 model/optimizer/extra/data 状态，共 11 个文件、`31,443,788,637` bytes；`latest_checkpointed_iteration.txt=50`。全项目仅此 1 个完整 GRPO checkpoint，符合 retention=1，`storagemgr` 为 `117.3/150 GB`。
- Simulator/trainer telemetry 各 1804/1803 个样本，显存峰值为 `87,576/97,887 MiB`（89.47%）和 `94,529/97,887 MiB`（96.57%），两卡最大利用率均 100%；trainer 余量已很小，不提高显存占用参数。
- 进度达到 100% 后出现与 pilot/resume 一致的 Ray DataLoader worker shutdown traceback；checkpoint、step-50/final validation metrics、manifest 与 Slurm exit 均成功，故记录为非致命 shutdown warning。公开安全的派生摘要归档于 `reports/cluster/F10-FORMAL-136868/`，原始对话日志不进入公开仓库。

#### 改进原因

- Pilot 已证明有效 outcome gradient、完整 checkpoint 和独立恢复；正式基线需要在不改写历史 gate FAIL 的前提下扩大到预注册的 250 steps，并在每个 50-step 边界保存和评测。
- 采用分段恢复是为配合 150 GB SSD 与最新 1 个完整 checkpoint 策略，同时让每个边界先做健康检查；模型、优化器和 extra state 连续，不改变训练语义。
- Step-50 实测的有效 outcome-gradient 比例为 42%，明显不同于“持续无有效梯度”的失败分支；虽然独立 dev 指标暂未上升，但当前证据支持继续同一 F10，而不是提前切换 F13 或模型。

#### 改进措施

- 排队/运行期间冻结执行代码、配置和所有科学/吞吐参数；监测 Slurm、manifest、双卡 telemetry、逐 step reward/advantage/gradient/KL/clip、step time、checkpoint 和 final validation。
- Step-50 PASS 要求完成 50/50、无 NaN/OOM/schema error、产生可恢复 `global_step_50` 且仍最多 1 个完整 checkpoint；单步同分允许存在，但需报告 50 steps 中有效 outcome-gradient 的频率与分布。完成并记录前不提交 step-100 resume 或 F11-F14。
- 上述 PASS 标准均已满足。保持 Git `9096f74` 对应执行路径、全部科学设置、caps `0.86/0.60` 和 retention `1/1` 不变；完成本 attempt 的四处记录、公开安全归档和 GitHub 推送后，允许从 `global_step_50` 人工提交 target=100 的独立 resume segment。

### Formal F10 step-100 submission attempt 1 / Slurm 137581

#### 实验设置

- 计划从 `f10_formal_20260903_stage19/global_step_50` 恢复到 total step 100；save/eval frequency `50/50`、retention `1/1`、全部科学/系统参数和同节点双卡拓扑继承 step-50 PASS 配置。
- 提交前 checkpoint 精确核对为 11 文件、`31,443,788,637` bytes，且为全项目唯一完整 GRPO checkpoint；Slurm `--test-only` 对相同 target=100 override 通过。

#### 执行结果

- `sbatch` 返回 Job `137581`，但提交器随后在登录 shell 执行 manifest update 时因 `python: command not found` 返回非零；原因是新 SSH 会话未激活 Conda，提交器隐式依赖 PATH 中存在 `python`。
- Job 尚处于 pending、未分配节点且 elapsed `00:00:00` 时已精确取消；Slurm 最终 `CANCELLED`，无 GPU、无 rollout、无 optimizer step，manifest 和 `submissions.tsv` 均未被错误更新，step-50 checkpoint 未修改。

#### 改进原因

- Submitter 在 `sbatch` 之后才调用裸 `python`，形成“作业已入队但生命周期元数据未更新”的非原子失败窗口；batch 脚本在启动 two-task `srun` 前也调用裸 `python`，因此不能仅在当前 shell 临时激活 Conda后重提。
- 该问题是提交封装/运行时路径缺陷，不是 checkpoint、资源、训练配置或科学假设失败。

#### 改进措施

- Formal submitter 与 F10 batch 脚本均显式解析并验证 `$GPU_ENV/bin/python`，所有 manifest/init 调用使用该绝对解释器，不再依赖交互 shell activation。
- 增加静态回归，要求 submitter 的 init/update 和 batch 的 running/final update 均使用 `PYTHON_BIN`。本地测试、远端 Bash syntax、无 Conda shell 下的解释器检查与 Slurm test-only 全部通过后，才以新 Job ID 重提 step 100。
- 存储峰值方面，veRL retention=1 已核实为新 checkpoint 成功保存后才删除旧 checkpoint；保留 CUDA/Triton/vLLM 编译缓存以避免减速，仅删除已确认无用且可再生的 69 MiB pip 下载缓存。模型、数据、F01/F02 结果、失败证据与 step-50 checkpoint 均保留。

### Formal F10 step-100 resume attempt 2 / Slurm 137588

#### 实验设置

- Run `experiments/f10_formal_20260903_stage19` 从 `global_step_50` 恢复到 total step 100；执行提交/生命周期代码为 Git `400794b`，训练代码、初始 manifest source/config digest 与 step-50 保持不变。
- Save/eval=`50/50`、retention=`1/1`、outcome GRPO、4x4 rollout、seed、LR、LoRA、长度、simulator、packed/chunked 路径、offload 和 caps `0.86/0.60` 全部冻结。Slurm 为同一物理节点 2x Pro 6000、2 tasks、8 CPU、180 GiB、12 小时，无 successor。

#### 执行结果

- 本地/远端 36 tests、Python compile、Bash syntax、三个同步文件 SHA-256、无 Conda shell 下的项目 Python 检查和 exact target=100 `sbatch --test-only` 均通过。
- Job `137588` 于 2026-09-03 11:49:18 UTC 提交，11:49:49 UTC 在 `gpu-pro6000-7` 启动；manifest 已原子更新为 `running`，allocation 记录物理 GPU indices `0,5` 与两个不同 GPU UUID。当前仍在 simulator/policy 初始化阶段，尚未产生 step 51 指标。

#### 改进原因

- Attempt 1 已证明裸 `python` 会让新 SSH shell 的提交/运行生命周期失败；attempt 2 必须同时验证登录端 submitted update 和 compute-node running/final update 均能通过项目绝对解释器执行。
- Step-50 的 42% 有效 outcome-gradient 比例满足继续 vanilla F10 的条件；当前没有切换 F13、修改超参数或迁移模型的证据。

#### 改进措施

- 运行期间冻结 Git `400794b` 对应执行脚本和所有训练设置，监测 checkpoint load、step 51--100 reward/advantage/gradient、双卡 telemetry、step-100 validation 与保存峰值。
- PASS 标准为从 step 50 加载 model/optimizer/extra，完成 100/100，无 NaN/OOM/schema error，成功写出 `global_step_100` 后自动删除 `global_step_50`，全项目仍仅 1 个完整 checkpoint。完成并记录前不提交 step 150 或 F11-F14。

## 结果记录要求

每个 run 单独保存 manifest、冻结配置、trajectory、训练指标、每个评测点的 CAR dev/BFCL 结果和失败样例；完整可恢复 checkpoint 在 SSD 仅保留最新 1 个，失败/被轮换 checkpoint 的小型元数据与结论继续归档，不删除 attempt 历史。
