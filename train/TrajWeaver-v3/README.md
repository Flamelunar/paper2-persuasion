# TrajWeaver-v3

v3 是基于 v1 思路的精简实现：保留静态目标与动态记忆，通过现有对话学习 Weaver，
再自动比较记忆的预测收益来学习 Trigger。代码独立存放，不依赖 v1/v2 的模型类或标签。

## 只保留两份数据

| 数据 | 来源与工作量 | 用途 |
| --- | --- | --- |
| Public response SFT | 自动转换 `data/CToMPersu/mysplit`，无需新增标注 | 训练 Weaver |
| Paired memory utility | SFT 后自动打分，默认训练 512 个 prefix、开发 128 个 prefix | 训练 Trigger |

第二份数据共 640 个前缀，每个前缀比较两个条件，共 1,280 次条件评分。
不生成新对话、不调用 Teacher、Simulator 或 Judge，也不要求人工提供 hidden states。
不再单独制作目标结构、心理状态转移、目标漂移偏好对或 token-level trigger 数据。

MemGen 原论文本身也不要求人工标注触发位置。它使用随机插入训练 Weaver，
再使用轨迹奖励和稀疏惩罚训练 Trigger。这里用离线参考回复损失差替代在线 RL 的采样成本；
这是 v3 的工程取舍，不是 MemGen 原论文的训练算法。

## 模型

```text
公开原始 goal ── Weaver LoRA ── 8 Goal tokens（每段对话缓存一次）
                                         │
公开对话历史 ── 冻结 Reasoner ── hidden-state hook
                                  │      │
                                  │      └─ Weaver LoRA + Goal ── 8 State tokens
                                  └─ 小型 MLP Trigger
                                         │
                       SKIP:   [Goal]
                       INVOKE: [Goal; State]
                                         │
                               冻结 Reasoner → 下一轮回复
```

- **静态目标**：只从原始公开 `goal` 编码，不读取后续历史；推理时缓存。
- **动态记忆**：由冻结 Reasoner 的公开前缀隐藏状态触发，经可学习投影、Weaver LoRA 和 queries 生成。
- **Trigger**：读取同一 hook 的最后状态，使用小型 MLP 二分类。采用轮次级决策，每条回复前判断一次。
- **真正的 SKIP**：只保留 Goal，不计算 State，也不放入 learned null-state tokens。
- **Reasoner 冻结**：回复预测与生成时关闭 LoRA；SFT 梯度仍通过其计算传到 memory 和 Weaver。
- **训练隔离**：SFT 只更新 Weaver LoRA、queries、投影；Trigger 阶段只更新 MLP。
- 默认 `llama31_8b.yaml` 是 4-bit NF4 QLoRA，主干计算 dtype 为 BF16；若要与 full-BF16 基线保持一致，使用 `configs/llama31_8b_bf16.yaml`（`load_in_4bit: false`）。

动态 8 tokens 统称 State Memory。没有状态标签时，不声称前四个一定表示 Belief、后四个一定表示 Desire。
移除了 v1 的 route/action/belief/desire/alignment 分类头与相应的标签覆盖率门槛。
因此 v1/v2 checkpoint 不能直接加载为 v3。

这是 **turn-level MemGen adaptation**。没有实现生成中途在标点位置插入 memory，
也没有实现 GRPO、DPO 或 continual learning。当前工作应验证的是：
生成式动态记忆在保留原目标的条件下，是否对回复有帮助，是否能够选择性使用。

## 训练目标

设 `L_skip` 为 `[Goal]` 条件下参考回复的平均 token NLL，
`L_invoke` 为 `[Goal; State]` 条件下同一回复的平均 token NLL。

```text
Weaver SFT:
    loss = (L_invoke + 0.25 * L_skip) / 1.25

固定 Weaver，生成 Trigger 标签：
    gain = L_skip - L_invoke
    net_gain = gain - cost
    net_gain > 0: INVOKE
    net_gain < 0: SKIP
    abs(net_gain) <= margin: 不用于 Trigger 训练

Trigger:
    cross_entropy(MLP(frozen_hook), trigger_label)
```

两种输入条件都在 SFT 中训练，避免训练只见过完整记忆而部署突然去掉 State。
标签生成在 `eval()` 和 `no_grad()` 下进行，两次评分使用同一目标文本，hook 只看当前公开历史。
参考回复用于计算离线标签，不作为 Trigger 输入；效用数据也不保留参考回复正文。

默认 `cost=0.01`、`margin=0.002`，单位均为 **nats / target token**。
cost 是调用偏好惩罚，不是测得的运行时间；margin 是标签过滤带，不是统计置信区间。
只能在开发集调整它们。若标签几乎全是同一类，先检查 `summary.json`；
不人为伪造另一类标签，也不以分类准确率证明 Trigger 有效。

SFT 根据开发集 loss 选择 checkpoint。Trigger 根据开发集 `utility_regret` 选择，
平局时比较分类 loss，同时报告 always-SKIP 与 always-INVOKE 的 regret。
效用 regret 定义为 `max(0, net_gain) - decision * net_gain`。

NLL 收益只表示更容易预测数据里的参考回复，**不等于真实说服成功率或目标保真度**。
原始回复质量会影响这个信号。v3 是否提升说服效果，仍需下游独立评测，不能从训练 loss 推断。

## 已转换数据

`data/sft/manifest.json` 记录原始文件 SHA256、划分、续行修复和去重数量。

| 划分 | 独立公开场景 | 回复样本 |
| --- | ---: | ---: |
| train | 5,341 | 19,434 |
| dev | 525 | 1,910 |
| test | 391 | 1,395 |

- 保留 `mysplit` 的场景归属，转换前检查 train/dev/test 场景交集。
- `preventive/generative/persona` 不进入训练、Trigger、Goal 编码器或公开输出。
- 同一 utterance 的无角色前缀续行合并到上一条消息；43 元素记录也正常解析，无需删除。
- 测试原始 525 行中有 134 个重复公开场景，v3 按 391 个独立场景评测。
- `data/sft/test.jsonl` 仅供检查或本地预测；训练和效用脚本只读取 train/dev。
- 标签与 checkpoint SHA256、数据 SHA256 和完整配置绑定。换 Weaver 后必须重新打分。

## 可选的 500 条 planner 标注

如果需要训练或检查一个独立的目标—顾虑 planner，可用
`generate_planner_data.py` 调用 `gpt-5.6-luna` 批量生成少量标注：

```bash
CHATANYWHERE_API=... \
/home1/liujianjian/anaconda3/envs/ljj/bin/python \
  train/TrajWeaver-v3/generate_planner_data.py
```

默认从 train 固定抽取 200 个场景、从 dev 抽取 50 个场景；每个场景生成
`initial` 和首个明确 persuadee 顾虑前缀各 1 条，共 500 条。每次请求输入 10
个场景并返回 20 条标注，默认计划 25 次请求；输入过长时会自动拆小。脚本会
断点跳过已校验批次，并将失败批次留在 `planner_quarantine.jsonl`，不会覆盖
`data/CToMPersu/mysplit`。

输出位于 `artifacts/`：

| 文件 | 内容 |
| --- | --- |
| `planner_raw_train.jsonl` / `planner_raw_dev.jsonl` | 400/100 条带 public prefix 的结构化标注 |
| `planner_sft_train.jsonl` / `planner_sft_dev.jsonl` | 同一批数据的 messages SFT 格式 |
| `planner_batches.jsonl` | 每次请求的模型、提示哈希、原始响应和校验状态 |
| `manifest.json` / `summary.json` | 采样、数据哈希、数量和完成状态 |

这些 JSON planner target 不是 persuader 回复，不能直接替代 `data/sft` 中的
参考回复去训练 Weaver；当前 v3 的 Weaver 仍使用 public response SFT，Trigger
仍使用 paired reference-NLL 自动标签。planner 文件可作为独立 controller/planner
的训练或离线分析输入。如果希望让它只影响 Trigger 的样本选择，可在效用阶段
显式指定 concern prefix；planner JSON 本身不会进入 NLL 或 Trigger 输入：

```bash
CUDA_VISIBLE_DEVICES=2 /home1/liujianjian/anaconda3/envs/ljj/bin/python \
  train/TrajWeaver-v3/run_pipeline.py \
  --config train/TrajWeaver-v3/configs/llama31_8b.yaml \
  --run-dir train/TrajWeaver-v3/runs/llama31_8b_500 \
  --train-prefixes 200 --dev-prefixes 50 \
  --planner-data train/TrajWeaver-v3/artifacts \
  --planner-prefix-type concern
```

这会把 200/50 条 concern 前缀映射回原始 public SFT 行，再由 Weaver checkpoint
对同一条参考回复计算 Goal-only 与 Goal+State 的 NLL；因此 500 条标注用于
关注心理顾虑的样本覆盖，Trigger 标签仍是本地 paired-NLL 标签。

## 一条命令运行

环境沿用项目的 `ljj`：PyTorch、Transformers、PEFT、bitsandbytes、PyYAML。
Llama 和 Qwen 配置均在训练与推理时使用 4-bit，同一个效用实验保持一致的数值配置。

```bash
cd /home1/liujianjian/2-paper-Coling-v2

CUDA_VISIBLE_DEVICES=0 /home1/liujianjian/anaconda3/envs/ljj/bin/python \
  train/TrajWeaver-v3/run_pipeline.py \
  --config train/TrajWeaver-v3/configs/llama31_8b.yaml \
  --run-dir train/TrajWeaver-v3/runs/llama31_8b
```

默认先使用 1,000 个训练场景、200 个开发场景做 SFT，再从原 train/dev 划分分别抽取
512/128 个场景，每个场景随机选择一个 prefix 生成效用标签。后一步允许覆盖 SFT 子集以外的训练场景。
两种规模独立控制，并记录采样 ID。全量 SFT 加 `--train-scenarios 0 --dev-scenarios 0`。
Qwen 换成 `configs/qwen25_7b.yaml` 和另一 `--run-dir`。

程序依次运行两轮 SFT、效用打分、三轮小型 Trigger 训练，自动选择各阶段 checkpoint。
最终路径打印到终端，同时保存在 `runs/<name>/trigger/best.json`。
完整训练不会在创建代码时自动启动；这条命令会使用选定 GPU。

## 分步运行与中断恢复

数据转换可单独运行：

```bash
/home1/liujianjian/anaconda3/envs/ljj/bin/python train/TrajWeaver-v3/prepare_data.py
```

各阶段入口：

```text
train.py --stage sft --config CONFIG --data-dir SFT_DIR --output WEAVER_DIR
build_trigger_data.py --config CONFIG --checkpoint SFT_CHECKPOINT --data-dir SFT_DIR --output UTILITY_DIR
train.py --stage trigger --config CONFIG --checkpoint SFT_CHECKPOINT --data-dir UTILITY_DIR --output TRIGGER_DIR
```

`CONFIG` 等大写项需换成实际路径。SFT_CHECKPOINT 取自 `WEAVER_DIR/best.json`。
效用打分可以原命令重跑，自动跳过已完成前缀；配置或 checkpoint 不一致时拒绝混合结果。
训练保存每个 epoch 的权重；当前未保存优化器状态，不支持精确训练续跑。
训练输出必须为空，避免重跑覆盖已有权重。

## 推理与对照

```text
predict.py --config CONFIG --checkpoint FINAL_CHECKPOINT --input data/sft/dev.jsonl --mode trigger
run_eval.py --config CONFIG --checkpoint FINAL_CHECKPOINT --output RESULT.jsonl --mode trigger
```

命令应使用 `train/TrajWeaver-v3/` 下的脚本路径。所有 Python 入口都支持 `--help`。

`predict.py` 完全本地。`run_eval.py` 使用项目现有固定 Persuadee API 和最终 Judge，
每个独立场景固定四轮，第一轮已接受也不提前停止；只在最后调用 Judge。
Simulator 可以读取原始私有字段，Persuader 和 Judge 沿用项目公开信息边界。
默认评测按 public scenario 去重；若要与主表的原始 525 行口径一致，使用
`run_eval.py ... --preserve-rows`，结果会为每个原始行保存唯一 `evaluation_id`，
包括 test 中重复的 public scenario。
评测继承 `CTOMPERSU_*` API 配置，实际运行会发生外部调用；代码交付过程未调用这些服务。

当 full-BF16 模型需要使用两张 24 GB 卡时，应让每张卡加载一份完整模型并切分测试行，
而不是用 `device_map` 把一个模型拆到两张卡。`--start-index` 包含、`--end-index` 不包含：

```bash
CUDA_VISIBLE_DEVICES=0 python train/TrajWeaver-v3/run_eval.py \
  --config train/TrajWeaver-v3/configs/llama31_8b_bf16.yaml \
  --checkpoint train/TrajWeaver-v3/trajweaver_v3/checkpoints/llama31_8b_500 \
  --dataset data/CToMPersu/mysplit/test.json \
  --output results/Meta-Llama-3.1-8B-Instruct/raw_shards/part0.jsonl \
  --mode trigger --preserve-rows --start-index 0 --end-index 263

CUDA_VISIBLE_DEVICES=1 python train/TrajWeaver-v3/run_eval.py \
  --config train/TrajWeaver-v3/configs/llama31_8b_bf16.yaml \
  --checkpoint train/TrajWeaver-v3/trajweaver_v3/checkpoints/llama31_8b_500 \
  --dataset data/CToMPersu/mysplit/test.json \
  --output results/Meta-Llama-3.1-8B-Instruct/raw_shards/part1.jsonl \
  --mode trigger --preserve-rows --start-index 263 --end-index 525

python train/TrajWeaver-v3/merge_eval_shards.py \
  --shard results/Meta-Llama-3.1-8B-Instruct/raw_shards/part0.jsonl \
          results/Meta-Llama-3.1-8B-Instruct/raw_shards/part1.jsonl \
  --output results/Meta-Llama-3.1-8B-Instruct/raw/TrajWeaver-v3.jsonl \
  --total 525
```

最小对照只需四种 `--mode`：

| mode | 记忆 | 用途 |
| --- | --- | --- |
| none | 0 tokens | 冻结基础 Reasoner |
| goal_only | 8 Goal tokens | 目标记忆对照 |
| always | 8 Goal + 8 State tokens | 动态记忆始终开启 |
| trigger | 8 或 16 tokens | v3 按效用选择动态记忆 |

建议四种模式使用同一个最终 checkpoint、同一个测试场景集合和相同 Simulator/Judge。
记录原目标成功率、调用率、延迟，并用项目现有评测检查目标漂移。
当前 final Judge 接口不产生目标漂移分数，不能把 `success` 当作 Goal Drift Rate。
v1 的成功早停结果、原始 525 行加权结果不能直接与 v3 的固定四轮去重结果混比。

## 验证

```bash
/home1/liujianjian/anaconda3/envs/ljj/bin/python \
  -m unittest discover -s train/TrajWeaver-v3/tests -v
```

测试覆盖 Llama/Qwen 梯度隔离、Goal 缓存、SKIP 不调用 State、无记忆与基础模型一致、
效用评分、checkpoint 完整性、数据泄漏与续行修复，以及 CPU 小模型的训练/打分/推理全流程。
评测流程测试用替身 Simulator/Judge 检查固定四轮、去重、缓存与续跑；不调用外部 API。
这些是功能验证，不是方法效果实验或 7B/8B 全量 GPU 训练结果。
