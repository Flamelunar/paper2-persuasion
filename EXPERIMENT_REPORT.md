# CToMPersu 劝说实验完整清单（组会讨论版）

本报告整理 Zero-shot、MA²P 和 TrajWeaver-v1 的方法、数据、训练、评测协议、主实验、机制消融和失败原因。所有数字均来自当前工作区已经生成的结果文件；没有把未运行的实验写成结果。

## 1. 实验要回答的问题

当前工作的核心问题不是“模型能否生成一段礼貌的回复”，而是：

1. 在 Persuadee 每轮提出新顾虑时，Persuader 能否继续推进原始目标；
2. 系统能否利用公开对话轨迹判断 Persuadee 的 belief/desire 变化；
3. 记忆模块是否能提高严格的最终说服成功，而不是只让回复听起来更温和；
4. 系统是否会为了迎合最新顾虑，把原始目标偷偷替换成更容易接受的目标。

TrajWeaver-v1 的首要机制假设是：原始目标必须保持静态，Persuadee 的当前 belief/desire 必须动态更新；二者不能混成一个会被对话污染的目标表示。

## 2. 数据与数据泄漏处理

### 2.1 评测集

- 数据集：CToMPersu Eval。
- 规模：525 条评测记录，固定原始索引。
- 每个方法最多 4 个 Persuader 回合；成功判定后提前停止。
- 同一场景由相同的 `gpt-4o-mini` Persuadee 模拟器交互。
- Persuader 只能看到公开场景字段和已经发生的可见对话；不得看到 `preventive`、`generative`、`persona` 等私有字段。

### 2.2 Full 数据过滤

当前数据清单位于 [train/TrajWeaver-v1/data/manifest.json](train/TrajWeaver-v1/data/manifest.json)。原始 Full 数据有 6,257 行；与 Eval 的公开场景/记录匹配的 391 条被剔除后，保留 5,866 条非 Eval 场景，再按固定种子 42 划分为：

| Split | 场景数 | turn 样本数 | Eval 重合 |
| --- | ---: | ---: | ---: |
| Train | 5,279 | 19,209 | 0 |
| Dev | 587 | 2,135 | 0 |
| 合计 | 5,866 | 21,344 | 0 |

评测行数为 525，但其唯一公开场景/记录数为 391，因此过滤以公开场景身份和记录身份进行，而不是简单按行号截断。

### 2.3 实际 Teacher 训练规模

第一版实际训练没有直接使用全部 5,279 个训练场景，而是采用固定 pilot：

| Component | Train | Dev | turn 标签总数 |
| --- | ---: | ---: | ---: |
| Weaver SFT | 2,000 场景 | 300 场景 | 7,269 / 1,087 |
| Trigger SFT | 2,000 场景 | 300 场景 | 7,269 / 1,087 |

Teacher 标签共 8,356 个 turn，由当前 Codex 插件中的 `gpt-5.6-terra` 直接生成，记录为 `teacher_model=codex_gpt-5.6-terra`；这一步不是 ChatAnywhere/OpenAI API 调用，也没有使用 API key。标签文件为 [annotations.jsonl](train/TrajWeaver-v1/data/terra/annotations.jsonl)，覆盖率为 100%。

## 3. 三个系统的实现

### 3.1 Zero-shot

代码：[train/ctompersu_zero_shot.py](train/ctompersu_zero_shot.py)。每轮只给本地 Llama-3.1-8B-Instruct 一个 public-only prompt：公开场景、可见历史、剩余轮数和“直接写下一句 Persuader 回复”的指令。没有规划器、心理状态标签、latent memory 或 Trigger。输出经过 `gpt-4o-mini` Persuadee 反馈后进入下一轮，独立 Judge 每轮检查是否达到严格成功。

Zero-shot 的作用是衡量“冻结本地模型直接反应”这一简单基线，不应被解释成论文中所有可能的 prompt engineering 版本。

### 3.2 MA²P

代码：[train/ctompersu_ma2p.py](train/ctompersu_ma2p.py)。当前复现包括：

1. Configurator：可选地选择高层 meta-strategy；本实验使用冷启动 `K=0`，没有在线更新知识库；
2. Perception：根据公开场景和可见历史抽取 belief/desire 线索；
3. World Model：生成少于五项的策略计划；
4. Persuader：根据当前策略生成下一句；
5. Short-Term Memory：保存历史、感知结果和上一轮策略。

所有 MA²P 子模块仍复用同一个本地 Llama 模型，Persuader 侧仍只看 public-only 信息；因此它是 prompt-level multi-agent baseline，而不是额外训练出的模型。

### 3.3 TrajWeaver-v1

实现目录：[train/TrajWeaver-v1](train/TrajWeaver-v1)，方法说明见 [method.md](train/TrajWeaver-v1/method.md)。

```text
原始公共目标 ── Weaver LoRA ──> G：8 个静态 Goal tokens（每个对话只编码一次）
公开场景 + 可见历史 ─ Weaver LoRA ──> B：4 个 belief tokens
                                      └─> D：4 个 desire/intention tokens
Trigger LoRA ──> SKIP / INVOKE
冻结 Reasoner <── [G; B; D] ──> Persuader 回复
```

具体参数和职责：

- backbone：本地 `/data1/liujianjian/model/Meta-Llama-3.1-8B-Instruct`，4-bit 加载；
- Reasoner：冻结，不训练；
- Weaver：独立 LoRA（rank 16、alpha 32），另训练 latent queries、projection、controller 和辅助状态头；
- Trigger：独立 LoRA（rank 8、alpha 16）和二分类 `SKIP/INVOKE` head；不更新 Weaver 或 Reasoner；
- 静态记忆：8 个 Goal tokens，只读取原始公开目标，同一对话内缓存不变；
- 动态记忆：4 个 Belief + 4 个 Desire/Intention tokens，每轮根据完整可见历史重新计算；
- 当前训练目标：gold response likelihood + route/action/belief/desire/alignment 辅助损失；没有 GRPO；DPO 代码存在但本次没有使用。

训练阶段和实际选择：

| 阶段 | 计划轮数 | 实际状态 |
| --- | ---: | --- |
| Weaver warm-up | 2 epochs | 已完成 |
| Weaver trajectory | 3 epochs | 已完成；`trajectory-epoch-2` dev loss 最低（1.505880） |
| Trigger SFT | 2 epochs | 已完成；`trigger-epoch-1` dev loss 0.011749，优于 epoch-2 的 0.011976 |

因此当前完整结果使用 `trigger-epoch-1`，不是 `trigger-epoch-2`。两个 epoch 都已保存，epoch-1 是按 dev loss 选择的 checkpoint，不是训练中断。

## 4. 统一评测协议与指标

统一设置：Persuadee=`gpt-4o-mini`；独立 Judge=`gpt-5.6-luna`；Judge prompt=`zero_shot_joint_quality_v3`；最大 4 轮；相同 525 条 Eval 记录。

主表保留以下互补指标：

| 指标 | 方向 | 定义 |
| --- | --- | --- |
| Success (%) | ↑ | Judge 在不超过 4 轮内确认 Persuadee 明确接受、打算采取或已开始执行原始目标；礼貌、模糊 maybe、仅表示想了解不算成功 |
| Acceptance Score (%) | ↑ | Persuadee 最终 0–5 acceptance level 除以 5 后的百分比，反映未跨过严格成功阈值但态度改善的样本 |
| Avg_Turn | ↓ | 成功取首次成功轮；失败取最大轮数 |
| ACR (%) | ↑ | 是否产生明确、目标相关、符合约束且可执行的行动承诺 |
| UNF (%) | ↑ | 是否回应 Persuadee 已公开表达的需求、顾虑和约束 |
| Groundedness (%) | ↑ | 关键断言/前提有场景或对话依据，且不与已知信息矛盾 |
| Goal Drift Rate (GDR, %) | ↓ | 一条对话中只要有一次实质目标稀释、替代、反向支持或部分目标坍缩，就计为偏移；合理小分支不计 |

`Persuasive/Logic/Helpful` 没有放入当前主表；`Persuadee` 和 `Judge` 作为表下注释而不是列。`acceptance_turn` 是模拟器自报的较宽接受信号，不等于独立 Judge 的严格 Success，所以两者不要求逐条一致：已有 16 条 Zero-shot、9 条 TrajWeaver 记录是 Judge 成功但 `acceptance_turn=null`；另有不少 `acceptance_turn` 非空但严格 Success=false。

## 5. 525 条主实验结果

完整主表：[results/tables/metric_allocation.md](results/tables/metric_allocation.md)。

| Model | Method | N | Max turns | Success (%) ↑ | Acceptance Score (%) ↑ | Avg_Turn ↓ | ACR (%) ↑ | UNF (%) ↑ | Groundedness (%) ↑ | GDR (%) ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | Zero-shot | 525 | 4 | 39.24 | 71.24 | 3.7886 | 26.67 | 81.71 | 61.00 | 14.10 |
|  | MA²P | 525 | 4 | 32.76 | 65.30 | 3.8610 | 20.00 | 69.67 | 62.14 | 17.52 |
|  | TrajWeaver-v1 | 525 | 4 | 38.48 | 69.60 | 3.8476 | 26.29 | 75.19 | 60.14 | 8.00 |
| gemma-4-E4B-it | Zero-shot | 525 | 4 | 26.48 | 60.19 | 3.7581 | 16.00 | 76.76 | 85.29 | 15.81 |
|  | MA²P | 525 | 4 | 28.57 | 64.04 | 3.7695 | 18.10 | 76.76 | 83.48 | 15.24 |
|  | TrajWeaver-v1 | 525 | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 5.1 Llama 内部比较

相对于 Llama Zero-shot，TrajWeaver-v1：

- Success：`-0.76 pp`（39.24 → 38.48）；
- Acceptance：`-1.64 pp`（71.24 → 69.60）；
- ACR：`-0.38 pp`；
- UNF：`-6.52 pp`，95% paired bootstrap CI `[-8.14, -4.86]`；
- Groundedness：`-0.86 pp`；
- GDR：`-6.10 pp`（14.10 → 8.00），95% paired bootstrap CI `[-9.33, -2.86]`；
- Success 的 paired bootstrap CI 为 `[-5.33, +3.81]`，不能说明成功率有提升；McNemar 配对计数为 Zero-shot-only=77、TrajWeaver-only=73。

因此当前方法的真实结论是：目标保持能力变好，但说服效果没有变好；同时对用户需求的满足程度下降。

## 6. 100 条固定机制消融

完整报告：[ablation_report.md](results/Meta-Llama-3.1-8B-Instruct/ablations/trajweaver-mechanism-100/ablation_report.md)。使用 Eval 固定 100 条子集，seed=`20260910`，selection hash=`849bd4bd1111ca49e931b722862fd838c981555f5bf17e1aab61341a4e3af78d`。这些结果是机制诊断，不替代 525 条主实验。

| Variant | Success (%) | Acceptance (%) | ACR (%) | UNF (%) | Groundedness (%) | GDR (%) ↓ | Mean output tokens/turn | “I understand” (%) | Trigger INVOKE (%) | Memory active (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| public-zero-shot | 41.00 | 71.20 | 26.00 | 82.25 | 58.25 | 10.00 | 58.86 | 14.92 | 0.00 | 0.00 |
| legacy-v1 | 37.00 | 69.00 | 28.00 | 75.75 | 60.50 | 5.00 | 47.73 | 54.95 | 73.96 | 100.00 |
| r0-no-memory | 35.00 | 71.40 | 27.00 | 79.25 | 56.50 | 9.00 | 63.91 | 15.00 | 0.00 | 0.00 |
| g8-static-goal | 29.00 | 62.20 | 25.00 | 71.75 | 62.25 | 17.00 | 43.94 | 5.91 | 74.29 | 100.00 |
| g8bd8-always | 36.00 | 69.40 | 18.00 | 76.75 | 59.25 | 7.00 | 48.01 | 52.97 | 100.00 | 100.00 |
| g8bd8-trigger-strict | 29.00 | 66.40 | 19.00 | 74.25 | 57.00 | 10.00 | 57.22 | 44.27 | 73.96 | 73.96 |

消融的版本含义：

- `public-zero-shot`：公共 Zero-shot 结果过滤到同一 100 条索引；
- `legacy-v1`：主实验使用的旧记忆路径，Trigger=SKIP 时仍插入静态 Goal 和 learned null-state，因此不是零干预；
- `r0-no-memory`：明确不插入任何 latent memory 的控制组；
- `g8-static-goal`：只插入 8 个静态 Goal tokens；
- `g8bd8-always`：每轮插入 8 Goal + 4 Belief + 4 Desire；
- `g8bd8-trigger-strict`：只有 Trigger=INVOKE 时插入 16 tokens，SKIP 时真正回退为零干预。

100 条结果没有任何变体带来可靠的 Success 提升。`g8-static-goal` 最差；`g8bd8-always` 的 Success 仅比 `r0-no-memory` 高 1 pp，但 ACR 低 9 pp；strict Trigger 已完成正确的零干预实现，但 Success 仍只有 29%，说明问题不只是 SKIP 路由。

## 7. 训练和推理诊断

### 7.1 Trigger 学到的是轮次位置，而不是“记忆是否有益”

Teacher 标签按 turn 的分布为：

| Turn | SKIP | INVOKE |
| ---: | ---: | ---: |
| 1 | 2,229 | 71 |
| 2 | 3 | 2,297 |
| 3 | 0 | 2,300 |
| 4 | 1 | 1,455 |

训练标签几乎等价于“第一轮 SKIP，后续 INVOKE”。训练 dev accuracy 达到 99.91%，但这更像位置规则拟合，而不是学到了 memory utility。

在 525 条 TrajWeaver 评测中：

- turn 1：521 SKIP、4 INVOKE；
- turn 2：525 INVOKE；
- turn 3：519 INVOKE；
- turn 4：451 INVOKE；
- 总计 1,499 次 INVOKE、521 次 SKIP；每条对话都至少触发过一次 INVOKE，521 条对话至少出现一次 SKIP。

### 7.2 Controller 退化为固定模板

全量 TrajWeaver 2,020 个已生成 turn 的动作统计：

| Action | 次数 |
| --- | ---: |
| PROBE | 525 |
| REPAIR | 1,159 |
| CONTINUE | 328 |
| COMMIT | 8 |
| REPLAN | 0 |

因此首轮 100% 是 `PROBE`，第二轮几乎固定 `REPAIR`，真正的 `COMMIT` 只有 8 次。输出没有把控制器动作可靠地转成语言层策略；action 目前只是 soft embedding，不能直接强制“回应顾虑—给具体证据—提出可执行承诺”。

状态头也有明显塌缩：2,020 个 turn 全部预测 `goal_alignment=ON_GOAL`，但外部 GDR Judge 仍发现 42/525 条对话发生目标偏移。说明 alignment head 的训练输出与最终语言行为脱节。

### 7.3 输出变短并模板化

| 方法 | 平均 output tokens/turn | “I understand” 占比 |
| --- | ---: | ---: |
| Zero-shot | 59.65 | 15.08% |
| TrajWeaver-v1 | 47.73 | 53.07% |

TrajWeaver 在首轮平均只有 30.49 output tokens，而 Zero-shot 为 53.30；第二轮 TrajWeaver 的 “I understand” 占比达到 81.3%。这与 Judge 对 UNF 下降的结论一致：系统更常用泛化确认和低风险措辞替代具体事实、例子、条件、成本、风险控制和行动安排。

### 7.4 Teacher 标签并不支持复杂的状态建模

8,356 个 teacher turn 标签的分布：

- `goal_alignment`：ON_GOAL 8,026，VALID_SUBSTEP 214，DILUTED 113，REVERSED 3；
- `trigger_action`：INVOKE 6,123，SKIP 2,233；
- `desire_state`：RELUCTANT 2,226，UNDECIDED 2,209，WILLING 2,159，RESISTANT 1,751，COMMITTED 11；
- `belief_state`：DOUBTFUL 2,371，UNKNOWN 2,243，REJECTS_PREMISE 2,002，MIXED 1,629，ACCEPTS_PREMISE 111；
- `route_state`：ON_ROUTE 5,914，REPAIRABLE 2,182，TERMINAL_READY 258，STRUCTURAL_FAILURE 2。

尤其是 `REVERSED=3`、`COMMITTED=11`，使模型几乎没有机会学习“真正的目标偏移”或“明确收束”这一关键边界。与此同时，Teacher turn 是 gold human trajectory，而线上 Eval 是本模型生成一轮后再与 `gpt-4o-mini` 交互，模型没有充分看到“自己的错误回复造成什么后果、下一轮如何修复”的训练样本。

## 8. 为什么 GDR 变好但 Success 没变好

这不是矛盾，而是当前目标函数和能力瓶颈共同造成的结果：

1. **模型学会了守住目标字符串，却没有学会完成目标。** 静态 Goal token 给了模型较强的“不要改目标”锚点，所以 GDR 从 14.10% 降到 8.00%；但目标一致不等于有证据、有行动方案、有承诺。
2. **记忆注入替代了语言规划。** 16 个 latent tokens 被直接拼到 Reasoner 输入，Reasoner 仍是冻结的；它没有被训练成“从这个状态向量生成更具体的说服策略”，因此 latent 控制更像额外噪声/风格条件。
3. **Trigger 触发标准错误。** 第一轮几乎总 SKIP、第二轮以后几乎总 INVOKE，模型没有判断“本轮 memory 是否带来增益”。这解释了为什么 strict trigger 修正后仍没有提升。
4. **动作监督和最终指标不对齐。** loss 主要是 gold response likelihood 和分类损失，不直接优化 Success、ACR、UNF、最终 commitment 或 Judge 的严格成功。
5. **训练输出分布过于模板化。** 高比例的 “I understand”、较短回复和低 `COMMIT` 率使 Persuadee 觉得“被理解”，但不一定愿意采取原始目标。
6. **Teacher 数据缺少 hard negatives 和反事实。** 标签几乎没有反向/替代样本，alignment head 于是对所有生成都预测 ON_GOAL；模型没有学到哪些看似礼貌的句子其实已经把目标换掉。
7. **训练—评测分布偏移。** 训练看到的是 Full 数据中的 gold dialogue；评测中历史由本模型产生，并由外部模拟器即时反馈。模型遇到自己的泛化错误时没有足够的 repair 轨迹。
8. **Goal specification 太弱。** 当前静态 Goal 主要编码原始自然语言目标，没有充分编码排除选项、必须保留的组成部分、允许的小步骤和不可接受的替代终点。

## 9. 组会讨论清单

### 需要先确认的事实

- 当前主实验 TrajWeaver 行是否应标为 `legacy-v1`，还是重新用 strict Trigger 跑 525 条；两者不能合并成一个方法名。
- 论文的核心 claim 是“目标保持”还是“最终劝服能力提升”。现有结果只支持前者，不能声称后者。
- `gpt-5.6-luna` Judge 的自动评分是否需要人工子集或第二个 Judge 校准。
- 4-turn protocol 与原论文 6-turn protocol 的差异是否需要在论文中单独做敏感性实验。

### 优先级 P0：先做机制定位（每个只跑固定 100 条）

1. strict zero-intervention 与 `r0-no-memory` 做完全一致的 paired run，确认 SKIP 是否真正回退到 Zero-shot；
2. 加入 `dynamic-only`、`static-only`、`frozen-controller` 和随机 Trigger 控制组；
3. 按首轮/后续轮、弱/中/强抵抗、领域和是否最终接受分层报告；
4. 对每个变体同时记录 Success、ACR、UNF、GDR、回复长度、Trigger 调用率和 action 分布；
5. 将 alignment head 的预测与外部 GDR Judge 逐条对照，确认 head 是否可用。

### 优先级 P1：若确认机制有效，再改训练

1. 重新构造 balanced teacher：提高 VALID_SUBSTEP、DILUTED、REVERSED、COMMITTED 和真实 repair turn 的比例；
2. 从模型自身 rollout 构造“具体回应/泛化确认”“守住原目标/替换目标”的 chosen-rejected 对，先做离线 DPO；
3. 让 Goal token 编码结构化规范：原始目标、排除选项、优先级、必须保留组成部分、合法小步骤、不可替代终点；
4. 把 action 从 soft embedding 改成可执行的生成约束或显式策略模板，增加具体证据、风险控制、下一步和 commit 的监督；
5. 对 `I understand`、重复理由、无具体数字/例子/计划的回复做负例或质量约束，但不能简单用长度奖励替代说服质量；
6. 训练时加入模型自身生成历史，让 Weaver 学习在错误后修复，而不是只拟合 gold trajectory。

### 优先级 P2：完成论文级证据

- 用完整 525 条 Eval 重跑最终配置；
- 对二元 Success 报 paired bootstrap 和 McNemar；
- 对 GDR、ACR、UNF 报逐条审计样本和人工盲评子集；
- 报告不同 persuadee 抵抗强度和宏平均领域成功率；
- 报告推理 token、调用次数和 latency，证明 memory 机制的成本；
- 不把 100 条诊断消融当作主结果，也不把 GDR 改善直接解释成说服能力改善。

## 10. 复现入口与产物

- 主指标表：[results/tables/metric_allocation.md](results/tables/metric_allocation.md)
- Llama Zero-shot raw/aligned：[results/Meta-Llama-3.1-8B-Instruct/raw/zero-shot.jsonl](results/Meta-Llama-3.1-8B-Instruct/raw/zero-shot.jsonl)、[aligned/zero-shot.json](results/Meta-Llama-3.1-8B-Instruct/aligned/zero-shot.json)
- Llama MA²P raw/aligned：[results/Meta-Llama-3.1-8B-Instruct/raw/MA2P.jsonl](results/Meta-Llama-3.1-8B-Instruct/raw/MA2P.jsonl)、[aligned/MA2P.json](results/Meta-Llama-3.1-8B-Instruct/aligned/MA2P.json)
- Llama TrajWeaver raw/aligned：[results/Meta-Llama-3.1-8B-Instruct/raw/TrajWeaver-v1.jsonl](results/Meta-Llama-3.1-8B-Instruct/raw/TrajWeaver-v1.jsonl)、[aligned/TrajWeaver-v1.json](results/Meta-Llama-3.1-8B-Instruct/aligned/TrajWeaver-v1.json)
- 盲评质量摘要：`results/Meta-Llama-3.1-8B-Instruct/quality/*-summary.json`
- 消融报告：[results/Meta-Llama-3.1-8B-Instruct/ablations/trajweaver-mechanism-100/ablation_report.md](results/Meta-Llama-3.1-8B-Instruct/ablations/trajweaver-mechanism-100/ablation_report.md)
- 训练日志：`train/TrajWeaver-v1/logs/`
- 单元测试：`python -m unittest discover -s train/TrajWeaver-v1/tests -v`（当前 12/12 通过）

### No-fabrication status

本报告只使用已存在的 525 条主实验、100 条固定消融、训练日志和标签统计。gemma TrajWeaver 主表的 TBD 没有从其他模型或部分运行结果推断，仍然保留 TBD。
