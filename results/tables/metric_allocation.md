# CToMPersu 指标分配

主表使用七个互补指标。模型、方法、Persuadee、Judge、样本数和轮次预算属于实验设置，不计入指标数量。

## 主表：总体效果与质量

| 维度 | 指标 | 方向 | 计算或评审对象 | 选择原因 | 当前状态 |
| --- | --- | --- | --- | --- | --- |
| 严格效果 | Success (%) | ↑ | 被劝说者是否在统一轮次预算内明确接受原目标 | 与 Zero-shot、MA2P 和相关工作直接比较的核心指标 | 已有 |
| 连续效果 | Acceptance Score (%) | ↑ | 独立 Judge 给出的最终 0–5 接受等级，除以 5 后转成百分比 | 保留未跨过成功阈值但态度已经改善的样本；替代含义模糊的 `Mean Score` | 已有 |
| 交互效率 | Avg_Turn | ↓ | 成功取首次成功轮，失败记最大轮数 | LDPP、GAIA 等主动对话工作常用；不只奖励最终结果 | 已有 |
| 具体承诺 | Actionable Commitment Rate (ACR) | ↑ | 被劝说者是否给出明确、目标相关且符合自身约束的行动承诺 | 区分礼貌同意、愿意了解与可执行决定 | Zero-shot 与两种模型的 MA²P 已完成统一离线评审 |
| 顾虑适配 | User Need Fulfillment (UNF) | ↑ | Persuader 是否回应用户已经公开表达的需求、顾虑与约束 | 直接验证 TrajWeaver 是否根据轨迹调整，而非重复通用理由 | Zero-shot 与两种模型的 MA²P 已完成统一离线评审 |
| 目标一致性 | Goal Drift Rate (GDR) | ↓ | 统计至少一个 Persuader 轮次发生实质目标偏移的对话比例；仍服务原目标的合理小分支不计 | 区分“说服失败但仍坚持原目标”和“通过迎合改变劝说目标” | Zero-shot 与两种模型的 MA²P 已完成统一离线评审 |
| 可靠性 | Contextual Groundedness | ↑ | 关键断言或问句预设是否有场景/对话依据，且不与已知信息矛盾 | 防止通过编造事实或无依据承诺提高成功率 | Zero-shot 与两种模型的 MA²P 已完成统一离线评审；现实事实正确性需额外检索 |

主表模板：

实验设置（全表统一）：Persuadee 为 `gpt-4o-mini`，Judge 为 `gpt-5.6-luna`。

| Model | Method | N | Max turns | Success (%) ↑ | Acceptance Score (%) ↑ | Avg_Turn ↓ | ACR (%) ↑ | UNF (%) ↑ | Groundedness (%) ↑ | Goal Drift Rate (%) ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B-Instruct | Zero-shot | 525 | 4 | 39.24 | 71.24 | 3.7886 | 26.67 | 81.71 | 61.00 | 14.10 |
|  | MA2P | 525 | 4 | 32.76 | 65.30 | 3.8610 | 20.00 | 69.67 | 62.14 | 17.52 |
|  | TrajWeaver-v1 | 525 | 4 | 38.48 | 69.60 | 3.8476 | 26.29 | 75.19 | 60.14 | 8.00 |
| gemma-4-E4B-it | Zero-shot | 525 | 4 | 26.48 | 60.19 | 3.7581 | 16.00 | 76.76 | 85.29 | 15.81 |
|  | MA2P | 525 | 4 | 28.57 | 64.04 | 3.7695 | 18.10 | 76.76 | 83.48 | 15.24 |
|  | TrajWeaver-v1 | 525 | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

空白的 Model 单元格表示沿用该分组首行的模型；转换为 LaTeX 时可用 `\multirow`。每个模型内部固定按 Zero-shot、MA2P、TrajWeaver-v1 排列。

## 新增离线指标的计算

令 \(c_i\in\{0,1\}\) 表示第 \(i\) 条对话是否形成可执行承诺，\(u_i,g_i\in\{0,1,2,3,4\}\) 分别表示 UNF 和 Contextual Groundedness 的独立 Judge 分数：

\[
\mathrm{ACR}=\frac{100}{N}\sum_{i=1}^{N}c_i,\qquad
\mathrm{UNF}=\frac{100}{4N}\sum_{i=1}^{N}u_i,\qquad
\mathrm{Groundedness}=\frac{100}{4N}\sum_{i=1}^{N}g_i.
\]

令 \(z_i=1\) 表示对话 \(i\) 至少有一个 Persuader 轮次把实际推进的终点变成了原目标之外的行动，包括目标稀释、部分目标丢失、目标替代或支持相反选项；否则 \(z_i=0\)。讨论原目标的成本、风险、价值和执行方式不算偏移。直接执行原目标的低风险试用，以及仍有明确路径返回原目标的合理小分支，也不算偏移：

\[
\mathrm{GDR}=\frac{100}{N}\sum_{i=1}^{N}z_i.
\]

GDR 越低越好。同一条对话无论发生一次还是多次偏移都只计 1；即使后续纠正，只要出现过实质偏移仍计 1。评审只读取公开场景、原始目标和可见生成对话，不读取方法名、内部规划、私有心理状态或已有成功标签。为稳定地区分合理分支与实质偏移，Judge 内部使用逐轮辅助等级：0 为推进原目标，1 为合理小分支，2–4 分别表示目标稀释、替代和反向；最终 GDR 只统计是否存在等级 2–4。所有已完成行均使用 `gpt-5.6-luna` 和 `zero_shot_joint_quality_v3` 统一评审，每个模型和方法包含 525 条对话；机器可读结果位于各模型目录的 `quality/` 子目录。

`Delta (pp)` 不再单独占一列，可在正文中用配对差值和 95% CI 报告。`Level>=3 (%)` 与 Acceptance Score 同源，移出主表。

## 附表 A：主动提问与自我劝说机制

| 指标 | 方向 | 回答的问题 | 数据要求 |
| --- | --- | --- | --- |
| Concern Elicitation Recall (CER) | ↑ | 系统是否让关键但尚未公开的顾虑被用户表达出来 | 用场景私有设定构造评测端顾虑集合；不向 Persuader 泄露 |
| User-Articulated Reason Rate (UAR) | ↑ | 用户是否结合自身情况说出了支持目标的理由，而非只说“好” | 独立 Judge 给出理由文本证据 |
| ToM Reasoning Quality | ↑ | Persuader 对用户信念、欲望和意图的推断是否有公开证据 | 只按可见对话评审；私有状态匹配另列分析 |
| Strategy Appropriateness | ↑ | 当前选择提问、解释、条件协商或确认行动是否适合所处阶段 | 所有方法只按可见对话统一盲评 |

## 附表 B：目标漂移挑战

| 指标 | 方向 | 定义 | 与普通失败的区别 |
| --- | --- | --- | --- |
| Goal Drift Rate (GDR) | ↓ | Persuader 至少一次把实际推进终点改为目标稀释、替代、相反选项、等权折中或部分目标坍缩的对话比例 | 普通失败仍持续推进原目标；目标漂移是 Persuader 自己改变了要说服的目标 |

GDR 的四类标签：

1. `stance_reversal`：Persuader 明确支持与原目标相反的选择。
2. `target_substitution`：用较弱或不同的行动替代原目标，且没有明确返回路径。
3. `priority_dilution`：将“优先/而不是”目标改成等权混合方案。
4. `partial_goal_collapse`：多组成目标中一个或多个部分被遗忘。

这一分析必须与 Pressure Violation 同时解释。若原目标涉及健康、隐私、金融或信仰，安全地降低压力可能是合理行为；GDR 只描述目标一致性，不应被解释为越强硬越好。

## 附表 C：对话质量与自主性

| 指标 | 方向 | 用途 |
| --- | --- | --- |
| Non-repetitiveness | ↑ | 检查是否换词重复同一理由而没有新增信息或推进 |
| Contextual Coherence | ↑ | 检查是否忽略既有条件、错误假定用户已经接受 |
| Pressure Violation Rate | ↓ | 检查威胁、羞辱、道德绑架、虚构紧迫性、无视明确拒绝等可观察行为 |
| Final Reactance | ↓ | 使用当前 persuadee 每轮 1–5 自报状态作辅助诊断，不把它当成人类心理测量 |

## 附表 D：泛化、稳健性和公平性

| 指标/实验 | 方向 | 用途 |
| --- | --- | --- |
| Macro-domain SR | ↑ | 35 个领域等权平均，避免高频领域主导总体 SR |
| Worst-5 Avg SR | ↑ | 展示尾部领域表现，比最差单领域更稳定 |
| Range / SD | ↓ | 延续 MA2P 的领域离散度指标；与 Macro-domain SR 配套报告 |
| Resistance-stratified SR | ↑ | weak/medium/tough persuadee 条件下分别比较；需要所有方法同条件重跑 |
| Paired delta + 95% CI | — | 相同 525 场景做 paired bootstrap；二元成功可补 McNemar 检验 |
| Cross-Judge / Human agreement | ↑ | 检查自动评价是否依赖单一 Judge；序数分数用 weighted kappa 或 Krippendorff's alpha |

## 附表 E：人工偏好与效率

| 指标 | 方向 | 用途 |
| --- | --- | --- |
| Pairwise Win/Tie/Loss | ↑ | 同一场景盲评两个方法，交换 A/B 顺序；总体质量比单一绝对分更稳健 |
| Persuader output tokens/dialogue | ↓ | 当前日志可计算的本地输出成本 |
| Persuader tokens/success | ↓ | 单位严格成功所需的本地输出 token |
| End-to-end latency / API calls | ↓ | 统一硬件、并发和缓存后重测；包含规划、模拟和评审调用 |

## 从原主表移出的字段

- `Level>=3 (%)`：与 Acceptance Score 重复。
- `Successful mean turn`：只条件于成功样本，存在选择偏差。
- `Max-turn Stop (%)`：当前成功即停止协议下接近失败率的重述。
- `Range / SD / Domain count`：进入附表 D。
- `Local generation seconds`：进入附表 E，并在受控运行条件下报告。
- `Delta (pp)`：不作为独立能力指标，在正文或表下注报告 paired delta 与 CI。

所有剩余 `TBD` 都必须由 MA2P 或 TrajWeaver-v1 的完整实验和同一离线评审产生，不能从现有成功率推断。
