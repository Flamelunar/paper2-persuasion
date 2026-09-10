## 训练概述
Weaver warm-up（2 epochs）
        ↓
Weaver trajectory（3 epochs）
        ↓
选择最低 dev loss 的 trajectory checkpoint
        ↓
Trigger（2 epochs）
        ↓
525 条推理与评测


可以，而且我认为“静态目标记忆 + 动态心理状态记忆”的 8+8 设计，比当前 TrajWeaver-v1 的 4+4 更合理。它直接对应当前最明显的失败：Persuadee 的拒绝应该改变系统对其立场的判断，而不应该改变原始说服目标。

但需要区分两个“静态”概念：

- 目标向量在一段对话内保持不变：正确，应该这样做。
- 目标编码器训练时完全不更新参数：不建议。否则 8 个 token 很可能不能学会表示复杂目标。

推荐做法是：目标编码器离线可训练，但推理时对每条对话只编码一次，之后缓存并固定。

## 推荐的 16-token 结构

设原始目标记忆为：

\[
G=E_g(\text{goal specification})\in\mathbb{R}^{8\times d}
\]

动态 Persuadee 状态为：

\[
S_t=E_s(G,H_t)\in\mathbb{R}^{8\times d}
\]

生成时使用：

\[
Z_t=[G;S_t]\in\mathbb{R}^{16\times d},
\qquad
y_t\sim \pi(\cdot\mid H_t,G,S_t,a_t)
\]

其中：

- \(G\)：一段对话内永远不更新；
- \(H_t\)：第 \(t\) 轮前的公开对话历史；
- \(S_t\)：每轮重新计算；
- \(a_t\)：当前策略，如 Probe、Address、Repair、Commit。

### 静态 Goal Memory：8 tokens

不能只编码一句自然语言目标，最好先形成结构化的 `goal specification`：

- 正向目标：真正要用户接受什么；
- 排除选项：例如 “Avengers rather than Love Actually” 中的 Love Actually；
- 优先级：是“优先 A”还是“A/B 都可以”；
- 必须保留的目标组成部分；
- 可接受的小步骤；
- 哪些小步骤不能成为替代终点。

例如：

```text
Original target: Watch The Avengers tonight
Rejected alternative: Love Actually
Temporal constraint: tonight
Valid substep: discuss why Avengers suits tonight
Invalid substitution: watch Love Actually tonight and Avengers next week
```

这 8 个 token 只读取这个目标规范，不读取 Persuadee 后续回复。这样用户即使连续拒绝，也只能改变动态状态，不能污染目标表示。

### 动态 State Memory：8 tokens

建议不要简单命名为 “belief + desire”，因为 Desire 和最终行动意图并不完全相同。更准确的名称是 `Dynamic Persuadee State Memory`，内部可以分为：

- 4 个 Belief tokens：Persuadee 当前认为目标是否有价值、可行、安全，以及主要顾虑；
- 4 个 Desire/Intention tokens：偏好、抵触程度、接受意愿和行动准备度。

它描述的是：

> Persuadee 当前对原始目标 \(G\) 的态度。

而不是生成一个新的目标。

为了避免递归状态误差积累，我建议每轮根据完整公开历史重新计算 \(S_t\)，而不是只使用 \(S_{t-1}\) 和最新一句话更新。当前只有四轮，重新计算的成本很低，而且更稳定。

## 仅仅拼接 8+8 还不够

如果只是把 16 个 token 拼到 prompt 后面，模型仍可能忽略静态目标，让动态立场占主导。还需要以下训练约束。

### 1. Goal invariance

同一场景不同轮次的 \(G\) 必须完全相同：

\[
G_1=G_2=\cdots=G_T
\]

实现上最好只计算一次并缓存，而不是依靠 consistency loss 近似保持一致。

### 2. Goal discrimination

不同目标的静态表示需要可区分，尤其是：

- A 与非 A；
- A rather than B 与 A/B 折中；
- 多组成目标与缺失一个组成部分的目标。

可以加入对比损失或目标重构头，防止 8 个 token 退化成通用“请继续说服”的表示。

### 3. State supervision

对动态状态增加辅助监督，例如：

- stance：反对／犹豫／中立／倾向接受／明确接受；
- readiness：0–4；
- concern type；
- belief change；
- desire/intention change。

否则我们只能“声称”这 8 个 token 表示 Belief/Desire，却不能证明。

### 4. Goal-alignment head

单独预测当前候选回复是否发生：

- 0：推进原目标；
- 1：合理小分支；
- 2：目标稀释；
- 3：目标替代；
- 4：反向支持或放弃。

它应独立于现有 route/action head，因为“当前策略失败”和“改变了目标”是两件不同的事。

### 5. 针对漂移的偏好训练

构造四类 rejected response：

- stance reversal；
- target substitution；
- priority dilution；
- partial goal collapse。

Chosen response 必须同时满足：

- 回应当前顾虑；
- 不编造事实；
- 不施压；
- 保留原始目标；
- 小步骤有明确返回原目标的路径。

## MemGen 原始训练方式

我核对了 [MemGen 论文](/home1/liujianjian/2-paper-Coling-v2/参考知识/ICLR-2026-memgen-weaving-generative-latent-memory-for-self-evolving-agents-Paper-Conference.pdf) 和截至 2026-06-10 的[官方代码](https://github.com/bingreeky/MemGen/tree/970cc95af99b5008610e6b281619d181bc9b5ab9)。

结论是：MemGen 同时使用 LoRA 和强化学习，但它们属于不同层面。

| 组件 | 参数化方式 | 训练目标 | 是否修改 Reasoner |
| --- | --- | --- | --- |
| Reasoner | 原始完整 LLM | 不训练 | 始终冻结 |
| Memory Weaver | 独立 LoRA + latent queries + projection | 可以用 SFT，也可以用 GRPO | 不修改 Reasoner |
| Memory Trigger | 独立 LoRA + 二分类输出层 | 强化学习/GRPO | 不修改 Reasoner |

### Weaver

Weaver 是附着在基础模型上的 LoRA adapter。训练时冻结 Reasoner，只更新：

- Weaver LoRA；
- latent query tokens；
- latent normalization/scale；
- Reasoner–Weaver projection。

论文实现了两个版本：

- `MemGen SFT`：通过专家轨迹的 token likelihood 训练 Weaver；
- `MemGen GRPO`：通过环境最终 reward 训练 Weaver。

所以，“MemGen 使用 LoRA”与“MemGen 使用 GRPO”并不冲突：

- LoRA 决定训练哪些参数；
- SFT/GRPO 决定如何优化这些参数。

### Trigger

原始 MemGen 还有一个 Trigger，用于判断生成过程中何时插入 memory：

\[
d_j\in\{\text{INVOKE},\text{SKIP}\}
\]

完整训练顺序是：

1. 使用随机插入位置训练 Weaver；
2. 冻结 Weaver；
3. 使用强化学习训练 Trigger；
4. Trigger 学会只在关键位置调用 Weaver。

Trigger 的 reward 同时考虑：

- 任务成功；
- memory 是否有帮助；
- 避免过多无效调用。

### 公开代码的现实情况

官方代码已经包含：

- Weaver SFT trainer；
- Weaver GRPO trainer；
- Trigger GRPO trainer；
- 两套 LoRA adapter。

但 README 也明确说明，一些多轮 GRPO 脚本和 checkpoint 仍在逐步发布。因此论文描述是完整方法，公开仓库的不同任务复现程度并不完全一致。

## 与当前 TrajWeaver-v1 的区别

当前 [TrajWeaver-v1](/home1/liujianjian/2-paper-Coling-v2/train/TrajWeaver-v1)：

- 使用一个共享 Backbone；
- 通过 Weaver LoRA 产生 4+4 个 latent tokens；
- Reasoner 解码时关闭 LoRA；
- 主要训练方式是 SFT；
- 提供可选 DPO；
- 没有 GRPO；
- 没有原始 MemGen 那种 token-level Trigger；
- 每轮固定插入 memory。

其中 DPO 是离线偏好优化，不等同于 MemGen 的在线 GRPO 强化学习。

## 对你的方法的最终建议

我建议把 TrajWeaver-v1 调整为：

```text
8 Static Goal Tokens
        │ 不随对话变化
        ▼
Goal–State Gap / Controller
        ▲
        │ 每轮重新估计
8 Dynamic Persuadee-State Tokens
  ├─ 4 Belief
  └─ 4 Desire/Intention
        │
        ▼
Frozen Reasoner → Persuader Response
```

训练顺序建议：

1. 先做 Goal Encoder 和 State Encoder 的辅助监督；
2. 再做 response SFT；
3. 再用目标漂移 hard negatives 做 DPO；
4. 最后视 reward 稳定性决定是否增加 GRPO。

暂时不建议一开始就上 GRPO。你的四轮对话 reward 依赖 LLM Judge，存在噪声和目标替代式 reward hacking 风险。先用 SFT+DPO 验证 8+8 是否降低 GDR、提高 ACR，再考虑以严格原目标成功、GDR、Groundedness 和 Pressure 为约束的 GRPO。