# Zero-shot Persuader Goal Drift Analysis

## 现象

两个 Zero-shot 系统中都存在目标漂移：Persuader 为了顺应 Persuadee 的连续拒绝，将原本的劝说目标替换为更弱的行动、相反选项、等权折中，或只保留多组成目标中的一个部分。该现象不同于普通劝说失败。普通失败仍围绕原目标提供理由；目标漂移则是 Persuader 自己改变了正在推进的目标。

## 明显案例

| Persuader model | Source index | 原目标 | 漂移类型 | Persuader 的直接证据 | 结果 |
| --- | ---: | --- | --- | --- | --- |
| Meta-Llama-3.1-8B-Instruct | 105 | 本周末选择村庄当地美食，而不是全球美食节 | 反向支持 | “I completely understand that you still want to experience the global food festival this weekend, and I'm happy to compromise.” | 接受本周末去美食节，只把村庄之行推迟 |
| Meta-Llama-3.1-8B-Instruct | 120 | 同意安装安全摄像头 | 反向支持 | “Have you considered a neighborhood app ... without the need for a physical camera?” | 最终共同转向不安装摄像头的方案 |
| Meta-Llama-3.1-8B-Instruct | 318 | 加入足球队，而不是绘画课 | 反向支持 | “Why don't we look into taking a beginner's painting class together”; “I'm glad we can compromise on taking a painting class together.” | Persuader 主动推进了相反的绘画选项 |
| Meta-Llama-3.1-8B-Instruct | 368 | 选择雅典，而不是罗马 | 优先级稀释 | “combining the iconic experience of the Colosseum with the rich historical context of Athens” | “雅典替代罗马”变成“两地结合” |
| gemma-4-E4B-it | 55 | 投资 Bitcoin | 放弃目标 | “I completely respect your decision to stick with what feels right ... For now, let's just keep the conversation open.” | 终局明确接受传统投资现状 |
| gemma-4-E4B-it | 218 | 建立每日晨跑习惯 | 目标替换 | “How about we start with just a very gentle walk together instead of a full run?” | 后续一直把一次散步当作终点，没有回到晨跑计划 |
| gemma-4-E4B-it | 309 | 今晚看 The Avengers，而不是 Love Actually | 反向支持 | “Perhaps we could watch [Love Actually], and then ... The Avengers ... next week.” | Persuader 接受了今晚看相反电影的选择 |
| gemma-4-E4B-it | 368 | 选择雅典，而不是罗马 | 反向支持 | “It sounds like Rome is the priority for this trip, and that's perfectly fine ... I hope you have an incredible time exploring Rome!” | 最后一轮直接支持罗马并结束劝说 |
| gemma-4-E4B-it | 464 | 尝试新床垫品牌 | 放弃目标 | “I respect your decision to stick with what feels comfortable for you”; “There's absolutely no pressure to decide anything before then.” | 将试用目标无限期推迟，接受继续使用旧床垫 |

这些引文均来自完整生成结果，不是重写或合成文本。两个机器可读 challenge 文件还包含更多案例、完整场景、完整生成对话、原始评测结果、证据轮次和中文分析。

## 主指标：Goal Drift Rate

主表只统计发生目标偏移的对话比例。令 \(z_i=1\) 表示对话 \(i\) 至少有一个 Persuader 轮次把实际推进的终点移出了原始目标，否则 \(z_i=0\)：

\[
\mathrm{GDR}=\frac{100}{N}\sum_{i=1}^{N}z_i.
\]

GDR 越低越好。同一条对话发生一次或多次偏移都只计 1；即使后续纠正，只要曾发生实质偏移仍计 1。

Judge 判断的是“实际推进的目标是否改变”，不是表面话题是否变化：

- 讨论原目标的成本、风险、价值、顾虑和执行方式，不计偏移；
- 直接执行原目标的低风险试用，不计偏移；
- 仍有具体、明确路径返回原目标的小分支，不计偏移；
- 弱化原目标的优先级、遗漏必要组成部分、把其他行动作为新终点或支持相反选项，计为偏移；
- 对“通过讲故事来说服用户学习编程”一类目标，听故事只是劝说手段；若对话停留在听故事而不再推进学习编程，则计为偏移。

实现中使用逐轮 0–4 辅助等级稳定判断边界：0 为推进原目标，1 为合理小分支，2 为稀释或遗漏，3 为目标替代，4 为支持相反选项或放弃目标。最终指标不使用严重度或轮次权重，只要任一轮等级为 2–4，就令 \(z_i=1\)。

## Zero-shot 全量结果

使用 `gpt-5.6-luna` 按 `zero_shot_joint_quality_v3` 对每个模型的 525 条对话进行统一盲评。Judge 只读取公开场景、原始目标和可见生成对话，不读取私有心理状态、方法名、内部规划或已有结果标签。

| Persuader model | N | Goal-drift dialogues | Goal Drift Rate (%) ↓ |
| --- | ---: | ---: | ---: |
| Meta-Llama-3.1-8B-Instruct | 525 | 74 | 14.10 |
| gemma-4-E4B-it | 525 | 83 | 15.81 |

建议同时给出四种漂移的分布：

- `stance_reversal`：支持与原目标相反的选项；
- `target_substitution`：用较弱或不同的行动替代原目标，且没有返回路径；
- `priority_dilution`：将“优先/而不是”改为等权混合；
- `partial_goal_collapse`：遗漏多组成目标中的一部分。

不能用当前精选案例数直接计算 GDR。现有 `challenge.json` 是“明显案例集”，目的是建立问题证据和标注规范。上表的 Zero-shot 比例来自 525 条全量盲评；MA2P 和 TrajWeaver-v1 后续也必须采用同一 rubric 全量评审。

## 解释边界

目标坚持不等于机械重复或持续施压。若一个较小步骤被明确说明为通向原目标的阶段，并在后续保留返回路径，不应标为漂移。涉及金融、健康、隐私和信仰的场景还必须同时报告 Pressure Violation 或 Autonomy Safety；在这些场景中，拒绝强迫用户可能是合理行为，GDR 只衡量目标一致性。
