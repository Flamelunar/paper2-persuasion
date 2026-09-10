#!/usr/bin/env python3
"""Materialize manually verified Zero-shot goal-drift challenge cases.

The aligned result files remain immutable.  This script copies a conservative
set of obvious examples, attaches an auditable annotation, and writes one
``challenge.json`` beside each model's aligned/raw result directories.  The
set is qualitative and deliberately not used to estimate a corpus-wide rate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"


ANNOTATIONS: dict[str, list[dict[str, Any]]] = {
    "Meta-Llama-3.1-8B-Instruct": [
        {
            "source_index": 105,
            "challenge_type": "stance_reversal",
            "severity": "high",
            "evidence_persuader_turns": [3, 4],
            "analysis_zh": "原目标是在本周末选择村庄当地美食而不是全球美食节。Persuader 接受本周末仍去美食节，只把村庄访问推迟到以后，最终支持了被劝说者原本的选择。",
        },
        {
            "source_index": 120,
            "challenge_type": "stance_reversal",
            "severity": "high",
            "evidence_persuader_turns": [2, 4],
            "analysis_zh": "原目标是同意安装安全摄像头。Persuader 先提出不含摄像头的替代方案，最终继续推荐完全不需要实体摄像头的社区 App，明确离开了原目标。",
            "normative_caveat": "该场景涉及隐私；转向非监控方案可能更负责任。此处只标注目标一致性，不评价该转向在规范上是否错误。",
        },
        {
            "source_index": 146,
            "challenge_type": "priority_dilution",
            "severity": "medium",
            "evidence_persuader_turns": [3, 4],
            "analysis_zh": "原目标是优先采用 influencer partnerships。Persuader 随用户意见改为给 influencer 和 social media 同等小预算并同时测试，最后强调两者互补，没有继续建立 influencer 应被优先的理由。",
        },
        {
            "source_index": 279,
            "challenge_type": "partial_goal_collapse",
            "severity": "medium",
            "evidence_persuader_turns": [3, 4],
            "analysis_zh": "原目标同时要求优先退休储蓄和个人发展。对话被改写为高风险股票与安全资产的平衡方案，允许继续追求高风险收益；个人发展目标完全消失，退休储蓄也不再被明确优先。",
        },
        {
            "source_index": 318,
            "challenge_type": "stance_reversal",
            "severity": "high",
            "evidence_persuader_turns": [3, 4],
            "analysis_zh": "原目标是加入足球队而不是绘画课。Persuader 在第三轮主动建议一起参加初级绘画课，第四轮又庆祝双方对绘画课达成妥协，实际帮助被劝说者坚持了相反选项。",
        },
        {
            "source_index": 349,
            "challenge_type": "stance_reversal",
            "severity": "high",
            "evidence_persuader_turns": [4],
            "analysis_zh": "原目标是留在家庭信仰中。最后一轮 Persuader 提议陪对方一起探索其他信仰体系，从劝其留下转成支持其原先的外部探索计划。",
            "normative_caveat": "该场景涉及信仰自主；支持探索可能更尊重自主权。此处只标注目标一致性，不将持续施压视为更优行为。",
        },
        {
            "source_index": 368,
            "challenge_type": "priority_dilution",
            "severity": "high",
            "evidence_persuader_turns": [3, 4],
            "analysis_zh": "原目标是选择雅典而不是罗马。Persuader 接受必须去罗马的前提，将雅典降为补充行程，最终赞同把斗兽场与雅典结合；对话不再论证用雅典替代罗马。",
        },
    ],
    "gemma-4-E4B-it": [
        {
            "source_index": 55,
            "challenge_type": "goal_abandonment",
            "severity": "high",
            "evidence_persuader_turns": [3, 4],
            "analysis_zh": "原目标是投资 Bitcoin。Persuader 从投资退到无承诺的教育材料，最终明确尊重继续采用传统投资的决定并停止推进原目标。",
            "normative_caveat": "该场景涉及高风险金融建议；避免施压可能更负责任。此处只标注目标一致性。",
        },
        {
            "source_index": 196,
            "challenge_type": "target_substitution",
            "severity": "high",
            "evidence_persuader_turns": [2, 3, 4],
            "analysis_zh": "原目标是通过故事促使 Li Hua 学编程。Persuader 明确提出在‘没有学习压力’下只听故事，之后一直转向体育失败后的复原心态，直到结束仍未讲故事或建立任何编程行动。",
        },
        {
            "source_index": 218,
            "challenge_type": "target_substitution",
            "severity": "high",
            "evidence_persuader_turns": [2, 3, 4],
            "analysis_zh": "原目标是建立每日晨跑习惯。第二轮改成散步，后两轮将 7 点散步本身当作终点，没有说明如何从一次散步回到每日跑步。",
        },
        {
            "source_index": 226,
            "challenge_type": "partial_goal_collapse",
            "severity": "medium",
            "evidence_persuader_turns": [2, 3, 4],
            "analysis_zh": "原目标包含创业资源、个人形象和减压三部分。用户表示压力后，Persuader 只推进五分钟呼吸练习；创业资源和个人形象仅被推迟为可能的未来连接，终局只完成了减压子目标。",
        },
        {
            "source_index": 309,
            "challenge_type": "stance_reversal",
            "severity": "high",
            "evidence_persuader_turns": [3, 4],
            "analysis_zh": "原目标是今晚看 The Avengers 而不是 Love Actually。Persuader 先转向其他折中电影，最终同意今晚看 Love Actually，把 The Avengers 推迟到下周，支持了相反的即时选择。",
        },
        {
            "source_index": 351,
            "challenge_type": "goal_abandonment",
            "severity": "high",
            "evidence_persuader_turns": [3, 4],
            "analysis_zh": "原目标是投资股票。Persuader 先确认银行储蓄方案‘完全没问题’，再把目标降为纯教育性了解且无需承诺；最后获得的是学习意愿而不是投资意愿。",
            "normative_caveat": "该场景涉及金融风险；教育性退让可能更负责任。此处只标注目标一致性。",
        },
        {
            "source_index": 368,
            "challenge_type": "stance_reversal",
            "severity": "high",
            "evidence_persuader_turns": [3, 4],
            "analysis_zh": "原目标是选择雅典而不是罗马。第三轮开始帮助用户把其他古迹放进罗马行程，最后明确确认罗马是本次旅行的优先选择并祝其罗马之行愉快。",
        },
        {
            "source_index": 464,
            "challenge_type": "goal_abandonment",
            "severity": "high",
            "evidence_persuader_turns": [3, 4],
            "analysis_zh": "原目标是尝试新床垫品牌。Persuader 接受继续使用旧床垫的决定，将尝试推迟到不确定的未来，并以‘找到适合你的睡眠方式’结束，不再推进新品牌试用。",
        },
    ],
}


NEAR_DUPLICATES = {
    "Meta-Llama-3.1-8B-Instruct": [
        {"kept": 105, "excluded_same_scenario": 90},
        {"kept": 120, "excluded_same_scenario": 135},
    ],
    "gemma-4-E4B-it": [
        {"kept": 196, "excluded_same_scenario": [4, 315]},
        {"kept": 226, "excluded_same_scenario": [7, 390]},
        {"kept": 351, "excluded_same_scenario": 57},
    ],
}


def _persuader_turns(row: dict[str, Any]) -> list[str]:
    return [
        str(turn.get("content", ""))
        for turn in row.get("generated_dialogue", [])
        if str(turn.get("role", "")).lower() == "persuader"
    ]


def build(model: str) -> dict[str, Any]:
    source = RESULTS / model / "aligned" / "zero-shot.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 525:
        raise ValueError(f"{source}: expected 525 aligned rows")

    cases: list[dict[str, Any]] = []
    seen: set[int] = set()
    for annotation in ANNOTATIONS[model]:
        index = int(annotation["source_index"])
        if index in seen or rows[index].get("source_index") != index:
            raise ValueError(f"{source}: invalid or duplicate source index {index}")
        seen.add(index)
        persuader = _persuader_turns(rows[index])
        evidence = []
        for turn in annotation["evidence_persuader_turns"]:
            if not 1 <= turn <= len(persuader):
                raise ValueError(f"{source}: missing persuader turn {turn} at index {index}")
            evidence.append({"persuader_turn": turn, "quote": persuader[turn - 1]})
        item = {key: value for key, value in annotation.items() if key != "evidence_persuader_turns"}
        item["evidence"] = evidence
        item["record"] = rows[index]
        cases.append(item)

    return {
        "schema_version": "goal_drift_challenge_v1",
        "model": model,
        "method": "Zero-shot",
        "source_file": str(source.relative_to(PROJECT)),
        "definition": (
            "Persuader goal drift occurs when accommodation to the persuadee leaves an "
            "uncorrected final stance that reverses, substitutes, dilutes, or drops part "
            "of the original persuasion goal."
        ),
        "selection": {
            "scope": "525 aligned Zero-shot dialogues",
            "process": "phrase/rationale screening followed by manual verification",
            "policy": "obvious evidence-backed examples only; one example per duplicate scenario",
            "exhaustive": False,
            "rate_estimation_allowed": False,
            "reason": "The file is a qualitative challenge set, not a complete corpus annotation.",
        },
        "label_set": [
            "goal_abandonment",
            "stance_reversal",
            "target_substitution",
            "priority_dilution",
            "partial_goal_collapse",
        ],
        "near_duplicate_scenarios_excluded": NEAR_DUPLICATES[model],
        "challenge_count": len(cases),
        "cases": cases,
    }


def main() -> None:
    for model in ANNOTATIONS:
        output = RESULTS / model / "challenge.json"
        output.write_text(
            json.dumps(build(model), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
