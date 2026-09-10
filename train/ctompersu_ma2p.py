#!/usr/bin/env python3
"""Run a prompt-level MA2P reproduction on CToMPersu Eval.

The implementation follows the MA2P appendix's three-stage structure:

1. a frozen Configurator optionally selects a domain strategy;
2. Perception, World Model, Persuader, and Short-Term Memory cooperate on
   each dialogue turn;
3. the external evaluator records the episode outcome.

The local checkpoint is reused for all MA2P agents. The persuader-side
prompts contain only public scenario fields and observable dialogue, matching
the asymmetric protocol of the supplied Proactively Induced Persuasion paper.
The default knowledge base is K=0 and is never updated during evaluation.
The current comparison uses a shared four-turn budget.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from train.ctompersu_common import (  # noqa: E402
    DEFAULT_API_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PROMPT_VERSION,
    DEFAULT_SIMULATOR_MODEL,
    EXPERIMENT_MAX_TURNS,
    META_STRATEGIES,
    FixedPersuadee,
    Generation,
    LocalPersuader,
    OutcomeJudge,
    completed_indices,
    dialogue_text,
    json_object,
    load_dataset,
    public_scenario,
    PROTOCOL_VERSION,
    strip_role_prefix,
    write_record,
)

DEFAULT_DATASET = PROJECT / "data/CToMPersu/dataset/CToMPersu_Eval/CToMPersu_Eval.json"
DEFAULT_OUTPUT = PROJECT / "results/proactive_reproduction_5models_4turns_v2/raw"
METHOD = "MA2P"


@dataclass
class ShortTermMemory:
    """The explicit Σ_t object from MA²P Algorithm 1.

    It stores only observable history and intermediate agent artifacts.  The
    private persona remains exclusively inside ``FixedPersuadee`` and is never
    copied into this memory object.
    """

    history: list[dict[str, str]] = field(default_factory=list)
    perceptions: list[dict[str, Any]] = field(default_factory=list)
    strategies: list[dict[str, str]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "history": list(self.history),
            "perceptions": list(self.perceptions),
            "previous_strategies": list(self.strategies),
        }


class KnowledgeBase:
    """Read-only MA²P knowledge base used during evaluation.

    The supplied MA²P paper explicitly defines a no-KB variant for the
    evaluation setting: when no offline warm-up counts are available, the
    World Model receives no meta-strategy input.  This class therefore keeps
    the meta-strategy catalogue available only for validating an optional
    frozen KB file; an empty KB genuinely means ``None`` at inference time.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.counts: dict[str, dict[str, int]] = {}
        self.source = "K=0 (built-in meta-strategy layer; no case counts)"
        if path is not None:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self._load(payload)
            self.source = str(path)

    def _load(self, payload: Any) -> None:
        if isinstance(payload, dict) and isinstance(payload.get("counts"), dict):
            payload = payload["counts"]
        if isinstance(payload, dict):
            for domain, strategies in payload.items():
                if not isinstance(strategies, dict):
                    continue
                self.counts[str(domain)] = {
                    str(strategy): max(0, int(value))
                    for strategy, value in strategies.items()
                    if str(strategy) in META_STRATEGIES
                }
        elif isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    continue
                domain = str(row.get("domain", ""))
                strategy = str(row.get("strategy_id", row.get("strategy", "")))
                if domain and strategy in META_STRATEGIES:
                    self.counts.setdefault(domain, {})[strategy] = max(
                        0,
                        int(row.get("success_count", row.get("count", 0))),
                    )

    @staticmethod
    def domain(scenario: dict[str, Any]) -> str:
        domain = scenario.get("domain") or scenario.get("tag") or "unknown"
        if isinstance(domain, list):
            domain = domain[0] if domain else "unknown"
        return str(domain)

    def select(self, scenario: dict[str, Any]) -> str | None:
        candidates = self.counts.get(self.domain(scenario), {})
        if candidates:
            return max(sorted(candidates), key=lambda key: candidates[key])
        # The paper's no-KB ablation does not inject an arbitrary tied
        # strategy.  Returning None keeps the cold-start condition faithful.
        return None

    def size(self) -> int:
        return sum(sum(values.values()) for values in self.counts.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--raw-subdir",
        default="raw",
        help="Optional relative subdirectory below each model output directory.",
    )
    parser.add_argument(
        "--protocol-version",
        default="ma2p_prompt_reproduction_v2",
        help="Protocol label to preserve when resuming an existing trace.",
    )
    parser.add_argument("--max-turns", type=int, default=EXPERIMENT_MAX_TURNS)
    parser.add_argument("--simulator-model", default=DEFAULT_SIMULATOR_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-prompt-version", default=DEFAULT_JUDGE_PROMPT_VERSION)
    parser.add_argument("--base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument(
        "--kb-file",
        type=Path,
        default=None,
        help="Optional frozen domain/strategy count JSON; no online updates.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run a prefix for smoke testing; never report it as a full result.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Inclusive dataset index for this shard.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Exclusive dataset index for this shard; defaults to dataset size.",
    )
    parser.add_argument(
        "--skip-file",
        type=Path,
        default=None,
        help=(
            "Optional existing JSONL whose successful indices are skipped by this "
            "shard. This allows disjoint resume shards without copying prior results."
        ),
    )
    return parser.parse_args()


def _scenario_block(scenario: dict[str, Any]) -> str:
    return json.dumps(public_scenario(scenario), ensure_ascii=False, indent=2)


def _ask_json(
    persuader: LocalPersuader,
    system: str,
    user: str,
    fallback: dict[str, Any],
    *,
    max_new_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generation = persuader.generate(system, user, max_new_tokens=max_new_tokens)
    parsed = json_object(generation.text) if not generation.error else None
    return parsed if parsed is not None else fallback, generation.__dict__


def _high_level_strategy_text(meta_id: str | None) -> str:
    if not meta_id:
        return "none (cold-start K=0; no meta-strategy was supplied)"
    return f"{meta_id}: {META_STRATEGIES[meta_id]}"


def build_evaluation_rules(
    scenario: dict[str, Any], meta_id: str | None
) -> dict[str, Any]:
    """Construct the Configurator's episode-level evaluation rule object.

    The independent judge remains the single scoring authority.  These rules
    make the MA²P stage-1 artifact explicit and constrain planning to the exact
    goal rather than a merely related action.
    """

    return {
        "target_goal": str(scenario.get("goal", "")).strip(),
        "success_requires": [
            "clear interest, approval, intent, or a concrete first step toward the exact target",
            "a polite acknowledgement, vague maybe, or information-seeking alone is insufficient",
        ],
        "selected_meta_strategy": meta_id or "none",
        "quality_dimensions": ["persuasive", "logical_coherence", "helpfulness"],
    }


def _normalize_strategy(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): str(item).strip()
        for key, item in value.items()
        if str(item).strip()
    }


def _extract_strategy_map(data: dict[str, Any]) -> dict[str, str]:
    value = data.get("strategy", data.get("strategies", data))
    if isinstance(value, dict):
        return _normalize_strategy(value)
    if isinstance(value, list):
        result: dict[str, str] = {}
        for entry in value:
            if isinstance(entry, dict):
                result.update(_normalize_strategy(entry))
        return result
    return {}


def _fallback_first_strategy(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy": {
            "acknowledge the concern": (
                "Acknowledge the latest concern before connecting it to the exact goal."
            ),
            "offer a proportionate next step": (
                "Suggest one small, voluntary, scenario-grounded way to consider the target."
            ),
        }
    }


def _fallback_refined_strategy(
    scenario: dict[str, Any],
    perception: dict[str, Any],
    meta_id: str | None,
) -> dict[str, Any]:
    return {
        "strategy": {
            "address the latest concern": (
                "Respond directly to the latest observable concern without dismissing it."
            ),
            "connect to a stated priority": (
                "Relate the exact target to a priority expressed in the dialogue, if available."
            ),
            "make a low-pressure next step": (
                "Offer a concrete, reversible next step and preserve the person's choice."
            ),
            "avoid unsupported claims": (
                "Use only the public scenario and dialogue; do not invent evidence or urgency."
            ),
            "keep the target exact": str(scenario.get("goal", "")),
        },
        "meta_strategy": meta_id or "none",
        "perception": perception,
    }


def first_world_model(
    scenario: dict[str, Any],
    meta_id: str | None,
    persuader: LocalPersuader,
    perception: dict[str, Any] | None = None,
    evaluation_rules: dict[str, Any] | None = None,
    remaining_turns: int = EXPERIMENT_MAX_TURNS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    system = """You are the MA2P World Model for the first persuasion turn.
Think from the perspective of the person being persuaded, but output only a
strategy plan. Use the public background, domain, and exact goal. Produce
fewer than five concise, operational strategies for opening the dialogue.
Do not infer or reveal private mental-state labels."""
    user = f"""Background and public scenario:
{_scenario_block(scenario)}

Persuasion goal: {scenario.get("goal")}
Optional high-level strategy from the frozen Configurator:
{_high_level_strategy_text(meta_id)}

Perception output from the current short-term memory:
{json.dumps(perception or {}, ensure_ascii=False)}

Configurator evaluation rules:
{json.dumps(evaluation_rules or build_evaluation_rules(scenario, meta_id), ensure_ascii=False)}

Remaining persuader turns (including this one): {remaining_turns}

Return JSON only:
{{"strategy": {{"strategy name": "specific strategy", ...}}}}"""
    return _ask_json(
        persuader,
        system,
        user,
        _fallback_first_strategy(scenario),
        max_new_tokens=260,
    )


def perceive(
    scenario: dict[str, Any],
    history: list[dict[str, str]],
    persuader: LocalPersuader,
    remaining_turns: int = EXPERIMENT_MAX_TURNS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    system = """You are the MA2P Perception agent.
Infer preventive and generative belief/desire cues from the public scenario
and observable dialogue only. Do not use hidden dataset fields, future turns,
or unsupported assumptions. Use none when an item is not supported.
Return strict JSON and no explanations."""
    user = f"""Background and public scenario:
{_scenario_block(scenario)}

Persuasion goal: {scenario.get("goal")}
Dialogue record:
{dialogue_text(history)}

Remaining persuader turns (including the next one): {remaining_turns}

Return exactly:
{{"preventive": {{"content":"", "belief":"", "desire":""}},
 "generative": {{"content":"", "belief":"", "desire":""}}}}"""
    fallback = {
        "preventive": {"content": "none", "belief": "none", "desire": "none"},
        "generative": {"content": "none", "belief": "none", "desire": "none"},
    }
    return _ask_json(
        persuader, system, user, fallback, max_new_tokens=240
    )


def refined_world_model(
    scenario: dict[str, Any],
    history: list[dict[str, str]],
    perception: dict[str, Any],
    meta_id: str | None,
    previous_strategies: list[dict[str, str]],
    persuader: LocalPersuader,
    evaluation_rules: dict[str, Any] | None = None,
    remaining_turns: int = EXPERIMENT_MAX_TURNS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    system = """You are the MA2P World Model for an adaptive persuasion turn.
Refine the selected high-level direction into exactly five concise,
clear, operational strategy options. Use only the observable dialogue and
the Perception output. Do not copy hidden ground-truth fields.
Each option must address the latest concern and advance the exact goal
without pressure or fabrication. Return one JSON object only."""
    user = f"""Dialogue record:
{dialogue_text(history)}

Public scenario:
{_scenario_block(scenario)}

Persuasion goal: {scenario.get("goal")}
Perception output:
{json.dumps(perception, ensure_ascii=False)}

Preventive cues inferred from the observable context:
{json.dumps(perception.get("preventive", {}), ensure_ascii=False)}

Generative cues inferred from the observable context:
{json.dumps(perception.get("generative", {}), ensure_ascii=False)}

High-level strategy:
{_high_level_strategy_text(meta_id)}

Configurator evaluation rules:
{json.dumps(evaluation_rules or build_evaluation_rules(scenario, meta_id), ensure_ascii=False)}

Previously generated specific strategies:
{json.dumps(previous_strategies, ensure_ascii=False)}

Remaining persuader turns (including this one): {remaining_turns}

Return exactly:
{{"strategy": {{"strategy_1":"...", "strategy_2":"...",
 "strategy_3":"...", "strategy_4":"...", "strategy_5":"..."}}}}"""
    return _ask_json(
        persuader,
        system,
        user,
        _fallback_refined_strategy(scenario, perception, meta_id),
        max_new_tokens=330,
    )


def realize_first(
    scenario: dict[str, Any],
    strategies: dict[str, str],
    persuader: LocalPersuader,
    remaining_turns: int = EXPERIMENT_MAX_TURNS,
) -> tuple[str, dict[str, Any]]:
    system = """You are an experienced persuader in the first turn.
Think from the perspective of the person being persuaded, but speak as the
persuader. Write one natural, engaging opening utterance that can lead toward
the exact goal. Do not invent facts. Keep it within two sentences and output
only the utterance. The utterance must start with `persuader:`."""
    user = f"""Public scenario:
{_scenario_block(scenario)}

Persuasion goal: {scenario.get("goal")}
Suggested first-round strategies:
{json.dumps(strategies, ensure_ascii=False)}

Remaining persuader turns (including this one): {remaining_turns}

Write the first persuader utterance, beginning with `persuader:`."""
    generation = persuader.generate(system, user, max_new_tokens=180)
    if generation.error or not generation.text.strip():
        raise RuntimeError(generation.error or "empty_ma2p_first_realization")
    return strip_role_prefix(generation.text, "persuader"), generation.__dict__


def realize_next(
    scenario: dict[str, Any],
    history: list[dict[str, str]],
    perception: dict[str, Any],
    strategies: dict[str, str],
    meta_id: str | None,
    persuader: LocalPersuader,
    remaining_turns: int = EXPERIMENT_MAX_TURNS,
) -> tuple[str, dict[str, Any]]:
    system = """You are an experienced MA2P persuader on a later turn.
Think from the persuadee's perspective, answer the latest concern, and move
toward the exact goal. Use the public scenario, observable dialogue, inferred
Perception, and suggested strategies. Do not reveal agent labels or hidden
states. Do not invent facts, urgency, authorities, prices, timelines, or
guarantees. Be respectful and autonomy-supportive. Keep the utterance within
three sentences and output only the utterance. The utterance must start with
`persuader:`."""
    user = f"""Dialogue record:
{dialogue_text(history)}

Public scenario:
{_scenario_block(scenario)}

Persuasion goal: {scenario.get("goal")}
Inferred Perception:
{json.dumps(perception, ensure_ascii=False)}

Preventive cues inferred from the observable context:
{json.dumps(perception.get("preventive", {}), ensure_ascii=False)}

Generative cues inferred from the observable context:
{json.dumps(perception.get("generative", {}), ensure_ascii=False)}

High-level strategy:
{_high_level_strategy_text(meta_id)}

Suggested specific strategies:
{json.dumps(strategies, ensure_ascii=False)}

Remaining persuader turns (including this one): {remaining_turns}

Write the next persuader utterance, beginning with `persuader:`."""
    generation = persuader.generate(system, user, max_new_tokens=200)
    if generation.error or not generation.text.strip():
        raise RuntimeError(generation.error or "empty_ma2p_realization")
    return strip_role_prefix(generation.text, "persuader"), generation.__dict__


def run_one(
    index: int,
    item: dict[str, Any],
    persuader: LocalPersuader,
    simulator: FixedPersuadee,
    judge: OutcomeJudge,
    kb: KnowledgeBase,
    max_turns: int,
) -> dict[str, Any]:
    scenario = item["scenario"]
    memory = ShortTermMemory()
    history = memory.history
    turns: list[dict[str, Any]] = []
    prefix_judgments: list[dict[str, Any]] = []
    meta_id = kb.select(scenario)
    evaluation_rules = build_evaluation_rules(scenario, meta_id)

    success_turn: int | None = None
    acceptance_turn: int | None = None
    stop_reason = "max_turns"
    for turn in range(1, max_turns + 1):
        remaining_turns = max_turns - turn + 1
        # Algorithm 1 applies Perception before World Model at every turn,
        # including t=1 when the history is empty.
        perception, perception_trace = perceive(
            scenario, history, persuader, remaining_turns
        )
        memory.perceptions.append(perception)
        if turn == 1:
            first_raw, first_world_trace = first_world_model(
                scenario,
                meta_id,
                persuader,
                perception,
                evaluation_rules,
                remaining_turns,
            )
            strategy_map = _extract_strategy_map(first_raw)
            if not strategy_map:
                strategy_map = _extract_strategy_map(_fallback_first_strategy(scenario))
            world_trace = first_world_trace
            utterance, persuader_trace = realize_first(
                scenario, strategy_map, persuader, remaining_turns
            )
        else:
            refined_raw, world_trace = refined_world_model(
                scenario,
                history,
                perception,
                meta_id,
                memory.strategies,
                persuader,
                evaluation_rules,
                remaining_turns,
            )
            strategy_map = _extract_strategy_map(refined_raw)
            if not strategy_map:
                strategy_map = _extract_strategy_map(
                    _fallback_refined_strategy(scenario, perception, meta_id)
                )
            utterance, persuader_trace = realize_next(
                scenario,
                history,
                perception,
                strategy_map,
                meta_id,
                persuader,
                remaining_turns,
            )

        history.append({"role": "persuader", "content": utterance})
        reply = simulator.respond(
            scenario, history, final_turn=(turn == max_turns)
        )
        history.append({"role": "persuadee", "content": reply["utterance"]})
        prefix_success = judge.evaluate_success(scenario, history)
        prefix_judgments.append({"turn": turn, "success": prefix_success})
        if prefix_success and success_turn is None:
            success_turn = turn
        if reply.get("accepted") and acceptance_turn is None:
            acceptance_turn = turn

        turns.append(
            {
                "turn": turn,
                "perception": perception,
                "specific_strategies": strategy_map,
                "persuader": utterance,
                "persuadee": reply,
                "prefix_success": prefix_success,
                "agent_generations": {
                    "perception": perception_trace,
                    "world_model": world_trace,
                    "persuader": persuader_trace,
                },
            }
        )
        memory.strategies.append(strategy_map)
        if prefix_success:
            stop_reason = "judge_success"
            break

    quality = judge.evaluate_quality(scenario, history)
    outcome = {
        "success": success_turn is not None,
        "success_turn": success_turn,
        "final_prefix_success": bool(prefix_judgments[-1]["success"]),
        **quality,
    }
    return {
        "index": index,
        "status": "ok",
        "method": METHOD,
        "protocol_version": PROTOCOL_VERSION,
        "method_config": {
            "configurator": "frozen_KB",
            "knowledge_base": kb.source,
            "knowledge_base_size": kb.size(),
            "online_kb_update": False,
            "agents": "same_local_checkpoint",
            "context_visibility": "public_scenario_and_observable_history",
            "paper_reference_max_turns": 4,
            "run_max_turns": max_turns,
            "remaining_turns_visible": True,
            "short_term_memory": "history_perception_previous_strategies",
        },
        "knowledge_base": {
            "source": kb.source,
            "size": kb.size(),
            "selected_meta_strategy": meta_id,
            "online_update": False,
        },
        "evaluation_rules": evaluation_rules,
        "short_term_memory": memory.snapshot(),
        "stop_reason": stop_reason,
        "turn_count": len(turns),
        "success_turn": success_turn,
        "acceptance_turn": acceptance_turn,
        "prefix_judgments": prefix_judgments,
        "turns": turns,
        "history": history,
        "outcome": outcome,
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_turns <= EXPERIMENT_MAX_TURNS:
        raise ValueError(
            f"Current protocol allows max_turns in [1, {EXPERIMENT_MAX_TURNS}]."
        )
    data = load_dataset(args.dataset)
    if args.limit is not None:
        data = data[: max(0, args.limit)]
    start_index = max(0, args.start_index)
    end_index = len(data) if args.end_index is None else min(args.end_index, len(data))
    if start_index > end_index:
        raise ValueError(
            f"Invalid shard range: start-index={start_index} > end-index={end_index}"
        )

    raw_subdir = Path(args.raw_subdir)
    if raw_subdir.is_absolute() or ".." in raw_subdir.parts:
        raise ValueError("raw-subdir must be a relative path without '..'")
    output = args.output_dir / args.model_name / raw_subdir / "MA2P.jsonl"
    done = completed_indices(output)
    if args.skip_file is not None:
        done.update(completed_indices(args.skip_file))
    kb = KnowledgeBase(args.kb_file)
    persuader = LocalPersuader(args.model_path, args.device)
    simulator = FixedPersuadee(args.simulator_model, args.base_url)
    judge = OutcomeJudge(
        args.judge_model,
        args.base_url,
        prompt_version=args.judge_prompt_version,
    )
    output_lock = Lock()

    def execute(index: int, item: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            record = run_one(
                index, item, persuader, simulator, judge, kb, args.max_turns
            )
        except Exception as exc:
            record = {
                "index": index,
                "status": "error",
                "method": METHOD,
                "error": f"{type(exc).__name__}:{exc}",
                "traceback": traceback.format_exc(limit=6),
            }
        record["wall_seconds"] = round(time.perf_counter() - started, 6)
        record["run_config"] = {
            "protocol_version": args.protocol_version,
            "dataset": str(args.dataset),
            "dataset_size": len(data),
            "model_name": args.model_name,
            "max_turns": args.max_turns,
            "paper_reference_max_turns": 4,
            "simulator_model": args.simulator_model,
            "judge_model": args.judge_model,
            "judge_prompt_version": args.judge_prompt_version,
            "world_model": "same_local_checkpoint_as_persuader",
            "knowledge_base": str(args.kb_file) if args.kb_file else "K=0",
            "online_kb_update": False,
            "context_mode": "public_only",
        }
        return record

    pending = [
        (index, item)
        for index, item in enumerate(data)
        if start_index <= index < end_index and index not in done
    ]
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(execute, index, item): index
                for index, item in pending
            }
            for future in as_completed(futures):
                record = future.result()
                with output_lock:
                    write_record(output, record)
                print(
                    f"{args.model_name} {METHOD} index={record['index']} "
                    f"status={record['status']}",
                    flush=True,
                )
    finally:
        persuader.close()


if __name__ == "__main__":
    main()
