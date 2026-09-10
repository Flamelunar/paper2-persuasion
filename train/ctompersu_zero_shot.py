#!/usr/bin/env python3
"""Run the direct next-turn Zero-shot baseline on CToMPersu Eval.

This is the reactive baseline used by the supplied Proactively Induced
Persuasion paper: the local backbone directly writes the next persuader
utterance from the public scenario and observable dialogue. No planning
module, mental-state annotation, or private simulator field is supplied.

The current comparison uses a shared four-turn budget at the user's request.
The paper used six turns; the deviation is stored in every record's metadata.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
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
    FixedPersuadee,
    LocalPersuader,
    OutcomeJudge,
    completed_indices,
    dialogue_text,
    load_dataset,
    public_scenario,
    PROTOCOL_VERSION,
    strip_role_prefix,
    write_record,
)

DEFAULT_DATASET = PROJECT / "data/CToMPersu/dataset/CToMPersu_Eval/CToMPersu_Eval.json"
DEFAULT_OUTPUT = PROJECT / "results/proactive_reproduction_5models_4turns_v2/raw"
METHOD = "Zero-shot"


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
        help="Relative result directory below each model directory.",
    )
    parser.add_argument("--max-turns", type=int, default=EXPERIMENT_MAX_TURNS)
    parser.add_argument("--simulator-model", default=DEFAULT_SIMULATOR_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-prompt-version", default=DEFAULT_JUDGE_PROMPT_VERSION)
    parser.add_argument("--base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run a prefix for smoke testing; never report it as a full result.",
    )
    return parser.parse_args()


def direct_prompt(
    scenario: dict[str, Any],
    history: list[dict[str, str]],
    remaining_turns: int = EXPERIMENT_MAX_TURNS,
) -> tuple[str, str]:
    """Build the public-only direct baseline prompt."""

    system = """You are the persuader in a natural multi-turn dialogue.
Generate only the next persuader utterance, in no more than three concise
sentences. Use the public scenario and observable dialogue to address the
latest concern and move toward the exact goal.

Be respectful, autonomy-supportive, and grounded. Do not invent facts,
authorities, urgency, prices, timelines, private motives, or guarantees.
Do not mention hidden labels or write analysis, JSON, or role instructions."""
    user = f"""Public scenario:
{public_scenario(scenario)}

Observable dialogue:
{dialogue_text(history)}

Remaining persuader turns (including this one): {remaining_turns}

Write the next persuader utterance only."""
    return system, user


def run_one(
    index: int,
    item: dict[str, Any],
    persuader: LocalPersuader,
    simulator: FixedPersuadee,
    judge: OutcomeJudge,
    max_turns: int,
) -> dict[str, Any]:
    scenario = item["scenario"]
    history: list[dict[str, str]] = []
    turns: list[dict[str, Any]] = []
    prefix_judgments: list[dict[str, Any]] = []
    success_turn: int | None = None
    acceptance_turn: int | None = None
    stop_reason = "max_turns"

    for turn in range(1, max_turns + 1):
        system, user = direct_prompt(
            scenario,
            history,
            remaining_turns=max_turns - turn + 1,
        )
        generation = persuader.generate(system, user, max_new_tokens=180)
        if generation.error or not generation.text.strip():
            raise RuntimeError(generation.error or "empty_zero_shot_generation")
        utterance = strip_role_prefix(generation.text, "persuader")
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
                "persuader": utterance,
                "persuadee": reply,
                "generation": generation.__dict__,
                "prefix_success": prefix_success,
            }
        )
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
            "planning": "none",
            "context_visibility": "public_scenario_and_observable_history",
            "paper_reference_max_turns": 6,
            "run_max_turns": max_turns,
            "remaining_turns_visible": True,
        },
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

    raw_subdir = Path(args.raw_subdir)
    if raw_subdir.is_absolute() or ".." in raw_subdir.parts:
        raise ValueError("raw-subdir must be a relative path without '..'")
    output = args.output_dir / args.model_name / raw_subdir / "zero-shot.jsonl"
    done = completed_indices(output)
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
                index, item, persuader, simulator, judge, args.max_turns
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
            "protocol_version": PROTOCOL_VERSION,
            "dataset": str(args.dataset),
            "dataset_size": len(data),
            "model_name": args.model_name,
            "max_turns": args.max_turns,
            "paper_reference_max_turns": 6,
            "simulator_model": args.simulator_model,
            "judge_model": args.judge_model,
            "judge_prompt_version": args.judge_prompt_version,
            "context_mode": "public_only",
        }
        return record

    pending = [
        (index, item) for index, item in enumerate(data) if index not in done
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
