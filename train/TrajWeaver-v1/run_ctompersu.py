#!/usr/bin/env python3
"""Evaluate a trained TrajWeaver-v1 checkpoint on CToMPersu Eval."""

from __future__ import annotations

import argparse
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from train.ctompersu_common import (  # noqa: E402
    DEFAULT_API_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PROMPT_VERSION,
    DEFAULT_SIMULATOR_MODEL,
    EXPERIMENT_MAX_TURNS,
    FixedPersuadee,
    Generation,
    OutcomeJudge,
    completed_indices,
    load_dataset,
    strip_role_prefix,
    write_record,
)
from trajweaver_v1.constants import (  # noqa: E402
    ACTIONS,
    BELIEF_STATES,
    DESIRE_STATES,
    GOAL_ALIGNMENT_STATES,
    METHOD,
    ROUTE_STATES,
    TRIGGER_ACTIONS,
)
from trajweaver_v1.data import public_scenario  # noqa: E402
from trajweaver_v1.modeling import (  # noqa: E402
    load_model_and_tokenizer,
    load_yaml,
)
from trajweaver_v1.trainer import tokenize_goal, tokenize_prompt  # noqa: E402

DEFAULT_DATASET = PROJECT / "data/CToMPersu/dataset/CToMPersu_Eval/CToMPersu_Eval.json"
DEFAULT_OUTPUT = PROJECT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Results root; output is <root>/<model>/<raw-subdir>/TrajWeaver-v1.jsonl",
    )
    parser.add_argument("--raw-subdir", default="raw")
    parser.add_argument("--max-turns", type=int, default=EXPERIMENT_MAX_TURNS)
    parser.add_argument("--simulator-model", default=DEFAULT_SIMULATOR_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-prompt-version", default=DEFAULT_JUDGE_PROMPT_VERSION)
    parser.add_argument("--base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    return parser.parse_args()


class LocalTrajWeaver:
    def __init__(self, config: dict[str, Any], checkpoint: Path) -> None:
        self.model, self.tokenizer = load_model_and_tokenizer(
            config, for_training=False, checkpoint=checkpoint
        )
        self.max_prompt_tokens = int(config["data"].get("max_prompt_tokens", 1024))
        self.max_goal_tokens = int(config["data"].get("max_goal_tokens", 256))
        self.lock = threading.Lock()

    def encode_goal(self, scenario: dict[str, Any]) -> Any:
        """Encode a goal once per dialogue; never mix in dialogue history."""

        import torch

        sample = {"scenario": public_scenario(scenario)}
        with self.lock, torch.no_grad():
            goal_ids = tokenize_goal(self.tokenizer, sample, self.max_goal_tokens)
            return self.model.encode_goal(goal_ids).detach()

    def generate(
        self,
        scenario: dict[str, Any],
        history: list[dict[str, str]],
        turn_index: int,
        max_turns: int,
        max_new_tokens: int,
        goal_tokens: Any,
    ) -> tuple[Generation, dict[str, Any]]:
        sample = {
            "scenario": public_scenario(scenario),
            "history": history,
            "turn_index": turn_index,
            "max_turns": max_turns,
        }
        started = time.perf_counter()
        try:
            with self.lock:
                prompt_ids = tokenize_prompt(self.tokenizer, sample, self.max_prompt_tokens)
                generated, control = self.model.generate(
                    prompt_ids,
                    goal_tokens=goal_tokens,
                    turn_index=turn_index,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
            result = Generation(
                text=text,
                input_tokens=int(prompt_ids.shape[1]),
                output_tokens=int(generated.shape[1]),
                latency_seconds=time.perf_counter() - started,
            )
            metadata = {
                "route_state": ROUTE_STATES[int(control.route_ids.item())],
                "action": ACTIONS[int(control.action_ids.item())],
                "route_probabilities": {
                    name: float(probability)
                    for name, probability in zip(
                        ROUTE_STATES,
                        control.route_logits.float().softmax(dim=-1)[0].cpu().tolist(),
                    )
                },
                "action_probabilities": {
                    name: float(probability)
                    for name, probability in zip(
                        ACTIONS,
                        control.action_logits.float().softmax(dim=-1)[0].cpu().tolist(),
                    )
                },
                "memory_tokens": int(control.memory_tokens.shape[1]),
                "static_goal_tokens": int(control.goal_tokens.shape[1]),
                "dynamic_belief_tokens": int(control.belief_tokens.shape[1]),
                "dynamic_desire_tokens": int(control.desire_tokens.shape[1]),
                "trigger_action": TRIGGER_ACTIONS[int(control.trigger_ids.item())],
                "trigger_probabilities": {
                    name: float(probability)
                    for name, probability in zip(
                        TRIGGER_ACTIONS,
                        control.trigger_logits.float().softmax(dim=-1)[0].cpu().tolist(),
                    )
                },
                "belief_state": BELIEF_STATES[int(control.belief_ids.item())],
                "desire_state": DESIRE_STATES[int(control.desire_ids.item())],
                "goal_alignment": GOAL_ALIGNMENT_STATES[
                    int(control.goal_alignment_ids.item())
                ],
            }
            return result, metadata
        except Exception as exc:
            return (
                Generation(
                    text="",
                    input_tokens=0,
                    output_tokens=0,
                    latency_seconds=time.perf_counter() - started,
                    error=f"{type(exc).__name__}:{exc}",
                ),
                {},
            )

    def close(self) -> None:
        import torch

        del self.model
        torch.cuda.empty_cache()


def run_one(
    index: int,
    item: dict[str, Any],
    persuader: LocalTrajWeaver,
    simulator: FixedPersuadee,
    judge: OutcomeJudge,
    max_turns: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    scenario = item["scenario"]
    history: list[dict[str, str]] = []
    turns: list[dict[str, Any]] = []
    prefix_judgments: list[dict[str, Any]] = []
    success_turn: int | None = None
    acceptance_turn: int | None = None
    stop_reason = "max_turns"
    # These eight tokens are immutable over all turns in this dialogue.
    # Only the belief/desire state tokens are recomputed each turn.
    goal_tokens = persuader.encode_goal(scenario)
    for turn in range(1, max_turns + 1):
        generation, controller = persuader.generate(
            scenario, history, turn - 1, max_turns, max_new_tokens, goal_tokens
        )
        if generation.error or not generation.text.strip():
            raise RuntimeError(generation.error or "empty_trajweaver_generation")
        utterance = strip_role_prefix(generation.text, "persuader")
        history.append({"role": "persuader", "content": utterance})
        reply = simulator.respond(scenario, history, final_turn=(turn == max_turns))
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
                # ``generate`` already converts the tensor controller output
                # into JSON-safe labels/probabilities.  Never write the
                # ``TrajWeaverOutput`` tensor container to the raw trace.
                "controller": controller,
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
        "protocol_version": "trajweaver_v1_public_only_4turn",
        "method_config": {
            "architecture": "frozen_reasoner_dual_adapter_16_latent_tokens",
            "goal_queries": 8,
            "belief_queries": 4,
            "desire_queries": 4,
            "goal_memory": "encoded_once_from_original_goal_and_cached_per_dialogue",
            "state_memory": "recomputed_from_public_history_each_turn",
            "weaver_adapter": "independent_lora",
            "trigger_adapter": "independent_lora_binary_skip_invoke",
            "first_action": "PROBE",
            "reasoner_decodes_per_turn": 1,
            "context_visibility": "public_scenario_and_observable_history",
            "run_max_turns": max_turns,
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
        raise ValueError(f"max_turns must be in [1, {EXPERIMENT_MAX_TURNS}]")
    data = load_dataset(args.dataset)
    if args.limit is not None:
        data = data[: max(0, args.limit)]
    raw_subdir = Path(args.raw_subdir)
    if raw_subdir.is_absolute() or ".." in raw_subdir.parts:
        raise ValueError("--raw-subdir must be a relative path without '..'")
    output = args.output_dir / args.model_name / raw_subdir / f"{METHOD}.jsonl"
    done = completed_indices(output)
    config = load_yaml(args.config)
    persuader = LocalTrajWeaver(config, args.checkpoint)
    simulator = FixedPersuadee(args.simulator_model, args.base_url)
    judge = OutcomeJudge(
        args.judge_model, args.base_url, prompt_version=args.judge_prompt_version
    )
    output_lock = threading.Lock()

    def execute(index: int, item: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            record = run_one(
                index, item, persuader, simulator, judge, args.max_turns, args.max_new_tokens
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
            "protocol_version": "trajweaver_v1_public_only_4turn",
            "dataset": str(args.dataset),
            "dataset_size": len(data),
            "raw_subdir": str(raw_subdir),
            "model_name": args.model_name,
            "max_turns": args.max_turns,
            "simulator_model": args.simulator_model,
            "judge_model": args.judge_model,
            "judge_prompt_version": args.judge_prompt_version,
            "context_mode": "public_only",
            "checkpoint": str(args.checkpoint),
        }
        return record

    pending = [(index, item) for index, item in enumerate(data) if index not in done]
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(execute, index, item): index for index, item in pending
            }
            for future in as_completed(futures):
                record = future.result()
                with output_lock:
                    write_record(output, record)
                print(
                    f"{args.model_name} {METHOD} index={record['index']} status={record['status']}",
                    flush=True,
                )
    finally:
        persuader.close()


if __name__ == "__main__":
    main()
