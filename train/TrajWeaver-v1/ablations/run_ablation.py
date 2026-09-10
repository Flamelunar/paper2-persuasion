#!/usr/bin/env python3
"""Run one TrajWeaver mechanism ablation on the frozen 100-row manifest."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
METHOD_ROOT = HERE.parent
PROJECT = METHOD_ROOT.parents[1]
for path in (PROJECT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
    latest_records,
    strip_role_prefix,
    write_record,
)
from trajweaver_v1.constants import (  # noqa: E402
    ACTIONS,
    BELIEF_STATES,
    DESIRE_STATES,
    GOAL_ALIGNMENT_STATES,
    ROUTE_STATES,
    TRIGGER_ACTIONS,
)
from trajweaver_v1.data import public_scenario  # noqa: E402
from trajweaver_v1.modeling import load_model_and_tokenizer, load_yaml  # noqa: E402
from trajweaver_v1.trainer import tokenize_goal, tokenize_prompt  # noqa: E402

from make_manifest import digest  # noqa: E402

DEFAULT_DATASET = PROJECT / "data/CToMPersu/dataset/CToMPersu_Eval/CToMPersu_Eval.json"
DEFAULT_MANIFEST = HERE / "manifests/eval100_seed20260910.json"
DEFAULT_CONFIG = METHOD_ROOT / "configs/llama31_8b.yaml"
DEFAULT_CHECKPOINT = METHOD_ROOT / "checkpoints/llama31-8b-terra-trigger/trigger-epoch-1"
DEFAULT_OUTPUT = PROJECT / "results/Meta-Llama-3.1-8B-Instruct/ablations/trajweaver-mechanism-100"
VARIANTS = ("r0-no-memory", "g8-static-goal", "g8bd8-always", "g8bd8-trigger-strict")
MODEL_NAME = "Meta-Llama-3.1-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-turns", type=int, default=EXPERIMENT_MAX_TURNS)
    parser.add_argument("--simulator-model", default=DEFAULT_SIMULATOR_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-prompt-version", default=DEFAULT_JUDGE_PROMPT_VERSION)
    parser.add_argument("--base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    return parser.parse_args()


def load_manifest(path: Path, dataset_path: Path) -> tuple[dict[str, Any], list[int]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    dataset = load_dataset(dataset_path)
    if manifest.get("dataset_size") != len(dataset):
        raise ValueError("Manifest dataset_size does not match the supplied dataset")
    if manifest.get("dataset_sha256") != digest(dataset):
        raise ValueError("Manifest dataset hash does not match the supplied dataset")
    indices = [int(value) for value in manifest.get("source_indices", [])]
    if len(indices) != 100 or len(set(indices)) != 100:
        raise ValueError("The ablation manifest must contain exactly 100 unique indices")
    if indices != sorted(indices) or any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError("Manifest indices must be sorted and in dataset range")
    for row, index in zip(manifest.get("rows", []), indices):
        public = public_scenario(dataset[index]["scenario"])
        if int(row.get("source_index", -1)) != index or row.get("scenario_hash") != digest(public):
            raise ValueError(f"Manifest scenario hash mismatch at source_index={index}")
    return manifest, indices


class AblationPersuader:
    def __init__(self, config: dict[str, Any], checkpoint: Path, variant: str) -> None:
        self.variant = variant
        self.memory_mode = {
            "r0-no-memory": "none",
            "g8-static-goal": "goal_only",
            "g8bd8-always": "always",
            "g8bd8-trigger-strict": "trigger_strict",
        }[variant]
        self.model, self.tokenizer = load_model_and_tokenizer(
            config, for_training=False, checkpoint=checkpoint
        )
        self.max_prompt_tokens = int(config["data"].get("max_prompt_tokens", 1024))
        self.max_goal_tokens = int(config["data"].get("max_goal_tokens", 256))
        self.lock = threading.Lock()

    def encode_goal(self, scenario: dict[str, Any]) -> Any:
        if self.memory_mode == "none":
            return None
        sample = {"scenario": public_scenario(scenario)}
        with self.lock:
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
                    memory_mode=self.memory_mode,
                )
                text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
            generation = Generation(
                text=text,
                input_tokens=int(prompt_ids.shape[1]),
                output_tokens=int(generated.shape[1]),
                latency_seconds=time.perf_counter() - started,
            )
            return generation, {
                "memory_mode": self.memory_mode,
                "route_state": ROUTE_STATES[int(control.route_ids.item())],
                "action": ACTIONS[int(control.action_ids.item())],
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
                "goal_alignment": GOAL_ALIGNMENT_STATES[int(control.goal_alignment_ids.item())],
            }
        except Exception as exc:
            return Generation(
                text="",
                input_tokens=0,
                output_tokens=0,
                latency_seconds=time.perf_counter() - started,
                error=f"{type(exc).__name__}:{exc}",
            ), {}

    def close(self) -> None:
        import torch

        del self.model
        torch.cuda.empty_cache()


def run_one(
    index: int,
    item: dict[str, Any],
    persuader: AblationPersuader,
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
    goal_tokens = persuader.encode_goal(scenario)
    for turn in range(1, max_turns + 1):
        generation, controller = persuader.generate(
            scenario, history, turn - 1, max_turns, max_new_tokens, goal_tokens
        )
        if generation.error or not generation.text.strip():
            raise RuntimeError(generation.error or "empty_ablation_generation")
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
        "source_index": index,
        "status": "ok",
        "method": f"TrajWeaver-ablation:{persuader.variant}",
        "protocol_version": "trajweaver_ablation_v1_100",
        "stop_reason": stop_reason,
        "turn_count": len(turns),
        "success_turn": success_turn,
        "acceptance_turn": acceptance_turn,
        "prefix_judgments": prefix_judgments,
        "turns": turns,
        "history": history,
        "outcome": outcome,
    }


def main() -> int:
    args = parse_args()
    if args.max_turns < 1 or args.max_turns > EXPERIMENT_MAX_TURNS:
        raise ValueError(f"max_turns must be in [1, {EXPERIMENT_MAX_TURNS}]")
    manifest, indices = load_manifest(args.manifest, args.dataset)
    dataset = load_dataset(args.dataset)
    selected = {index: dataset[index] for index in indices}
    output = args.output_root / args.variant / "raw.jsonl"
    if output.exists():
        for index, row in latest_records(output).items():
            config_row = row.get("run_config")
            if not isinstance(config_row, dict):
                raise ValueError(f"Refusing to resume unproven ablation record at index={index}")
            if config_row.get("variant") != args.variant:
                raise ValueError(f"Refusing to mix ablation variants at index={index}")
            if config_row.get("selection_sha256") != manifest["selection_sha256"]:
                raise ValueError(f"Refusing to mix ablation manifests at index={index}")
            if str(Path(config_row.get("checkpoint", "")).resolve()) != str(args.checkpoint.resolve()):
                raise ValueError(f"Refusing to mix checkpoints at index={index}")
    done = completed_indices(output)
    config = load_yaml(args.config)
    persuader = AblationPersuader(config, args.checkpoint, args.variant)
    simulator = FixedPersuadee(args.simulator_model, args.base_url)
    judge = OutcomeJudge(args.judge_model, args.base_url, prompt_version=args.judge_prompt_version)
    output_lock = threading.Lock()

    def execute(index: int) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            record = run_one(
                index, selected[index], persuader, simulator, judge,
                args.max_turns, args.max_new_tokens,
            )
        except Exception as exc:
            record = {
                "index": index,
                "source_index": index,
                "status": "error",
                "method": f"TrajWeaver-ablation:{args.variant}",
                "error": f"{type(exc).__name__}:{exc}",
                "traceback": traceback.format_exc(limit=6),
            }
        record["wall_seconds"] = round(time.perf_counter() - started, 6)
        record["run_config"] = {
            "protocol_version": "trajweaver_ablation_v1_100",
            "variant": args.variant,
            "memory_mode": persuader.memory_mode,
            "manifest": str(args.manifest.resolve()),
            "selection_sha256": manifest["selection_sha256"],
            "dataset": str(args.dataset.resolve()),
            "dataset_size": len(dataset),
            "selected_size": len(indices),
            "model_name": args.model_name,
            "max_turns": args.max_turns,
            "simulator_model": args.simulator_model,
            "judge_model": args.judge_model,
            "judge_prompt_version": args.judge_prompt_version,
            "checkpoint": str(args.checkpoint.resolve()),
        }
        return record

    pending = [index for index in indices if index not in done]
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(execute, index): index for index in pending}
            for future in as_completed(futures):
                record = future.result()
                with output_lock:
                    write_record(output, record)
                print(f"{args.variant} source_index={record['index']} status={record['status']}", flush=True)
    finally:
        persuader.close()
    latest = completed_indices(output)
    if not set(indices).issubset(latest):
        missing = sorted(set(indices) - latest)
        raise SystemExit(f"Ablation incomplete; missing successful indices: {missing}")
    print(json.dumps({"variant": args.variant, "n": len(indices), "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
