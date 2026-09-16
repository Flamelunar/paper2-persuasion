"""Four-turn evaluation with the project's fixed simulator and final Judge."""

import argparse
import json
import sys
import time
from pathlib import Path

from trajweaver_v3.data import encode, file_digest, public_scenario, read_jsonl, scenario_id, write_json

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=PROJECT / "data/CToMPersu/mysplit/test.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("none", "goal_only", "always", "trigger"), default="trigger")
    parser.add_argument("--limit", type=int, default=0,
                        help="0: all selected rows (or unique scenarios without --preserve-rows)")
    parser.add_argument("--start-index", type=int, default=0,
                        help="zero-based inclusive start in the selected dataset (for independent shards)")
    parser.add_argument("--end-index", type=int, default=None,
                        help="zero-based exclusive end in the selected dataset; default: dataset end")
    parser.add_argument("--preserve-rows", action="store_true",
                        help="evaluate every original dataset row, including duplicate public scenarios")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--simulator-model", default=None)
    parser.add_argument("--judge-model", default=None)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.start_index < 0 or (args.end_index is not None and args.end_index < 0):
        parser.error("--start-index/--end-index must be nonnegative")
    if args.end_index is not None and args.end_index < args.start_index:
        parser.error("--end-index must be >= --start-index")
    if args.limit and (args.start_index or args.end_index is not None):
        parser.error("--limit cannot be combined with --start-index/--end-index")
    sys.path.insert(0, str(PROJECT))
    from train.ctompersu_common import (FixedPersuadee, OutcomeJudge, DEFAULT_SIMULATOR_MODEL,
        DEFAULT_JUDGE_MODEL, DEFAULT_JUDGE_PROMPT_VERSION, DEFAULT_API_BASE_URL, strip_role_prefix)
    from trajweaver_v3.model import load_config, load_model
    import torch

    config = load_config(args.config)
    simulator_model = args.simulator_model or DEFAULT_SIMULATOR_MODEL
    judge_model = args.judge_model or DEFAULT_JUDGE_MODEL
    manifest = {"method": "TrajWeaver-v3", "protocol": "public_only_fixed_4turn_final_judge",
        "config": config, "checkpoint_sha256": file_digest(args.checkpoint / "trajweaver.pt"),
        "dataset_sha256": file_digest(args.dataset), "mode": args.mode,
        "simulator_model": simulator_model, "judge_model": judge_model,
        "judge_prompt_version": DEFAULT_JUDGE_PROMPT_VERSION, "base_url": DEFAULT_API_BASE_URL,
        "max_new_tokens": args.max_new_tokens, "limit": args.limit,
        "start_index": args.start_index, "end_index": args.end_index,
        "preserve_rows": args.preserve_rows,
        "duplicate_policy": "one result per original row" if args.preserve_rows else "one result per unique public scenario"}
    metadata_path = args.output.with_suffix(".manifest.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != manifest:
            raise ValueError("Evaluation settings changed; use a different output file")
    elif args.output.exists():
        raise ValueError("Existing evaluation output has no manifest")
    write_json(metadata_path, manifest)
    completed = read_jsonl(args.output) if args.output.exists() else []
    done_key = "evaluation_id" if args.preserve_rows else "scenario_id"
    done = {row[done_key] for row in completed}
    if len(done) != len(completed):
        raise ValueError("Duplicate completed scenarios")
    raw = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.preserve_rows:
        selected = []
        for index, record in enumerate(raw):
            sid = scenario_id(record["scenario"])
            selected.append((f"{sid}:row{index}", sid, index, record))
    else:
        unique = {}
        for index, record in enumerate(raw):
            unique.setdefault(scenario_id(record["scenario"]), (index, record))
        selected = [(sid, sid, index, record) for sid, (index, record) in unique.items()]
    selected = selected[:args.limit or None]
    end_index = args.end_index if args.end_index is not None else len(selected)
    if args.start_index > len(selected) or end_index > len(selected):
        parser.error(f"row range [{args.start_index}, {end_index}) exceeds {len(selected)} selected rows")
    selected = selected[args.start_index:end_index]
    if not done <= {key for key, _, _, _ in selected}:
        raise ValueError("Completed scenarios are outside selected evaluation set")
    if len(done) == len(selected):
        print(f"Already complete: {len(done)} unique scenarios")
        return
    model, tokenizer, metadata = load_model(config, args.checkpoint)
    if args.mode == "trigger" and not metadata.get("trigger_trained"):
        raise ValueError("Train the Trigger before evaluating --mode trigger")
    model.set_component("frozen")
    model.eval()
    simulator, judge = FixedPersuadee(model=simulator_model), OutcomeJudge(model=judge_model)
    eos = model.backbone.generation_config.eos_token_id
    with args.output.open("a", encoding="utf-8") as handle:
        for evaluation_id, sid, index, record in selected:
            if evaluation_id in done:
                continue
            history, turns, goal_cache = [], [], None
            for turn in range(4):
                row = {"sample_id": f"{sid}:t{turn+1}", "scenario": public_scenario(record["scenario"]),
                       "history": history, "turn_index": turn, "max_turns": 4}
                prompt, goal_ids, _ = encode(tokenizer, row, config["data"])
                started = time.perf_counter()
                if goal_cache is None and args.mode != "none":
                    with torch.no_grad():
                        goal_cache = model.encode_goal(goal_ids).detach()
                tokens, control = model.generate(prompt, goal=goal_cache, mode=args.mode,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=eos if eos is not None else tokenizer.eos_token_id)
                text = strip_role_prefix(tokenizer.decode(tokens[0], skip_special_tokens=True), "persuader")
                latency = time.perf_counter() - started
                if not text.strip():
                    raise ValueError(f"Empty response at {sid}:t{turn+1}")
                history.append({"role": "persuader", "content": text})
                reply = simulator.respond(record["scenario"], history, final_turn=turn == 3)
                history.append({"role": "persuadee", "content": reply["utterance"]})
                turns.append({"turn": turn+1, "persuader": text, "persuadee": reply,
                    "controller": control, "generation": {"input_tokens": prompt.shape[1],
                    "output_tokens": tokens.shape[1], "latency_seconds": latency}})
            result = {"index": index, "evaluation_id": evaluation_id, "scenario_id": sid,
                      "status": "ok", "method": "TrajWeaver-v3",
                      "protocol_version": manifest["protocol"], "scenario": public_scenario(record["scenario"]),
                      "stop_reason": "max_turns", "turn_count": 4, "turns": turns,
                      "history": history, "outcome": judge.evaluate_final(record["scenario"], history)}
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"completed {sid}", flush=True)


if __name__ == "__main__":
    main()
