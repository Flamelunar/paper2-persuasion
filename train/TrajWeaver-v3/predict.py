"""Generate a next response from one prepared public JSONL sample, locally."""

import argparse
import json
from pathlib import Path

from trajweaver_v3.data import encode, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--mode", choices=("none", "goal_only", "always", "trigger"), default="trigger")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()
    from trajweaver_v3.model import load_config, load_model
    config = load_config(args.config)
    model, tokenizer, metadata = load_model(config, args.checkpoint)
    if args.mode == "trigger" and not metadata.get("trigger_trained"):
        raise ValueError("Untrained Trigger; select --mode always or train Trigger first")
    model.set_component("frozen")
    model.eval()
    row = {key: value for key, value in read_jsonl(args.input)[args.index].items() if key != "target"}
    prompt, goal, _ = encode(tokenizer, row, config["data"])
    eos = model.backbone.generation_config.eos_token_id
    tokens, control = model.generate(prompt, goal_ids=goal, mode=args.mode,
        max_new_tokens=args.max_new_tokens, eos_token_id=eos if eos is not None else tokenizer.eos_token_id)
    print(json.dumps({"sample_id": row["sample_id"],
        "response": tokenizer.decode(tokens[0], skip_special_tokens=True),
        "controller": control}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
