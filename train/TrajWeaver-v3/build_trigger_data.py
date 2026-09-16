"""Score paired Goal/Goal+State conditions locally; resume with exact provenance."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from trajweaver_v3.data import (assert_public, encode, file_digest, read_jsonl, write_json)
from trajweaver_v3.utility import label_utility, select_prefixes

HERE = Path(__file__).resolve().parent


def select_planner_prefixes(rows, planner_dir, split, limit, prefix_type):
    """Map validated planner prefixes to public SFT rows.

    The planner target is never copied into utility data.  It only chooses a
    concern-focused observable prefix; paired NLL still supplies the Trigger
    label.  This keeps the Weaver target as the original persuader reply.
    """

    planner_path = planner_dir / f"planner_raw_{split}.jsonl"
    if not planner_path.exists():
        raise ValueError(f"Planner data is missing {planner_path}")
    planner_rows = read_jsonl(planner_path)
    by_source_turn = {}
    for row in rows:
        key = (int(row["source_index"]), int(row["turn_index"]))
        if key in by_source_turn:
            raise ValueError(f"Duplicate SFT source prefix {split}:{key}")
        by_source_turn[key] = row
    selected = []
    seen_scenarios = set()
    for annotation in planner_rows:
        if annotation.get("split") != split or annotation.get("prefix_type") != prefix_type:
            continue
        history = annotation.get("history")
        if not isinstance(history, list):
            raise ValueError(f"Invalid planner history in {planner_path}: {annotation.get('sample_id')}")
        turn_index = sum(1 for message in history if message.get("role") == "persuader")
        key = (int(annotation["source_index"]), turn_index)
        row = by_source_turn.get(key)
        if row is None:
            raise ValueError(f"Planner prefix {annotation.get('sample_id')} does not map to {split}:{key}")
        if row["scenario_id"] in seen_scenarios:
            raise ValueError(f"Duplicate planner scenario {annotation.get('sample_id')}")
        seen_scenarios.add(row["scenario_id"])
        selected.append(row)
    if limit > 0:
        selected = selected[:limit]
    if not selected:
        raise ValueError(f"No planner prefixes for split={split}, prefix_type={prefix_type}")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data/sft")
    parser.add_argument("--output", type=Path, default=HERE / "data/trigger")
    parser.add_argument("--train-prefixes", type=int, default=512)
    parser.add_argument("--dev-prefixes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner-data", type=Path, default=None,
                        help="validated planner artifact directory; selects concern prefixes without using planner targets")
    parser.add_argument("--planner-prefix-type", choices=("concern", "initial"), default="concern")
    args = parser.parse_args()
    from trajweaver_v3.model import load_config, load_model

    if min(args.train_prefixes, args.dev_prefixes) < 0:
        parser.error("Prefix counts must be nonnegative; 0 uses all scenarios")
    config = load_config(args.config)
    settings = config["utility"]
    provenance = {"schema": "v3_paired_reference_nll", "config": config,
        "checkpoint_sha256": file_digest(args.checkpoint / "trajweaver.pt"),
        "checkpoint": str(args.checkpoint.resolve()), "seed": args.seed,
        "train_prefixes": args.train_prefixes, "dev_prefixes": args.dev_prefixes,
        "planner_data": str(args.planner_data.resolve()) if args.planner_data else None,
        "planner_prefix_type": args.planner_prefix_type if args.planner_data else None,
        "planner_sha256": ({split: file_digest(args.planner_data / f"planner_raw_{split}.jsonl")
                            for split in ("train", "dev")} if args.planner_data else None),
        "data_sha256": {split: file_digest(args.data_dir / f"{split}.jsonl") for split in ("train", "dev")}}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != provenance:
            raise ValueError("Existing utility data uses different weights/config/data; choose a new output")
    elif any((args.output / f"{split}.jsonl").exists() for split in ("train", "dev")):
        raise ValueError("Existing utility rows have no provenance manifest")
    write_json(manifest, provenance)
    model, tokenizer, metadata = load_model(config, args.checkpoint)
    if metadata.get("stage") != "sft":
        raise ValueError("Utility labels require a fixed SFT checkpoint, before Trigger training")
    model.set_component("frozen")
    model.eval()
    summary = {}
    for split, limit in (("train", args.train_prefixes), ("dev", args.dev_prefixes)):
        output = args.output / f"{split}.jsonl"
        rows = read_jsonl(args.data_dir / f"{split}.jsonl")
        if any(row["split"] != split for row in rows):
            raise ValueError(f"Wrong split inside {split} source")
        if args.planner_data:
            selected = select_planner_prefixes(rows, args.planner_data, split, limit, args.planner_prefix_type)
            selection_source = f"planner:{args.planner_prefix_type}"
        else:
            selected = select_prefixes(rows, limit, args.seed + int(split == "dev"))
            selection_source = "random_one_prefix_per_scenario"
        records = read_jsonl(output) if output.exists() else []
        done = {row["sample_id"] for row in records}
        expected = {row["sample_id"] for row in selected}
        if len(done) != len(records) or not done <= expected:
            raise ValueError("Duplicate or foreign utility records")
        with output.open("a", encoding="utf-8") as handle:
            for row in selected:
                if row["sample_id"] in done:
                    continue
                prompt, goal, target = encode(tokenizer, row, config["data"])
                scores = model.utility(prompt, goal, target)
                record = {key: value for key, value in row.items() if key != "target"}
                record.update(label_utility(scores["skip_nll"], scores["invoke_nll"], **settings))
                record["reference_sha256"] = hashlib.sha256(row["target"].encode()).hexdigest()
                assert_public(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)
                print(f"{split} {len(records)}/{len(selected)} gain={scores['gain']:.5f}", flush=True)
        summary[split] = {"prefixes": len(records),
            "usable": sum(row["use_for_training"] for row in records),
            "labels": dict(Counter(row["trigger_action"] for row in records if row["use_for_training"])),
            "mean_gain": sum(row["gain"] for row in records) / max(1, len(records)),
            "selection_source": selection_source}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
