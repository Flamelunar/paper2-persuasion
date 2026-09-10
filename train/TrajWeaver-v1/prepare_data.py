#!/usr/bin/env python3
"""Build leakage-free train/dev turn data from CToMPersu Full minus Eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajweaver_v1.data import (
    iter_turn_samples,
    load_json_list,
    repair_dialogue,
    scenario_key,
    split_full_excluding_eval,
    write_jsonl,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        type=Path,
        default=PROJECT / "data/CToMPersu/dataset/CToMPersu_Full/CToMPersu.json",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=PROJECT / "data/CToMPersu/dataset/CToMPersu_Eval/CToMPersu_Eval.json",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "data")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_rows = load_json_list(args.full)
    eval_rows = load_json_list(args.eval)
    train_rows, dev_rows, manifest = split_full_excluding_eval(
        full_rows, eval_rows, train_ratio=args.train_ratio, seed=args.seed
    )
    eval_keys = {scenario_key(row["scenario"]) for row in eval_rows}
    if any(scenario_key(row["scenario"]) in eval_keys for row in train_rows + dev_rows):
        raise AssertionError("Eval overlap remains after filtering")

    train_count = write_jsonl(
        args.output_dir / "train.jsonl",
        (sample for row in train_rows for sample in iter_turn_samples(row, "train")),
    )
    dev_count = write_jsonl(
        args.output_dir / "dev.jsonl",
        (sample for row in dev_rows for sample in iter_turn_samples(row, "dev")),
    )
    manifest.update(
        {
            "train_turn_samples": train_count,
            "dev_turn_samples": dev_count,
            "total_turn_samples": train_count + dev_count,
            "train_round_distribution": {
                str(rounds): sum(
                    len(repair_dialogue(row["dialog"])) // 2 == rounds
                    for row in train_rows
                )
                for rounds in (3, 4)
            },
            "dev_round_distribution": {
                str(rounds): sum(
                    len(repair_dialogue(row["dialog"])) // 2 == rounds
                    for row in dev_rows
                )
                for rounds in (3, 4)
            },
            "eval_overlap_after_filter": 0,
            "full_path": str(args.full.resolve()),
            "eval_path": str(args.eval.resolve()),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
