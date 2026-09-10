#!/usr/bin/env python3
"""Build scenario-safe Weaver-SFT and Trigger-SFT datasets.

The canonical leakage-free turn files remain the source of truth.  This script
materializes only the requested pilot scale and writes scenario-ID manifests
for 1k, 2k, and full-scale runs, avoiding three duplicated copies of the
31-MiB canonical training file.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from trajweaver_v1.constants import TRIGGER_TO_ID  # noqa: E402
from trajweaver_v1.data import read_jsonl, write_jsonl  # noqa: E402
from trajweaver_v1.teacher_adapter import merge_annotations  # noqa: E402
from trajweaver_v1.terra import (  # noqa: E402
    TERRA_LABEL_SOURCE,
    TERRA_PROMPT_VERSION,
    TERRA_TEACHER_MODEL,
    read_terra_annotations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=HERE / "data/train.jsonl")
    parser.add_argument("--dev", type=Path, default=HERE / "data/dev.jsonl")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=HERE / "data/terra/annotations.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "data/lora")
    parser.add_argument(
        "--train-scenarios",
        type=int,
        default=2000,
        help="Use 0 for every canonical training scenario.",
    )
    parser.add_argument("--dev-scenarios", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ordered_scenario_ids(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for row in rows:
        sid = str(row["scenario_id"])
        if sid not in seen:
            seen.add(sid)
            result.append(sid)
    return result


def select_ids(
    rows: list[dict[str, Any]], count: int, *, seed: int
) -> tuple[list[str], set[str]]:
    ids = ordered_scenario_ids(rows)
    random.Random(seed).shuffle(ids)
    selected = ids if count <= 0 else ids[: min(count, len(ids))]
    return selected, set(selected)


def subset(rows: list[dict[str, Any]], selected: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["scenario_id"]) in selected]


def weaver_rows(rows: list[dict[str, Any]], scale: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["lora_component"] = "memory_weaver"
        row["dataset_scale"] = scale
        output.append(row)
    return output


def trigger_rows(rows: list[dict[str, Any]], scale: str) -> list[dict[str, Any]]:
    keep = (
        "sample_id",
        "scenario_id",
        "split",
        "scenario",
        "history",
        "turn_index",
        "turn_count",
        "max_turns",
        "remaining_turns",
        "label_source",
    )
    output: list[dict[str, Any]] = []
    for source in rows:
        row = {key: source[key] for key in keep if key in source}
        if int(source.get("trigger_id", -100)) >= 0:
            trigger_id = int(source["trigger_id"])
            trigger_action = str(source.get("trigger_action", ""))
            label_source = str(source.get("label_source", "teacher"))
        else:
            # Bootstrap only: no dynamic state exists before the first reply;
            # after a reply, the state Weaver is useful. Main training requires
            # teacher coverage unless explicitly run as a smoke test.
            trigger_action = "SKIP" if int(source["turn_index"]) == 0 else "INVOKE"
            trigger_id = TRIGGER_TO_ID[trigger_action]
            label_source = "bootstrap_turn_boundary"
        row.update(
            {
                "trigger_action": trigger_action,
                "trigger_id": trigger_id,
                "trigger_label_source": label_source,
                "lora_component": "memory_trigger",
                "dataset_scale": scale,
            }
        )
        output.append(row)
    return output


def write_ids(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(args.train)
    dev_rows = read_jsonl(args.dev)
    annotations_loaded = args.annotations.exists()
    if annotations_loaded:
        annotations = (
            read_terra_annotations(args.annotations)
            if args.annotations.parent.name == "terra"
            else read_jsonl(args.annotations)
        )
        train_rows = merge_annotations(train_rows, annotations)
        dev_rows = merge_annotations(dev_rows, annotations)

    all_train_ids = ordered_scenario_ids(train_rows)
    shuffled_train_ids = list(all_train_ids)
    random.Random(args.seed).shuffle(shuffled_train_ids)
    train_ids, train_set = select_ids(
        train_rows, args.train_scenarios, seed=args.seed
    )
    dev_ids, dev_set = select_ids(
        dev_rows, args.dev_scenarios, seed=args.seed + 1
    )
    selected_train = subset(train_rows, train_set)
    selected_dev = subset(dev_rows, dev_set)
    train_scale = f"scenarios-{len(train_ids)}"
    dev_scale = f"scenarios-{len(dev_ids)}"

    weaver_dir = args.output_dir / "weaver_sft"
    trigger_dir = args.output_dir / "trigger_sft"
    weaver_train_path = weaver_dir / f"train.{train_scale}.jsonl"
    weaver_dev_path = weaver_dir / f"dev.{dev_scale}.jsonl"
    trigger_train_path = trigger_dir / f"train.{train_scale}.jsonl"
    trigger_dev_path = trigger_dir / f"dev.{dev_scale}.jsonl"
    counts = {
        "weaver_train_turns": write_jsonl(
            weaver_train_path, weaver_rows(selected_train, train_scale)
        ),
        "weaver_dev_turns": write_jsonl(
            weaver_dev_path, weaver_rows(selected_dev, dev_scale)
        ),
        "trigger_train_turns": write_jsonl(
            trigger_train_path, trigger_rows(selected_train, train_scale)
        ),
        "trigger_dev_turns": write_jsonl(
            trigger_dev_path, trigger_rows(selected_dev, dev_scale)
        ),
    }

    split_dir = args.output_dir / "scenario_ids"
    write_ids(split_dir / "train.scenarios-1000.txt", shuffled_train_ids[:1000])
    write_ids(split_dir / "train.scenarios-2000.txt", shuffled_train_ids[:2000])
    write_ids(split_dir / f"train.scenarios-{len(all_train_ids)}.txt", shuffled_train_ids)
    write_ids(split_dir / f"dev.scenarios-{len(dev_ids)}.txt", dev_ids)

    manifest = {
        "seed": args.seed,
        "canonical_train": str(args.train.resolve()),
        "canonical_dev": str(args.dev.resolve()),
        "annotations": str(args.annotations.resolve()) if annotations_loaded else None,
        "annotation_status": "terra_merged" if annotations_loaded else "bootstrap_only",
        "teacher_model": TERRA_TEACHER_MODEL if annotations_loaded else None,
        "teacher_prompt_version": TERRA_PROMPT_VERSION if annotations_loaded else None,
        "annotation_label_source": TERRA_LABEL_SOURCE if annotations_loaded else None,
        "recommended_first_run": "2000 training scenarios",
        "data_contract": {
            "weaver_component": "weaver_lora + latent_queries + projections + controller",
            "weaver_teacher_labels_required_for_main_run": [
                "belief_state",
                "desire_state",
                "goal_alignment",
                "trigger_action",
            ],
            "trigger_component": "trigger_lora + binary_skip_invoke_head",
            "bootstrap_trigger_policy": "turn_1=SKIP; later_turns=INVOKE; smoke_only",
            "full_scale_selection": "data/lora/scenario_ids/train.scenarios-5279.txt",
        },
        "selected_train_scenarios": len(train_ids),
        "selected_dev_scenarios": len(dev_ids),
        "canonical_train_scenarios": len(all_train_ids),
        "weaver_train": str(weaver_train_path.resolve()),
        "weaver_dev": str(weaver_dev_path.resolve()),
        "trigger_train": str(trigger_train_path.resolve()),
        "trigger_dev": str(trigger_dev_path.resolve()),
        **counts,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
