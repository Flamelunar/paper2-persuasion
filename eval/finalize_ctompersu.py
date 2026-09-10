#!/usr/bin/env python3
"""Finalize the two-model CToMPersu reproduction after all runs finish.

The script is deliberately fail-closed: it refuses to write the final
publication-style table when any requested model/method is incomplete. Raw
training traces remain append-only JSONL; this command creates the readable
deduplicated and dataset-aligned artifacts only after the 525-index contract
has been satisfied.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from eval.ctompersu_eval import (  # noqa: E402
    aggregate,
    canonicalize,
    check_complete,
    materialize,
    validate_aligned_artifact,
)

DEFAULT_DATASET = PROJECT / "data/CToMPersu/dataset/CToMPersu_Eval/CToMPersu_Eval.json"
# The clean reproduction has its own root; the earlier v2 traces remain
# preserved and must be finalized only with explicit legacy paths.
DEFAULT_RAW = PROJECT / "results"
DEFAULT_OUTPUT = PROJECT / "results/tables"
DEFAULT_ALIGNED = PROJECT / "results"
DEFAULT_MODELS = ("gemma-4-E4B-it", "Meta-Llama-3.1-8B-Instruct")
METHODS = ("Zero-shot", "MA2P")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--aligned-dir", type=Path, default=DEFAULT_ALIGNED)
    parser.add_argument("--canonical-dir", type=Path)
    parser.add_argument("--raw-subdir", default="")
    parser.add_argument("--aligned-subdir", default="")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import json

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or len(dataset) != 525:
        raise ValueError(f"Expected exactly 525 dataset rows, got {len(dataset) if isinstance(dataset, list) else type(dataset).__name__}")

    checks = []
    for model in args.models:
        for method in METHODS:
            result = check_complete(
                args.raw_dir,
                model,
                method,
                total=len(dataset),
                raw_subdir=args.raw_subdir,
            )
            checks.append(result)
    incomplete = [result for result in checks if not result["complete"]]
    if incomplete:
        print("Finalization refused: at least one requested run is incomplete.", file=sys.stderr)
        return 2

    requested_models = set(args.models)
    rows = aggregate(
        args.raw_dir,
        args.dataset,
        args.output_dir,
        models=requested_models,
        methods=set(METHODS),
        raw_subdir=args.raw_subdir,
    )
    expected = {(model, method) for model in args.models for method in METHODS}
    actual = {(row["Model"], row["Method"]) for row in rows}
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(f"Final table is missing requested rows: {missing}")

    aligned_count = materialize(
        args.raw_dir,
        args.dataset,
        args.aligned_dir,
        models=requested_models,
        methods=set(METHODS),
        raw_subdir=args.raw_subdir,
        aligned_subdir=args.aligned_subdir,
    )
    for model in args.models:
        for method, filename in (("Zero-shot", "zero-shot.jsonl"), ("MA2P", "MA2P.jsonl")):
            aligned = (
                args.aligned_dir / model / Path(args.aligned_subdir) / filename.replace(".jsonl", ".json")
            )
            validate_aligned_artifact(aligned, dataset)
    canonical_count = 0
    if args.canonical_dir is not None:
        canonical_count = canonicalize(
            args.raw_dir,
            args.canonical_dir,
            models=requested_models,
            methods=set(METHODS),
            raw_subdir=args.raw_subdir,
        )
    print(f"Finalized {len(rows)} aggregate rows.")
    print(f"Aligned files: {aligned_count} -> {args.aligned_dir}")
    if args.canonical_dir is not None:
        print(f"Canonical files: {canonical_count} -> {args.canonical_dir}")
    print(f"Table files -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
