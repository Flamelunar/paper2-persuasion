#!/usr/bin/env python3
"""Create the immutable 100-example manifest used by all ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "data/CToMPersu/dataset/CToMPersu_Eval/CToMPersu_Eval.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "manifests/eval100_seed20260910.json"
PUBLIC_FIELDS = ("tag", "background", "persuadee", "persuader", "goal", "domain")


def digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(dataset_path: Path, *, sample_size: int, seed: int) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Eval dataset must be a non-empty JSON list")
    if sample_size < 1 or sample_size > len(dataset):
        raise ValueError(f"sample_size must be in [1, {len(dataset)}]")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(dataset)), sample_size))
    rows = []
    for index in indices:
        scenario = dataset[index].get("scenario")
        if not isinstance(scenario, dict) or not str(scenario.get("goal", "")).strip():
            raise ValueError(f"Missing public scenario/goal at source_index={index}")
        public = {field: scenario.get(field) for field in PUBLIC_FIELDS}
        rows.append(
            {
                "source_index": index,
                "scenario_hash": digest(public),
                "goal": str(scenario["goal"]),
            }
        )
    return {
        "schema_version": "trajweaver_ablation_manifest_v1",
        "dataset": str(dataset_path.resolve()),
        "dataset_size": len(dataset),
        "dataset_sha256": digest(dataset),
        "sampling": {"kind": "random_sample_sorted", "seed": seed, "size": sample_size},
        "source_indices": indices,
        "rows": rows,
        "selection_sha256": digest(indices),
    }


def load_manifest(path: Path, dataset_path: Path) -> tuple[dict[str, Any], list[int]]:
    """Load and fail closed on a manifest/data mismatch."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or manifest.get("dataset_size") != len(dataset):
        raise ValueError("Manifest dataset_size does not match the supplied dataset")
    if manifest.get("dataset_sha256") != digest(dataset):
        raise ValueError("Manifest dataset hash does not match the supplied dataset")
    indices = [int(value) for value in manifest.get("source_indices", [])]
    if len(indices) != int(manifest.get("sampling", {}).get("size", 0)):
        raise ValueError("Manifest source_indices length does not match sampling.size")
    if len(indices) != 100 or len(set(indices)) != 100:
        raise ValueError("The fast ablation manifest must contain exactly 100 unique indices")
    if indices != sorted(indices) or any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError("Manifest indices must be sorted and in dataset range")
    for row, index in zip(manifest.get("rows", []), indices):
        scenario = dataset[index].get("scenario", {})
        public = {field: scenario.get(field) for field in PUBLIC_FIELDS}
        if int(row.get("source_index", -1)) != index or row.get("scenario_hash") != digest(public):
            raise ValueError(f"Manifest scenario hash mismatch at source_index={index}")
    return manifest, indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    manifest = build_manifest(args.dataset, sample_size=args.size, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("dataset_size", "sampling", "selection_sha256")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
