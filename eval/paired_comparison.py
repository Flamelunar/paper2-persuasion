#!/usr/bin/env python3
"""Create paired per-scenario comparisons against the Zero-shot baseline.

The output keeps the 525 source indices paired, rather than subtracting two
aggregate rates.  It is deliberately read-only with respect to raw/aligned/
quality artifacts and writes a separate ``paired/`` directory.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        index = int(item["source_index"] if "source_index" in item else item["index"])
        rows[index] = item
    return rows


def _quality_value(item: dict[str, Any], key: str) -> float:
    value = item.get(key)
    if isinstance(value, dict):
        value = value.get("label" if key == "acr" else "score")
    return float(value)


def _drift(item: dict[str, Any]) -> int:
    drift = item.get("goal_drift", {})
    return int(any(int(turn.get("score", 0)) >= 2 for turn in drift.get("turn_scores", [])))


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sample")
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _bootstrap(values: list[float], *, seed: int, samples: int) -> dict[str, float]:
    if not values:
        raise ValueError("No paired values")
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    return {
        "mean": sum(values) / len(values),
        "ci95_low": _percentile(means, 0.025),
        "ci95_high": _percentile(means, 0.975),
    }


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = args.results_root / args.model
    zero_raw = _read_jsonl(root / args.raw_subdir / "zero-shot.jsonl")
    traj_raw = _read_jsonl(root / args.raw_subdir / "TrajWeaver-v1.jsonl")
    zero_quality = _read_jsonl(root / args.quality_subdir / "zero-shot.jsonl")
    traj_quality = _read_jsonl(root / args.quality_subdir / "TrajWeaver-v1.jsonl")
    expected = set(range(525))
    sources = {
        "zero_raw": zero_raw,
        "traj_raw": traj_raw,
        "zero_quality": zero_quality,
        "traj_quality": traj_quality,
    }
    for name, rows in sources.items():
        if set(rows) != expected:
            raise ValueError(f"{name} does not contain exactly the 525 paired indices")
        if any(row.get("status") != "ok" for row in rows.values()):
            raise ValueError(f"{name} contains non-ok records")

    records: list[dict[str, Any]] = []
    metrics: dict[str, list[float]] = {
        "success_delta_pp": [],
        "acceptance_delta_pp": [],
        "acr_delta_pp": [],
        "unf_delta_pp": [],
        "groundedness_delta_pp": [],
        "goal_drift_delta_pp": [],
    }
    for index in range(525):
        zr, tr = zero_raw[index], traj_raw[index]
        zq, tq = zero_quality[index], traj_quality[index]
        zo = zr.get("outcome", {})
        to = tr.get("outcome", {})
        row = {
            "source_index": index,
            "zero_shot_success": int(bool(zo.get("success"))),
            "trajweaver_success": int(bool(to.get("success"))),
            "success_delta_pp": 100 * (int(bool(to.get("success"))) - int(bool(zo.get("success")))),
            "zero_shot_acceptance_score_percent": 20 * float(zo.get("acceptance_level", 0)),
            "trajweaver_acceptance_score_percent": 20 * float(to.get("acceptance_level", 0)),
            "zero_shot_acr": 100 * _quality_value(zq, "acr"),
            "trajweaver_acr": 100 * _quality_value(tq, "acr"),
            "zero_shot_unf": 25 * _quality_value(zq, "unf"),
            "trajweaver_unf": 25 * _quality_value(tq, "unf"),
            "zero_shot_groundedness": 25 * _quality_value(zq, "groundedness"),
            "trajweaver_groundedness": 25 * _quality_value(tq, "groundedness"),
            "zero_shot_goal_drift": _drift(zq),
            "trajweaver_goal_drift": _drift(tq),
        }
        row.update(
            {
                "acceptance_delta_pp": row["trajweaver_acceptance_score_percent"] - row["zero_shot_acceptance_score_percent"],
                "acr_delta_pp": row["trajweaver_acr"] - row["zero_shot_acr"],
                "unf_delta_pp": row["trajweaver_unf"] - row["zero_shot_unf"],
                "groundedness_delta_pp": row["trajweaver_groundedness"] - row["zero_shot_groundedness"],
                "goal_drift_delta_pp": 100 * (row["trajweaver_goal_drift"] - row["zero_shot_goal_drift"]),
            }
        )
        records.append(row)
        for key in metrics:
            metrics[key].append(float(row[key]))

    summary = {
        "model": args.model,
        "baseline": "Zero-shot",
        "treatment": "TrajWeaver-v1",
        "n": 525,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "paired_metrics": {key: _bootstrap(value, seed=args.seed + i, samples=args.bootstrap_samples) for i, (key, value) in enumerate(metrics.items())},
        "mcnemar": {
            "zero_only": sum(row["zero_shot_success"] == 1 and row["trajweaver_success"] == 0 for row in records),
            "trajweaver_only": sum(row["zero_shot_success"] == 0 and row["trajweaver_success"] == 1 for row in records),
        },
    }
    return records, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--results-root", type=Path, default=PROJECT / "results")
    parser.add_argument("--raw-subdir", default="raw")
    parser.add_argument("--quality-subdir", default="quality")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")
    output_dir = args.output_dir or args.results_root / args.model / "paired"
    records, summary = build(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "zero-shot_vs_TrajWeaver-v1.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "zero-shot_vs_TrajWeaver-v1-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
