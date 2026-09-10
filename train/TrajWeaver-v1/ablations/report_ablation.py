#!/usr/bin/env python3
"""Summarize the fixed-100 mechanism ablations without touching the main table."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = ROOT / "results/Meta-Llama-3.1-8B-Instruct/ablations/trajweaver-mechanism-100"
VARIANTS = ("public-zero-shot", "legacy-v1", "r0-no-memory", "g8-static-goal", "g8bd8-always", "g8bd8-trigger-strict")
UNDERSTAND = re.compile(r"\bi\s+understand\b", re.I)


def latest(path: Path, key: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        if line.strip():
            row = json.loads(line)
            value = row.get(key, row.get("index"))
            if value is not None:
                rows[int(value)] = row
    return rows


def bootstrap(values: list[float], *, seed: int, n: int = 10000) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = random.Random(seed)
    means = [sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(n)]
    means.sort()
    return {
        "mean": sum(values) / len(values),
        "ci95_low": means[int(0.025 * (n - 1))],
        "ci95_high": means[int(0.975 * (n - 1))],
    }


def quality_value(row: dict[str, Any], key: str) -> float:
    value = row[key]
    if isinstance(value, dict):
        value = value.get("label" if key == "acr" else "score")
    return float(value)


def summarize_variant(root: Path, indices: list[int]) -> tuple[dict[str, Any], dict[int, dict[str, float]]]:
    raw = latest(root / "raw.jsonl", "index")
    quality = latest(root / "quality.jsonl", "source_index")
    if set(raw) != set(indices) or set(quality) != set(indices):
        raise ValueError(f"Incomplete variant directory: {root}")
    rows: dict[int, dict[str, float]] = {}
    turn_count = 0
    token_count = 0
    understand_count = 0
    invoke_count = 0
    memory_active_count = 0
    memory_count = 0
    total_success = 0
    acceptance = []
    for index in indices:
        r = raw[index]
        q = quality[index]
        outcome = r.get("outcome", {})
        turns = r.get("turns", [])
        total_success += int(bool(outcome.get("success")))
        acceptance.append(20.0 * float(outcome.get("acceptance_level", 0)))
        turn_success = int(bool(outcome.get("success")))
        row = {
            "success": float(turn_success),
            "acceptance": acceptance[-1],
            "acr": 100.0 * quality_value(q, "acr"),
            "unf": 25.0 * quality_value(q, "unf"),
            "groundedness": 25.0 * quality_value(q, "groundedness"),
            "gdr": 100.0 * float(any(int(item.get("score", 0)) >= 2 for item in q["goal_drift"]["turn_scores"])),
        }
        rows[index] = row
        for turn in turns:
            turn_count += 1
            generation = turn.get("generation", {})
            token_count += int(generation.get("output_tokens", 0))
            understand_count += int(bool(UNDERSTAND.search(str(turn.get("persuader", "")))))
            controller = turn.get("controller", {})
            invoke_count += int(controller.get("trigger_action") == "INVOKE")
            memory_tokens = int(controller.get("memory_tokens", 0))
            memory_active_count += int(memory_tokens > 0)
            memory_count += memory_tokens
    n = len(indices)
    return {
        "n": n,
        "success_percent": 100.0 * total_success / n,
        "acceptance_percent": sum(acceptance) / n,
        "acr_percent": sum(row["acr"] for row in rows.values()) / n,
        "unf_percent": sum(row["unf"] for row in rows.values()) / n,
        "groundedness_percent": sum(row["groundedness"] for row in rows.values()) / n,
        "gdr_percent": sum(row["gdr"] for row in rows.values()) / n,
        "mean_generated_tokens_per_turn": token_count / turn_count if turn_count else 0.0,
        "i_understand_rate_percent": 100.0 * understand_count / turn_count if turn_count else 0.0,
        "invoke_rate_percent": 100.0 * invoke_count / turn_count if turn_count else 0.0,
        "memory_active_rate_percent": 100.0 * memory_active_count / turn_count if turn_count else 0.0,
        "mean_memory_tokens": memory_count / turn_count if turn_count else 0.0,
    }, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "manifests/eval100_seed20260910.json")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    indices = [int(index) for index in manifest["source_indices"]]
    summaries: dict[str, dict[str, Any]] = {}
    per_variant: dict[str, dict[int, dict[str, float]]] = {}
    for variant in VARIANTS:
        summary, rows = summarize_variant(args.root / variant, indices)
        summaries[variant] = summary
        per_variant[variant] = rows

    baseline = per_variant["r0-no-memory"]
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for variant, rows in per_variant.items():
        deltas[variant] = {}
        for metric in ("success", "acceptance", "acr", "unf", "groundedness", "gdr"):
            values = [rows[index][metric] - baseline[index][metric] for index in indices]
            deltas[variant][metric] = bootstrap(values, seed=20260910 + len(deltas[variant]), n=args.bootstrap_samples)

    result = {
        "schema_version": "trajweaver_ablation_report_v1",
        "manifest": str(args.manifest.resolve()),
        "selection_sha256": manifest["selection_sha256"],
        "n": len(indices),
        "baseline_for_deltas": "r0-no-memory",
        "summaries": summaries,
        "delta_vs_r0_no_memory": deltas,
    }
    (args.root / "ablation_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# TrajWeaver-v1 mechanism ablation (fixed 100)",
        "",
        f"Manifest: `{args.manifest}`  |  n={len(indices)}  |  selection_sha256=`{manifest['selection_sha256']}`",
        "",
        "This is a diagnostic subset, not a replacement for the 525-row main result. Existing public-zero-shot and legacy-v1 rows are filtered from their completed canonical runs; they are not recomputed.",
        "",
        "| Variant | Success (%) | Acceptance (%) | ACR (%) | UNF (%) | Groundedness (%) | GDR (%) ↓ | Mean tokens/turn | I understand (%) | Trigger INVOKE (%) | Memory active (%) | Mean memory tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        s = summaries[variant]
        lines.append(
            f"| {variant} | {s['success_percent']:.2f} | {s['acceptance_percent']:.2f} | {s['acr_percent']:.2f} | {s['unf_percent']:.2f} | {s['groundedness_percent']:.2f} | {s['gdr_percent']:.2f} | {s['mean_generated_tokens_per_turn']:.2f} | {s['i_understand_rate_percent']:.2f} | {s['invoke_rate_percent']:.2f} | {s['memory_active_rate_percent']:.2f} | {s['mean_memory_tokens']:.2f} |"
        )
    lines += [
        "",
        "Delta columns below are scenario-paired bootstrap estimates against `r0-no-memory`; they are for mechanism screening only.",
        "",
        "| Variant | ΔSuccess (pp) | ΔAcceptance (pp) | ΔACR (pp) | ΔUNF (pp) | ΔGroundedness (pp) | ΔGDR (pp) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        d = deltas[variant]
        def cell(name: str) -> str:
            x = d[name]
            return f"{x['mean']:+.2f} [{x['ci95_low']:+.2f}, {x['ci95_high']:+.2f}]"
        lines.append(f"| {variant} | {cell('success')} | {cell('acceptance')} | {cell('acr')} | {cell('unf')} | {cell('groundedness')} | {cell('gdr')} |")
    (args.root / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.root / "ablation_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
