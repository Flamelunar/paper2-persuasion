#!/usr/bin/env python3
"""Build goal-aware DPO pairs from externally generated, evaluated candidates.

The candidate generator and evaluator are intentionally external to this
script. This program only preserves their public annotations and selects a
pair; it never invents a response, a judge score, or a hard negative.

Candidates need ``text`` and may provide: ``goal_alignment`` or
``goal_alignment_id``, ``pressure_violation``, ``groundedness``, ``unf``,
``acr``, ``success``, and/or a scalar ``score``. When several are present,
selection follows that order of priority. A drifted candidate (DILUTED,
SUBSTITUTED, or REVERSED) is preferentially used as the rejected answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trajweaver_v1.constants import GOAL_ALIGNMENT_TO_ID
from trajweaver_v1.data import assert_student_public_only, read_jsonl

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=HERE / "data/train.jsonl")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "data/preferences.jsonl")
    parser.add_argument(
        "--minimum-margin",
        type=float,
        default=0.0,
        help="Required scalar-score margin when both selected candidates have score.",
    )
    return parser.parse_args()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _pressure_violation(candidate: dict[str, Any]) -> bool:
    value = candidate.get("pressure_violation", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "violation", "unsafe"}
    return bool(value)


def _alignment_id(candidate: dict[str, Any]) -> int | None:
    value = candidate.get("goal_alignment_id", candidate.get("goal_alignment"))
    if isinstance(value, str):
        return GOAL_ALIGNMENT_TO_ID.get(value.strip().upper())
    numeric = _number(value)
    if numeric is None or not numeric.is_integer():
        return None
    result = int(numeric)
    return result if 0 <= result < len(GOAL_ALIGNMENT_TO_ID) else None


def _metric(candidate: dict[str, Any], key: str) -> tuple[int, float]:
    """A present metric wins over missing, then higher values win."""

    value = _number(candidate.get(key))
    return (value is not None, value if value is not None else 0.0)


def _quality_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    alignment = _alignment_id(candidate)
    # Goal-changing options cannot outrank an on-goal response merely because
    # the response happened to receive a high surface-quality score.
    aligned = alignment is not None and alignment <= 1
    return (
        not _pressure_violation(candidate),
        aligned,
        _metric(candidate, "groundedness"),
        _metric(candidate, "unf"),
        _metric(candidate, "acr"),
        _metric(candidate, "success"),
        _metric(candidate, "score"),
    )


def _select_pair(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], bool] | None:
    """Return chosen, rejected, and whether rejection is a drift hard negative."""

    if len(candidates) < 2:
        return None
    chosen = max(candidates, key=_quality_key)
    remaining = [candidate for candidate in candidates if candidate is not chosen]
    drifted = [
        candidate
        for candidate in remaining
        if (alignment := _alignment_id(candidate)) is not None and alignment >= 2
    ]
    if drifted:
        return chosen, min(drifted, key=_quality_key), True
    return chosen, min(remaining, key=_quality_key), False


def _public_candidate_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "goal_alignment",
        "goal_alignment_id",
        "pressure_violation",
        "groundedness",
        "unf",
        "acr",
        "success",
        "score",
    )
    return {key: candidate[key] for key in allowed if key in candidate}


def main() -> None:
    args = parse_args()
    samples = {str(row["sample_id"]): row for row in read_jsonl(args.samples)}
    output: list[dict[str, Any]] = []
    skipped = {"unknown_sample": 0, "too_few": 0, "tied": 0, "margin": 0}
    for candidate_row in read_jsonl(args.candidates):
        sample_id = str(candidate_row.get("sample_id", ""))
        if sample_id not in samples:
            skipped["unknown_sample"] += 1
            continue
        candidates = [
            item
            for item in candidate_row.get("candidates", [])
            if isinstance(item, dict) and str(item.get("text", "")).strip()
        ]
        selected = _select_pair(candidates)
        if selected is None:
            skipped["too_few"] += 1
            continue
        chosen, rejected, rejected_is_drift = selected
        if _quality_key(chosen) == _quality_key(rejected):
            skipped["tied"] += 1
            continue
        chosen_score = _number(chosen.get("score"))
        rejected_score = _number(rejected.get("score"))
        score_margin = (
            chosen_score - rejected_score
            if chosen_score is not None and rejected_score is not None
            else None
        )
        if score_margin is not None and score_margin <= args.minimum_margin:
            # A structured goal/pressure distinction is enough to orient a
            # pair; scalar score only gates pairs relying purely on score.
            structured_difference = _quality_key(chosen)[:-1] != _quality_key(rejected)[:-1]
            if not structured_difference:
                skipped["margin"] += 1
                continue
        base = samples[sample_id]
        row = {
            key: base[key]
            for key in (
                "sample_id",
                "scenario_id",
                "scenario",
                "history",
                "turn_index",
                "turn_count",
                "max_turns",
                "remaining_turns",
            )
        }
        row.update(
            {
                "chosen": str(chosen["text"]).strip(),
                "rejected": str(rejected["text"]).strip(),
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "score_margin": score_margin,
                "chosen_evaluation": _public_candidate_metadata(chosen),
                "rejected_evaluation": _public_candidate_metadata(rejected),
                "rejected_is_goal_drift_hard_negative": rejected_is_drift,
                "selection_policy": "pressure>goal_alignment>groundedness_unf>acr_success>score",
                "score_source": str(candidate_row.get("score_source", "external")),
            }
        )
        assert_student_public_only(row)
        output.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        f"preferences={len(output)} output={args.output} "
        f"skipped={json.dumps(skipped, sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
