#!/usr/bin/env python3
"""Update the TrajWeaver row in the CToMPersu paper metrics table.

This is deliberately fail-closed: a raw run must contain exactly the 525
successful records and the blind quality judge must contain 525 valid rows
before any ``TBD`` cell is replaced.  Existing Zero-shot/MA²P values are not
recomputed or rewritten.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from eval.ctompersu_eval import METHOD_FILES, summarize, validate_aligned_artifact  # noqa: E402
from train.ctompersu_common import latest_records  # noqa: E402


DATASET = PROJECT / "data/CToMPersu/dataset/CToMPersu_Eval/CToMPersu_Eval.json"
TABLE = PROJECT / "results/tables/metric_allocation.md"

# Result directories keep the Hugging Face repository name, while the paper
# table uses the shorter display name for the Llama group.  Keep this mapping
# local to the table writer so raw/aligned/quality paths and paired artifacts
# continue to use their canonical directory names.
TABLE_MODEL_NAMES = {
    "Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", default="TrajWeaver-v1", choices=("TrajWeaver-v1",))
    parser.add_argument("--results-root", type=Path, default=PROJECT / "results")
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--raw-subdir", default="raw")
    parser.add_argument("--quality-subdir", default="quality")
    parser.add_argument("--write", action="store_true", help="Write the markdown table; otherwise print the computed row")
    return parser.parse_args()


def _quality_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing blind quality summary: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Quality summary is not an object: {path}")
    if int(value.get("n", -1)) != 525 or int(value.get("requested_n", -1)) != 525:
        raise ValueError(f"Quality summary is incomplete: n={value.get('n')} requested_n={value.get('requested_n')}")
    required = (
        "acr_percent",
        "unf_percent",
        "contextual_groundedness_percent",
        "goal_drift_rate_percent",
    )
    if any(key not in value for key in required):
        raise ValueError(f"Quality summary lacks required metrics: {path}")
    return value


def _quality_rows(path: Path) -> dict[int, dict[str, Any]]:
    """Fail closed on the actual append-only blind-judge artifact."""

    if not path.exists():
        raise FileNotFoundError(f"Missing blind quality JSONL: {path}")
    rows: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Quality row {path}:{line_number} is not an object")
        index = int(value.get("source_index", -1))
        if index in rows:
            # A duplicate is acceptable only when the latest append is a
            # valid success; materialized evaluation still must have one
            # unique record per source index.
            if value.get("status") != "ok":
                raise ValueError(f"Quality row {path}:{line_number} duplicates index {index} with an error")
        rows[index] = value
    expected = set(range(525))
    if set(rows) != expected:
        raise ValueError(f"Quality artifact is not a complete 525-row cover: {path}")
    errors = [index for index, row in rows.items() if row.get("status") != "ok"]
    if errors:
        raise ValueError(f"Quality artifact contains errors at indices {errors[:10]}: {path}")
    return rows


def compute_row(args: argparse.Namespace) -> dict[str, Any]:
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or len(dataset) != 525:
        raise ValueError("CToMPersu Eval must contain exactly 525 rows")
    filename = METHOD_FILES[args.method]
    raw_path = args.results_root / args.model / args.raw_subdir / filename
    records = latest_records(raw_path)
    expected = set(range(len(dataset)))
    in_range = {idx: row for idx, row in records.items() if idx in expected}
    if len(in_range) != 525 or any(row.get("status") != "ok" for row in in_range.values()):
        raise ValueError(
            f"Raw run is incomplete: {raw_path}; records={len(in_range)}/525, "
            f"errors={sum(row.get('status') != 'ok' for row in in_range.values())}"
        )
    aggregate = summarize(records, dataset, args.model, args.method)
    if aggregate.get("N completed") != "525/525" or aggregate.get("Success (%)") is None:
        raise ValueError(f"Raw aggregate is incomplete: {aggregate}")

    aligned_path = args.results_root / args.model / "aligned" / filename.replace(".jsonl", ".json")
    if not aligned_path.exists():
        raise FileNotFoundError(f"Missing aligned artifact: {aligned_path}")
    validate_aligned_artifact(aligned_path, dataset)
    quality_jsonl = args.results_root / args.model / args.quality_subdir / filename
    _quality_rows(quality_jsonl)
    # ``filename`` is a JSONL filename (e.g. ``TrajWeaver-v1.jsonl``).
    # Using ``[:-5]`` leaves the separating dot (``TrajWeaver-v1.``), which
    # produces a non-existent ``TrajWeaver-v1.-summary.json`` path.  Use the
    # actual stem so the summary artifact is resolved exactly.
    quality_path = args.results_root / args.model / args.quality_subdir / f"{Path(filename).stem}-summary.json"
    quality = _quality_summary(quality_path)
    row = {
        "Model": TABLE_MODEL_NAMES.get(args.model, args.model),
        "Method": args.method,
        "N": 525,
        "Max turns": int(aggregate["Max turns"]),
        "Success (%)": float(aggregate["Success (%)"]),
        "Acceptance Score (%)": round(100.0 * float(aggregate["Mean Score"]), 2),
        "Avg_Turn": float(aggregate["Avg_Turn"]),
        "ACR (%)": round(float(quality["acr_percent"]), 2),
        "UNF (%)": round(float(quality["unf_percent"]), 2),
        "Groundedness (%)": round(float(quality["contextual_groundedness_percent"]), 2),
        "Goal Drift Rate (%)": round(float(quality["goal_drift_rate_percent"]), 2),
        "raw_path": str(raw_path),
        "quality_path": str(quality_path),
        "aligned_path": str(aligned_path),
    }
    return row


def _fmt(value: Any, *, decimals: int | None = None) -> str:
    """Format one paper-facing cell without guessing from its magnitude."""

    if isinstance(value, float):
        if decimals is not None:
            return f"{value:.{decimals}f}"
        return str(value)
    return str(value)


def update_table(path: Path, row: dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    current_model: str | None = None
    replaced = False
    for index, line in enumerate(lines):
        if not line.startswith("|") or line.startswith("| ---"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] == "维度":
            continue
        if cells[0]:
            current_model = cells[0]
        if current_model != row["Model"] or cells[1] != row["Method"]:
            continue
        if len(cells) < 11:
            raise ValueError(f"Unexpected main-table row layout at line {index + 1}")
        values = [
            row["Model"], row["Method"], row["N"], row["Max turns"],
            _fmt(row["Success (%)"], decimals=2),
            _fmt(row["Acceptance Score (%)"], decimals=2),
            _fmt(row["Avg_Turn"], decimals=4),
            _fmt(row["ACR (%)"], decimals=2),
            _fmt(row["UNF (%)"], decimals=2),
            _fmt(row["Groundedness (%)"], decimals=2),
            _fmt(row["Goal Drift Rate (%)"], decimals=2),
        ]
        # Keep grouped-model layout: only the first row repeats the model.
        values[0] = "" if index > 0 and lines[index - 1].startswith("|") else row["Model"]
        lines[index] = "| " + " | ".join(str(value) for value in values) + " |"
        replaced = True
        break
    if not replaced:
        raise ValueError(f"Could not find {row['Model']} / {row['Method']} row in {path}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    row = compute_row(args)
    print(json.dumps(row, ensure_ascii=False, indent=2))
    if args.write:
        update_table(args.table, row)
        print(f"updated={args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
