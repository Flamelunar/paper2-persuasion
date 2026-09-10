#!/usr/bin/env python3
"""Single production evaluator for the CToMPersu baseline experiments.

Train scripts write append-only JSONL traces. This module owns aggregation,
completion checks, paper-style metrics, tables, and aligned exports. Tests
import these functions and do not duplicate evaluation logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from train.ctompersu_common import EXPERIMENT_MAX_TURNS, latest_records

DEFAULT_DATASET = PROJECT / "data/CToMPersu/dataset/CToMPersu_Eval/CToMPersu_Eval.json"
DEFAULT_RAW = PROJECT / "results"
DEFAULT_TABLES = PROJECT / "results/tables"
METHOD_FILES = {
    "Zero-shot": "zero-shot.jsonl",
    "MA2P": "MA2P.jsonl",
    "TrajWeaver-v1": "TrajWeaver-v1.jsonl",
}


def _relative_subdir(value: str | Path | None, *, label: str) -> Path:
    """Return a safe optional relative layout component."""

    text = str(value or "").strip()
    if text in {"", "."}:
        return Path()
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a relative directory without '..': {value!r}")
    return path


def _method_path(raw_dir: Path, model: str, filename: str, raw_subdir: str | Path = "") -> Path:
    return raw_dir / model / _relative_subdir(raw_subdir, label="raw_subdir") / filename


def _config(row: dict[str, Any], key: str, default: str = "unknown") -> str:
    value = row.get("run_config")
    return str(value.get(key, default)) if isinstance(value, dict) else default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _success_turn(row: dict[str, Any], max_turns: int) -> int | None:
    value = row.get("success_turn")
    if value is None and isinstance(row.get("outcome"), dict):
        value = row["outcome"].get("success_turn")
    try:
        turn = int(value)
    except (TypeError, ValueError):
        return None
    return turn if 1 <= turn <= max_turns else None


def _pct(numerator: float, denominator: int) -> float | None:
    return round(100.0 * numerator / denominator, 2) if denominator else None


def _mean(values: list[float], digits: int = 4) -> float | None:
    return round(statistics.fmean(values), digits) if values else None


def _domain_names(scenario: dict[str, Any]) -> list[str]:
    """Return every CToMPersu domain label attached to an instance.

    CToMPersu uses multi-label domain lists.  The MA²P paper's Range and SD
    are over all 35 domain labels, so selecting only the first label changes
    the statistic and is not a faithful reproduction.
    """

    value = scenario.get("domain") or scenario.get("tag") or "unknown"
    values = value if isinstance(value, list) else [value]
    names = [str(item) for item in values if str(item).strip()]
    return names or ["unknown"]


def _max_turns(records: list[dict[str, Any]]) -> int:
    values: list[int] = []
    for row in records:
        try:
            value = int(_config(row, "max_turns", str(EXPERIMENT_MAX_TURNS)))
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return max(values, default=EXPERIMENT_MAX_TURNS)


def summarize(
    records: dict[int, dict[str, Any]] | list[dict[str, Any]],
    dataset: list[dict[str, Any]],
    model: str,
    method: str,
) -> dict[str, Any]:
    """Compute one result row; incomplete runs never get a full success rate."""
    total = len(dataset)
    latest = records if isinstance(records, dict) else {int(x["index"]): x for x in records}
    expected_indices = set(range(total))
    in_range = {index: row for index, row in latest.items() if index in expected_indices}
    complete = [x for x in in_range.values() if x.get("status") == "ok"]
    errors = [x for x in in_range.values() if x.get("status") != "ok"]
    pending = len(expected_indices - set(in_range))
    out_of_range = len(set(latest) - expected_indices)
    max_turns = _max_turns(complete)
    outcomes = [x.get("outcome") if isinstance(x.get("outcome"), dict) else {} for x in complete]
    success_count = sum(bool(x.get("success")) for x in outcomes)
    acceptance = [_number(x.get("acceptance_level")) for x in outcomes]
    success_turns = [
        float(turn)
        for row in complete
        if (turn := _success_turn(row, max_turns)) is not None
    ]
    paper_turns = [float(_success_turn(row, max_turns) or max_turns) for row in complete]
    max_stops = sum(x.get("stop_reason") == "max_turns" for x in complete)

    by_domain: dict[str, list[int]] = defaultdict(list)
    for index, row in latest.items():
        if row.get("status") != "ok" or not 0 <= index < len(dataset):
            continue
        outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
        for domain in _domain_names(dataset[index]["scenario"]):
            by_domain[domain].append(int(bool(outcome.get("success"))))
    domain_rates = [sum(values) / len(values) for values in by_domain.values() if values]
    complete_run = (
        len(complete) == total
        and not errors
        and pending == 0
        and out_of_range == 0
    )
    return {
        "Model": model,
        "Method": method,
        "N completed": f"{len(complete)}/{total}",
        "Max turns": max_turns,
        "Persuadee": ", ".join(sorted({_config(x, "simulator_model") for x in complete})),
        "Judge": ", ".join(sorted({_config(x, "judge_model") for x in complete})),
        "Success (%)": _pct(success_count, total) if complete_run else None,
        "Mean Score": _mean([x / 5.0 for x in acceptance]),
        "Level>=3 (%)": _pct(sum(x >= 3 for x in acceptance), len(acceptance)),
        "Avg_Turn": _mean(paper_turns),
        "Successful mean turn": _mean(success_turns),
        "Max-turn Stop (%)": _pct(max_stops, len(complete)),
        # Exact names used by MA²P Table 1.
        "Range": round(max(domain_rates) - min(domain_rates), 4) if domain_rates else None,
        "SD": (
            round(statistics.pstdev(domain_rates), 4)
            if len(domain_rates) > 1
            else 0.0 if domain_rates else None
        ),
        "Domain count": len(domain_rates),
        "Local generation seconds": _mean([_local_generation_seconds(x) for x in complete]),
    }


def _local_generation_seconds(row: dict[str, Any]) -> float:
    traces: list[dict[str, Any]] = []
    for turn in row.get("turns", []):
        if not isinstance(turn, dict):
            continue
        if isinstance(turn.get("generation"), dict):
            traces.append(turn["generation"])
        agents = turn.get("agent_generations")
        if isinstance(agents, dict):
            traces.extend(x for x in agents.values() if isinstance(x, dict))
    return sum(_number(x.get("latency_seconds")) for x in traces)


def check_complete(
    raw_dir: Path,
    model: str,
    method: str,
    total: int = 525,
    *,
    raw_subdir: str | Path = "",
) -> dict[str, Any]:
    if method not in METHOD_FILES:
        raise ValueError(f"Unknown method: {method}")
    path = _method_path(raw_dir, model, METHOD_FILES[method], raw_subdir)
    records = latest_records(path)
    expected = set(range(total))
    in_range = {index: row for index, row in records.items() if index in expected}
    ok = [x for x in in_range.values() if x.get("status") == "ok"]
    errors = [x for x in in_range.values() if x.get("status") != "ok"]
    missing = sorted(expected - set(in_range))
    out_of_range = sorted(set(records) - expected)
    result = {
        "path": str(path),
        "records": len(records),
        "ok": len(ok),
        "errors": len(errors),
        "missing": len(missing),
        "out_of_range": len(out_of_range),
        "complete": len(ok) == total and not errors and not missing and not out_of_range,
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


def aligned_rows(
    dataset: list[dict[str, Any]],
    records: dict[int, dict[str, Any]],
    model: str,
    method: str,
) -> list[dict[str, Any]]:
    """Create a readable, source-index-aligned result dataset.

    The source corpus calls its reference conversation ``dialog`` while the
    run trace calls the generated conversation ``history``.  Those names make
    it too easy to confuse a gold dialogue with a model trajectory.  The
    export therefore presents the model trajectory with a consistent
    ``role``/``content`` turn schema as ``generated_dialogue``.  The reference
    dialogue remains in the immutable source CToMPersu file and is linked by
    ``source_index`` rather than copied 525 times.  The remaining trace
    metadata is retained under ``evaluation``.
    """

    def to_turns(value: Any) -> list[dict[str, str]]:
        """Normalize CToMPersu's ``role: utterance`` strings to turn objects."""

        if not isinstance(value, list):
            return []
        turns: list[dict[str, str]] = []
        for item in value:
            if isinstance(item, dict):
                role = str(item.get("role", "unknown")).strip().lower()
                content = str(item.get("content", "")).strip()
            else:
                text = str(item).strip()
                role_text, separator, content_text = text.partition(":")
                role = role_text.strip().lower() if separator else "unknown"
                content = content_text.strip() if separator else text
            turns.append({"role": role, "content": content})
        return turns

    output: list[dict[str, Any]] = []
    for index, source in enumerate(dataset):
        evaluation = dict(records.get(index, {"index": index, "status": "pending"}))
        evaluation.setdefault("model_name", model)
        evaluation.setdefault("method", method)
        generated = to_turns(evaluation.pop("history", []))
        output.append(
            {
                "source_index": index,
                "scenario": source.get("scenario", {}),
                "generated_dialogue": generated,
                "evaluation": evaluation,
            }
        )
    return output


def materialize(
    raw_dir: Path,
    dataset_path: Path,
    aligned_dir: Path,
    *,
    models: set[str] | None = None,
    methods: set[str] | None = None,
    raw_subdir: str | Path = "",
    aligned_subdir: str | Path = "",
) -> int:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or len(dataset) != 525:
        raise ValueError("The aligned CToMPersu dataset must contain exactly 525 rows")
    written = 0
    for model_dir in sorted(raw_dir.iterdir()) if raw_dir.exists() else []:
        if not model_dir.is_dir():
            continue
        if models is not None and model_dir.name not in models:
            continue
        for method, filename in METHOD_FILES.items():
            if methods is not None and method not in methods:
                continue
            path = model_dir / _relative_subdir(raw_subdir, label="raw_subdir") / filename
            if not path.exists():
                continue
            output = (
                aligned_dir
                / model_dir.name
                / _relative_subdir(aligned_subdir, label="aligned_subdir")
                / filename.replace(".jsonl", ".json")
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(aligned_rows(dataset, latest_records(path), model_dir.name, method), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            written += 1
    return written


def validate_aligned_artifact(path: Path, dataset: list[dict[str, Any]]) -> None:
    """Fail closed unless one aligned export satisfies the final data contract."""

    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != len(dataset):
        raise ValueError(f"{path}: expected {len(dataset)} aligned rows")
    for index, (row, source) in enumerate(zip(rows, dataset)):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {index} is not an object")
        if row.get("source_index") != index:
            raise ValueError(f"{path}: source_index mismatch at row {index}")
        if row.get("scenario") != source.get("scenario"):
            raise ValueError(f"{path}: scenario mismatch at row {index}")
        if "golden_dialogue" in row or "dialog" in row:
            raise ValueError(f"{path}: duplicated golden dialogue at row {index}")
        generated = row.get("generated_dialogue")
        if not isinstance(generated, list) or len(generated) > 2 * EXPERIMENT_MAX_TURNS:
            raise ValueError(f"{path}: invalid generated_dialogue at row {index}")
        if any(
            not isinstance(turn, dict)
            or not isinstance(turn.get("role"), str)
            or not isinstance(turn.get("content"), str)
            for turn in generated
        ):
            raise ValueError(f"{path}: malformed dialogue turn at row {index}")
        evaluation = row.get("evaluation")
        if not isinstance(evaluation, dict) or evaluation.get("status") != "ok":
            raise ValueError(f"{path}: incomplete evaluation at row {index}")
        if "history" in evaluation:
            raise ValueError(f"{path}: duplicated history at row {index}")


def canonicalize(
    raw_dir: Path,
    output_dir: Path,
    *,
    models: set[str] | None = None,
    methods: set[str] | None = None,
    raw_subdir: str | Path = "",
) -> int:
    """Write one de-duplicated JSON array per method for inspection.

    Training remains append-only so an interrupted job can resume safely.  A
    canonical snapshot keeps only the newest record for each dataset index and
    is therefore the human-readable counterpart of the raw JSONL trace.
    """

    written = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_dir in sorted(raw_dir.iterdir()) if raw_dir.exists() else []:
        if not model_dir.is_dir():
            continue
        if models is not None and model_dir.name not in models:
            continue
        for method, filename in METHOD_FILES.items():
            if methods is not None and method not in methods:
                continue
            path = model_dir / _relative_subdir(raw_subdir, label="raw_subdir") / filename
            if not path.exists():
                continue
            records = latest_records(path)
            rows = [records[index] for index in sorted(records)]
            output = output_dir / model_dir.name / filename.replace(".jsonl", ".json")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            written += 1
    return written


def aggregate(
    raw_dir: Path,
    dataset_path: Path,
    output_dir: Path,
    *,
    models: set[str] | None = None,
    methods: set[str] | None = None,
    raw_subdir: str | Path = "",
) -> list[dict[str, Any]]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or len(dataset) != 525:
        raise ValueError("The evaluation dataset must contain exactly 525 rows")
    rows: list[dict[str, Any]] = []
    for model_dir in sorted(raw_dir.iterdir()) if raw_dir.exists() else []:
        if not model_dir.is_dir():
            continue
        if models is not None and model_dir.name not in models:
            continue
        for method, filename in METHOD_FILES.items():
            if methods is not None and method not in methods:
                continue
            path = model_dir / _relative_subdir(raw_subdir, label="raw_subdir") / filename
            if path.exists():
                rows.append(summarize(latest_records(path), dataset, model_dir.name, method))
    baselines = {x["Model"]: x for x in rows if x["Method"] == "Zero-shot"}
    for row in rows:
        base = baselines.get(row["Model"])
        delta = (
            round(float(row["Success (%)"]) - float(base["Success (%)"]), 2)
            if base and row["Success (%)"] is not None and base["Success (%)"] is not None
            else None
        )
        row["Delta (pp)"] = delta

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["Model", "Method"]
    with (output_dir / "ctompersu_main.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# CToMPersu Zero-shot and MA²P comparison",
        "",
        "Dataset: original CToMPersu Eval split (525 rows). The supplied Proactively Induced Persuasion paper uses six turns; this run uses a shared four-turn budget. Persuadee and judge models are fixed across methods.",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    lines.extend(
        "| " + " | ".join("NA" if row.get(field) is None else str(row.get(field)) for field in fields) + " |"
        for row in rows
    )
    (output_dir / "ctompersu_main.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "ctompersu_main.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--method", choices=list(METHOD_FILES))
    parser.add_argument("--raw-subdir", default="")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--aligned-dir", type=Path)
    parser.add_argument("--aligned-subdir", default="")
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        help="Also write de-duplicated per-method JSON arrays from append-only JSONL traces.",
    )
    args = parser.parse_args()
    if args.check:
        if not args.model or not args.method:
            parser.error("--check requires --model and --method")
        result = check_complete(args.raw_dir, args.model, args.method, raw_subdir=args.raw_subdir)
        raise SystemExit(0 if result["complete"] else 2)
    rows = aggregate(args.raw_dir, args.dataset, args.output_dir, raw_subdir=args.raw_subdir)
    if args.materialize:
        aligned_dir = args.aligned_dir or args.raw_dir.parent / "aligned"
        count = materialize(
            args.raw_dir,
            args.dataset,
            aligned_dir,
            raw_subdir=args.raw_subdir,
            aligned_subdir=args.aligned_subdir,
        )
        print(f"Materialized {count} aligned files to {aligned_dir}")
    if args.canonical_dir:
        count = canonicalize(args.raw_dir, args.canonical_dir, raw_subdir=args.raw_subdir)
        print(f"Canonicalized {count} files to {args.canonical_dir}")
    print(f"Wrote {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
