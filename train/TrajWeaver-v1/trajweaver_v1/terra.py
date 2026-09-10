"""Validation and coverage contracts for direct Codex Terra annotations.

The direct teacher pass happens outside this module.  This module deliberately
has no network or model-client dependency: it accepts append-only label rows,
validates them against the public turn source, and produces auditable coverage
metadata.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    ACTION_TO_ID,
    BELIEF_TO_ID,
    DESIRE_TO_ID,
    GOAL_ALIGNMENT_TO_ID,
    PRIVATE_FIELDS,
    ROUTE_TO_ID,
    TRIGGER_TO_ID,
)
from .data import assert_student_public_only, read_jsonl
from .teacher_adapter import (
    normalize_action,
    normalize_belief,
    normalize_desire,
    normalize_goal_alignment,
    normalize_route,
    normalize_trigger,
)


# The teacher is the model exposed by the local Codex plug-in.  The
# ``codex_`` prefix is provenance metadata; this is deliberately not an HTTP
# API model name and no API client is used by the direct runner.
TERRA_TEACHER_MODEL = "codex_gpt-5.6-terra"
TERRA_PROMPT_VERSION = "trajweaver-v1-terra-direct-v1"
TERRA_LABEL_SOURCE = "terra_direct"
REQUIRED_LABELS = (
    "route_state",
    "action",
    "belief_state",
    "desire_state",
    "goal_alignment",
    "trigger_action",
)
REQUIRED_IDS = (
    "route_id",
    "action_id",
    "belief_id",
    "desire_id",
    "goal_alignment_id",
    "trigger_id",
)


def terra_prompt(sample: dict[str, Any]) -> str:
    """Return the fixed public-only prompt used by the direct Terra pass."""

    public_sample = {
        "sample_id": sample.get("sample_id"),
        "scenario": sample.get("scenario"),
        "history": sample.get("history"),
        "target": sample.get("target"),
        "turn_index": sample.get("turn_index"),
        "turn_count": sample.get("turn_count"),
        "max_turns": sample.get("max_turns"),
    }
    return f"""You are Codex GPT-5.6 Terra, directly labeling one public training turn.
Read only the JSON data below. The scenario, history, and gold next response are
observable inputs. Do not use hidden persona, preventive, generative, or other
private dataset fields. Return one JSON object with exactly the six categorical
labels below; do not include private reasoning or copied scenario text.

Allowed route_state values: ON_ROUTE, REPAIRABLE, STRUCTURAL_FAILURE, TERMINAL_READY.
Allowed action values: PROBE, CONTINUE, REPAIR, REPLAN, COMMIT.
Allowed belief_state values: UNKNOWN, REJECTS_PREMISE, DOUBTFUL, MIXED, ACCEPTS_PREMISE.
Allowed desire_state values: RESISTANT, RELUCTANT, UNDECIDED, WILLING, COMMITTED.
Allowed goal_alignment values: ON_GOAL, VALID_SUBSTEP, DILUTED, SUBSTITUTED, REVERSED.
Allowed trigger_action values: SKIP, INVOKE.

For goal_alignment, judge the gold next response against the exact original goal:
ON_GOAL directly advances it; VALID_SUBSTEP keeps an explicit path back; DILUTED
weakens a priority or drops a required component; SUBSTITUTED makes another
action the endpoint; REVERSED supports the opposed option or abandons the goal.
For trigger_action, INVOKE means dynamic belief/desire memory is useful before
writing this response; SKIP means the frozen reasoner can proceed from visible
history and immutable goal alone. Turn one must use PROBE.

PUBLIC TURN:
{json.dumps(public_sample, ensure_ascii=False, sort_keys=True)}

Return exactly one JSON object with these keys:
{{"sample_id":"<the supplied sample_id>",
"teacher_model":"codex_gpt-5.6-terra",
"prompt_version":"trajweaver-v1-terra-direct-v1",
"route_state":"...","action":"...","belief_state":"...",
"desire_state":"...","goal_alignment":"...","trigger_action":"..."}}"""


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validate_control_belief(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("control_belief must be an object when present")
    allowed = {"on_route", "repairable", "structural_failure", "terminal_ready"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown control_belief keys: {sorted(unknown)}")
    if any(not _finite_number(item) for item in value.values()):
        raise ValueError("control_belief values must be finite numbers")


def normalize_terra_annotation(
    annotation: dict[str, Any],
    sample: dict[str, Any],
    *,
    teacher_model: str = TERRA_TEACHER_MODEL,
    prompt_version: str = TERRA_PROMPT_VERSION,
) -> dict[str, Any]:
    """Validate one direct label and return the compact student-facing row."""

    if not isinstance(annotation, dict):
        raise ValueError("Terra annotation must be an object")
    sample_id = str(annotation.get("sample_id", "")).strip()
    expected_id = str(sample.get("sample_id", "")).strip()
    if not sample_id or sample_id != expected_id:
        raise ValueError(f"sample_id mismatch: {sample_id!r} != {expected_id!r}")
    if str(annotation.get("teacher_model", teacher_model)) != teacher_model:
        raise ValueError("Annotation teacher_model is not the fixed Terra model")
    if str(annotation.get("prompt_version", prompt_version)) != prompt_version:
        raise ValueError("Annotation prompt_version does not match the fixed Terra prompt")
    route = normalize_route(annotation.get("route_state"), annotation.get("control_belief"))
    action = normalize_action(annotation.get("action"))
    belief = normalize_belief(annotation.get("belief_state"))
    desire = normalize_desire(annotation.get("desire_state"))
    alignment = normalize_goal_alignment(annotation.get("goal_alignment"))
    trigger = normalize_trigger(annotation.get("trigger_action"))
    if int(sample.get("turn_index", -1)) == 0 and action != "PROBE":
        raise ValueError("The first turn must be labeled PROBE")
    _validate_control_belief(annotation.get("control_belief"))

    compact = {
        "sample_id": sample_id,
        "route_state": route,
        "route_id": ROUTE_TO_ID[route],
        "action": action,
        "action_id": ACTION_TO_ID[action],
        "belief_state": belief,
        "belief_id": BELIEF_TO_ID[belief],
        "desire_state": desire,
        "desire_id": DESIRE_TO_ID[desire],
        "goal_alignment": alignment,
        "goal_alignment_id": GOAL_ALIGNMENT_TO_ID[alignment],
        "trigger_action": trigger,
        "trigger_id": TRIGGER_TO_ID[trigger],
        "teacher_model": teacher_model,
        "prompt_version": prompt_version,
        "label_source": TERRA_LABEL_SOURCE,
    }
    if "control_belief" in annotation:
        compact["control_belief"] = annotation["control_belief"]
    # The annotation is merged into a public student row before training.
    assert_student_public_only({**sample, **compact})
    return compact


def read_terra_annotations(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL without silently accepting malformed or duplicate rows."""

    source = Path(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not source.exists():
        return rows
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Terra JSONL at {source}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Terra row {source}:{line_number} is not an object")
        sample_id = str(value.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"Terra row {source}:{line_number} has no sample_id")
        if sample_id in seen:
            raise ValueError(f"Duplicate Terra sample_id {sample_id} at {source}:{line_number}")
        seen.add(sample_id)
        rows.append(value)
    return rows


def append_terra_annotations(
    output: str | Path,
    samples: Iterable[dict[str, Any]],
    supplied: Iterable[dict[str, Any]],
    *,
    teacher_model: str = TERRA_TEACHER_MODEL,
    prompt_version: str = TERRA_PROMPT_VERSION,
) -> dict[str, int]:
    """Validate and append direct labels; rejected rows never enter training."""

    sample_by_id = {str(row["sample_id"]): row for row in samples}
    supplied_rows = list(supplied)
    by_id: dict[str, dict[str, Any]] = {}
    rejected = 0
    for row in supplied_rows:
        sample_id = str(row.get("sample_id", "")).strip() if isinstance(row, dict) else ""
        try:
            if sample_id in by_id:
                raise ValueError("duplicate supplied sample_id")
            if sample_id not in sample_by_id:
                raise ValueError("sample_id is not in the selected public source")
            by_id[sample_id] = normalize_terra_annotation(
                row,
                sample_by_id[sample_id],
                teacher_model=teacher_model,
                prompt_version=prompt_version,
            )
        except (TypeError, ValueError):
            rejected += 1

    destination = Path(output)
    existing = read_terra_annotations(destination)
    existing_ids = {str(row["sample_id"]) for row in existing}
    append_rows = [row for key, row in by_id.items() if key not in existing_ids]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for row in append_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    return {
        "supplied": len(supplied_rows),
        "accepted": len(by_id),
        "appended": len(append_rows),
        "already_present": len(by_id) - len(append_rows),
        "rejected": rejected,
    }


def coverage_report(
    sources: dict[str, str | Path],
    annotation_path: str | Path,
    *,
    minimum: float = 0.95,
) -> dict[str, Any]:
    """Report valid Terra coverage by split and fail only at the caller's gate."""

    annotations = read_terra_annotations(annotation_path)
    all_samples: list[dict[str, Any]] = []
    for path in sources.values():
        all_samples.extend(read_jsonl(path))
    sample_by_id = {str(row["sample_id"]): row for row in all_samples}
    by_id: dict[str, dict[str, Any]] = {}
    invalid_rows: list[dict[str, str]] = []
    for row in annotations:
        sample_id = str(row.get("sample_id", ""))
        try:
            serialized = json.dumps(row, ensure_ascii=False).lower()
            leaked = [name for name in PRIVATE_FIELDS if f'"{name}"' in serialized]
            if leaked:
                raise ValueError(f"private field names leaked: {leaked}")
            if sample_id not in sample_by_id:
                raise ValueError("sample_id is not in the selected public source")
            normalized = normalize_terra_annotation(row, sample_by_id[sample_id])
            if row.get("teacher_model") != TERRA_TEACHER_MODEL:
                raise ValueError("teacher_model is missing or incorrect")
            if row.get("prompt_version") != TERRA_PROMPT_VERSION:
                raise ValueError("prompt_version is missing or incorrect")
            by_id[sample_id] = normalized
        except (TypeError, ValueError) as exc:
            invalid_rows.append({"sample_id": sample_id, "error": str(exc)})
    result: dict[str, Any] = {
        "teacher_model": TERRA_TEACHER_MODEL,
        "prompt_version": TERRA_PROMPT_VERSION,
        "annotation_path": str(Path(annotation_path).resolve()),
        "minimum_coverage": minimum,
        "splits": {},
        "duplicate_sample_ids": 0,
        "extra_annotation_rows": 0,
        "invalid_annotation_rows": invalid_rows,
    }
    source_ids: set[str] = set()
    for split, path in sources.items():
        samples = read_jsonl(path)
        ids = [str(row["sample_id"]) for row in samples]
        counts = Counter(ids)
        duplicate_ids = sorted(sample_id for sample_id, count in counts.items() if count > 1)
        expected = set(ids)
        source_ids.update(expected)
        covered = expected.intersection(by_id)
        split_result = {
            "source": str(Path(path).resolve()),
            "turns": len(ids),
            "scenarios": len({str(row["scenario_id"]) for row in samples}),
            "valid_annotations": len(covered),
            "missing": len(expected - by_id.keys()),
            "duplicate_source_sample_ids": len(duplicate_ids),
            "coverage": len(covered) / max(len(ids), 1),
            "passed": (
                len(duplicate_ids) == 0
                and len(covered) / max(len(ids), 1) >= minimum
            ),
        }
        result["splits"][split] = split_result
    result["extra_annotation_rows"] = len(set(row.get("sample_id", "") for row in annotations) - source_ids)
    result["passed"] = (
        result["extra_annotation_rows"] == 0
        and not result["invalid_annotation_rows"]
        and bool(result["splits"])
        and all(item["passed"] for item in result["splits"].values())
    )
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def require_training_coverage(
    train_path: str | Path,
    dev_path: str | Path,
    annotation_path: str | Path,
    *,
    minimum: float = 0.95,
) -> dict[str, Any]:
    """Raise before model loading unless both formal splits pass the Terra gate."""

    report = coverage_report(
        {"train": train_path, "dev": dev_path},
        annotation_path,
        minimum=minimum,
    )
    if not report["passed"]:
        split_text = ", ".join(
            f"{name}={value['coverage']:.2%}"
            for name, value in report["splits"].items()
        )
        raise RuntimeError(
            "Terra annotation coverage is below the formal-training gate: "
            f"{split_text}; required={minimum:.2%}. "
            f"See {Path(annotation_path).with_name('coverage.json')}"
        )
    return report


def write_terra_metadata(
    path: str | Path,
    *,
    sources: dict[str, str | Path],
    manifest: str | Path,
    seed: int,
    annotation_path: str | Path,
) -> dict[str, Any]:
    metadata = {
        "teacher_model": TERRA_TEACHER_MODEL,
        "prompt_version": TERRA_PROMPT_VERSION,
        "label_source": TERRA_LABEL_SOURCE,
        "annotation_path": str(Path(annotation_path).resolve()),
        "append_only": True,
        "api_calls": False,
        "data_manifest": str(Path(manifest).resolve()),
        "source_files": {key: str(Path(value).resolve()) for key, value in sources.items()},
        "seed": int(seed),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "public_only_fields": ["scenario", "history", "target"],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
