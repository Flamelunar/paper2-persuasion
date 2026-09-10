"""Adapters for proactive-persuasion teacher labels.

Teacher artifacts may contain rich private reasoning, but the returned Student
row contains only discrete control labels and public text.  Raw teacher fields
are never copied into a training prompt.
"""

from __future__ import annotations

from typing import Any, Iterable

from .constants import (
    ACTION_TO_ID,
    BELIEF_TO_ID,
    DESIRE_TO_ID,
    GOAL_ALIGNMENT_TO_ID,
    ROUTE_TO_ID,
    TRIGGER_TO_ID,
)
from .data import assert_student_public_only, scenario_id

A_TO_ACTION = {
    "A1": "PROBE",
    "A2": "CONTINUE",
    "A3": "REPAIR",
    "A4": "REPLAN",
    "A5": "COMMIT",
}


def normalize_route(value: Any, belief: dict[str, Any] | None = None) -> str:
    aliases = {
        "on_route": "ON_ROUTE",
        "repairable": "REPAIRABLE",
        "structural_failure": "STRUCTURAL_FAILURE",
        "terminal_ready": "TERMINAL_READY",
    }
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in aliases:
        return aliases[text]
    if belief:
        candidates = {
            aliases[key]: float(belief.get(key, 0.0))
            for key in aliases
            if key in belief
        }
        if candidates:
            return max(candidates, key=candidates.get)
    raise ValueError(f"Unknown route state: {value!r}")


def normalize_action(value: Any) -> str:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if text in A_TO_ACTION:
        return A_TO_ACTION[text]
    if text in ACTION_TO_ID:
        return text
    raise ValueError(f"Unknown control action: {value!r}")


def _normalize_named(value: Any, mapping: dict[str, int], label: str) -> str:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if text in mapping:
        return text
    raise ValueError(f"Unknown {label}: {value!r}")


def normalize_belief(value: Any) -> str:
    return _normalize_named(value, BELIEF_TO_ID, "belief state")


def normalize_desire(value: Any) -> str:
    return _normalize_named(value, DESIRE_TO_ID, "desire state")


def normalize_goal_alignment(value: Any) -> str:
    if isinstance(value, int) and 0 <= value < len(GOAL_ALIGNMENT_TO_ID):
        return next(name for name, index in GOAL_ALIGNMENT_TO_ID.items() if index == value)
    return _normalize_named(value, GOAL_ALIGNMENT_TO_ID, "goal alignment")


def normalize_trigger(value: Any) -> str:
    if isinstance(value, bool):
        return "INVOKE" if value else "SKIP"
    return _normalize_named(value, TRIGGER_TO_ID, "trigger action")


def overlay_annotation(sample: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    """Apply distilled control/target labels without copying teacher internals."""

    # Legacy teacher overlays (and hand-written fixtures) may omit an id;
    # direct Terra rows are still required to carry and validate ``sample_id``
    # in ``normalize_terra_annotation``.  Only compare ids when the overlay
    # actually supplies one so the public adapter remains backwards compatible.
    annotation_sample_id = str(annotation.get("sample_id", "")).strip()
    if annotation_sample_id and annotation_sample_id != str(sample.get("sample_id", "")):
        raise ValueError(
            "Annotation/sample mismatch: "
            f"{annotation.get('sample_id')!r} != {sample.get('sample_id')!r}"
        )
    row = dict(sample)
    route = normalize_route(annotation.get("route_state"), annotation.get("control_belief"))
    action = normalize_action(annotation.get("action", annotation.get("system_action")))
    row.update(
        {
            "route_state": route,
            "route_id": ROUTE_TO_ID[route],
            "action": action,
            "action_id": ACTION_TO_ID[action],
            "label_source": str(annotation.get("label_source", "proactive_teacher")),
        }
    )
    for provenance_key in ("teacher_model", "prompt_version"):
        if annotation.get(provenance_key) is not None:
            row[provenance_key] = annotation[provenance_key]
    optional_labels = (
        ("belief_state", "belief_id", normalize_belief, BELIEF_TO_ID),
        ("desire_state", "desire_id", normalize_desire, DESIRE_TO_ID),
        (
            "goal_alignment",
            "goal_alignment_id",
            normalize_goal_alignment,
            GOAL_ALIGNMENT_TO_ID,
        ),
        ("trigger_action", "trigger_id", normalize_trigger, TRIGGER_TO_ID),
    )
    for source, target, normalizer, mapping in optional_labels:
        if annotation.get(source) is not None:
            normalized = normalizer(annotation[source])
            row[source] = normalized
            row[target] = mapping[normalized]
    teacher_target = str(annotation.get("target", annotation.get("next_question", ""))).strip()
    if teacher_target:
        row["target"] = teacher_target
    assert_student_public_only(row)
    return row


def _run_public_scenario(run: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("scenario", "public_scenario", "task"):
        candidate = run.get(key)
        if isinstance(candidate, dict):
            if "persuasion_goal" in candidate and "goal" not in candidate:
                continue
            return candidate
    return None


def official_run_annotations(
    run: dict[str, Any], *, scenario_id_override: str | None = None
) -> list[dict[str, Any]]:
    """Convert one supplied Proactively Induced Persuasion run.

    The initial A1 anchor supervises turn one.  Each subsequent ``turns[i]``
    decision is aligned to its ``next_question``, which follows that record's
    user/persuadee response in the observable prefix.
    """

    # The supplied batch runner wraps each result as
    # {"index": ..., "status": "ok", "result": {...}}.
    payload = run.get("result") if isinstance(run.get("result"), dict) else run
    sid = str(
        scenario_id_override
        or run.get("scenario_id")
        or payload.get("scenario_id")
        or ""
    ).strip()
    scenario = _run_public_scenario(payload)
    if not sid and scenario is not None:
        sid = scenario_id(scenario)
    if not sid:
        raise ValueError("Teacher run needs scenario_id or a public scenario")
    results: list[dict[str, Any]] = []
    initial = payload.get("initial_anchor", {})
    if isinstance(initial, dict) and str(initial.get("question", "")).strip():
        results.append(
            {
                "sample_id": f"{sid}:t1",
                "route_state": "ON_ROUTE",
                "action": "PROBE",
                "target": str(initial["question"]).strip(),
                "label_source": "proactive_teacher_initial_anchor",
            }
        )
    for index, turn in enumerate(payload.get("turns", []), start=2):
        if not isinstance(turn, dict):
            continue
        target = turn.get("next_question")
        if not str(target or "").strip():
            continue
        belief = turn.get("control_belief")
        if not isinstance(belief, dict):
            belief = turn.get("belief") if isinstance(turn.get("belief"), dict) else None
        if not isinstance(belief, dict):
            bayesian = turn.get("bayesian_control") or turn.get("bayesian") or {}
            candidate = bayesian.get("posterior_belief") if isinstance(bayesian, dict) else None
            belief = candidate if isinstance(candidate, dict) else None
        action = turn.get("system_action")
        # Control action supplies a conservative fallback when a run omitted
        # its explicit four-way belief serialization.
        fallback_route = {
            "A2": "ON_ROUTE",
            "A3": "REPAIRABLE",
            "A4": "STRUCTURAL_FAILURE",
            "A5": "TERMINAL_READY",
        }.get(str(action).upper(), "ON_ROUTE")
        results.append(
            {
                "sample_id": f"{sid}:t{index}",
                "route_state": normalize_route(turn.get("route_state", fallback_route), belief),
                "control_belief": belief,
                "action": normalize_action(action),
                "target": str(target).strip(),
                "label_source": "proactive_teacher_turn",
            }
        )
    return results


def merge_annotations(
    samples: Iterable[dict[str, Any]], annotations: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in annotations:
        sample_id = str(row["sample_id"])
        if sample_id in by_id:
            raise ValueError(f"Duplicate annotation sample_id: {sample_id}")
        by_id[sample_id] = row
    return [
        overlay_annotation(sample, by_id[str(sample["sample_id"])])
        if str(sample["sample_id"]) in by_id
        else sample
        for sample in samples
    ]
