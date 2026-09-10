"""Leakage-safe CToMPersu data preparation and prompt construction."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .constants import (
    ACTION_TO_ID,
    EXPERIMENT_MAX_TURNS,
    GOAL_ENCODER_SYSTEM_PROMPT,
    PRIVATE_FIELDS,
    PUBLIC_FIELDS,
    ROUTE_TO_ID,
    SYSTEM_PROMPT,
)

ROLE_RE = re.compile(r"^\s*(persuader|persuadee)\s*:\s*(.*)$", re.I | re.S)


def _normalized(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalized(value[key]) for key in sorted(value)}
    return value


def public_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """Copy only fields that are observable by the evaluated persuader."""

    return {field: scenario.get(field) for field in PUBLIC_FIELDS}


def scenario_key(scenario: dict[str, Any]) -> str:
    """Canonical key used to remove every Eval scenario from Full."""

    payload = _normalized(public_scenario(scenario))
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def scenario_id(scenario: dict[str, Any]) -> str:
    return hashlib.sha256(scenario_key(scenario).encode("utf-8")).hexdigest()[:16]


def repair_dialogue(lines: Sequence[Any]) -> list[dict[str, str]]:
    """Parse roles and fold newline-split utterance fragments into the prior turn.

    Some Full records store one utterance as several array entries.  A fragment
    without a role prefix is continuation text, including the case where the
    preceding ``persuader:`` or ``persuadee:`` entry has empty content.
    """

    turns: list[dict[str, str]] = []
    for raw in lines:
        text = str(raw or "").strip()
        if not text:
            continue
        matched = ROLE_RE.match(text)
        if matched:
            turns.append(
                {"role": matched.group(1).lower(), "content": matched.group(2).strip()}
            )
        elif turns:
            turns[-1]["content"] = " ".join(
                part for part in (turns[-1]["content"], text) if part
            ).strip()
        else:
            raise ValueError("Dialogue starts with text that has no role prefix")

    if not turns:
        raise ValueError("Dialogue is empty after normalization")
    for index, turn in enumerate(turns):
        expected = "persuader" if index % 2 == 0 else "persuadee"
        if turn["role"] != expected:
            raise ValueError(
                f"Non-alternating dialogue at message {index}: expected {expected}, "
                f"got {turn['role']}"
            )
        if not turn["content"]:
            raise ValueError(f"Empty {expected} utterance at message {index}")
    if len(turns) not in {6, 8}:
        raise ValueError(f"Expected three or four persuasion rounds, got {len(turns) / 2:g}")
    return turns


def _bootstrap_control_label(turn_index: int, turn_count: int) -> tuple[str, str]:
    """Weak labels for the controller warm-up; teacher labels override these."""

    if turn_index == 0:
        return "ON_ROUTE", "PROBE"
    if turn_index == turn_count - 1:
        return "TERMINAL_READY", "COMMIT"
    return "ON_ROUTE", "CONTINUE"


def iter_turn_samples(row: dict[str, Any], split: str) -> Iterator[dict[str, Any]]:
    scenario = public_scenario(row["scenario"])
    sid = scenario_id(scenario)
    dialogue = repair_dialogue(row["dialog"])
    persuader_turns = sum(turn["role"] == "persuader" for turn in dialogue)
    history: list[dict[str, str]] = []
    turn_index = 0
    for message in dialogue:
        if message["role"] == "persuader":
            route, action = _bootstrap_control_label(turn_index, persuader_turns)
            yield {
                "sample_id": f"{sid}:t{turn_index + 1}",
                "scenario_id": sid,
                "split": split,
                "scenario": scenario,
                "history": [dict(item) for item in history],
                "target": message["content"],
                "turn_index": turn_index,
                "turn_count": persuader_turns,
                "max_turns": EXPERIMENT_MAX_TURNS,
                "remaining_turns": EXPERIMENT_MAX_TURNS - turn_index,
                "route_state": route,
                "route_id": ROUTE_TO_ID[route],
                "action": action,
                "action_id": ACTION_TO_ID[action],
                "label_source": "bootstrap_position",
                # Auxiliary state labels are deliberately absent until a
                # public-only teacher annotation is available.
                "belief_id": -100,
                "desire_id": -100,
                "goal_alignment_id": -100,
                "trigger_id": -100,
            }
            turn_index += 1
        history.append(dict(message))


def load_json_list(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Expected a JSON list of objects: {path}")
    return value


def split_full_excluding_eval(
    full_rows: Sequence[dict[str, Any]],
    eval_rows: Sequence[dict[str, Any]],
    *,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Exclude by public-scenario identity, then make a deterministic scenario split."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between zero and one")
    eval_keys = {scenario_key(row["scenario"]) for row in eval_rows}
    full_record_keys = {
        json.dumps(_normalized(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in full_rows
    }
    eval_record_keys = [
        json.dumps(_normalized(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in eval_rows
    ]
    retained: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_full = 0
    for row in full_rows:
        key = scenario_key(row["scenario"])
        if key in eval_keys:
            continue
        if key in seen:
            duplicate_full += 1
            continue
        seen.add(key)
        retained.append(row)
    random.Random(seed).shuffle(retained)
    cut = int(len(retained) * train_ratio)
    train_rows, dev_rows = retained[:cut], retained[cut:]
    manifest = {
        "full_rows": len(full_rows),
        "full_unique_records": len(full_record_keys),
        "eval_rows": len(eval_rows),
        "eval_unique_records": len(set(eval_record_keys)),
        "eval_rows_present_exactly_in_full": sum(key in full_record_keys for key in eval_record_keys),
        "eval_unique_public_scenarios": len(eval_keys),
        "excluded_full_rows": len(full_rows) - len(retained) - duplicate_full,
        "duplicate_non_eval_public_scenarios": duplicate_full,
        "retained_scenarios": len(retained),
        "train_scenarios": len(train_rows),
        "dev_scenarios": len(dev_rows),
        "train_ratio": train_ratio,
        "seed": seed,
        "public_key_fields": list(PUBLIC_FIELDS),
        "private_fields_excluded": list(PRIVATE_FIELDS),
    }
    return train_rows, dev_rows, manifest


def assert_student_public_only(sample: dict[str, Any]) -> None:
    encoded = json.dumps(sample, ensure_ascii=False).lower()
    forbidden = set(PRIVATE_FIELDS).intersection(sample.get("scenario", {}))
    if forbidden:
        raise ValueError(f"Private scenario keys leaked into sample: {sorted(forbidden)}")
    # The field names themselves must not be serialized into model-facing rows.
    for name in PRIVATE_FIELDS:
        if f'"{name}"' in encoded:
            raise ValueError(f"Private field name leaked into student sample: {name}")


def dialogue_text(history: Sequence[dict[str, str]]) -> str:
    if not history:
        return "(no dialogue yet)"
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)


def goal_specification(sample_or_scenario: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable, public target specification for one dialogue.

    Prepared rows may later contain a richer teacher-produced specification,
    but the exact dataset goal always remains the source of truth.  Dynamic
    dialogue history is intentionally excluded from this object.
    """

    scenario = sample_or_scenario.get("scenario", sample_or_scenario)
    if not isinstance(scenario, dict):
        raise ValueError("A scenario mapping is required to encode the goal")
    goal = str(scenario.get("goal", "")).strip()
    if not goal:
        raise ValueError("Public scenario has no non-empty goal")
    supplied = sample_or_scenario.get("goal_specification", {})
    supplied = supplied if isinstance(supplied, dict) else {}
    return {
        "exact_goal": goal,
        "required_components": supplied.get("required_components", []),
        "contrasted_or_excluded_options": supplied.get(
            "contrasted_or_excluded_options", []
        ),
        "priority_or_time_constraints": supplied.get(
            "priority_or_time_constraints", []
        ),
        "valid_substeps": supplied.get("valid_substeps", []),
        "invalid_substitutions": supplied.get("invalid_substitutions", []),
    }


def goal_messages(sample_or_scenario: dict[str, Any]) -> list[dict[str, str]]:
    specification = json.dumps(
        goal_specification(sample_or_scenario), ensure_ascii=False, sort_keys=True
    )
    return [
        {"role": "system", "content": GOAL_ENCODER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Immutable goal specification:\n{specification}"},
    ]


def student_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    scenario = json.dumps(sample["scenario"], ensure_ascii=False, sort_keys=True)
    turn_index = int(
        sample.get(
            "turn_index",
            sum(turn.get("role") == "persuader" for turn in sample.get("history", [])),
        )
    )
    max_turns = int(sample.get("max_turns", EXPERIMENT_MAX_TURNS))
    remaining = max(1, max_turns - turn_index)
    user = f"""Public scenario:
{scenario}

Observable dialogue:
{dialogue_text(sample['history'])}

Public turn budget: this is persuader turn {turn_index + 1} of {max_turns};
{remaining} persuader response(s) remain including the current response.

Write the next persuader utterance only."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            assert_student_public_only(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count
