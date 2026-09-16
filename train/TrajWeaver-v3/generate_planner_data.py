#!/usr/bin/env python3
"""Generate a small, source-grounded planner dataset with batched Luna calls.

The generator deliberately keeps this annotation task separate from the
TrajWeaver-v3 reply SFT target.  It creates planner supervision that can be
used by a future planner/controller, while the existing Weaver still trains
on public reference replies.

Each selected source scenario contributes two examples:

* ``initial``: no dialogue history;
* ``concern``: the prefix ending at the first explicit persuadee concern.

The default 200 train + 50 dev scenarios therefore produce 500 examples in
25 requests when the default batch size is 10 scenarios (20 examples).
Every request is resumable and every returned item is checked against the
public source record before it enters a training file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_SOURCE = PROJECT / "data/CToMPersu/mysplit"
DEFAULT_OUTPUT = HERE / "artifacts"
MODEL = "gpt-5.6-luna"
PUBLIC_FIELDS = ("tag", "background", "persuadee", "persuader", "goal", "domain")
PREFIX_TYPES = ("initial", "concern")
MODES = {"REPAIR", "FOLLOW", "PASS"}
CONCERN_CATEGORIES = {
    "efficacy_speed",
    "cost_effort",
    "risk_loss",
    "time_convenience",
    "identity_values",
    "social_acceptance",
    "autonomy_reactance",
    "uncertainty",
    "unknown",
}

# These cues are intentionally conservative.  If none is found, the second
# persuadee turn is used because CToMPersu's dialogue template normally places
# its first explicit generative concern there.
EXPLICIT_CONCERN = re.compile(
    r"\b(?:concern(?:ed|ing)?|worr(?:y|ied|ies)|afraid|fear(?:ful)?|"
    r"risk(?:y)?|expensive|costly|burden|time-consuming|time consuming|"
    r"too much|too difficult|difficult|hard to|not sure|unsure|uncertain|"
    r"skeptic(?:al)?|hesitat(?:e|ion|ed)|reluctant|uncomfortable|"
    r"inconvenien(?:t|ce)|privacy|overwhelm(?:ed|ing)?|drawback|downside|"
    r"danger|loss|harm|problem|issue|negative|sacrifice|compromise|"
    r"disrupt|strain|waste|ineffective|not convinced|don['’]t think|"
    r"not enough|not able|can['’]t|cannot|wouldn['’]t|won['’]t|"
    r"don['’]t want|rather not|prefer not|lack|limited|shortage)\b",
    re.IGNORECASE,
)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(compact_json(value).encode("utf-8")).hexdigest()


def public_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    return {key: scenario.get(key) for key in PUBLIC_FIELDS}


def repair_dialogue(lines: list[Any]) -> list[dict[str, str]]:
    """Repair continuation-only lines without importing training code."""

    pattern = re.compile(r"^\s*(persuader|persuadee)\s*:\s*(.*)$", re.I | re.S)
    messages: list[dict[str, str]] = []
    for line in lines:
        text = str(line or "").strip()
        if not text:
            continue
        match = pattern.match(text)
        if match:
            messages.append({"role": match[1].lower(), "content": match[2].strip()})
        elif messages:
            messages[-1]["content"] = (messages[-1]["content"] + " " + text).strip()
        else:
            raise ValueError("dialogue starts with an unlabelled fragment")
    if len(messages) not in (6, 8):
        raise ValueError(f"expected 3/4 rounds, got {len(messages)} messages")
    for index, message in enumerate(messages):
        expected = "persuader" if index % 2 == 0 else "persuadee"
        if message["role"] != expected or not message["content"]:
            raise ValueError(f"invalid role/content at message {index}")
    return messages


def concern_prefix(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], str, str]:
    """Return history through the first explicit persuadee concern.

    The returned reference is stable within a scenario and can be cited by
    the model in ``evidence_ref``.  A template fallback keeps all 500 examples
    grounded even when a concern is expressed without a lexical cue.
    """

    persuadee_turn = 0
    candidates: list[tuple[int, int, str]] = []
    for index, message in enumerate(messages):
        if message["role"] != "persuadee":
            continue
        persuadee_turn += 1
        if EXPLICIT_CONCERN.search(message["content"]):
            candidates.append((index, persuadee_turn, "lexical"))
    if candidates:
        index, turn, detection = candidates[0]
    else:
        # The source format alternates roles; the second persuadee response
        # is the safest fallback for a non-lexical concern.
        persuadee_indices = [i for i, m in enumerate(messages) if m["role"] == "persuadee"]
        index = persuadee_indices[min(1, len(persuadee_indices) - 1)]
        turn = persuadee_indices.index(index) + 1
        detection = "template_fallback"
    ref = f"persuadee_turn_{turn}"
    history = [
        {"ref": f"{message['role']}_turn_{(i // 2) + 1}",
         "role": message["role"], "content": message["content"]}
        for i, message in enumerate(messages[: index + 1])
    ]
    quote = messages[index]["content"]
    return history, ref, detection


def load_sources(source_dir: Path, split: str) -> list[dict[str, Any]]:
    path = source_dir / f"{split}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    result = []
    for source_index, record in enumerate(data):
        if not isinstance(record, dict) or not isinstance(record.get("scenario"), dict):
            raise ValueError(f"{split}:{source_index} has no scenario object")
        scenario = public_scenario(record["scenario"])
        for field in ("background", "persuadee", "persuader", "goal"):
            if not str(scenario.get(field) or "").strip():
                raise ValueError(f"{split}:{source_index} has empty public {field}")
        messages = repair_dialogue(record.get("dialog", []))
        sid = f"{split}_{source_index:05d}_{digest(scenario)[:10]}"
        history, evidence_ref, detection = concern_prefix(messages)
        result.append({
            "source_id": sid,
            "split": split,
            "source_index": source_index,
            "scenario": scenario,
            "concern_history": history,
            "concern_evidence_ref": evidence_ref,
            "concern_evidence_quote": next(
                item["content"] for item in history if item["ref"] == evidence_ref
            ),
            "concern_detection": detection,
        })
    return result


def select_sources(source_dir: Path, train_count: int, dev_count: int, seed: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for split, count, offset in (("train", train_count, 0), ("dev", dev_count, 1)):
        rows = load_sources(source_dir, split)
        if count < 0 or count > len(rows):
            raise ValueError(f"{split} count {count} is outside 0..{len(rows)}")
        rng = random.Random(seed + offset)
        rng.shuffle(rows)
        selected.extend(rows[:count])
    return selected


def cases_for_source(source: dict[str, Any]) -> list[dict[str, Any]]:
    scenario = source["scenario"]
    return [
        {
            "source_id": source["source_id"],
            "split": source["split"],
            "source_index": source["source_index"],
            "prefix_type": "initial",
            "scenario": scenario,
            "history": [],
        },
        {
            "source_id": source["source_id"],
            "split": source["split"],
            "source_index": source["source_index"],
            "prefix_type": "concern",
            "scenario": scenario,
            "history": source["concern_history"],
        },
    ]


def all_cases(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in selected:
        result.extend(cases_for_source(source))
    return result


def batch_cases(cases: list[dict[str, Any]], batch_size: int, max_input_chars: int) -> list[list[dict[str, Any]]]:
    """Pack whole source pairs, splitting further only for oversized prompts."""

    by_source = [cases[i : i + 2] for i in range(0, len(cases), 2)]
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for pair in by_source:
        candidate = current + pair
        if current and len(current) // 2 >= batch_size:
            batches.append(current)
            current = []
            candidate = pair
        if current and estimate_prompt_chars(candidate) > max_input_chars:
            batches.append(current)
            current = []
            candidate = pair
        current = candidate
    if current:
        batches.append(current)
    return batches


def estimate_prompt_chars(cases: list[dict[str, Any]]) -> int:
    return len(compact_json(cases)) + 6500


def prompt_for(cases: list[dict[str, Any]]) -> str:
    payload = [
        {
            "source_id": case["source_id"],
            "prefix_type": case["prefix_type"],
            "scenario": case["scenario"],
            "history": case["history"],
        }
        for case in cases
    ]
    return """You are annotating a small training set for a multi-turn persuasion planner.
Return exactly one JSON object with an `items` array, one item for every input case,
and no prose outside JSON. Never use private fields; only the supplied public
scenario and observable history are evidence.

For each case output:
{
  "source_id": "copied exactly",
  "prefix_type": "initial|concern",
  "mode": "REPAIR|FOLLOW|PASS",
  "goal": "copied exactly from scenario.goal",
  "conditions": {"expression": "short faithful condition summary", "items": ["..."]},
  "concern": {"category": "efficacy_speed|cost_effort|risk_loss|time_convenience|identity_values|social_acceptance|autonomy_reactance|uncertainty|unknown", "status": "explicit|unknown", "evidence_ref": "persuadee_turn_N or none", "evidence_quote": "exact substring of the cited persuadee utterance or empty"},
  "target_condition": "one concrete condition to resolve or preserve next",
  "path": [{"goal_id": "original_goal", "current_subgoal": "...", "evidence_refs": ["persuadee_turn_N"], "action": "...", "advance_when": "...", "repair_when": "..."}],
  "guidance": "concise next-step guidance for a persuader",
  "reason": "one concise reason tied to the visible evidence"
}

Rules: preserve the exact original goal and all qualifiers; do not declare a
conditional interest to be full acceptance. For `initial`, concern.status must
be `unknown`, evidence_ref must be `none`, and evidence_quote must be empty.
For `concern`, cite only a persuadee message present in that case's history;
copy its quote exactly and use its supplied `persuadee_turn_N` reference. Do not
cite a future turn or a persuader utterance. Keep conditions and guidance
specific to the person's expressed concern (cost/effort, efficacy/speed,
risk/loss, time/convenience, identity/values, social acceptance,
autonomy/reactance, or uncertainty). `PASS` means the visible evidence does
not justify a special repair; `FOLLOW` advances a valid path; `REPAIR` changes
an unsupported or stale strategy. Keep strings brief (normally under 35 words)
so the batch remains compact. Output exactly one item per input case.

INPUT CASES:
""" + compact_json(payload)


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        value = None
        for start, char in enumerate(text):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                value = candidate
                break
    return value if isinstance(value, dict) else None


def evidence_by_ref(case: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {item["ref"]: item for item in case["history"] if item["role"] == "persuadee"}


def validate_item(item: Any, case: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(item, dict):
        return None, "item is not an object"
    required = {"source_id", "prefix_type", "mode", "goal", "conditions", "concern",
                "target_condition", "path", "guidance", "reason"}
    missing = required - set(item)
    if missing:
        return None, f"missing keys: {sorted(missing)}"
    if item["source_id"] != case["source_id"]:
        return None, "source_id mismatch"
    if item["prefix_type"] != case["prefix_type"]:
        return None, "prefix_type mismatch"
    if item["mode"] not in MODES:
        return None, "invalid mode"
    if item["goal"] != case["scenario"]["goal"]:
        return None, "goal is not an exact copy"
    if not isinstance(item["conditions"], dict) or not isinstance(item["conditions"].get("items"), list):
        return None, "invalid conditions"
    if not isinstance(item["concern"], dict):
        return None, "invalid concern"
    concern = item["concern"]
    if concern.get("category") not in CONCERN_CATEGORIES:
        return None, "invalid concern category"
    if concern.get("status") not in {"explicit", "unknown"}:
        return None, "invalid concern status"
    if not isinstance(item["conditions"].get("expression"), str) or not item["conditions"]["expression"].strip():
        return None, "empty conditions.expression"
    for key in ("target_condition", "guidance", "reason"):
        if not isinstance(item.get(key), str) or not item[key].strip():
            return None, f"empty {key}"
    if not isinstance(item["path"], list) or not item["path"]:
        return None, "path must be a non-empty list"
    valid_refs = evidence_by_ref(case)
    if case["prefix_type"] == "initial":
        if concern.get("status") != "unknown" or concern.get("evidence_ref") != "none" or concern.get("evidence_quote") != "":
            return None, "initial case must have unknown concern"
    else:
        ref = concern.get("evidence_ref")
        quote = concern.get("evidence_quote")
        if concern.get("status") != "explicit" or ref not in valid_refs or not isinstance(quote, str):
            return None, "concern case must cite a persuadee message"
        if quote != valid_refs[ref]["content"] and quote not in valid_refs[ref]["content"]:
            return None, "evidence_quote is not an exact observed substring"
    for step in item["path"]:
        if not isinstance(step, dict):
            return None, "path step is not an object"
        for key in ("goal_id", "current_subgoal", "action", "advance_when", "repair_when"):
            if not isinstance(step.get(key), str) or not step[key].strip():
                return None, f"empty path.{key}"
        refs = step.get("evidence_refs", [])
        if not isinstance(refs, list) or any(ref not in valid_refs for ref in refs):
            return None, "path has invalid evidence reference"
    # Keep generated records auditable and prevent private labels from being
    # smuggled into nested target objects.
    forbidden = {"preventive", "generative", "persona", "accepted", "reactance", "acceptance_level"}
    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            return any(key in forbidden or walk(v) for key, v in value.items())
        if isinstance(value, list):
            return any(walk(v) for v in value)
        return False
    if walk(item):
        return None, "private simulator key present"
    return item, None


def normalize_item(item: Any, case: dict[str, Any]) -> Any:
    """Fill the one unambiguous omission common on empty initial prefixes."""

    if not isinstance(item, dict):
        return item
    normalized = json.loads(json.dumps(item, ensure_ascii=False))
    conditions = normalized.get("conditions")
    if isinstance(conditions, dict) and not str(conditions.get("expression") or "").strip():
        if case["prefix_type"] == "initial":
            conditions["expression"] = "No explicit persuadee concern is visible in this prefix."
        else:
            conditions["expression"] = "The cited persuadee message contains the active concern."
    return normalized


def api_call(prompt: str, model: str, base_url: str, retries: int) -> str:
    # Reuse the project's OpenAI-compatible wrapper so Luna's completion
    # parameter conventions and CHATANYWHERE_API handling stay consistent.
    import sys
    sys.path.insert(0, str(PROJECT))
    from train.ctompersu_common import api_text

    last: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            return api_text(prompt, model, base_url, 12000, 1, temperature=1)
        except Exception as exc:  # retry only transient failures
            last = exc
            text = str(exc).lower()
            if any(marker in text for marker in ("authentication", "permission", "insufficient", "balance", "余额", "充值")):
                raise
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"planner API call failed: {type(last).__name__}: {last}")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def make_sft_row(case: dict[str, Any], annotation: dict[str, Any], batch_id: str) -> dict[str, Any]:
    user = {"scenario": case["scenario"], "history": case["history"], "prefix_type": case["prefix_type"]}
    system = "You are a source-grounded multi-turn persuasion planner. Preserve the exact public goal and cite only visible evidence. Return the planner JSON schema."
    return {
        "sample_id": f"{case['source_id']}:{case['prefix_type']}",
        "source_id": case["source_id"],
        "split": case["split"],
        "prefix_type": case["prefix_type"],
        "batch_id": batch_id,
        "scenario": case["scenario"],
        "history": case["history"],
        "input": user,
        "target": annotation,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": compact_json(user)},
            {"role": "assistant", "content": compact_json(annotation)},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-scenarios", type=int, default=200)
    parser.add_argument("--dev-scenarios", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=10, help="source scenarios per request")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=os.getenv("CTOMPERSU_API_BASE_URL", "https://api.chatanywhere.tech/v1"))
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-input-chars", type=int, default=48000,
                        help="split a batch above roughly 12k input tokens")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="debug limit; 0 processes all batches")
    args = parser.parse_args()
    if args.batch_size <= 0 or min(args.train_scenarios, args.dev_scenarios, args.seed) < 0:
        parser.error("counts, seed, and batch-size must be nonnegative (batch-size > 0)")
    if not os.environ.get("CHATANYWHERE_API") and not os.environ.get("OPENAI_API_KEY"):
        parser.error("CHATANYWHERE_API or OPENAI_API_KEY is required for live generation")

    selected = select_sources(args.source, args.train_scenarios, args.dev_scenarios, args.seed)
    cases = all_cases(selected)
    batches = batch_cases(cases, args.batch_size, args.max_input_chars)
    expected_cases = {f"{case['source_id']}:{case['prefix_type']}": case for case in cases}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "planner_concern_annotation_v1",
        "model": args.model,
        "source": str(args.source.resolve()),
        "source_public_fields": list(PUBLIC_FIELDS),
        "seed": args.seed,
        "train_scenarios": args.train_scenarios,
        "dev_scenarios": args.dev_scenarios,
        "expected_annotations": len(cases),
        "batch_size_scenarios": args.batch_size,
        "planned_batches": len(batches),
        "max_input_chars": args.max_input_chars,
        "batch_policy": "one request returns initial+concern for each source scenario",
        "hidden_fields_excluded": ["preventive", "generative", "persona"],
        "source_sha256": {split: hashlib.sha256((args.source / f"{split}.json").read_bytes()).hexdigest()
                          for split in ("train", "dev")},
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Existing planner output was made with a different selection/config; choose a new --output")
    write_json(manifest_path, manifest)

    batch_log = output / "planner_batches.jsonl"
    quarantine_path = output / "planner_quarantine.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    batch_for_key: dict[str, str] = {}
    for row in read_jsonl(batch_log):
        if row.get("status") == "ok":
            for annotation in row.get("annotations", []):
                key = f"{annotation.get('source_id')}:{annotation.get('prefix_type')}"
                completed[key] = annotation
                batch_for_key[key] = str(row.get("batch_id", "resumed"))
    raw_by_split = {split: output / f"planner_raw_{split}.jsonl" for split in ("train", "dev")}
    sft_by_split = {split: output / f"planner_sft_{split}.jsonl" for split in ("train", "dev")}
    # Existing raw files are authoritative on resume; reject foreign rows.
    for path in (*raw_by_split.values(), *sft_by_split.values()):
        for row in read_jsonl(path):
            key = f"{row.get('source_id')}:{row.get('prefix_type')}"
            if key not in expected_cases:
                raise ValueError(f"foreign planner row in {path}: {key}")
            if row.get("annotation") is not None:
                completed.setdefault(key, row["annotation"])
                batch_for_key.setdefault(key, str(row.get("batch_id", "resumed")))
            elif row.get("target") is not None:
                completed.setdefault(key, row["target"])
                batch_for_key.setdefault(key, str(row.get("batch_id", "resumed")))

    processed = 0
    for batch_index, batch in enumerate(batches):
        batch_id = f"batch_{batch_index:03d}"
        needed = [case for case in batch if f"{case['source_id']}:{case['prefix_type']}" not in completed]
        if not needed:
            processed += 1
            continue
        # A missing member can only occur when a previous response was partial;
        # send the complete source pairs so the model sees both contexts.
        source_ids = {case["source_id"] for case in needed}
        request_cases = [case for case in batch if case["source_id"] in source_ids]
        prompt = prompt_for(request_cases)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        print(f"{batch_id}: {len(request_cases)//2} scenarios / {len(request_cases)} annotations / {len(prompt)} input chars", flush=True)
        raw: str | None = None
        try:
            raw = api_call(prompt, args.model, args.base_url, args.retries)
            parsed = parse_json_object(raw)
            raw_items = parsed.get("items") if isinstance(parsed, dict) else None
            if not isinstance(raw_items, list):
                raise ValueError("response has no items array")
            by_key: dict[str, dict[str, Any]] = {}
            errors: list[str] = []
            for item in raw_items:
                key = f"{item.get('source_id')}:{item.get('prefix_type')}" if isinstance(item, dict) else "<non-object>"
                case = next((c for c in request_cases if f"{c['source_id']}:{c['prefix_type']}" == key), None)
                # Luna occasionally appends a repeated hash suffix while
                # copying a long identifier.  Canonicalize only when exactly
                # one expected ID is a strict prefix; never guess a missing
                # or unrelated source ID.
                if case is None and isinstance(item, dict):
                    candidates = [
                        c for c in request_cases
                        if item.get("prefix_type") == c["prefix_type"]
                        and isinstance(item.get("source_id"), str)
                        and item["source_id"].startswith(c["source_id"])
                    ]
                    if len(candidates) == 1:
                        item = dict(item)
                        item["source_id"] = candidates[0]["source_id"]
                        case = candidates[0]
                        key = f"{item['source_id']}:{item['prefix_type']}"
                if case is None:
                    errors.append(f"foreign or malformed item {key}")
                    continue
                if key in by_key:
                    errors.append(f"duplicate item {key}")
                    continue
                valid, error = validate_item(normalize_item(item, case), case)
                if valid is None:
                    errors.append(f"{key}: {error}")
                else:
                    by_key[key] = valid
            missing = [f"{c['source_id']}:{c['prefix_type']}" for c in request_cases
                       if f"{c['source_id']}:{c['prefix_type']}" not in by_key]
            if missing:
                errors.append("missing items: " + ", ".join(missing))
            if errors:
                raise ValueError("; ".join(errors))
            for key, annotation in by_key.items():
                completed[key] = annotation
                batch_for_key[key] = batch_id
            append_jsonl(batch_log, {"batch_id": batch_id, "status": "ok", "model": args.model,
                                     "prompt_sha256": prompt_sha, "source_ids": sorted(source_ids),
                                     "annotations": list(by_key.values()), "raw_response": raw})
            print(f"{batch_id}: validated {len(by_key)} annotations", flush=True)
        except Exception as exc:
            append_jsonl(batch_log, {"batch_id": batch_id, "status": "error", "model": args.model,
                                     "prompt_sha256": prompt_sha, "source_ids": sorted(source_ids),
                                     "error": f"{type(exc).__name__}: {exc}",
                                     **({"raw_response": raw} if raw is not None else {})})
            append_jsonl(quarantine_path, {"batch_id": batch_id, "source_ids": sorted(source_ids),
                                           "error": f"{type(exc).__name__}: {exc}"})
            print(f"{batch_id}: ERROR {type(exc).__name__}: {exc}", flush=True)
        processed += 1
        if args.max_batches and processed >= args.max_batches:
            break

    # Rewrite per-split files atomically from validated records, so a resumed
    # run never duplicates a sample after a partial process termination.
    for split in ("train", "dev"):
        raw_rows = []
        sft_rows = []
        for case in cases:
            if case["split"] != split or case["prefix_type"] not in PREFIX_TYPES:
                continue
            prefix_type = case["prefix_type"]
            key = f"{case['source_id']}:{prefix_type}"
            annotation = completed.get(key)
            if annotation is None:
                continue
            raw_rows.append({"sample_id": key, "source_id": case["source_id"],
                             "split": split, "source_index": case["source_index"],
                             "prefix_type": prefix_type, "scenario": case["scenario"],
                             "history": case["history"], "annotation": annotation})
            sft_rows.append(make_sft_row(case, annotation, batch_for_key.get(key, "resumed")))
        write_jsonl(output / f"planner_raw_{split}.jsonl", raw_rows)
        write_jsonl(sft_by_split[split], sft_rows)

    counts = {split: sum(1 for key in completed if key.startswith(split + "_")) for split in ("train", "dev")}
    summary = {"validated_annotations": len(completed), "expected_annotations": len(cases),
               "missing_annotations": len(cases) - len(completed), "by_split": counts,
               "planned_batches": len(batches), "processed_batches": processed}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(completed) != len(cases) and not args.max_batches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
