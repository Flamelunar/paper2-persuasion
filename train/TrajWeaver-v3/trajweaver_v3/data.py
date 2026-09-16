"""Public-only CToMPersu conversion and model inputs; no teacher labels."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

PUBLIC_FIELDS = ("tag", "background", "persuadee", "persuader", "goal", "domain")
PRIVATE_FIELDS = {"preventive", "generative", "persona"}
ROLE = re.compile(r"^\s*(persuader|persuadee)\s*:\s*(.*)$", re.I | re.S)
SYSTEM = """You are the persuader in a natural multi-turn dialogue.
Write only the next persuader utterance, at most three concise sentences.
Address the latest stated concern and preserve the exact original goal,
including its alternatives, priorities, and time constraints. A small step
must keep a path to that goal. Be respectful and preserve the person's choice.
Do not invent facts, private motives, guarantees, authorities, or urgency."""


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def file_digest(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def public_scenario(scenario: dict) -> dict:
    return {key: scenario.get(key) for key in PUBLIC_FIELDS}


def assert_public(value: Any) -> None:
    if isinstance(value, dict):
        if PRIVATE_FIELDS.intersection(value):
            raise ValueError("Private simulator fields in public data")
        for item in value.values():
            assert_public(item)
    elif isinstance(value, list):
        for item in value:
            assert_public(item)


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    return value


def scenario_id(scenario: dict) -> str:
    return digest(_normalize(public_scenario(scenario)))[:24]


def repair_dialogue(lines: list) -> list[dict]:
    messages = []
    for line in lines:
        line = str(line or "").strip()
        if not line:
            continue
        match = ROLE.match(line)
        if match:
            messages.append({"role": match[1].lower(), "content": match[2].strip()})
        elif messages:
            messages[-1]["content"] = (messages[-1]["content"] + " " + line).strip()
        else:
            raise ValueError("Dialogue starts with an unlabelled fragment")
    if len(messages) not in (6, 8):
        raise ValueError(f"Expected 3/4 rounds after fragment repair, got {len(messages)} messages")
    for index, message in enumerate(messages):
        expected = "persuader" if index % 2 == 0 else "persuadee"
        if message["role"] != expected or not message["content"]:
            raise ValueError(f"Invalid role/content at message {index}")
    return messages


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            assert_public(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare(source: Path, output: Path) -> dict:
    rows_by_split, identities, counts = {}, {}, {}
    for split in ("train", "dev", "test"):
        raw = json.loads((source / f"{split}.json").read_text(encoding="utf-8"))
        seen, turns, duplicate_count, repairs = set(), [], 0, 0
        for index, record in enumerate(raw):
            scenario = public_scenario(record["scenario"])
            if not str(scenario.get("goal") or "").strip():
                raise ValueError(f"Empty goal at {split}:{index}")
            sid = scenario_id(scenario)
            if sid in seen:
                duplicate_count += 1
                continue
            seen.add(sid)
            try:
                history = repair_dialogue(record["dialog"])
            except ValueError as exc:
                raise ValueError(f"{split}:{index}: {exc}") from exc
            repairs += int(len(history) != len(record["dialog"]))
            for offset in range(0, len(history), 2):
                turn = offset // 2
                turns.append({
                    "sample_id": f"{sid}:t{turn + 1}", "scenario_id": sid,
                    "source_index": index, "split": split, "scenario": scenario,
                    "history": history[:offset], "target": history[offset]["content"],
                    "turn_index": turn, "max_turns": 4, "remaining_turns": 4 - turn,
                })
        rows_by_split[split], identities[split] = turns, seen
        counts[split] = {"source_rows": len(raw), "unique_scenarios": len(seen),
                         "turn_samples": len(turns), "duplicates_removed": duplicate_count,
                         "fragment_repaired_scenarios": repairs,
                         "source_sha256": file_digest(source / f"{split}.json")}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = identities[left] & identities[right]
        if overlap:
            raise ValueError(f"Scenario leakage between {left}/{right}: {len(overlap)}")
    manifest = {"method": "TrajWeaver-v3", "source": str(source.resolve()),
                "public_fields": list(PUBLIC_FIELDS), "splits": counts,
                "policy": "preserve mysplit membership; deduplicate by public scenario"}
    for split, rows in rows_by_split.items():
        write_jsonl(output / f"{split}.jsonl", rows)
    write_json(output / "manifest.json", manifest)
    return manifest


def messages(row: dict) -> list[dict]:
    assert_public(row)
    scenario = public_scenario(row["scenario"])
    history = "\n".join(f"{item['role']}: {item['content']}" for item in row["history"])
    remaining = int(row["max_turns"]) - int(row["turn_index"])
    if remaining <= 0:
        raise ValueError("No remaining persuader turns")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
        f"Public scenario:\n{json.dumps(scenario, ensure_ascii=False)}\n\n"
        f"Observable dialogue:\n{history or '(no dialogue yet)'}\n\n"
        f"{remaining} persuader response(s) remain including this one."}]


def encode(tokenizer: Any, row: dict, config: dict) -> tuple:
    import torch

    def chat(items: list[dict], maximum: int):
        ids = tokenizer.apply_chat_template(items, tokenize=True,
                                            add_generation_prompt=True, return_tensors="pt")
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        elif isinstance(ids, dict):
            ids = ids["input_ids"]
        ids = torch.as_tensor(ids, dtype=torch.long).reshape(1, -1)
        # Failing is preferable to silently losing the immutable goal or a concern.
        if ids.shape[1] > maximum:
            raise ValueError(f"{row.get('sample_id')}: prompt length {ids.shape[1]} > {maximum}")
        return ids

    prompt = chat(messages(row), int(config["max_prompt_tokens"]))
    goal = chat([{"role": "system", "content": "Encode the exact immutable target, including all constraints."},
                 {"role": "user", "content": str(row["scenario"]["goal"])}],
                int(config["max_goal_tokens"]))
    target = None
    if "target" in row:
        target = tokenizer(row["target"], add_special_tokens=False, return_tensors="pt").input_ids
        eos = tokenizer.eos_token_id
        if eos is not None and (target.numel() == 0 or target[0, -1].item() != eos):
            target = torch.cat((target, torch.tensor([[eos]], dtype=torch.long)), dim=1)
        if target.numel() == 0 or target.shape[1] > int(config["max_target_tokens"]):
            raise ValueError(f"{row.get('sample_id')}: invalid target length {target.shape[1]}")
    return prompt, goal, target
