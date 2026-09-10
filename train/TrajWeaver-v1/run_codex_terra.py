#!/usr/bin/env python3
"""Generate direct Codex GPT-5.6 Terra annotations without an API client.

This runner invokes the local ``codex exec`` CLI, not ChatAnywhere/OpenAI.
Each batch has an auditable raw response and a parsed JSONL file.  Completed
batches are reused on later invocations; accepted rows are passed through the
existing Terra validator before entering the append-only annotation artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from trajweaver_v1.data import read_jsonl
from trajweaver_v1.terra import (
    TERRA_PROMPT_VERSION,
    TERRA_TEACHER_MODEL,
    append_terra_annotations,
    terra_prompt,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_TRAIN = HERE / "data/lora/weaver_sft/train.scenarios-2000.jsonl"
DEFAULT_DEV = HERE / "data/lora/weaver_sft/dev.scenarios-300.jsonl"
DEFAULT_OUTPUT = HERE / "data/terra/annotations.jsonl"
DEFAULT_BATCH_DIR = HERE / "data/terra/codex_batches"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "max"),
        default="max",
        help="Codex effort for the fixed Terra teacher (max corresponds to extra-high)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry", type=int, default=2)
    return parser.parse_args()


def _public_prompt(sample: dict[str, Any]) -> str:
    return terra_prompt(sample)


def make_batch_prompt(samples: list[dict[str, Any]]) -> str:
    blocks = []
    for index, sample in enumerate(samples, 1):
        blocks.append(f"TASK {index}\n{_public_prompt(sample)}")
    return """You are Codex GPT-5.6 Terra performing a direct offline teacher pass.
There is no API call and no hidden dataset state. Complete every task below.
Return one JSON array only, with exactly one object per task, in task order.
Every object must contain sample_id, teacher_model, prompt_version, route_state,
action, belief_state, desire_state, goal_alignment, and trigger_action. Use
teacher_model=codex_gpt-5.6-terra and prompt_version=trajweaver-v1-terra-direct-v1.
Do not include markdown, explanations, evidence, or copied scenario text.

""" + "\n\n".join(blocks)


def _extract_json_values(text: str) -> list[dict[str, Any]]:
    """Extract one JSON array or JSONL objects from Codex's final message."""

    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "")
    decoder = json.JSONDecoder()
    values: list[Any] = []
    for match in re.finditer(r"[\[{]", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            values.extend(value)
            break
        if isinstance(value, dict):
            values.append(value)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        sample_id = str(value.get("sample_id", "")).strip()
        if sample_id and sample_id not in seen:
            result.append(value)
            seen.add(sample_id)
    return result


def _batch_paths(batch_dir: Path, batch_index: int) -> tuple[Path, Path, Path]:
    stem = f"batch-{batch_index:04d}"
    return (
        batch_dir / f"{stem}.prompt.txt",
        batch_dir / f"{stem}.response.txt",
        batch_dir / f"{stem}.jsonl",
    )


def run_batch(
    batch_index: int,
    samples: list[dict[str, Any]],
    batch_dir: Path,
    *,
    model: str,
    reasoning_effort: str,
    retry: int,
) -> dict[str, Any]:
    prompt_path, response_path, parsed_path = _batch_paths(batch_dir, batch_index)
    expected = {str(sample["sample_id"]) for sample in samples}
    if parsed_path.exists():
        try:
            cached = read_jsonl(parsed_path)
            if {str(row.get("sample_id", "")) for row in cached} == expected:
                return {"batch": batch_index, "status": "cached", "rows": len(cached)}
        except (OSError, ValueError):
            pass
    prompt = make_batch_prompt(samples)
    prompt_path.write_text(prompt, encoding="utf-8")
    last_error = ""
    for attempt in range(max(1, retry)):
        with tempfile.NamedTemporaryFile(prefix="terra-final-", suffix=".txt", delete=False) as handle:
            final_path = Path(handle.name)
        try:
            command = [
                "codex",
                "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--model",
                model,
                "-c",
                f"model_reasoning_effort={reasoning_effort}",
                "-o",
                str(final_path),
                "-",
            ]
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                cwd=PROJECT,
                capture_output=True,
                check=False,
                timeout=900,
            )
            response = final_path.read_text(encoding="utf-8") if final_path.exists() else completed.stderr
            response_path.write_text(response, encoding="utf-8")
            parsed = _extract_json_values(response)
            if {str(row.get("sample_id", "")) for row in parsed} == expected:
                with parsed_path.open("w", encoding="utf-8") as output:
                    for row in parsed:
                        output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                return {"batch": batch_index, "status": "generated", "rows": len(parsed)}
            last_error = f"expected {len(expected)} rows, parsed {len(parsed)}"
        except (OSError, subprocess.SubprocessError) as exc:
            last_error = f"{type(exc).__name__}:{exc}"
        finally:
            try:
                final_path.unlink(missing_ok=True)
            except OSError:
                pass
    return {"batch": batch_index, "status": "error", "error": last_error}


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 1:
        raise ValueError("batch-size and workers must be positive")
    if args.model != "gpt-5.6-terra":
        raise ValueError("TrajWeaver-v1 teacher is fixed to the local Codex model gpt-5.6-terra")
    train = read_jsonl(args.train)
    dev = read_jsonl(args.dev)
    samples = train + dev
    if args.limit is not None:
        samples = samples[: max(0, args.limit)]
    batches = [
        samples[start : start + args.batch_size]
        for start in range(0, len(samples), args.batch_size)
    ]
    args.batch_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_batch,
                index,
                batch,
                args.batch_dir,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                retry=args.retry,
            ): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    errors = [result for result in results if result["status"] == "error"]
    if errors:
        print(f"terra batch failures={len(errors)}; rerun to resume", flush=True)
        return 2

    # Merge in deterministic order after all workers finish; the validator
    # rejects malformed, private, duplicate, or out-of-source rows.
    accepted = rejected = 0
    for index, batch in enumerate(batches):
        _, _, parsed_path = _batch_paths(args.batch_dir, index)
        supplied = read_jsonl(parsed_path)
        stats = append_terra_annotations(args.output, samples, supplied)
        accepted += stats["accepted"]
        rejected += stats["rejected"]
    print(
        json.dumps(
            {
                "teacher_model": TERRA_TEACHER_MODEL,
                "prompt_version": TERRA_PROMPT_VERSION,
                "samples": len(samples),
                "batches": len(batches),
                "accepted": accepted,
                "rejected": rejected,
                "annotation_path": str(args.output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
