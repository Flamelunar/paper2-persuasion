#!/usr/bin/env python3
"""Materialize and blindly judge one 100-row ablation result."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from eval.evaluate_zero_shot_quality import (  # noqa: E402
    PROMPT_VERSION,
    aggregate,
    append_jsonl,
    call_judge,
)
from train.ctompersu_common import (  # noqa: E402
    DEFAULT_API_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    load_dataset,
)
from make_manifest import load_manifest  # noqa: E402

DEFAULT_DATASET = PROJECT / "data/CToMPersu/dataset/CToMPersu_Eval/CToMPersu_Eval.json"
DEFAULT_OUTPUT = PROJECT / "results/Meta-Llama-3.1-8B-Instruct/ablations/trajweaver-mechanism-100"
DEFAULT_MANIFEST = HERE / "manifests/eval100_seed20260910.json"
MODEL_NAME = "Meta-Llama-3.1-8B-Instruct"
_WRITE_LOCK = threading.Lock()


def latest(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        index = row.get("source_index", row.get("index"))
        if index is not None:
            rows[int(index)] = row
    return rows


def to_turns(value: Any) -> list[dict[str, str]]:
    turns = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict):
            turns.append({"role": str(item.get("role", "")).strip().lower(), "content": str(item.get("content", "")).strip()})
        else:
            role, separator, content = str(item).partition(":")
            turns.append({
                "role": role.strip().lower() if separator else "unknown",
                "content": content.strip() if separator else str(item).strip(),
            })
    return turns


def materialize_aligned(root: Path, indices: list[int], dataset: list[dict[str, Any]], raw: dict[int, dict[str, Any]]) -> Path:
    rows = []
    for index in indices:
        record = raw[index]
        rows.append({
            "source_index": index,
            "scenario": dataset[index]["scenario"],
            "generated_dialogue": to_turns(record.get("history")),
            "evaluation": record,
        })
    path = root / "aligned.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest, indices = load_manifest(args.manifest, args.dataset)
    dataset = load_dataset(args.dataset)
    root = args.output_root / args.variant
    raw = latest(root / "raw.jsonl")
    if set(raw) != set(indices) or any(row.get("status") != "ok" for row in raw.values()):
        raise ValueError(f"Raw ablation output is not complete for {args.variant}")
    aligned_path = materialize_aligned(root, indices, dataset, raw)
    aligned = {row["source_index"]: row for row in json.loads(aligned_path.read_text(encoding="utf-8"))}
    quality_path = root / "quality.jsonl"
    existing = latest(quality_path)
    expected_method = f"TrajWeaver-ablation:{args.variant}"
    for index, cached in existing.items():
        if cached.get("method") not in {None, expected_method}:
            raise ValueError(f"Refusing to mix quality variants at source_index={index}")
    pending = []
    for index in indices:
        cached = existing.get(index)
        if (
            not args.force
            and cached is not None
            and cached.get("status") == "ok"
            and cached.get("judge_model") == args.judge_model
            and cached.get("prompt_version") == PROMPT_VERSION
            and cached.get("method") == expected_method
        ):
            continue
        pending.append(index)
    print(f"{args.variant}: selected={len(indices)} cached={len(indices)-len(pending)} pending={len(pending)}", flush=True)

    def work(index: int) -> dict[str, Any]:
        started = time.monotonic()
        base = {
            "source_index": index,
            "model": MODEL_NAME,
            "method": f"TrajWeaver-ablation:{args.variant}",
            "judge_model": args.judge_model,
            "prompt_version": PROMPT_VERSION,
        }
        try:
            judgment, attempts, latency = call_judge(
                aligned[index],
                judge_model=args.judge_model,
                base_url=args.base_url,
                max_retries=args.max_retries,
                max_completion_tokens=args.max_completion_tokens,
            )
            return {**base, "status": "ok", "attempts": attempts, "latency_seconds": latency, **judgment}
        except Exception as exc:
            return {**base, "status": "error", "error": f"{type(exc).__name__}:{exc}", "wall_seconds": time.monotonic()-started}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(work, index): index for index in pending}
        for future in as_completed(futures):
            record = future.result()
            with _WRITE_LOCK:
                append_jsonl(quality_path, record)
            print(f"{args.variant} quality source_index={record['source_index']} status={record['status']}", flush=True)

    latest_quality = latest(quality_path)
    records = [latest_quality[index] for index in indices if latest_quality.get(index, {}).get("status") == "ok"]
    summary = {
        "schema_version": "trajweaver_ablation_quality_v1",
        "variant": args.variant,
        "model": MODEL_NAME,
        "judge_model": args.judge_model,
        "prompt_version": PROMPT_VERSION,
        "manifest": str(args.manifest.resolve()),
        "selection_sha256": manifest["selection_sha256"],
        "requested_n": len(indices),
        **aggregate(records),
    }
    (root / "quality-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if len(records) != len(indices):
        raise SystemExit(f"Quality incomplete: {len(records)}/{len(indices)}")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
