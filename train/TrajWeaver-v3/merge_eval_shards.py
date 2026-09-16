#!/usr/bin/env python3
"""Validate and merge independent TrajWeaver evaluation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total", type=int, default=525)
    args = parser.parse_args()
    if args.total < 1:
        parser.error("--total must be positive")

    rows_by_index: dict[int, dict[str, Any]] = {}
    manifests: list[dict[str, Any]] = []
    for shard in args.shard:
        manifest_path = shard.with_suffix(".manifest.json")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing shard manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"Shard manifest is not an object: {manifest_path}")
        manifests.append(manifest)
        for row in read_jsonl(shard):
            index = int(row.get("index", -1))
            if not 0 <= index < args.total:
                raise ValueError(f"{shard}: index {index} is outside 0..{args.total - 1}")
            if row.get("status") != "ok":
                raise ValueError(f"{shard}: index {index} is not successful")
            if index in rows_by_index:
                raise ValueError(f"Duplicate index across shards: {index}")
            evaluation_id = row.get("evaluation_id")
            if not isinstance(evaluation_id, str) or not evaluation_id.endswith(f":row{index}"):
                raise ValueError(f"{shard}: invalid evaluation_id for index {index}")
            rows_by_index[index] = row

    expected = set(range(args.total))
    actual = set(rows_by_index)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Shard coverage incomplete: missing={missing[:10]}, extra={extra[:10]}")

    comparable = ("method", "protocol", "config", "checkpoint_sha256", "dataset_sha256",
                  "mode", "simulator_model", "judge_model", "judge_prompt_version",
                  "base_url", "max_new_tokens", "preserve_rows", "duplicate_policy")
    reference = {key: manifests[0].get(key) for key in comparable}
    for manifest in manifests[1:]:
        for key, value in reference.items():
            if manifest.get(key) != value:
                raise ValueError(f"Shard manifest mismatch in {key}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for index in range(args.total):
            handle.write(json.dumps(rows_by_index[index], ensure_ascii=False) + "\n")
    temporary.replace(args.output)

    merged_manifest = dict(manifests[0])
    merged_manifest.update({"start_index": 0, "end_index": None,
                            "shards": [str(path.resolve()) for path in args.shard],
                            "merged_rows": args.total})
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(merged_manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest": str(manifest_path),
                      "rows": args.total, "shards": len(args.shard)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
