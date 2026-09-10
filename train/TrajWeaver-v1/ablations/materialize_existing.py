#!/usr/bin/env python3
"""Filter already-computed 525-row baselines onto the frozen manifest."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from make_manifest import load_manifest  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "data/CToMPersu/dataset/CToMPersu_Eval/CToMPersu_Eval.json"
CANONICAL = ROOT / "results/Meta-Llama-3.1-8B-Instruct"
OUTPUT = CANONICAL / "ablations/trajweaver-mechanism-100"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "manifests/eval100_seed20260910.json"
SOURCES = {
    "public-zero-shot": ("raw/zero-shot.jsonl", "quality/zero-shot.jsonl", "Zero-shot"),
    "legacy-v1": ("raw/TrajWeaver-v1.jsonl", "quality/TrajWeaver-v1.jsonl", "TrajWeaver-v1"),
}


def latest(path: Path, key: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get(key)
        if value is not None:
            rows[int(value)] = row
    return rows


def turns(value: Any) -> list[dict[str, str]]:
    result = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict):
            result.append({"role": str(item.get("role", "")).lower(), "content": str(item.get("content", ""))})
        else:
            role, separator, content = str(item).partition(":")
            result.append({"role": role.lower() if separator else "unknown", "content": content if separator else str(item)})
    return result


def materialize(variant: str, raw_rel: str, quality_rel: str, method: str, manifest_path: Path) -> Path:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    manifest, indices = load_manifest(manifest_path, DATASET)
    raw = latest(CANONICAL / raw_rel, "index")
    quality = latest(CANONICAL / quality_rel, "source_index")
    if any(index not in raw or raw[index].get("status") != "ok" for index in indices):
        raise ValueError(f"Missing successful raw record for {variant}")
    if any(index not in quality or quality[index].get("status") != "ok" for index in indices):
        raise ValueError(f"Missing successful quality record for {variant}")
    root = OUTPUT / variant
    root.mkdir(parents=True, exist_ok=True)
    selected_raw = []
    aligned = []
    selected_quality = []
    for index in indices:
        record = dict(raw[index])
        record["source_index"] = index
        record["ablation_variant"] = variant
        selected_raw.append(record)
        aligned.append(
            {
                "source_index": index,
                "scenario": dataset[index]["scenario"],
                "generated_dialogue": turns(record.get("history")),
                "evaluation": record,
            }
        )
        selected_quality.append(dict(quality[index]))
    (root / "raw.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected_raw),
        encoding="utf-8",
    )
    (root / "aligned.json").write_text(json.dumps(aligned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "quality.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected_quality),
        encoding="utf-8",
    )
    metadata = {
        "variant": variant,
        "source_method": method,
        "source_raw": str((CANONICAL / raw_rel).resolve()),
        "source_quality": str((CANONICAL / quality_rel).resolve()),
        "manifest": str(manifest_path.resolve()),
        "selection_sha256": manifest["selection_sha256"],
        "n": len(indices),
        "recomputed": False,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--variant", choices=tuple(SOURCES), nargs="+", default=list(SOURCES))
    args = parser.parse_args()
    for variant in args.variant:
        print(materialize(variant, *SOURCES[variant], args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
