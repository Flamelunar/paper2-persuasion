#!/usr/bin/env python3
"""Ingest direct Codex Terra labels without any API or model call.

The direct Codex pass supplies JSON objects through ``--labels``. This command
only validates and appends them, writes provenance metadata, and reports the
coverage gate used by the training entrypoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from trajweaver_v1.data import read_jsonl  # noqa: E402
from trajweaver_v1.terra import (  # noqa: E402
    TERRA_PROMPT_VERSION,
    TERRA_TEACHER_MODEL,
    append_terra_annotations,
    coverage_report,
    read_terra_annotations,
    terra_prompt,
    write_terra_metadata,
)


DEFAULT_TRAIN = HERE / "data/lora/weaver_sft/train.scenarios-2000.jsonl"
DEFAULT_DEV = HERE / "data/lora/weaver_sft/dev.scenarios-300.jsonl"
DEFAULT_OUTPUT = HERE / "data/terra/annotations.jsonl"
DEFAULT_METADATA = HERE / "data/terra/metadata.json"
DEFAULT_REPORT = HERE / "data/terra/coverage.json"
DEFAULT_MANIFEST = HERE / "data/lora/manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, help="JSONL emitted by the direct Codex Terra pass")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--coverage-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    parser.add_argument("--print-prompts", action="store_true", help="Print fixed prompts for a direct Codex pass")
    parser.add_argument("--check", action="store_true", help="Only validate the existing append-only artifact")
    return parser.parse_args()


def read_labels(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Label row {path}:{line_number} is not an object")
        rows.append(value)
    return rows


def main() -> int:
    args = parse_args()
    sources = {"train": args.train, "dev": args.dev}
    samples = read_jsonl(args.train) + read_jsonl(args.dev)
    if args.print_prompts:
        for sample in samples:
            print(terra_prompt(sample))
    if args.labels is not None and not args.check:
        stats = append_terra_annotations(args.output, samples, read_labels(args.labels))
        print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    elif not args.check and args.labels is None and not args.print_prompts:
        raise SystemExit("Provide --labels from the direct Codex pass, or use --check")

    write_terra_metadata(
        args.metadata,
        sources=sources,
        manifest=args.manifest,
        seed=args.seed,
        annotation_path=args.output,
    )
    report = coverage_report(sources, args.output, minimum=args.minimum_coverage)
    args.coverage_report.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
