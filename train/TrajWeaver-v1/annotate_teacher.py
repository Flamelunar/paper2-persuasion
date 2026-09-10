#!/usr/bin/env python3
"""Compatibility entrypoint for the direct Codex Terra teacher.

This file intentionally contains no HTTP/API fallback. ``--model
gpt-5.6-terra`` is forwarded to :mod:`run_codex_terra`, which invokes the
locally installed ``codex exec`` plug-in and writes auditable batch artifacts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", choices=("minimal", "low", "medium", "high", "max"), default="max")
    parser.add_argument("--train", type=Path, default=HERE / "data/lora/weaver_sft/train.scenarios-2000.jsonl")
    parser.add_argument("--dev", type=Path, default=HERE / "data/lora/weaver_sft/dev.scenarios-300.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "data/terra/annotations.jsonl")
    parser.add_argument("--batch-dir", type=Path, default=HERE / "data/terra/codex_batches")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model != "gpt-5.6-terra":
        raise SystemExit(
            "TrajWeaver-v1 teacher is fixed to gpt-5.6-terra and must run through the local Codex plug-in"
        )
    command = [
        sys.executable,
        str(HERE / "run_codex_terra.py"),
        "--model",
        args.model,
        "--reasoning-effort",
        args.reasoning_effort,
        "--train",
        str(args.train),
        "--dev",
        str(args.dev),
        "--output",
        str(args.output),
        "--batch-dir",
        str(args.batch_dir),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--retry",
        str(args.retry),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return subprocess.call(command, cwd=PROJECT)


if __name__ == "__main__":
    raise SystemExit(main())
