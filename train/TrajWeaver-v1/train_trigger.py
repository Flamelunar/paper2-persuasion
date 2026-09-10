#!/usr/bin/env python3
"""Train the independent binary Memory Trigger with supervised labels."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

from trajweaver_v1.modeling import load_model_and_tokenizer, load_yaml
from trajweaver_v1.trainer import TriggerDataset, TriggerTrainer
from trajweaver_v1.terra import require_training_coverage

HERE = Path(__file__).resolve().parent


class _Tee:
    """Keep terminal output while persisting a copy for long training runs."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _install_log(output_dir: Path) -> None:
    log_dir = HERE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{output_dir.name}.train.log"
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, handle)
    sys.stderr = _Tee(sys.__stderr__, handle)
    print(f"log_file={log_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weaver-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--train-data",
        type=Path,
        default=HERE / "data/lora/trigger_sft/train.scenarios-2000.jsonl",
    )
    parser.add_argument(
        "--dev-data",
        type=Path,
        default=HERE / "data/lora/trigger_sft/dev.scenarios-300.jsonl",
    )
    parser.add_argument(
        "--annotations", type=Path, default=HERE / "data/terra/annotations.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "checkpoints/trigger")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-bootstrap-only", action="store_true")
    parser.add_argument("--minimum-annotation-coverage", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _install_log(args.output_dir)
    config = load_yaml(args.config)
    seed = int(config["training"].get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if not args.allow_bootstrap_only:
        report = require_training_coverage(
            args.train_data,
            args.dev_data,
            args.annotations,
            minimum=args.minimum_annotation_coverage,
        )
        print(f"terra_coverage={report['splits']}", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        config, for_training=True, checkpoint=args.weaver_checkpoint
    )
    annotations = args.annotations if args.annotations.exists() else None
    max_prompt_tokens = int(config["data"].get("max_prompt_tokens", 1024))
    train_data = TriggerDataset(
        args.train_data,
        tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        annotations=annotations,
    )
    dev_data = TriggerDataset(
        args.dev_data,
        tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        annotations=annotations,
    )
    if not train_data.rows or not dev_data.rows:
        raise RuntimeError("Trigger train/dev data contains no labeled rows")
    if args.limit is not None:
        train_data.rows = train_data.rows[: max(1, args.limit)]
        dev_data.rows = dev_data.rows[: max(1, args.limit // 10)]
    TriggerTrainer(
        model, tokenizer, config, output_dir=args.output_dir
    ).run(train_data, dev_data)


if __name__ == "__main__":
    main()
