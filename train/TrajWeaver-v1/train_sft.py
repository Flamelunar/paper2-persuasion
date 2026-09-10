#!/usr/bin/env python3
"""Train TrajWeaver-v1 controller warm-up and trajectory distillation."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

from trajweaver_v1.modeling import load_model_and_tokenizer, load_yaml
from trajweaver_v1.trainer import Phase, SFTTrainer, TurnDataset
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
    parser.add_argument(
        "--train-data",
        type=Path,
        default=HERE / "data/lora/weaver_sft/train.scenarios-2000.jsonl",
    )
    parser.add_argument(
        "--dev-data",
        type=Path,
        default=HERE / "data/lora/weaver_sft/dev.scenarios-300.jsonl",
    )
    parser.add_argument(
        "--annotations", type=Path, default=HERE / "data/terra/annotations.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "checkpoints/qwen25-7b")
    parser.add_argument("--resume-components", type=Path, default=None)
    parser.add_argument(
        "--resume-phase",
        choices=("warmup", "trajectory"),
        default=None,
        help="Phase containing --resume-components; inferred from metadata when omitted.",
    )
    parser.add_argument(
        "--resume-epoch",
        type=int,
        default=None,
        help="Completed absolute epoch in --resume-phase; inferred from metadata when omitted.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--allow-bootstrap-only",
        action="store_true",
        help="Allow a smoke test without teacher belief/desire/drift labels.",
    )
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
        config, for_training=True, checkpoint=args.resume_components
    )
    # Checkpoints intentionally contain adapter/module weights only (not an
    # optimizer state).  We still recover the phase/epoch/global-step metadata
    # so an interrupted run resumes at the next absolute epoch and never
    # overwrites an already completed checkpoint.
    resume_phase = args.resume_phase
    resume_epoch = args.resume_epoch
    resume_global_step = 0
    if args.resume_components is not None:
        metadata_path = args.resume_components / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Resume checkpoint lacks metadata.json: {args.resume_components}")
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        inferred_phase = str(metadata.get("phase", ""))
        if resume_phase is None:
            if inferred_phase not in {"warmup", "trajectory"}:
                raise ValueError(
                    f"Cannot infer an SFT phase from checkpoint metadata: {inferred_phase!r}"
                )
            resume_phase = inferred_phase
        if resume_epoch is None:
            resume_epoch = int(metadata.get("epoch", 0))
        if resume_epoch < 1:
            raise ValueError(f"Resume epoch must be >=1, got {resume_epoch}")
        resume_global_step = int(metadata.get("global_step", 0))
        if inferred_phase and inferred_phase != resume_phase:
            raise ValueError(
                f"Resume phase mismatch: metadata={inferred_phase!r}, argument={resume_phase!r}"
            )
    data_cfg = config["data"]
    annotations = args.annotations if args.annotations.exists() else None
    train_data = TurnDataset(
        args.train_data,
        tokenizer,
        max_goal_tokens=int(data_cfg.get("max_goal_tokens", 256)),
        max_prompt_tokens=int(data_cfg.get("max_prompt_tokens", 1024)),
        max_target_tokens=int(data_cfg.get("max_target_tokens", 192)),
        annotations=annotations,
    )
    dev_data = TurnDataset(
        args.dev_data,
        tokenizer,
        max_goal_tokens=int(data_cfg.get("max_goal_tokens", 256)),
        max_prompt_tokens=int(data_cfg.get("max_prompt_tokens", 1024)),
        max_target_tokens=int(data_cfg.get("max_target_tokens", 192)),
        annotations=annotations,
    )
    if args.limit is not None:
        train_data.rows = train_data.rows[: max(0, args.limit)]
        dev_data.rows = dev_data.rows[: max(1, args.limit // 10)]
    training = config["training"]
    warmup_epochs = int(training.get("warmup_epochs", 2))
    trajectory_epochs = int(training.get("trajectory_epochs", 3))
    phases = [
        Phase(
            "warmup",
            warmup_epochs if resume_phase is None else (0 if resume_phase == "trajectory" else max(0, warmup_epochs - int(resume_epoch or 0))),
            float(training.get("warmup_language_weight", 0.5)),
            float(training.get("warmup_route_weight", 1.0)),
            float(training.get("warmup_action_weight", 1.0)),
            start_epoch=(int(resume_epoch or 0) + 1 if resume_phase == "warmup" else 1),
        ),
        Phase(
            "trajectory",
            trajectory_epochs if resume_phase is None else (max(0, trajectory_epochs - int(resume_epoch or 0)) if resume_phase == "trajectory" else trajectory_epochs),
            float(training.get("language_weight", 1.0)),
            float(training.get("route_weight", 0.25)),
            float(training.get("action_weight", 0.25)),
            start_epoch=(int(resume_epoch or 0) + 1 if resume_phase == "trajectory" else 1),
        ),
    ]
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"trainable={trainable:,} total={total:,} ratio={trainable / total:.6%}")
    print(
        f"resume_phase={resume_phase or 'none'} resume_epoch={resume_epoch or 0} "
        f"resume_global_step={resume_global_step}",
        flush=True,
    )
    SFTTrainer(
        model,
        tokenizer,
        config,
        output_dir=args.output_dir,
        initial_global_step=resume_global_step,
    ).run(
        train_data, dev_data, phases
    )


if __name__ == "__main__":
    main()
