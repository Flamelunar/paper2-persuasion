#!/usr/bin/env python3
"""Run memory-conditioned DPO with an in-place precomputed SFT reference."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from trajweaver_v1.checkpoint import save_checkpoint
from trajweaver_v1.data import read_jsonl
from trajweaver_v1.modeling import load_model_and_tokenizer, load_yaml
from trajweaver_v1.trainer import tokenize_goal, tokenize_prompt, tokenize_target

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preferences", type=Path, default=HERE / "data/preferences.jsonl")
    parser.add_argument("--output-dir", type=Path, default=HERE / "checkpoints/dpo")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def tensors(tokenizer: Any, row: dict[str, Any], config: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    data = config["data"]
    goal = tokenize_goal(tokenizer, row, int(data.get("max_goal_tokens", 256)))
    prompt = tokenize_prompt(tokenizer, row, int(data.get("max_prompt_tokens", 1024)))
    chosen = tokenize_target(tokenizer, row["chosen"], int(data.get("max_target_tokens", 192)))
    rejected = tokenize_target(tokenizer, row["rejected"], int(data.get("max_target_tokens", 192)))
    return (
        goal,
        torch.ones_like(goal),
        prompt,
        torch.ones_like(prompt),
        chosen,
        torch.ones_like(chosen),
        rejected,
        torch.ones_like(rejected),
    )


@torch.no_grad()
def reference_logprobs(
    model: Any, tokenizer: Any, rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, tuple[float, float]]:
    """Snapshot reference scores before the first optimizer step, using no second 7B copy."""

    model.eval()
    result: dict[str, tuple[float, float]] = {}
    for index, row in enumerate(rows):
        goal, goal_mask, prompt, prompt_mask, chosen, chosen_mask, rejected, rejected_mask = tensors(
            tokenizer, row, config
        )
        woven = model.weave(
            prompt,
            prompt_mask,
            goal_ids=goal,
            goal_attention_mask=goal_mask,
            force_probe=int(row["turn_index"]) == 0,
            force_invoke=True,
        )
        chosen_lp = model.target_logprobs(
            prompt, chosen, woven.memory_tokens, prompt_attention_mask=prompt_mask,
            target_attention_mask=chosen_mask,
        )
        rejected_lp = model.target_logprobs(
            prompt, rejected, woven.memory_tokens, prompt_attention_mask=prompt_mask,
            target_attention_mask=rejected_mask,
        )
        result[str(row["sample_id"])] = (float(chosen_lp.item()), float(rejected_lp.item()))
        if (index + 1) % 100 == 0:
            print(f"reference={index + 1}/{len(rows)}", flush=True)
    model.train()
    return result


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed = int(config["training"].get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    rows = read_jsonl(args.preferences)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    if not rows:
        raise ValueError("No preference pairs were loaded")
    model, tokenizer = load_model_and_tokenizer(
        config, for_training=True, checkpoint=args.checkpoint
    )
    model.set_trainable_component("weaver")
    references = reference_logprobs(model, tokenizer, rows, config)
    dpo_cfg = config.get("dpo", {})
    beta = float(dpo_cfg.get("beta", 0.1))
    epochs = int(dpo_cfg.get("epochs", 1))
    accumulation = int(config["training"].get("gradient_accumulation_steps", 16))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(dpo_cfg.get("learning_rate", 5e-6)),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    optimizer.zero_grad(set_to_none=True)
    step = 0
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(rows)
        total = 0.0
        for index, row in enumerate(rows):
            goal, goal_mask, prompt, prompt_mask, chosen, chosen_mask, rejected, rejected_mask = tensors(
                tokenizer, row, config
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                woven = model.weave(
                    prompt,
                    prompt_mask,
                    goal_ids=goal,
                    goal_attention_mask=goal_mask,
                    force_probe=int(row["turn_index"]) == 0,
                    # DPO optimizes the Weaver only.  The independently trained
                    # Trigger is deliberately not a preference-learning target.
                    force_invoke=True,
                )
                chosen_lp = model.target_logprobs(
                    prompt, chosen, woven.memory_tokens, prompt_attention_mask=prompt_mask,
                    target_attention_mask=chosen_mask,
                )
                rejected_lp = model.target_logprobs(
                    prompt, rejected, woven.memory_tokens, prompt_attention_mask=prompt_mask,
                    target_attention_mask=rejected_mask,
                )
                ref_chosen, ref_rejected = references[str(row["sample_id"])]
                advantage = (chosen_lp - rejected_lp) - (ref_chosen - ref_rejected)
                loss = -F.logsigmoid(beta * advantage).mean()
            (loss / accumulation).backward()
            total += float(loss.detach())
            boundary = (index + 1) % accumulation == 0 or index + 1 == len(rows)
            if boundary:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"epoch={epoch + 1} step={step} dpo={float(loss):.6f}", flush=True)
        output = args.output_dir / f"epoch-{epoch + 1}"
        save_checkpoint(
            output,
            model,
            {
                "stage": "dpo",
                "epoch": epoch + 1,
                "global_step": step,
                "mean_loss": total / len(rows),
                "beta": beta,
                "reference": "precomputed_from_input_sft_checkpoint_before_updates",
                "config": config,
            },
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "reference_logprobs.json").write_text(
            json.dumps(references, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
