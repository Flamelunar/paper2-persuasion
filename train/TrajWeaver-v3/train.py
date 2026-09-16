"""Train Weaver from existing responses, then a small Trigger from paired NLL."""

import argparse
import json
import math
import random
from pathlib import Path

from trajweaver_v3.data import assert_public, encode, file_digest, read_jsonl, write_json

HERE = Path(__file__).resolve().parent


def subset(rows, count, seed):
    ids = sorted({row["scenario_id"] for row in rows})
    random.Random(seed).shuffle(ids)
    selected = set(ids[:count] if count > 0 else ids)
    return [row for row in rows if row["scenario_id"] in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sft", "trigger"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-scenarios", type=int, default=0)
    parser.add_argument("--dev-scenarios", type=int, default=0)
    args = parser.parse_args()
    import torch
    from torch.nn import functional as F
    from trajweaver_v3.model import load_config, load_model, save_checkpoint

    if min(args.train_scenarios, args.dev_scenarios) < 0:
        parser.error("Scenario counts must be nonnegative; 0 uses all")
    config = load_config(args.config)
    cfg = config["training"][args.stage]
    seed = config["training"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Training output must be empty; existing checkpoints are not overwritten")
    if args.stage == "trigger" and args.checkpoint is None:
        parser.error("Trigger requires --checkpoint from SFT")
    if args.stage == "sft" and args.checkpoint is not None:
        parser.error("SFT starts a new v3 run; optimizer resume is not implemented")
    data_dir = args.data_dir or HERE / "data" / ("sft" if args.stage == "sft" else "trigger")
    source = {split: read_jsonl(data_dir / f"{split}.jsonl") for split in ("train", "dev")}
    for split, rows in source.items():
        for row in rows:
            assert_public(row)
            if row["split"] != split:
                raise ValueError(f"Wrong split in {split} data")
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate sample IDs in {split}")
    if {r["scenario_id"] for r in source["train"]} & {r["scenario_id"] for r in source["dev"]}:
        raise ValueError("Train/dev scenario leakage")
    provenance = None
    if args.stage == "trigger":
        provenance = json.loads((data_dir / "manifest.json").read_text())
        if provenance["checkpoint_sha256"] != file_digest(args.checkpoint / "trajweaver.pt"):
            raise ValueError("Utility labels were scored with another Weaver checkpoint")
        if provenance["config"] != config:
            raise ValueError("Utility labels/config mismatch")
        for rows in source.values():
            if any(row.get("label_source") != "paired_reference_nll" for row in rows):
                raise ValueError("Trigger requires measured paired-reference labels")
        source = {split: [row for row in rows if row["use_for_training"]]
                  for split, rows in source.items()}
    selected = {"train": subset(source["train"], args.train_scenarios, seed),
                "dev": subset(source["dev"], args.dev_scenarios, seed + 1)}
    if not selected["train"] or not selected["dev"]:
        raise ValueError("Empty usable train/dev data; inspect utility summary or prepare SFT data")
    model, tokenizer, metadata = load_model(config, args.checkpoint)
    if args.stage == "trigger" and metadata.get("stage") != "sft":
        raise ValueError("Trigger training requires an SFT checkpoint")
    model.set_component("weaver" if args.stage == "sft" else "trigger")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], weight_decay=0.01)
    accumulation, epochs = int(cfg["gradient_accumulation"]), int(cfg["epochs"])
    if min(accumulation, epochs) < 1:
        raise ValueError("Epochs and accumulation must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "run.json", {"config": config, "stage": args.stage,
        "input_sha256": {split: file_digest(data_dir / f"{split}.jsonl") for split in source},
        "sample_ids": {split: [row["sample_id"] for row in rows] for split, rows in selected.items()},
        "utility_provenance": provenance})
    # Tokenize once, failing on overlength examples instead of changing the target.
    encoded = {split: [(row, encode(tokenizer, row, config["data"])) for row in rows]
               for split, rows in selected.items()}

    def loss_for(item):
        row, (prompt, goal, target) = item
        if args.stage == "sft":
            return model.sft_loss(prompt, goal, target, cfg["goal_only_weight"]), None
        logits = model.trigger_logits(model.hook(prompt))
        target_label = torch.tensor([row["trigger_id"]], device=model.device)
        return F.cross_entropy(logits, target_label), int(logits.argmax(-1).item())

    best = (math.inf, math.inf)
    for epoch in range(1, epochs + 1):
        model.train()
        # Frozen hook features should not depend on train/eval dropout.
        model.backbone.eval()
        order = list(encoded["train"])
        random.Random(seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        train_total = 0.0
        for index, item in enumerate(order):
            loss, _ = loss_for(item)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            group_start = (index // accumulation) * accumulation
            group_size = min(accumulation, len(order) - group_start)
            (loss / group_size).backward()
            train_total += float(loss.detach())
            if (index + 1) % accumulation == 0 or index + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if (index + 1) % 50 == 0:
                print(f"epoch={epoch} step={index + 1}/{len(order)} loss={train_total/(index+1):.5f}", flush=True)
        model.eval()
        dev_total, correct, regret = 0.0, 0, 0.0
        with torch.no_grad():
            for item in encoded["dev"]:
                loss, prediction = loss_for(item)
                dev_total += float(loss)
                if prediction is not None:
                    correct += int(prediction == item[0]["trigger_id"])
                    net = item[0]["net_gain"]
                    regret += max(0.0, net) - prediction * net
        dev_loss = dev_total / len(encoded["dev"])
        if not math.isfinite(dev_loss):
            raise ValueError("Non-finite dev loss")
        report = {"stage": args.stage, "epoch": epoch, "train_loss": train_total/len(order),
                  "dev_loss": dev_loss, "trigger_trained": args.stage == "trigger",
                  "utility_provenance": provenance}
        if args.stage == "trigger":
            report.update(dev_accuracy=correct/len(encoded["dev"]),
                          dev_utility_regret=regret/len(encoded["dev"]),
                          dev_always_skip_regret=sum(max(0.0, item[0]["net_gain"])
                              for item in encoded["dev"])/len(encoded["dev"]),
                          dev_always_invoke_regret=sum(max(0.0, -item[0]["net_gain"])
                              for item in encoded["dev"])/len(encoded["dev"]))
        selection = ((report["dev_utility_regret"], dev_loss) if args.stage == "trigger"
                     else (dev_loss, dev_loss))
        report["selection_metric"] = "dev_utility_regret_then_loss" if args.stage == "trigger" else "dev_loss"
        destination = args.output / f"{args.stage}-epoch-{epoch}"
        save_checkpoint(destination, model, config, report)
        if selection < best:
            best = selection
            write_json(args.output / "best.json", {"checkpoint": str(destination.resolve()), **report})
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
