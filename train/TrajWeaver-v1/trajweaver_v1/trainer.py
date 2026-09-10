"""Minimal single-GPU trainers for SFT and memory-conditioned DPO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .checkpoint import save_checkpoint
from .data import goal_messages, read_jsonl, student_messages
from .teacher_adapter import merge_annotations
from .terra import TERRA_PROMPT_VERSION, TERRA_TEACHER_MODEL


def _truncate(ids: Tensor, maximum: int, keep_prefix: int = 128) -> Tensor:
    if ids.shape[-1] <= maximum:
        return ids
    prefix = min(keep_prefix, maximum // 4)
    return torch.cat([ids[:, :prefix], ids[:, -(maximum - prefix) :]], dim=-1)


def tokenize_prompt(tokenizer: Any, sample: dict[str, Any], maximum: int) -> Tensor:
    ids = tokenizer.apply_chat_template(
        student_messages(sample), tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif isinstance(ids, dict):
        ids = ids["input_ids"]
    if not isinstance(ids, Tensor):
        ids = torch.tensor(ids, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
    return _truncate(ids.long(), maximum)


def tokenize_goal(tokenizer: Any, sample: dict[str, Any], maximum: int) -> Tensor:
    ids = tokenizer.apply_chat_template(
        goal_messages(sample), tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif isinstance(ids, dict):
        ids = ids["input_ids"]
    if not isinstance(ids, Tensor):
        ids = torch.tensor(ids, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
    return _truncate(ids.long(), maximum, keep_prefix=64)


def tokenize_target(tokenizer: Any, text: str, maximum: int) -> Tensor:
    ids = tokenizer(str(text), add_special_tokens=False, return_tensors="pt").input_ids.long()
    eos = tokenizer.eos_token_id
    if eos is not None and (ids.numel() == 0 or int(ids[0, -1]) != eos):
        ids = torch.cat([ids, torch.tensor([[eos]], dtype=torch.long)], dim=1)
    return ids[:, :maximum]


class TurnDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        *,
        max_goal_tokens: int,
        max_prompt_tokens: int,
        max_target_tokens: int,
        annotations: str | Path | None = None,
    ) -> None:
        rows = read_jsonl(path)
        if annotations is not None and Path(annotations).exists():
            rows = merge_annotations(rows, read_jsonl(annotations))
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_goal_tokens = max_goal_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.max_target_tokens = max_target_tokens
        # Route/action-only traces are useful historical artifacts but cannot
        # supervise the claimed Belief/Desire/goal-alignment representation or
        # the trigger decision. Count a row as fully annotated only when all
        # four public teacher labels survived the merge.
        required_labels = (
            "belief_id",
            "desire_id",
            "goal_alignment_id",
            "trigger_id",
        )
        self.annotation_count = sum(
            str(row.get("label_source", "")) == "terra_direct"
            and row.get("teacher_model") == TERRA_TEACHER_MODEL
            and row.get("prompt_version") == TERRA_PROMPT_VERSION
            and all(int(row.get(key, -100)) >= 0 for key in required_labels)
            for row in rows
        )

    @property
    def annotation_coverage(self) -> float:
        return self.annotation_count / max(len(self.rows), 1)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "sample_id": row["sample_id"],
            "goal_ids": tokenize_goal(self.tokenizer, row, self.max_goal_tokens)[0],
            "prompt_ids": tokenize_prompt(self.tokenizer, row, self.max_prompt_tokens)[0],
            "target_ids": tokenize_target(self.tokenizer, row["target"], self.max_target_tokens)[0],
            "route_id": int(row["route_id"]),
            "action_id": int(row["action_id"]),
            "belief_id": int(row.get("belief_id", -100)),
            "desire_id": int(row.get("desire_id", -100)),
            "goal_alignment_id": int(row.get("goal_alignment_id", -100)),
            "trigger_id": int(row.get("trigger_id", -100)),
            "force_probe": int(row["turn_index"]) == 0,
        }


class TriggerDataset(Dataset[dict[str, Any]]):
    """Compact public-only rows for supervised SKIP/INVOKE training."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        *,
        max_prompt_tokens: int,
        annotations: str | Path | None = None,
    ) -> None:
        rows = read_jsonl(path)
        if annotations is not None and Path(annotations).exists():
            rows = merge_annotations(rows, read_jsonl(annotations))
        self.rows = [row for row in rows if int(row.get("trigger_id", -100)) >= 0]
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        self.annotation_count = sum(
            str(row.get("label_source", "")) == "terra_direct"
            and row.get("teacher_model") == TERRA_TEACHER_MODEL
            and row.get("prompt_version") == TERRA_PROMPT_VERSION
            for row in self.rows
        )

    @property
    def annotation_coverage(self) -> float:
        return self.annotation_count / max(len(self.rows), 1)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "sample_id": row["sample_id"],
            "prompt_ids": tokenize_prompt(
                self.tokenizer, row, self.max_prompt_tokens
            )[0],
            "trigger_id": int(row["trigger_id"]),
        }


def _left_pad(sequences: Sequence[Tensor], value: int) -> tuple[Tensor, Tensor]:
    maximum = max(item.shape[0] for item in sequences)
    padded = torch.full((len(sequences), maximum), value, dtype=torch.long)
    mask = torch.zeros((len(sequences), maximum), dtype=torch.long)
    for index, item in enumerate(sequences):
        padded[index, -item.shape[0] :] = item
        mask[index, -item.shape[0] :] = 1
    return padded, mask


def _right_pad(sequences: Sequence[Tensor], value: int) -> tuple[Tensor, Tensor]:
    maximum = max(item.shape[0] for item in sequences)
    padded = torch.full((len(sequences), maximum), value, dtype=torch.long)
    mask = torch.zeros((len(sequences), maximum), dtype=torch.long)
    for index, item in enumerate(sequences):
        padded[index, : item.shape[0]] = item
        mask[index, : item.shape[0]] = 1
    return padded, mask


def make_collator(pad_token_id: int):
    def collate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        goal_ids, goal_mask = _left_pad([row["goal_ids"] for row in rows], pad_token_id)
        prompt_ids, prompt_mask = _left_pad([row["prompt_ids"] for row in rows], pad_token_id)
        target_ids, target_mask = _right_pad([row["target_ids"] for row in rows], pad_token_id)
        return {
            "sample_ids": [row["sample_id"] for row in rows],
            "goal_ids": goal_ids,
            "goal_mask": goal_mask,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "route_labels": torch.tensor([row["route_id"] for row in rows]),
            "action_labels": torch.tensor([row["action_id"] for row in rows]),
            "belief_labels": torch.tensor([row["belief_id"] for row in rows]),
            "desire_labels": torch.tensor([row["desire_id"] for row in rows]),
            "goal_alignment_labels": torch.tensor(
                [row["goal_alignment_id"] for row in rows]
            ),
            "trigger_labels": torch.tensor([row["trigger_id"] for row in rows]),
            "force_probe": torch.tensor([row["force_probe"] for row in rows]),
        }

    return collate


def make_trigger_collator(pad_token_id: int):
    def collate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        prompt_ids, prompt_mask = _left_pad(
            [row["prompt_ids"] for row in rows], pad_token_id
        )
        return {
            "sample_ids": [row["sample_id"] for row in rows],
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "trigger_labels": torch.tensor([row["trigger_id"] for row in rows]),
        }

    return collate


@dataclass
class Phase:
    name: str
    epochs: int
    language_weight: float
    route_weight: float
    action_weight: float
    # Absolute checkpoint epoch to use for the first epoch in this run.  This
    # allows an interrupted phase to continue without overwriting an existing
    # checkpoint (e.g. resume trajectory epoch 2 from trajectory-epoch-1).
    start_epoch: int = 1


class SFTTrainer:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: dict[str, Any],
        *,
        output_dir: str | Path,
        initial_global_step: int = 0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = Path(output_dir)
        train_cfg = config["training"]
        model.set_trainable_component("weaver")
        lora_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith("backbone.") and parameter.requires_grad
        ]
        auxiliary_parameters = list(model.auxiliary_parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {"params": lora_parameters, "lr": float(train_cfg.get("lora_lr", 1e-5))},
                {"params": auxiliary_parameters, "lr": float(train_cfg.get("module_lr", 1e-4))},
            ],
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        )
        self.grad_accum = int(train_cfg.get("gradient_accumulation_steps", 16))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.global_step = int(initial_global_step)

    def _loader(self, dataset: Dataset[Any], shuffle: bool) -> DataLoader[Any]:
        cfg = self.config["training"]
        generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
        return DataLoader(
            dataset,
            batch_size=int(cfg.get("micro_batch_size", 1)),
            shuffle=shuffle,
            generator=generator,
            num_workers=int(cfg.get("num_workers", 0)),
            collate_fn=make_collator(self.tokenizer.pad_token_id),
        )

    def _loss(self, batch: dict[str, Any], phase: Phase) -> tuple[Tensor, dict[str, Tensor]]:
        training = self.config["training"]
        alignment = batch["goal_alignment_labels"]
        language_weights = torch.ones_like(alignment, dtype=torch.float)
        known_drift = alignment.ge(2)
        language_weights[known_drift] = float(
            training.get("drift_target_language_weight", 0.0)
        )
        trigger = batch["trigger_labels"]
        force_invoke = torch.where(trigger.ge(0), trigger, torch.ones_like(trigger)).bool()
        return self.model.supervised_loss(
            batch["prompt_ids"],
            batch["target_ids"],
            goal_ids=batch["goal_ids"],
            goal_attention_mask=batch["goal_mask"],
            prompt_attention_mask=batch["prompt_mask"],
            target_attention_mask=batch["target_mask"],
            route_labels=batch["route_labels"],
            action_labels=batch["action_labels"],
            belief_labels=batch["belief_labels"],
            desire_labels=batch["desire_labels"],
            goal_alignment_labels=batch["goal_alignment_labels"],
            language_sample_weights=language_weights,
            force_probe=batch["force_probe"],
            force_invoke=force_invoke,
            language_weight=phase.language_weight,
            route_weight=phase.route_weight,
            action_weight=phase.action_weight,
            belief_weight=float(training.get("belief_weight", 0.2)),
            desire_weight=float(training.get("desire_weight", 0.2)),
            goal_alignment_weight=float(training.get("goal_alignment_weight", 0.3)),
        )

    @torch.no_grad()
    def evaluate(self, dataset: Dataset[Any], phase: Phase, limit: int | None = None) -> float:
        self.model.eval()
        losses: list[float] = []
        for index, batch in enumerate(self._loader(dataset, shuffle=False)):
            if limit is not None and index >= limit:
                break
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                loss, _ = self._loss(batch, phase)
            losses.append(float(loss))
        self.model.train()
        return sum(losses) / max(len(losses), 1)

    def run(self, train_data: Dataset[Any], dev_data: Dataset[Any], phases: Sequence[Phase]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer.zero_grad(set_to_none=True)
        trajectory_checkpoints: list[Path] = []
        for phase in phases:
            if phase.epochs <= 0:
                continue
            for epoch in range(phase.epochs):
                absolute_epoch = phase.start_epoch + epoch
                loader = self._loader(train_data, shuffle=True)
                running = 0.0
                for batch_index, batch in enumerate(loader):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                        loss, parts = self._loss(batch, phase)
                    (loss / self.grad_accum).backward()
                    running += float(loss.detach())
                    boundary = (batch_index + 1) % self.grad_accum == 0 or batch_index + 1 == len(loader)
                    if boundary:
                        torch.nn.utils.clip_grad_norm_(
                            [p for group in self.optimizer.param_groups for p in group["params"]],
                            self.max_grad_norm,
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1
                        if self.global_step % 10 == 0:
                            values = " ".join(f"{name}={float(value):.4f}" for name, value in parts.items())
                            print(
                                f"phase={phase.name} epoch={absolute_epoch} step={self.global_step} {values}",
                                flush=True,
                            )
                dev_loss = self.evaluate(dev_data, phase)
                checkpoint = self.output_dir / f"{phase.name}-epoch-{absolute_epoch}"
                save_checkpoint(
                    checkpoint,
                    self.model,
                    {
                        "phase": phase.name,
                        "epoch": absolute_epoch,
                        "global_step": self.global_step,
                        "train_mean_loss": running / max(len(loader), 1),
                        "dev_loss": dev_loss,
                        "dev_batches": len(self._loader(dev_data, shuffle=False)),
                        "config": self.config,
                    },
                )
                if phase.name == "trajectory":
                    trajectory_checkpoints.append(checkpoint)
                print(f"saved={checkpoint} dev_loss={dev_loss:.6f}", flush=True)
        if trajectory_checkpoints:
            from .checkpoint import select_lowest_checkpoint

            # Include checkpoints from an earlier invocation when this run is
            # resuming a phase.  The configured total is the completeness
            # guard: selection must see all expected trajectory epochs before
            # the pipeline can advance to Trigger/evaluation.
            expected_trajectory_epochs = int(
                self.config.get("training", {}).get("trajectory_epochs", 0)
            ) or None
            selected, metadata = select_lowest_checkpoint(
                self.output_dir,
                phase="trajectory",
                expected_epochs=expected_trajectory_epochs,
            )
            print(
                f"selected_weaver={selected} dev_loss={metadata['selected_dev_loss']:.6f}",
                flush=True,
            )


class TriggerTrainer:
    """Train only the Trigger LoRA and its binary classification head."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: dict[str, Any],
        *,
        output_dir: str | Path,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = Path(output_dir)
        model.set_trainable_component("trigger")
        trigger_cfg = config.get("trigger_training", {})
        parameters = list(model.component_parameters("trigger"))
        if not parameters:
            raise ValueError("No Trigger parameters are trainable")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(trigger_cfg.get("learning_rate", 1e-5)),
            weight_decay=float(trigger_cfg.get("weight_decay", 0.01)),
        )
        self.grad_accum = int(trigger_cfg.get("gradient_accumulation_steps", 16))
        self.max_grad_norm = float(trigger_cfg.get("max_grad_norm", 1.0))
        self.global_step = 0

    def _loader(self, dataset: Dataset[Any], shuffle: bool) -> DataLoader[Any]:
        cfg = self.config.get("trigger_training", {})
        generator = torch.Generator().manual_seed(
            int(self.config["training"].get("seed", 42))
        )
        return DataLoader(
            dataset,
            batch_size=int(cfg.get("micro_batch_size", 1)),
            shuffle=shuffle,
            generator=generator,
            num_workers=int(cfg.get("num_workers", 0)),
            collate_fn=make_trigger_collator(self.tokenizer.pad_token_id),
        )

    @torch.no_grad()
    def evaluate(self, dataset: Dataset[Any]) -> tuple[float, float]:
        self.model.eval()
        losses: list[float] = []
        correct = total = 0
        for batch in self._loader(dataset, shuffle=False):
            loss, logits = self.model.trigger_loss(
                batch["prompt_ids"],
                batch["trigger_labels"],
                prompt_attention_mask=batch["prompt_mask"],
            )
            losses.append(float(loss))
            predictions = logits.argmax(dim=-1).cpu()
            labels = batch["trigger_labels"]
            correct += int(predictions.eq(labels).sum())
            total += int(labels.numel())
        self.model.train()
        return sum(losses) / max(len(losses), 1), correct / max(total, 1)

    def run(self, train_data: Dataset[Any], dev_data: Dataset[Any]) -> None:
        cfg = self.config.get("trigger_training", {})
        epochs = int(cfg.get("epochs", 2))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(epochs):
            loader = self._loader(train_data, shuffle=True)
            running = 0.0
            for batch_index, batch in enumerate(loader):
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
                ):
                    loss, _ = self.model.trigger_loss(
                        batch["prompt_ids"],
                        batch["trigger_labels"],
                        prompt_attention_mask=batch["prompt_mask"],
                    )
                (loss / self.grad_accum).backward()
                running += float(loss.detach())
                boundary = (batch_index + 1) % self.grad_accum == 0 or (
                    batch_index + 1 == len(loader)
                )
                if boundary:
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for group in self.optimizer.param_groups
                            for parameter in group["params"]
                        ],
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
            dev_loss, dev_accuracy = self.evaluate(dev_data)
            checkpoint = self.output_dir / f"trigger-epoch-{epoch + 1}"
            save_checkpoint(
                checkpoint,
                self.model,
                {
                    "phase": "trigger_sft",
                    "epoch": epoch + 1,
                    "global_step": self.global_step,
                    "train_mean_loss": running / max(len(loader), 1),
                    "dev_loss": dev_loss,
                    "dev_accuracy": dev_accuracy,
                    "config": self.config,
                },
            )
            print(
                f"saved={checkpoint} dev_loss={dev_loss:.6f} "
                f"dev_accuracy={dev_accuracy:.4f}",
                flush=True,
            )
        from .checkpoint import select_lowest_trigger_checkpoint

        selected, metadata = select_lowest_trigger_checkpoint(
            self.output_dir, expected_epochs=epochs
        )
        print(
            f"selected_trigger={selected} dev_loss={metadata['selected_dev_loss']:.6f}",
            flush=True,
        )
