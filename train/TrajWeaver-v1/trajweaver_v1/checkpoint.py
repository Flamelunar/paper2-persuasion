"""Compact TrajWeaver checkpoint I/O (adapter plus small modules only)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, model: Any, metadata: dict[str, Any]) -> None:
    from peft import get_peft_model_state_dict

    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    auxiliary = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if not name.startswith("backbone.")
    }
    adapters = {
        name: {
            key: tensor.detach().cpu()
            for key, tensor in get_peft_model_state_dict(
                model.backbone, adapter_name=name
            ).items()
        }
        for name in ("weaver", "trigger")
    }
    torch.save(
        {
            "format_version": 2,
            "auxiliary": auxiliary,
            "adapters": adapters,
        },
        output / "trajweaver.pt",
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_checkpoint(path: str | Path, model: Any) -> dict[str, Any]:
    from peft import set_peft_model_state_dict

    source = Path(path)
    payload = torch.load(source / "trajweaver.pt", map_location="cpu", weights_only=False)
    if payload.get("format_version") != 2:
        raise ValueError(
            f"Unsupported checkpoint format in {source}; the 16-token dual-adapter "
            "architecture requires format_version=2."
        )
    incompatible = model.load_state_dict(payload["auxiliary"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"Unexpected auxiliary checkpoint keys: {incompatible.unexpected_keys}"
        )
    missing_auxiliary = [
        key for key in incompatible.missing_keys if not key.startswith("backbone.")
    ]
    if missing_auxiliary:
        raise ValueError(f"Missing auxiliary checkpoint keys: {missing_auxiliary}")
    for name in ("weaver", "trigger"):
        if name not in payload["adapters"]:
            raise ValueError(f"Checkpoint is missing the {name} adapter")
        result = set_peft_model_state_dict(
            model.backbone, payload["adapters"][name], adapter_name=name
        )
        if getattr(result, "unexpected_keys", None):
            raise ValueError(
                f"Unexpected {name} adapter keys: {result.unexpected_keys}"
            )
    metadata_path = source / "metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}


def select_lowest_checkpoint(
    root: str | Path,
    *,
    phase: str = "trajectory",
    expected_epochs: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Select a completed checkpoint using its recorded full-dev loss."""

    directory = Path(root)
    candidates: list[tuple[float, int, Path, dict[str, Any]]] = []
    for checkpoint in sorted(directory.glob(f"{phase}-epoch-*")):
        if not checkpoint.is_dir() or not (checkpoint / "trajweaver.pt").exists():
            continue
        metadata_path = checkpoint / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("phase") != phase:
            continue
        try:
            epoch = int(metadata["epoch"])
            dev_loss = float(metadata["dev_loss"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Incomplete checkpoint metadata: {checkpoint}") from exc
        if not math.isfinite(dev_loss):
            raise ValueError(f"Non-finite dev loss in checkpoint: {checkpoint}")
        candidates.append((dev_loss, epoch, checkpoint, metadata))
    if expected_epochs is not None and len(candidates) != expected_epochs:
        raise ValueError(
            f"Expected {expected_epochs} {phase} checkpoints in {directory}, "
            f"found {len(candidates)}"
        )
    if not candidates:
        raise FileNotFoundError(f"No completed {phase} checkpoints in {directory}")
    selected_loss, selected_epoch, selected_path, selected_metadata = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    pointer = {
        "phase": phase,
        "selected_epoch": selected_epoch,
        "selected_checkpoint": str(selected_path.resolve()),
        "selected_dev_loss": selected_loss,
        "candidate_checkpoints": [
            {
                "epoch": epoch,
                "checkpoint": str(path.resolve()),
                "dev_loss": loss,
            }
            for loss, epoch, path, _ in sorted(candidates, key=lambda item: item[1])
        ],
    }
    (directory / "selected_weaver_checkpoint.json").write_text(
        json.dumps(pointer, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selected_path, {**selected_metadata, **pointer}


def select_lowest_trigger_checkpoint(
    root: str | Path,
    *,
    expected_epochs: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Select the Trigger epoch with the lowest complete dev loss."""

    directory = Path(root)
    candidates: list[tuple[float, int, Path, dict[str, Any]]] = []
    for checkpoint in sorted(directory.glob("trigger-epoch-*")):
        if not checkpoint.is_dir() or not (checkpoint / "trajweaver.pt").exists():
            continue
        metadata_path = checkpoint / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("phase") != "trigger_sft":
            continue
        try:
            epoch = int(metadata["epoch"])
            dev_loss = float(metadata["dev_loss"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Incomplete Trigger checkpoint metadata: {checkpoint}") from exc
        if not math.isfinite(dev_loss):
            raise ValueError(f"Non-finite Trigger dev loss in checkpoint: {checkpoint}")
        candidates.append((dev_loss, epoch, checkpoint, metadata))
    if expected_epochs is not None and len(candidates) != expected_epochs:
        raise ValueError(
            f"Expected {expected_epochs} Trigger checkpoints in {directory}, found {len(candidates)}"
        )
    if not candidates:
        raise FileNotFoundError(f"No completed Trigger checkpoints in {directory}")
    selected_loss, selected_epoch, selected_path, selected_metadata = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    pointer = {
        "phase": "trigger_sft",
        "selected_epoch": selected_epoch,
        "selected_checkpoint": str(selected_path.resolve()),
        "selected_dev_loss": selected_loss,
        "candidate_checkpoints": [
            {
                "epoch": epoch,
                "checkpoint": str(path.resolve()),
                "dev_loss": loss,
            }
            for loss, epoch, path, _ in sorted(candidates, key=lambda item: item[1])
        ],
    }
    (directory / "selected_trigger_checkpoint.json").write_text(
        json.dumps(pointer, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selected_path, {**selected_metadata, **pointer}
