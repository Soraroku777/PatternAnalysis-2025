from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch import amp  # noqa: E402

from dataset import SPLIT_TO_FOLDERS, create_dataloaders  # noqa: E402
from modules import UNet  # noqa: E402

# -----------------------------------------------------------------------------#
# Configuration (edit values directly; argparse is not used as per AGENT.md)
# -----------------------------------------------------------------------------#

DATA_ROOT = Path("/home/groups/comp3710/OASIS")
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("outputs")

MODEL_CONFIG = {
    "in_channels": 1,
    "num_classes": 4,
    "base_channels": 32,
    "depth": 4,
    "dropout": 0.1,
    "use_bilinear": False,
}

TRAINING_CONFIG = {
    "batch_size": 8,
    "epochs": 80,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "num_workers": 4,
    "pin_memory": True,
    "mixed_precision": True,
    "dice_threshold": 0.90,
}

CHECKPOINT_PATH = CHECKPOINT_DIR / "unet_oasis.pth"
CURVES_PATH = OUTPUT_DIR / "training_curves.png"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

# -----------------------------------------------------------------------------#


def ensure_dataset_structure(data_root: Path) -> None:
    missing: List[Path] = []
    for split in ("train", "val", "test"):
        folders = SPLIT_TO_FOLDERS[split]
        image_dir = data_root / folders["images"]
        mask_dir = data_root / folders["masks"]
        if not image_dir.exists():
            missing.append(image_dir)
        if not mask_dir.exists():
            missing.append(mask_dir)
    if missing:
        formatted = "\n".join(f" - {p}" for p in missing)
        raise FileNotFoundError(
            "Expected the OASIS dataset to follow the structure:\n"
            "/home/groups/comp3710/OASIS/\n"
            "  keras_png_slices_train             keras_png_slices_seg_train\n"
            "  keras_png_slices_validate          keras_png_slices_seg_validate\n"
            "  keras_png_slices_test              keras_png_slices_seg_test\n"
            "Missing paths:\n"
            f"{formatted}"
        )


def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-class and mean Dice scores.
    Both logits and targets are expected to have shape (N, C, H, W).
    """
    probs = torch.sigmoid(logits)
    probs = probs.clamp(min=eps, max=1 - eps)
    targets = targets.float()

    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    union = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    mean_dice = dice.mean()
    return dice, mean_dice


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if targets.ndimension() == 3:
        targets = torch.nn.functional.one_hot(targets.long(), num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    cardinality = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2 * intersection + eps) / (cardinality + eps)
    return 1 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets.float())
    dsc = dice_loss(logits, targets)
    return bce + dsc


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler | None,
    device: torch.device,
    use_mixed_precision: bool,
) -> Tuple[float, torch.Tensor]:
    model.train()
    running_loss = 0.0
    collected_dice: List[torch.Tensor] = []

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(device_type=device.type, enabled=use_mixed_precision):
            logits = model(images)
            loss = segmentation_loss(logits, masks)

        if scaler is not None and use_mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        with torch.no_grad():
            per_class, _ = dice_coefficient(logits.detach(), masks)
            collected_dice.append(per_class.cpu())

    epoch_loss = running_loss / len(loader.dataset)
    mean_dice = torch.stack(collected_dice).mean(dim=0)
    return epoch_loss, mean_dice


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> Tuple[float, torch.Tensor]:
    model.eval()
    running_loss = 0.0
    collected_dice: List[torch.Tensor] = []

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = segmentation_loss(logits, masks)
        running_loss += loss.item() * images.size(0)

        per_class, _ = dice_coefficient(logits, masks)
        collected_dice.append(per_class.cpu())

    epoch_loss = running_loss / len(loader.dataset)
    mean_dice = torch.stack(collected_dice).mean(dim=0)
    return epoch_loss, mean_dice