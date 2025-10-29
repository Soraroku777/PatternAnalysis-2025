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


def plot_curves(history: Dict[str, List[float]], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_dice"], label="Train")
    plt.plot(epochs, history["val_dice"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Dice")
    plt.title("Dice Curves")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_metrics(history: Dict[str, List[float]], per_class_names: List[str], test_dice: torch.Tensor) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {key: [float(v) for v in values] for key, values in history.items()}
    data["test_dice"] = {name: float(score) for name, score in zip(per_class_names, test_dice)}
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    ensure_dataset_structure(DATA_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders = create_dataloaders(
        data_root=DATA_ROOT,
        num_classes=MODEL_CONFIG["num_classes"],
        batch_size=TRAINING_CONFIG["batch_size"],
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=TRAINING_CONFIG["pin_memory"] and device.type == "cuda",
        augment=True,
    )

    model = UNet(**MODEL_CONFIG).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"], weight_decay=TRAINING_CONFIG["weight_decay"])
    use_mixed_precision = TRAINING_CONFIG["mixed_precision"] and device.type == "cuda"
    scaler = amp.GradScaler(device="cuda", enabled=use_mixed_precision) if use_mixed_precision else None

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

    for epoch in range(1, TRAINING_CONFIG["epochs"] + 1):
        train_loss, train_dice = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            device,
            use_mixed_precision,
        )
        val_loss, val_dice = evaluate(model, loaders["val"], device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(float(train_dice.mean()))
        history["val_dice"].append(float(val_dice.mean()))

        dice_details = ", ".join(f"{score:.3f}" for score in val_dice)
        print(
            f"Epoch {epoch:03d}/{TRAINING_CONFIG['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Dice Mean: {val_dice.mean():.4f} | per-class: [{dice_details}]"
        )

    print("Training complete. Evaluating on test set...")
    test_loss, test_dice = evaluate(model, loaders["test"], device)
    print(f"Test Loss: {test_loss:.4f}")
    for idx, score in enumerate(test_dice):
        print(f" - Class {idx}: Dice = {score:.4f}")

    min_dice = float(test_dice.min())
    if min_dice < TRAINING_CONFIG["dice_threshold"]:
        raise RuntimeError(
            f"Dice similarity requirement not met. Minimum per-class Dice {min_dice:.3f} "
            f"< {TRAINING_CONFIG['dice_threshold']:.2f}. Consider further training or tuning."
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": MODEL_CONFIG,
            "training_config": TRAINING_CONFIG,
            "test_dice": test_dice.tolist(),
        },
        CHECKPOINT_PATH,
    )
    print(f"Model checkpoint saved to {CHECKPOINT_PATH.resolve()}")

    plot_curves(history, CURVES_PATH)
    print(f"Training curves saved to {CURVES_PATH.resolve()}")

    class_names = [f"class_{i}" for i in range(MODEL_CONFIG["num_classes"])]
    save_metrics(history, class_names, test_dice)
    print(f"Metrics saved to {METRICS_PATH.resolve()}")


if __name__ == "__main__":
    main()
