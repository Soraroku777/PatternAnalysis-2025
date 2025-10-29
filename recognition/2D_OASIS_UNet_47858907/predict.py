from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from dataset import BrainSegmentationDataset, DEFAULT_DATA_ROOT  # noqa: E402
from modules import UNet  # noqa: E402

CHECKPOINT_PATH = Path("checkpoints/unet_oasis.pth")
OUTPUT_DIR = Path("outputs")
PREDICTION_FIGURE = OUTPUT_DIR / "prediction_example.png"
EXAMPLE_INDEX = 0


def load_model(checkpoint_path: Path, device: torch.device) -> UNet:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint at {checkpoint_path}. Train the model with train.py before running predictions."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("model_config", {})
    required_keys = {"in_channels", "num_classes"}
    if not required_keys.issubset(config):
        raise KeyError(f"Checkpoint missing configuration keys: {required_keys - set(config)}")

    model = UNet(**config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    model.num_classes = config["num_classes"]  # type: ignore[attr-defined]
    return model


@torch.no_grad()
def compute_dice(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits).clamp(min=eps, max=1 - eps)
    targets = targets.float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    union = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return dice


@torch.no_grad()
def evaluate_model(model: UNet, dataset: BrainSegmentationDataset, device: torch.device) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    dice_scores = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        dice = compute_dice(logits, masks)
        dice_scores.append(dice.cpu())
    return torch.stack(dice_scores).mean(dim=0)


@torch.no_grad()
def infer_example(model: UNet, dataset: BrainSegmentationDataset, device: torch.device, index: int) -> dict:
    sample = dataset[index]
    image = sample["image"].unsqueeze(0).to(device)
    logits = model(image)
    probs = torch.sigmoid(logits)[0].cpu()
    predicted = torch.argmax(probs, dim=0)

    mask_tensor = sample["mask"]
    if mask_tensor.ndimension() == 3:
        ground_truth = torch.argmax(mask_tensor, dim=0)
    else:
        ground_truth = mask_tensor

    return {
        "id": sample["id"],
        "image": sample["image"][0].cpu(),
        "prediction": predicted,
        "ground_truth": ground_truth,
    }


def plot_example(example: dict, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(example["image"], cmap="gray")
    axes[0].set_title("Input Slice")
    axes[1].imshow(example["ground_truth"], cmap="tab20")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(example["prediction"], cmap="tab20")
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Example ID: {example['id']}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved prediction visualisation to {save_path.resolve()}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(CHECKPOINT_PATH, device)

    normalization = T.Normalize(mean=(0.5,), std=(0.5,))

    test_dataset = BrainSegmentationDataset(
        split="test",
        data_root=DEFAULT_DATA_ROOT,
        num_classes=model.num_classes,  # type: ignore[arg-type]
        augmentations=None,
        transform=normalization,
        target_transform=None,
    )

    dice_scores = evaluate_model(model, test_dataset, device)
    print("Per-class Dice scores on the test set:")
    for idx, score in enumerate(dice_scores):
        print(f" - Class {idx}: {score:.4f}")

    example = infer_example(model, test_dataset, device, EXAMPLE_INDEX)
    plot_example(example, PREDICTION_FIGURE)


if __name__ == "__main__":
    main()