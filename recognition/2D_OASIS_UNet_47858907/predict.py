"""
Prediction and visualisation script for trained U-Net brain segmentation model.

This script loads a trained U-Net model and generates predictions on the test set,
computing evaluation metrics and creating visualisation examples.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import torch  
from torch.utils.data import DataLoader  
from torchvision import transforms as T  

from dataset import BrainSegmentationDataset, DEFAULT_DATA_ROOT  
from modules import UNet 

# Configuration paths
CHECKPOINT_PATH = Path("checkpoints/unet_oasis.pth")
OUTPUT_DIR = Path("outputs")
PREDICTION_FIGURE = OUTPUT_DIR / "prediction_example.png"

# Index of the test image visualised in the README (adjust to inspect other cases)
EXAMPLE_INDEX = 0


def load_model(checkpoint_path: Path, device: torch.device) -> UNet:
    """
    Load a trained U-Net model from checkpoint.
    
    Args:
        checkpoint_path: Path to the saved model checkpoint
        device: Device to load the model onto
        
    Returns:
        Loaded U-Net model in evaluation mode
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        KeyError: If checkpoint is missing required configuration
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint at {checkpoint_path}. Train the model with train.py before running predictions."
        )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("model_config", {})
    
    # Validate required configuration keys
    required_keys = {"in_channels", "num_classes"}
    if not required_keys.issubset(config):
        raise KeyError(f"Checkpoint missing configuration keys: {required_keys - set(config)}")

    # Create and load model
    model = UNet(**config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Store number of classes for later use
    model.num_classes = config["num_classes"]  # type: ignore[attr-defined]
    
    return model


@torch.no_grad()
def compute_dice(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute per-class Dice similarity coefficients.
    
    Args:
        logits: Model predictions with shape (N, C, H, W)
        targets: Ground truth masks with shape (N, C, H, W)
        eps: Small epsilon for numerical stability
        
    Returns:
        Per-class Dice scores tensor with shape (C,)
    """
    probs = torch.sigmoid(logits).clamp(min=eps, max=1 - eps)
    targets = targets.float()
    
    # Sum over batch, height, and width dimensions
    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    union = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    
    return dice


@torch.no_grad()
def evaluate_model(model: UNet, dataset: BrainSegmentationDataset, device: torch.device) -> torch.Tensor:
    """
    Evaluate model performance on the entire dataset.
    
    Args:
        model: Trained U-Net model
        dataset: Dataset to evaluate on
        device: Device for computations
        
    Returns:
        Mean per-class Dice scores across all samples
    """
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    dice_scores = []
    
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        
        # Forward pass
        logits = model(images)
        dice = compute_dice(logits, masks)
        dice_scores.append(dice.cpu())
    
    # Calculate mean Dice scores across all batches
    return torch.stack(dice_scores).mean(dim=0)


@torch.no_grad()
def infer_example(model: UNet, dataset: BrainSegmentationDataset, device: torch.device, index: int) -> dict:
    """
    Generate prediction for a specific sample and prepare for visualisation.
    
    Args:
        model: Trained model
        dataset: Dataset containing the sample
        device: Device for computations
        index: Index of the sample to predict
        
    Returns:
        Dictionary containing image, prediction, ground truth, and sample ID
    """
    sample = dataset[index]
    
    # Forward pass
    image = sample["image"].unsqueeze(0).to(device)  # Add batch dimension
    logits = model(image)
    probs = torch.sigmoid(logits)[0].cpu()           # Remove batch dimension
    predicted = torch.argmax(probs, dim=0)           # Get class predictions

    # Prepare ground truth for comparison
    mask_tensor = sample["mask"]
    if mask_tensor.ndimension() == 3:  # One-hot encoded
        ground_truth = torch.argmax(mask_tensor, dim=0)
    else:  # Class indices
        ground_truth = mask_tensor

    return {
        "id": sample["id"],
        "image": sample["image"][0].cpu(),      # Remove channel dimension for visualisation
        "prediction": predicted,
        "ground_truth": ground_truth,
    }


def plot_example(example: dict, save_path: Path) -> None:
    """
    Create and save a visualisation comparing input, ground truth, and prediction.
    
    Args:
        example: Dictionary containing image data and predictions
        save_path: Path to save the figure
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Input image (greyscale)
    axes[0].imshow(example["image"], cmap="gray")
    axes[0].set_title("Input Slice")
    
    # Ground truth segmentation (colour-coded)
    axes[1].imshow(example["ground_truth"], cmap="tab20")
    axes[1].set_title("Ground Truth")
    
    # Model prediction (colour-coded)
    axes[2].imshow(example["prediction"], cmap="tab20")
    axes[2].set_title("Prediction")
    
    # Remove axis ticks for cleaner appearance
    for ax in axes:
        ax.axis("off")
    
    # Add sample ID as figure title
    fig.suptitle(f"Example ID: {example['id']}")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)  # Free memory
    
    print(f"Saved prediction visualisation to {save_path.resolve()}")


def main() -> None:
    """Main prediction function that loads model and generates evaluation results."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trained model
    model = load_model(CHECKPOINT_PATH, device)

    # Setup data preprocessing (same as training)
    normalization = T.Normalize(mean=(0.5,), std=(0.5,))

    # Create test dataset
    test_dataset = BrainSegmentationDataset(
        split="test",
        data_root=DEFAULT_DATA_ROOT,
        num_classes=model.num_classes,  # type: ignore[arg-type]
        augmentations=None,             # No augmentations for evaluation
        transform=normalization,
        target_transform=None,
    )

    # Evaluate model performance on entire test set
    dice_scores = evaluate_model(model, test_dataset, device)
    print("Per-class Dice scores on the test set:")
    for idx, score in enumerate(dice_scores):
        print(f" - Class {idx}: {score:.4f}")

    # Generate and visualise an example prediction
    example = infer_example(model, test_dataset, device, EXAMPLE_INDEX)
    plot_example(example, PREDICTION_FIGURE)


if __name__ == "__main__":
    main()
