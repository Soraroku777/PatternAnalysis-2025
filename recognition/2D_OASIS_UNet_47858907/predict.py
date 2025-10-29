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
