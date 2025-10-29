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