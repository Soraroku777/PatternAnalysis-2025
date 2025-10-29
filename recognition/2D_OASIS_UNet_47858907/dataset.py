from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# Directory mapping for each dataset split inside /home/groups/comp3710/OASIS
SPLIT_TO_FOLDERS: Dict[str, Dict[str, str]] = {
    "train": {"images": "keras_png_slices_train", "masks": "keras_png_slices_seg_train"},
    "val": {"images": "keras_png_slices_validate", "masks": "keras_png_slices_seg_validate"},
    "validation": {"images": "keras_png_slices_validate", "masks": "keras_png_slices_seg_validate"},
    "test": {"images": "keras_png_slices_test", "masks": "keras_png_slices_seg_test"},
}

# Default location of the dataset on the shared filesystem.
DEFAULT_DATA_ROOT = Path("/home/groups/comp3710/OASIS")


def _load_image(path: Path) -> torch.Tensor:
    """Load a grayscale PNG image and normalise it to [0, 1]."""
    image = Image.open(path).convert("F")
    array = np.asarray(image, dtype=np.float32)
    array = np.nan_to_num(array)
    if array.max() > array.min():
        array = (array - array.min()) / (array.max() - array.min())
    array = array[None, ...]  # add channel dimension
    return torch.from_numpy(array)


def _load_mask(path: Path, num_classes: Optional[int] = None) -> torch.Tensor:
    """Load a segmentation mask and map values to a contiguous label set."""
    mask = Image.open(path)
    array = np.asarray(mask, dtype=np.int64)
    unique_values = np.unique(array)

    # Reindex arbitrary label values (e.g. {0, 85, 170, 255}) to sequential ids.
    value_to_index = {value: idx for idx, value in enumerate(sorted(unique_values))}
    array = np.vectorize(value_to_index.get)(array).astype(np.int64)

    if num_classes is not None and len(value_to_index) > num_classes:
        raise ValueError(
            f"Found {len(value_to_index)} unique labels in mask '{path.name}' "
            f"but num_classes={num_classes}. Update num_classes to match the dataset."
        )

    return torch.from_numpy(array)


def _default_augmentations(image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply simple random flips for data augmentation."""
    if torch.rand(1).item() > 0.5:
        image = torch.flip(image, dims=[2])
        mask = torch.flip(mask, dims=[1])
    if torch.rand(1).item() > 0.5:
        image = torch.flip(image, dims=[1])
        mask = torch.flip(mask, dims=[0])
    return image, mask