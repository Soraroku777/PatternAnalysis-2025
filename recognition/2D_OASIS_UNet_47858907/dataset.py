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

class BrainSegmentationDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    Dataset wrapper for OASIS PNG slices.

    Expects the following directory layout under the chosen data root:

    data_root/
        keras_png_slices_train/
        keras_png_slices_seg_train/
        keras_png_slices_validate/
        keras_png_slices_seg_validate/
        keras_png_slices_test/
        keras_png_slices_seg_test/
    """

    def __init__(
        self,
        split: str,
        *,
        data_root: Path | str = DEFAULT_DATA_ROOT,
        num_classes: int,
        image_loader: Callable[[Path], torch.Tensor] = _load_image,
        mask_loader: Callable[[Path], torch.Tensor] = _load_mask,
        augmentations: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        split_key = split.lower()
        if split_key not in SPLIT_TO_FOLDERS:
            valid = ", ".join(sorted(SPLIT_TO_FOLDERS))
            raise ValueError(f"Unknown split '{split}'. Choose from: {valid}.")

        self.data_root = Path(data_root)
        self.split = split_key
        self.num_classes = num_classes
        self.image_loader = image_loader
        self.mask_loader = mask_loader
        self.augmentations = augmentations
        self.transform = transform
        self.target_transform = target_transform

        folders = SPLIT_TO_FOLDERS[split_key]
        image_dir = self.data_root / folders["images"]
        mask_dir = self.data_root / folders["masks"]

        if not image_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(
                f"Missing expected directories for split '{split}':\n"
                f" - images -> {image_dir}\n"
                f" - masks  -> {mask_dir}\n"
                "Ensure the OASIS dataset is available at /home/groups/comp3710/OASIS."
            )

        self.image_paths = sorted(image_dir.glob("*.png"))
        self.mask_paths = sorted(mask_dir.glob("*.png"))
        if not self.image_paths or not self.mask_paths:
            raise RuntimeError(f"No PNG files found in {image_dir} or {mask_dir}.")
        if len(self.image_paths) != len(self.mask_paths):
            raise RuntimeError("Number of images and masks does not match for the selected split.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = self.image_loader(image_path)
        mask = self.mask_loader(mask_path, self.num_classes)

        if self.augmentations is not None:
            image, mask = self.augmentations(image, mask)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            mask = self.target_transform(mask)
        else:
            mask = torch.nn.functional.one_hot(mask.long(), num_classes=self.num_classes).permute(2, 0, 1).float()

        return {"image": image, "mask": mask, "id": image_path.stem}