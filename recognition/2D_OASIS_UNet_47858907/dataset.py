"""
Dataset utilities for OASIS brain segmentation data.

This module provides PyTorch Dataset and DataLoader implementations for loading
and preprocessing OASIS brain MRI slices with corresponding segmentation masks.
"""

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
    """
    Load a greyscale PNG image and normalise it to [0, 1].
    
    Args:
        path: Path to the PNG image file
        
    Returns:
        Tensor with shape (1, H, W) containing normalised pixel values
    """
    image = Image.open(path).convert("F")  # Convert to 32-bit greyscale
    array = np.asarray(image, dtype=np.float32)
    array = np.nan_to_num(array)  # Replace any NaN values with 0
    
    # Min-max normalisation to [0, 1] range
    if array.max() > array.min():
        array = (array - array.min()) / (array.max() - array.min())
    
    array = array[None, ...]  # Add channel dimension: (H, W) -> (1, H, W)
    return torch.from_numpy(array)


def _load_mask(path: Path, num_classes: Optional[int] = None) -> torch.Tensor:
    """
    Load a segmentation mask and map values to a contiguous label set.
    
    Converts arbitrary label values (e.g., {0, 85, 170, 255}) to sequential 
    class indices (e.g., {0, 1, 2, 3}) for compatibility with PyTorch.
    
    Args:
        path: Path to the mask PNG file
        num_classes: Expected number of classes for validation
        
    Returns:
        Tensor with shape (H, W) containing sequential class labels
        
    Raises:
        ValueError: If mask contains more classes than expected
    """
    mask = Image.open(path)
    array = np.asarray(mask, dtype=np.int64)
    unique_values = np.unique(array)

    # Reindex arbitrary label values to sequential indices
    value_to_index = {value: idx for idx, value in enumerate(sorted(unique_values))}
    array = np.vectorize(value_to_index.get)(array).astype(np.int64)

    # Validate number of classes matches expectation
    if num_classes is not None and len(value_to_index) > num_classes:
        raise ValueError(
            f"Found {len(value_to_index)} unique labels in mask '{path.name}' "
            f"but num_classes={num_classes}. Update num_classes to match the dataset."
        )

    return torch.from_numpy(array)


def _default_augmentations(image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply simple random flips for data augmentation.
    
    Randomly applies horizontal and/or vertical flips to both image and mask
    to increase training data diversity whilst preserving anatomical structure.
    
    Args:
        image: Input image tensor with shape (C, H, W)
        mask: Corresponding mask tensor with shape (H, W)
        
    Returns:
        Tuple of (augmented_image, augmented_mask)
    """
    # Random horizontal flip (50% chance)
    if torch.rand(1).item() > 0.5:
        image = torch.flip(image, dims=[2])  # Flip width dimension
        mask = torch.flip(mask, dims=[1])    # Flip width dimension
    
    # Random vertical flip (50% chance)
    if torch.rand(1).item() > 0.5:
        image = torch.flip(image, dims=[1])  # Flip height dimension
        mask = torch.flip(mask, dims=[0])    # Flip height dimension
    
    return image, mask


class BrainSegmentationDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    PyTorch Dataset for OASIS brain segmentation PNG slices.

    Loads brain MRI slices and corresponding segmentation masks from the
    OASIS dataset, applying preprocessing and optional augmentations.

    Expected directory layout under data_root:
        keras_png_slices_train/
        keras_png_slices_seg_train/
        keras_png_slices_validate/
        keras_png_slices_seg_validate/
        keras_png_slices_test/
        keras_png_slices_seg_test/
        
    Args:
        split: Dataset split ('train', 'val', 'validation', or 'test')
        data_root: Root directory containing the OASIS dataset
        num_classes: Number of segmentation classes
        image_loader: Function to load image files
        mask_loader: Function to load mask files
        augmentations: Optional augmentation function for training
        transform: Optional transforms applied to images
        target_transform: Optional transforms applied to masks
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

        # Resolve image and mask directories for this split
        folders = SPLIT_TO_FOLDERS[split_key]
        image_dir = self.data_root / folders["images"]
        mask_dir = self.data_root / folders["masks"]

        # Validate directories exist
        if not image_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(
                f"Missing expected directories for split '{split}':\n"
                f" - images -> {image_dir}\n"
                f" - masks  -> {mask_dir}\n"
                "Ensure the OASIS dataset is available at /home/groups/comp3710/OASIS."
            )

        # Collect all PNG files and validate dataset integrity
        self.image_paths = sorted(image_dir.glob("*.png"))
        self.mask_paths = sorted(mask_dir.glob("*.png"))
        if not self.image_paths or not self.mask_paths:
            raise RuntimeError(f"No PNG files found in {image_dir} or {mask_dir}.")
        if len(self.image_paths) != len(self.mask_paths):
            raise RuntimeError("Number of images and masks does not match for the selected split.")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess a single sample.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary containing:
                - 'image': Preprocessed image tensor (C, H, W)
                - 'mask': One-hot encoded mask tensor (num_classes, H, W)
                - 'id': Sample identifier (filename stem)
        """
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # Load raw image and mask
        image = self.image_loader(image_path)
        mask = self.mask_loader(mask_path, self.num_classes)

        # Apply augmentations if specified (typically for training)
        if self.augmentations is not None:
            image, mask = self.augmentations(image, mask)

        # Apply image transforms (e.g., normalisation)
        if self.transform is not None:
            image = self.transform(image)

        # Apply mask transforms or convert to one-hot encoding
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        else:
            # Convert class indices to one-hot encoding: (H, W) -> (num_classes, H, W)
            mask = torch.nn.functional.one_hot(mask.long(), num_classes=self.num_classes).permute(2, 0, 1).float()

        return {"image": image, "mask": mask, "id": image_path.stem}
    

def create_dataloaders(
    *,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    num_classes: int,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    augment: bool = True,
) -> Dict[str, DataLoader]:
    """
    Factory function that creates PyTorch DataLoaders for all dataset splits.
    
    Sets up consistent preprocessing and creates train/validation/test loaders
    with appropriate configurations for each split.
    
    Args:
        data_root: Root directory of the OASIS dataset
        num_classes: Number of segmentation classes
        batch_size: Samples per batch
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        augment: Whether to apply data augmentations to training set
        
    Returns:
        Dictionary with keys 'train', 'val', 'test' containing DataLoaders
    """
    # Standard normalisation to [-1, 1] range
    normalization = T.Normalize(mean=(0.5,), std=(0.5,))

    def mask_to_one_hot(mask: torch.Tensor) -> torch.Tensor:
        """Convert mask to one-hot encoding if needed."""
        if mask.ndimension() == 2:
            mask = torch.nn.functional.one_hot(mask.long(), num_classes=num_classes).permute(2, 0, 1)
        return mask.float()

    # Common dataset configuration
    dataset_kwargs = dict(
        data_root=data_root,
        num_classes=num_classes,
        transform=normalization,
        target_transform=mask_to_one_hot,
    )

    # Create datasets for each split
    train_dataset = BrainSegmentationDataset(
        split="train",
        augmentations=_default_augmentations if augment else None,  # Augment training data
        **dataset_kwargs,
    )
    val_dataset = BrainSegmentationDataset(split="val", augmentations=None, **dataset_kwargs)  # No augmentation
    test_dataset = BrainSegmentationDataset(split="test", augmentations=None, **dataset_kwargs)  # No augmentation

    # Create DataLoaders with appropriate settings
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,        # Shuffle training data for better generalisation
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,       # Keep validation order consistent
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,       # Keep test order consistent
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    return loaders


__all__ = ["BrainSegmentationDataset", "create_dataloaders", "DEFAULT_DATA_ROOT", "SPLIT_TO_FOLDERS"]