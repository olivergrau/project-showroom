"""
Data utilities for PneumoniaMNIST medical imaging dataset.

This module provides comprehensive data loading, preprocessing, and exploration
utilities specifically designed for the PneumoniaMNIST binary classification task.
Handles clinical chest X-ray data with medical imaging best practices and
exploratory analysis tools for understanding dataset characteristics.

Key features:
    - Medical imaging data loading with clinical preprocessing
    - Balanced subset creation for rapid experimentation
    - Comprehensive dataset analysis and visualization
    - Clinical context integration for medical imaging applications
"""

import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from medmnist import PneumoniaMNIST

import os
import json
from torch.utils.data import DataLoader

# Simple in-process cache so we do not recompute stats on every call
_XRAY_STATS_CACHE: Dict[Tuple[int, bool], Dict[str, List[float]]] = {}


def _compute_pneumoniamnist_channel_stats(
    size: int,
    as_rgb: bool,
    split: str = "train",
    download: bool = True,
    batch_size: int = 128,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
) -> Tuple[List[float], List[float]]:
    """
    Compute per-channel mean/std for PneumoniaMNIST in [0,1] tensor space (after ToTensor).

    Uses a streaming estimator:
      mean = sum(x) / N
      var  = sum(x^2) / N - mean^2
      std  = sqrt(var)

    Args:
        size: dataset resolution
        as_rgb: True -> 3 channels, False -> 1 channel
        split: typically 'train'
        max_samples: optionally cap samples for faster estimation (None = full split)
    """
    ds = PneumoniaMNIST(
        split=split,
        transform=transforms.ToTensor(),
        download=download,
        size=size,
        as_rgb=as_rgb
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Accumulators in float64 for numerical stability
    sum_c = None
    sumsq_c = None
    count = 0

    for batch_idx, (x, _) in enumerate(dl):
        # x: [B, C, H, W] in float32, range approx [0,1]
        x = x.double()
        b, c, h, w = x.shape

        pixels = b * h * w
        # Sum over batch+spatial dims -> [C]
        batch_sum = x.sum(dim=(0, 2, 3))
        batch_sumsq = (x * x).sum(dim=(0, 2, 3))

        if sum_c is None:
            sum_c = batch_sum
            sumsq_c = batch_sumsq
        else:
            sum_c += batch_sum
            sumsq_c += batch_sumsq

        count += pixels

        if max_samples is not None and (batch_idx + 1) * b >= max_samples:
            break

    if sum_c is None or count == 0:
        raise RuntimeError("Failed to compute dataset statistics (empty dataset or loader).")

    mean = (sum_c / count).tolist()
    var = (sumsq_c / count - (sum_c / count) ** 2).clamp(min=0.0)
    std = torch.sqrt(var).tolist()

    return mean, std


def _get_or_compute_xray_stats(
    size: int,
    as_rgb: bool,
    download: bool = True,
    cache_to_disk: bool = True,
    cache_path: str = "./xray_stats_pneumoniamnist.json",
    max_samples: Optional[int] = None
) -> Tuple[List[float], List[float]]:
    """
    Returns cached stats if present, otherwise computes them on the train split.
    Optionally caches to disk so the next run is instant.
    """
    key = (size, as_rgb)

    # 1) in-memory cache
    if key in _XRAY_STATS_CACHE:
        stats = _XRAY_STATS_CACHE[key]
        return stats["mean"], stats["std"]

    # 2) disk cache
    if cache_to_disk and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            k = f"{size}_{'rgb' if as_rgb else 'gray'}"
            if k in payload:
                mean = payload[k]["mean"]
                std = payload[k]["std"]
                _XRAY_STATS_CACHE[key] = {"mean": mean, "std": std}
                return mean, std
        except Exception:
            # If cache is corrupt, ignore and recompute
            pass

    # 3) compute
    mean, std = _compute_pneumoniamnist_channel_stats(
        size=size,
        as_rgb=as_rgb,
        split="train",
        download=download,
        max_samples=max_samples
    )

    _XRAY_STATS_CACHE[key] = {"mean": mean, "std": std}

    # 4) save to disk
    if cache_to_disk:
        try:
            payload: Dict[str, Any] = {}
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            k = f"{size}_{'rgb' if as_rgb else 'gray'}"
            payload[k] = {"mean": mean, "std": std}
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            # Disk caching failure should not break training
            pass

    return mean, std



# Clinical class mapping for pneumonia detection
PNEUMONIA_CLASSES = {
    0: "Normal",
    1: "Pneumonia"
}




def load_pneumoniamnist(split: str = "train", download: bool = True, 
                          size: int = 28, batch_size: int = 32, 
                          as_rgb: bool = True, subset_size: Optional[int] = None, use_xray_stats: bool = False,
    stats_max_samples: Optional[int] = None) -> DataLoader:
    """
    Load PneumoniaMNIST dataset with clinical preprocessing for binary pneumonia detection.
    
    Creates standardized data loaders with medical imaging preprocessing optimized
    for transfer learning from ImageNet pretrained models. Includes balanced subset
    creation for rapid experimentation and development.
    
    Args:
        split: Dataset split to load ('train', 'val', or 'test')
        download: Whether to download the dataset if not locally available
        size: Target image resolution (28, 64, 128, or 224 pixels)
        batch_size: Batch size for training/inference optimization
        as_rgb: Convert grayscale medical images to RGB for pretrained model compatibility
        subset_size: Optional size limit for rapid experimentation. If provided,
                    creates a balanced subset maintaining class distribution.
        use_xray_stats: Whether to use X-ray specific normalization statistics
        stats_max_samples: Max samples to use when computing X-ray stats (None = all)
        
    Returns:
        Configured DataLoader with medical imaging preprocessing and optimization
    
    Note:
        ImageNet normalization is applied for optimal transfer learning performance
        with pretrained models. RGB conversion ensures compatibility with standard
        computer vision architectures while preserving medical image information.
        
    Example:
        >>> # Load full training set
        >>> train_loader = load_pneumoniamnist('train', size=224, batch_size=16)
        >>> 
        >>> # Load balanced subset for rapid experimentation
        >>> dev_loader = load_pneumoniamnist('train', subset_size=1000, batch_size=32)
        >>> print(f"Development set: {len(dev_loader.dataset)} samples")
    """

    # Choose normalization statistics
    if use_xray_stats:
        mean, std = _get_or_compute_xray_stats(
            size=size,
            as_rgb=as_rgb,
            download=download,
            cache_to_disk=True,
            cache_path="./xray_stats_pneumoniamnist.json",
            max_samples=stats_max_samples
        )
    else:
        # ImageNet normalization for pretrained model compatibility
        mean = [0.485, 0.456, 0.406] if as_rgb else [0.485]
        std  = [0.229, 0.224, 0.225] if as_rgb else [0.229]

    # Define medical imaging preprocessing pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        # ImageNet normalization for optimal pretrained model performance
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Load PneumoniaMNIST with clinical configuration
    dataset = PneumoniaMNIST(
        split=split,
        transform=transform,
        download=download,
        size=size,
        as_rgb=as_rgb  # RGB conversion for pretrained model compatibility
    )
    
    # Create balanced subset if requested for development
    if subset_size is not None:
        dataset = _create_subset(dataset, subset_size)
    
    # Configure DataLoader with medical imaging optimizations
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(split == 'train'),  # Shuffle training data for generalization
        num_workers=4,               # Parallel data loading for performance
        pin_memory=True              # GPU memory optimization for CUDA
    )
    
    return dataloader


def _create_subset(dataset, subset_size: int) -> Subset:
    """
    Create a balanced subset maintaining clinical class distribution.
    
    Ensures representative sampling across normal and pneumonia cases for
    rapid development and experimentation while preserving the underlying
    class distribution critical for medical imaging validation.
    
    Args:
        dataset: Original PneumoniaMNIST dataset
        subset_size: Target number of samples for the subset
        
    Returns:
        Balanced Subset maintaining original class proportions
        
    Note:
        Maintains clinical class balance to ensure development results
        translate to full dataset performance. Critical for medical
        imaging where class imbalance affects diagnostic accuracy.
    """
    total_size = len(dataset)
    
    if subset_size >= total_size:
        warnings.warn(f"Requested subset size ({subset_size}) >= total size ({total_size}), using full dataset")
        return dataset
    
    # Separate indices by clinical class for balanced sampling
    indices = list(range(total_size))
    normal_indices: List[int] = []
    pneumonia_indices: List[int] = []
    
    # Classify samples by medical diagnosis
    for idx in indices:
        _, label = dataset[idx]
        if label.item() == 0:
            normal_indices.append(idx)
        else:
            pneumonia_indices.append(idx)
    
    # Calculate balanced subset composition preserving clinical distribution
    total_normal = len(normal_indices)
    total_pneumonia = len(pneumonia_indices)
    pneumonia_ratio = total_pneumonia / total_size
    
    # Maintain clinical class proportions in subset
    subset_pneumonia = min(int(subset_size * pneumonia_ratio), total_pneumonia)
    subset_normal = min(subset_size - subset_pneumonia, total_normal)
    
    # Random sampling within each clinical class
    random.shuffle(normal_indices)
    random.shuffle(pneumonia_indices)
    
    selected_indices = (pneumonia_indices[:subset_pneumonia] + 
                       normal_indices[:subset_normal])
    random.shuffle(selected_indices)  # Final randomization for training
    
    print(f"Created balanced clinical subset: {len(selected_indices)} samples "
          f"({subset_pneumonia} pneumonia, {subset_normal} normal)")
    
    return Subset(dataset, selected_indices)


def get_dataset_info(use_binary: bool = True) -> Dict[str, Any]:
    """
    Retrieve comprehensive PneumoniaMNIST dataset information and clinical context.
    
    Provides detailed metadata about the medical imaging dataset including
    clinical context, sample counts, and preprocessing specifications for
    research documentation and clinical validation planning.
    
    Args:
        use_binary: Whether to return binary classification information (deprecated parameter)
        
    Returns:
        Dictionary containing comprehensive dataset metadata:
            - Clinical context and medical significance
            - Sample counts and class distribution
            - Technical specifications and processing options
    
    Example:
        >>> info = get_dataset_info()
        >>> print(f"Dataset: {info['name']}")
        >>> print(f"Clinical task: {info['task']}")
        >>> print(f"Total samples: {info['total_samples']:,}")
    """
    return {
        "name": "PneumoniaMNIST",
        "task": "Binary classification (Normal vs Pneumonia)",
        "classes": ["Normal", "Pneumonia"],
        "num_classes": 2,
        "source": "Chest X-ray Images (Pneumonia) from Kaggle",
        "original_size": "Various sizes (medical imaging protocols vary)",
        "processed_sizes": [28, 64, 128, 224],  # Available preprocessing resolutions
        "medical_context": "Pneumonia detection in chest X-rays for clinical decision support",
        "samples": {
            "train": 4708, 
            "val": 524, 
            "test": 624
        },
        "total_samples": 5856,
        "clinical_significance": "Early pneumonia detection critical for patient outcomes",
        "preprocessing_note": "ImageNet normalization applied for transfer learning optimization"
    }


def get_sample_batch(dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract a single batch for testing, profiling, and development purposes.
    
    Provides convenient access to representative data batch for model testing,
    performance profiling, and rapid development iteration.
    
    Args:
        dataloader: Configured DataLoader for batch extraction
        
    Returns:
        Tuple containing:
            - Input tensor batch with shape (batch_size, channels, height, width)
            - Label tensor batch with shape (batch_size,) containing class indices
    
    Example:
        >>> train_loader = load_pneumoniamnist('train', batch_size=8)
        >>> images, labels = get_sample_batch(train_loader)
        >>> print(f"Batch shape: {images.shape}, Labels: {labels.shape}")
        >>> # Use for model testing: output = model(images)
    """
    return next(iter(dataloader))


def explore_dataset_splits(train_loader: DataLoader, val_loader: DataLoader, 
                          test_loader: DataLoader) -> Dict[str, Dict[str, Union[int, float, Dict[str, int]]]]:
    """
    Comprehensive analysis of dataset split sizes and clinical class distribution.
    
    Provides detailed statistical analysis of dataset composition across training,
    validation, and test splits. Essential for understanding data balance and
    ensuring representative clinical evaluation.
    
    Args:
        train_loader: Training data loader for analysis
        val_loader: Validation data loader for analysis
        test_loader: Test data loader for analysis
        
    Returns:
        Dictionary containing detailed split analysis:
            - Sample counts per split
            - Class distribution (normal vs pneumonia)
            - Clinical balance ratios for medical validation
    
    Note:
        Class balance analysis is critical for medical imaging where imbalanced
        datasets can lead to biased models that miss rare but critical cases.
        
    Example:
        >>> train_loader = load_pneumoniamnist('train')
        >>> val_loader = load_pneumoniamnist('val')
        >>> test_loader = load_pneumoniamnist('test')
        >>> stats = explore_dataset_splits(train_loader, val_loader, test_loader)
        >>> print(f"Training pneumonia ratio: {stats['train']['pneumonia_ratio']:.1%}")
    """
    def count_samples_and_classes(loader: DataLoader) -> Dict[str, Union[int, float, Dict[str, int]]]:
        """Analyze sample counts and clinical class distribution for a data loader."""
        total_samples = 0
        normal_count = 0
        pneumonia_count = 0
        
        # Iterate through all batches for complete analysis
        for _, labels in loader:
            total_samples += labels.size(0)
            for label in labels:
                if label.item() == 0:
                    normal_count += 1
                else:
                    pneumonia_count += 1
        
        # Clinical class statistics
        class_stats = {
            "normal": normal_count,
            "pneumonia": pneumonia_count,
            "type": "binary"
        }
        
        return {
            "total": total_samples,
            "class_stats": class_stats,
            "normal": normal_count,
            "pneumonia": pneumonia_count,
            "pneumonia_ratio": pneumonia_count / total_samples if total_samples > 0 else 0
        }
    
    # Comprehensive analysis of each clinical dataset split
    train_stats = count_samples_and_classes(train_loader)
    val_stats = count_samples_and_classes(val_loader)
    test_stats = count_samples_and_classes(test_loader)
    
    return {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats
    }

def _extract_normalize_from_transform(tf) -> Optional[Tuple[List[float], List[float]]]:
    """
    Try to find torchvision.transforms.Normalize(mean, std) inside a transform pipeline.
    Returns (mean, std) if found, else None.
    """
    if tf is None:
        return None

    # Direct Normalize
    if isinstance(tf, transforms.Normalize):
        return list(tf.mean), list(tf.std)

    # Compose([...])
    if isinstance(tf, transforms.Compose):
        for t in tf.transforms:
            res = _extract_normalize_from_transform(t)
            if res is not None:
                return res

    # Unknown structure
    return None


def _denormalize(img: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """
    Reverse torchvision Normalize for a single image tensor [C,H,W].
    """
    device = img.device
    mean_t = torch.tensor(mean, device=device, dtype=img.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=img.dtype).view(-1, 1, 1)
    return img * std_t + mean_t

def visualize_sample_images(dataloader: DataLoader, num_samples: int = 8) -> None:
    """
    Visualize representative medical imaging samples with clinical annotations.

    Denormalizes using the dataset's actual Normalize(mean,std) if present,
    instead of assuming ImageNet stats.
    """
    all_images: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for images, labels in dataloader:
        all_images.extend([img for img in images])
        all_labels.extend([label for label in labels])
        if len(all_images) >= num_samples * 2:
            break

    normal_samples = [(img, label) for img, label in zip(all_images, all_labels) if label.item() == 0]
    pneumonia_samples = [(img, label) for img, label in zip(all_images, all_labels) if label.item() == 1]

    half_samples = num_samples // 2
    selected_samples = normal_samples[:half_samples] + pneumonia_samples[:half_samples]
    random.shuffle(selected_samples)

    cols = 4
    rows = (len(selected_samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))

    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.ravel()

    # ---- NEW: derive stats from the dataset transform ----
    mean_std = None
    ds = getattr(dataloader, "dataset", None)
    if ds is not None:
        tf = getattr(ds, "transform", None)
        mean_std = _extract_normalize_from_transform(tf)

    if mean_std is None:
        # Fallback: do not distort images. Show them "as is" and clamp.
        # (If you really want ImageNet fallback, do it explicitly, but for X-rays
        # it is safer to avoid hard-coded stats.)
        mean = std = None
        print("NOTE: No Normalize(mean,std) found in dataset transform. Displaying without denormalization.")
    else:
        mean, std = mean_std

    normal_count = 0
    pneumonia_count = 0

    for i, (img, label) in enumerate(selected_samples):
        # Denormalize using actual stats if available
        if mean is not None and std is not None:
            img = _denormalize(img, mean, std)

        img = torch.clamp(img, 0, 1)

        # Convert to numpy for display
        if img.shape[0] == 1:
            # grayscale
            img_np = img.squeeze(0).numpy()
            axes[i].imshow(img_np, cmap="gray")
        else:
            # RGB
            img_np = img.permute(1, 2, 0).numpy()
            axes[i].imshow(img_np)

        class_name = PNEUMONIA_CLASSES[label.item()]
        title_color = "red" if label.item() == 1 else "green"

        if label.item() == 0:
            normal_count += 1
        else:
            pneumonia_count += 1

        axes[i].set_title(class_name, color=title_color, fontweight="bold")
        axes[i].axis("off")

    for i in range(len(selected_samples), len(axes)):
        axes[i].axis("off")

    plt.suptitle(
        f'PneumoniaMNIST Clinical Samples ({normal_count} Normal, {pneumonia_count} Pneumonia)',
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    print(f"Displayed {len(selected_samples)} images: {normal_count} normal, {pneumonia_count} pneumonia")


# def visualize_sample_images(dataloader: DataLoader, num_samples: int = 8) -> None:
#     """
#     Visualize representative medical imaging samples with clinical annotations.
    
#     Creates a clinical visualization grid showing normal and pneumonia cases
#     with medical annotations for dataset understanding and clinical validation.
#     Essential for verifying data quality and understanding medical imaging characteristics.
    
#     Args:
#         dataloader: DataLoader containing medical images for visualization
#         num_samples: Number of samples to display (recommended: 4-16 for clarity)
        
#     Returns:
#         None: Displays medical imaging visualization with clinical context
    
#     Note:
#         Visualization includes denormalization for proper medical image display
#         and clinical color coding (green for normal, red for pneumonia) to aid
#         in medical interpretation and validation.
        
#     Example:
#         >>> train_loader = load_pneumoniamnist('train', size=224)
#         >>> visualize_sample_images(train_loader, num_samples=12)
#         >>> # Displays 12 medical images with clinical annotations
#     """
#     # Collect diverse samples for medical visualization
#     all_images: List[torch.Tensor] = []
#     all_labels: List[torch.Tensor] = []
    
#     for images, labels in dataloader:
#         all_images.extend([img for img in images])
#         all_labels.extend([label for label in labels])
        
#         # Ensure sufficient samples for balanced selection
#         if len(all_images) >= num_samples * 2:
#             break
    
#     # Create balanced clinical sample selection
#     normal_samples = [(img, label) for img, label in zip(all_images, all_labels) if label.item() == 0]
#     pneumonia_samples = [(img, label) for img, label in zip(all_images, all_labels) if label.item() == 1]
    
#     # Balanced sampling for clinical representation
#     half_samples = num_samples // 2
#     selected_samples = (normal_samples[:half_samples] + 
#                        pneumonia_samples[:half_samples])
#     random.shuffle(selected_samples)  # Randomize for natural presentation
    
#     # Configure visualization grid for medical display
#     cols = 4
#     rows = (len(selected_samples) + cols - 1) // cols
#     fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    
#     if rows == 1:
#         axes = axes.reshape(1, -1)
#     axes = axes.ravel()
    
#     # Clinical sample tracking
#     normal_count = 0
#     pneumonia_count = 0
    
#     # Display medical images with clinical annotations
#     for i, (img, label) in enumerate(selected_samples):
#         # Denormalize medical images for proper visualization
#         # Reverse ImageNet normalization to display original intensities
#         img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
#               torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#         img = torch.clamp(img, 0, 1)  # Ensure valid intensity range
        
#         # Convert to numpy format for matplotlib display
#         img_np = img.permute(1, 2, 0).numpy()
#         axes[i].imshow(img_np)
        
#         # Clinical annotation with color coding
#         class_name = PNEUMONIA_CLASSES[label.item()]
#         title_color = 'red' if label.item() == 1 else 'green'  # Clinical color coding
        
#         # Track clinical class distribution
#         if label.item() == 0:
#             normal_count += 1
#         else:
#             pneumonia_count += 1
        
#         # Apply clinical annotation
#         axes[i].set_title(class_name, color=title_color, fontweight='bold')
#         axes[i].axis('off')  # Remove axes for clean medical display
    
#     # Hide unused subplot areas
#     for i in range(len(selected_samples), len(axes)):
#         axes[i].axis('off')
    
#     # Clinical visualization summary
#     plt.suptitle(f'PneumoniaMNIST Clinical Samples ({normal_count} Normal, {pneumonia_count} Pneumonia)', 
#                  fontsize=14, fontweight='bold')
#     plt.tight_layout()
#     plt.show()
    
#     # Clinical analysis summary
#     print(f"Displayed {len(selected_samples)} images: {normal_count} normal, {pneumonia_count} pneumonia")
