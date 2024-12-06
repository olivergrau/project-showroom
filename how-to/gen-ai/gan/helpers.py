import random
import torch
from torch import nn


def set_seeds(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def initialize_weights(model):
    """Custom weight initialization."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


def get_positive_labels(size, device, smoothing=True, random_flip=0.05):
    """
    Generate smoothed positive labels for training a GAN.

    Args:
        size (int): Number of labels to generate.
        device (torch.device): Device to generate the labels on.
        smoothing (bool): Whether to apply label smoothing.
        random_flip (float): Proportion of labels to randomly flip to 0.

    Returns:
        torch.Tensor: Tensor of smoothed positive labels.
    """
    if smoothing:
        # Random positive numbers between 0.8 and 1.0 (valid range for BCELoss)
        labels = 0.8 + 0.2 * torch.rand(size, device=device)
    else:
        labels = torch.full((size,), 1.0, device=device)

    if random_flip > 0:
        # Optionally flip some labels to 0 to make it harder for the discriminator
        num_to_flip = int(random_flip * labels.size(0))
        indices = torch.randperm(labels.size(0))[:num_to_flip]
        labels[indices] = 0

    return labels


def get_negative_labels(size, device):
    return torch.full((size,), 0.0, device=device)