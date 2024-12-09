import torch
import torch.nn as nn
import torch.nn.functional as F

## Visual Representation
#
# Input (1 channel)
# ↓
# Conv1 (→ 64 channels)
# ↓
# Pool1
# ↓
# Conv2 (→ 192 channels)
# ↓
# Pool2
# ↓
# Inception3a (192 → 256 channels)
# ↓
# Inception3b (256 → 480 channels)
# ↓
# Pool3
# ↓
# Inception4a (480 → 512 channels)
# ↓
# Inception4b (512 → 512 channels)
# ↓
# Inception4c (512 → 512 channels)
# ↓
# Inception4d (512 → 528 channels)
# ↓
# Inception4e (528 → 832 channels)
# ↓
# Pool4
# ↓
# Inception5a (832 → 832 channels)
# ↓
# Inception5b (832 → 1024 channels)
# ↓
# Adaptive Pooling
# ↓
# Fully Connected Layer (→ 136 outputs)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from modules import Resnet18SpatialTransformer, EnhancedAuxiliaryClassifier


class StaticInceptionNet(nn.Module):
    def __init__(self, spatial_transform=False, use_auxiliary_classifier=False):
        super().__init__()

        self.spatial_transform = spatial_transform

        # Adding Spatial Transformer
        if self.spatial_transform:
            self.stn = Resnet18SpatialTransformer()
            print(f"Using spatial transformer ({self.stn.__class__.__name__}) in the model.")

        self.use_auxiliary_classifier = use_auxiliary_classifier
        
        if use_auxiliary_classifier:            
            self.aux1 = EnhancedAuxiliaryClassifier(in_channels=512, num_keypoints=68)
            self.aux2 = EnhancedAuxiliaryClassifier(in_channels=528, num_keypoints=68)
            print(f"Using auxiliary classifier ({self.aux1.__class__.__name__}) in the model.")        
        
        # Initial convolution layers with moderate downsampling
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Standard max pooling for downsampling
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Moderate downsampling

        # Inception modules with partial downsampling
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Further downsampling

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Final downsampling

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        # Adaptive pooling to retain spatial information at a higher resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))  # Adjusted to retain more spatial detail

        # Fully connected layer for regression (for 136 keypoints)
        self.fc = nn.Linear(1024 * 6 * 6, 136)

    def forward(self, x, writer=None, global_step=None, logging=False):
        # Apply Spatial Transformer
        if self.spatial_transform:
            if logging and writer and global_step is not None:
                writer.add_image("Input_Image_Before_STN", x[0], global_step=global_step, dataformats="CHW")
            x, theta = self.stn(x, writer, global_step, logging)
            if logging and writer and global_step is not None:
                writer.add_image("Input_Image_After_STN", x[0], global_step=global_step, dataformats="CHW")
        else:
            theta = None
        
        aux_output1 = None
        aux_output2 = None
        
        # Initial layers with max pooling
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        # Inception modules with intermediate pooling
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)

        x = self.inception4a(x)
        
        if self.use_auxiliary_classifier:
            aux_output1 = self.aux1(x) if self.training else None
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.use_auxiliary_classifier:
            aux_output2 = self.aux2(x) if self.training else None
        
        x = self.inception4e(x)
        x = self.pool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        # Adaptive pooling and flatten
        x = self.adaptive_pool(x)  # Higher spatial detail with 6x6 output
        x = x.view(x.size(0), -1)

        # Final fully connected layer to predict keypoints
        x = self.fc(x)
        if logging and writer and global_step is not None:
            writer.add_histogram("FC_Output", x, global_step=global_step)

        return x, aux_output1, aux_output2, theta


class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super().__init__()

        # 1x1 convolution branch with BatchNorm and ReLU
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.Mish()
        )

        # 1x1 convolution followed by 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(ch3x3_reduce),
            nn.Mish(),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.Mish()
        )

        # 1x1 convolution followed by 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(ch5x5_reduce),
            nn.ELU(),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.Mish()
        )

        # 3x3 max pooling followed by 1x1 convolution branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.Mish()
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)        
        branch3x3 = self.branch3x3(x)        
        branch5x5 = self.branch5x5(x)        
        branch_pool = self.branch_pool(x)

        # Concatenate along channel axis
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class ScalableInceptionNet(nn.Module):
    def __init__(self, spatial_transform=False, use_auxiliary_classifier=False, scale_factor=1):
        super().__init__()

        self.spatial_transform = spatial_transform

        # Adding Spatial Transformer
        if self.spatial_transform:
            self.stn = Resnet18SpatialTransformer()
            print(f"Using spatial transformer ({self.stn.__class__.__name__}) in the model.")

        self.use_auxiliary_classifier = use_auxiliary_classifier

        if use_auxiliary_classifier:
            self.aux1 = EnhancedAuxiliaryClassifier(in_channels=int(512 * scale_factor), num_keypoints=68)
            self.aux2 = EnhancedAuxiliaryClassifier(in_channels=int(528 * scale_factor), num_keypoints=68)
            print(f"Using auxiliary classifier ({self.aux1.__class__.__name__}) in the model.")

            # Initial convolution layers with moderate downsampling
        self.conv1 = nn.Conv2d(1, int(64 * scale_factor), kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(int(64 * scale_factor), int(192 * scale_factor), kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception modules with scaled parameters
        self.inception3a = InceptionModule(int(192 * scale_factor), int(64 * scale_factor), int(96 * scale_factor), int(128 * scale_factor), int(16 * scale_factor), int(32 * scale_factor), int(32 * scale_factor))
        self.inception3b = InceptionModule(int(256 * scale_factor), int(128 * scale_factor), int(128 * scale_factor), int(192 * scale_factor), int(32 * scale_factor), int(96 * scale_factor), int(64 * scale_factor))
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(int(480 * scale_factor), int(192 * scale_factor), int(96 * scale_factor), int(208 * scale_factor), int(16 * scale_factor), int(48 * scale_factor), int(64 * scale_factor))
        self.inception4b = InceptionModule(int(512 * scale_factor), int(160 * scale_factor), int(112 * scale_factor), int(224 * scale_factor), int(24 * scale_factor), int(64 * scale_factor), int(64 * scale_factor))
        self.inception4c = InceptionModule(int(512 * scale_factor), int(128 * scale_factor), int(128 * scale_factor), int(256 * scale_factor), int(24 * scale_factor), int(64 * scale_factor), int(64 * scale_factor))
        self.inception4d = InceptionModule(int(512 * scale_factor), int(112 * scale_factor), int(144 * scale_factor), int(288 * scale_factor), int(32 * scale_factor), int(64 * scale_factor), int(64 * scale_factor))
        self.inception4e = InceptionModule(int(528 * scale_factor), int(256 * scale_factor), int(160 * scale_factor), int(320 * scale_factor), int(32 * scale_factor), int(128 * scale_factor), int(128 * scale_factor))
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionModule(int(832 * scale_factor), int(256 * scale_factor), int(160 * scale_factor), int(320 * scale_factor), int(32 * scale_factor), int(128 * scale_factor), int(128 * scale_factor))
        self.inception5b = InceptionModule(int(832 * scale_factor), int(384 * scale_factor), int(192 * scale_factor), int(384 * scale_factor), int(48 * scale_factor), int(128 * scale_factor), int(128 * scale_factor))

        # Adaptive pooling to retain spatial information at a higher resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))  # Adjusted to retain more spatial detail

        # Fully connected layer for regression (for 136 keypoints)
        self.fc = nn.Linear(int(1024 * scale_factor) * 6 * 6, 136)

    def forward(self, x, writer=None, global_step=None, logging=False):
        # Apply Spatial Transformer
        if self.spatial_transform:
            if logging and writer and global_step is not None:
                writer.add_image("Input_Image_Before_STN", x[0], global_step=global_step, dataformats="CHW")
            x, theta = self.stn(x, writer, global_step, logging)
            if logging and writer and global_step is not None:
                writer.add_image("Input_Image_After_STN", x[0], global_step=global_step, dataformats="CHW")
        else:
            theta = None

        aux_output1 = None
        aux_output2 = None

        # Initial layers with max pooling
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        # Inception modules with intermediate pooling
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)

        x = self.inception4a(x)

        if self.use_auxiliary_classifier:
            aux_output1 = self.aux1(x) if self.training else None

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.use_auxiliary_classifier:
            aux_output2 = self.aux2(x) if self.training else None

        x = self.inception4e(x)
        x = self.pool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        # Adaptive pooling and flatten
        x = self.adaptive_pool(x)  # Higher spatial detail with 6x6 output
        x = x.view(x.size(0), -1)

        # Final fully connected layer to predict keypoints
        x = self.fc(x)
        if logging and writer and global_step is not None:
            writer.add_histogram("FC_Output", x, global_step=global_step)

        return x, aux_output1, aux_output2, theta

