import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

## I tried out some modules: spatial transformer, SEBlock, auxiliary classifier

## My manually created transformer yields also good results, but I tried nevertheless one with an existing ResNet18 weights.
class Resnet18SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # Load a pretrained ResNet model
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Modify the first conv layer to accept 1-channel input (grayscale)
        self.localization = nn.Sequential(*list(resnet.children())[:-2])
        self.localization[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust first conv layer

        # Initialize weights for the modified layer
        with torch.no_grad():
            self.localization[0].weight = nn.Parameter(resnet.conv1.weight[:, :1] * (3 / 1))  # Scale if necessary

        # Regressor for the 6 affine parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.Mish(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 6)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[3].weight.data.zero_()
        self.fc_loc[3].bias.data.copy_(torch.tensor([0.9, 0, 0, 0, 0.9, 0], dtype=torch.float))

    def forward(self, x, writer=None, global_step=None, logging=False):
        # Forward pass through the localization network
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)  # Flatten for the fully connected layer

        # Regressor for affine parameters
        theta = self.fc_loc(xs)
        theta = torch.tanh(theta) * 1
        theta = theta.view(-1, 2, 3)

        # Optional logging
        if logging and writer and global_step is not None:
            writer.add_scalar("STN/Theta_Mean", theta.mean().item(), global_step=global_step)
            writer.add_scalar("STN/Theta_StdDev", theta.std().item(), global_step=global_step)
            writer.add_histogram("STN/Theta_Histogram", theta.clone().detach().cpu(), global_step=global_step)

        # Apply affine transformation
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x, theta


class RevisedSpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # Localization network with added batch norm, dropout, and an extra convolutional layer
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, stride=2),
            nn.BatchNorm2d(8),
            nn.Mish(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.Mish(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.Mish(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # Extra convolutional layer
            nn.BatchNorm2d(64),
            nn.Mish(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Regressor for the 6 affine parameters with added dropout and random initialization
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),  # Adjusted to match the added conv layer
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Dropout added here
            nn.Linear(128, 6)
        )

        # Initialize weights and bias with identity transformation
        nn.init.normal_(self.fc_loc[3].weight, mean=0, std=0.01)  # Random init for weights
        self.fc_loc[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x, writer=None, global_step=None, logging=False):
        # Localization network forward pass
        xs = self.localization(x)
        xs = xs.view(-1, 64 * 4 * 4)  # Adjusted for added conv layer

        # Affine parameters (theta) computation with scaling
        theta = self.fc_loc(xs)
        theta = torch.tanh(theta) * 1  # Scale to [-0.5, 0.5]

        theta = theta.view(-1, 2, 3)

        # Optional logging
        if logging and writer and global_step is not None:
            writer.add_scalar("STN/Theta_Mean", theta.mean().item(), global_step=global_step)
            writer.add_scalar("STN/Theta_StdDev", theta.std().item(), global_step=global_step)
            writer.add_histogram("STN/Theta_Histogram", theta.clone().detach().cpu(), global_step=global_step)

        # Create affine grid and apply transformation
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x, theta


class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, stride=2),
            nn.Mish(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),  # Increased channels and stride
            nn.Mish(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # Added an extra conv layer
            nn.Mish(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # Reduced output size to (32, 4, 4)
        )

        # Regressor for the 6 affine parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.Mish(inplace=True),
            nn.Dropout(0.3),  # Add dropout here
            nn.Linear(64, 6)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[3].weight.data.zero_()
        self.fc_loc[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x, writer=None, global_step=None, logging=False):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 4 * 4)

        theta = self.fc_loc(xs)
        theta = torch.tanh(theta) * 1 # no scaling here

        theta = theta.view(-1, 2, 3)

        if logging and writer and global_step is not None:
            writer.add_scalar("STN/Theta_Mean", theta.mean().item(), global_step=global_step)
            writer.add_histogram("STN/Theta", theta.clone().detach().cpu(), global_step=global_step)

        # Apply affine transformation
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        theta = torch.tanh(theta) * 1  # Scales transformations to be between 1 and -1 (default)

        return x, theta

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        scale = torch.mean(x, dim=(2, 3), keepdim=True)  # Global pooling
        scale = self.fc1(scale.view(x.size(0), -1))
        scale = F.relu(scale)
        scale = self.fc2(scale)
        scale = torch.sigmoid(scale).view(x.size(0), -1, 1, 1)
        return x * scale

class EnhancedAuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super().__init__()

        # Adaptive pooling to handle variable input sizes
        self.average_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Multi-scale convolution for richer feature extraction
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        ])

        # Convolutional layer with SE Block
        self.conv = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1),  # 192 from multi-scale concat
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        self.se_block = SEBlock(128)  # Squeeze-and-Excitation block

        # Fully connected layers with batch normalization and dropout
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True)
        )

        # Output layer for keypoint predictions
        self.fc3 = nn.Linear(512, num_keypoints * 2)

        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)

    def forward(self, x):
        x = self.average_pool(x)

        # Multi-scale convolution
        branches = [conv(x) for conv in self.multi_scale_conv]
        x = torch.cat(branches, dim=1)  # Concatenate along channel dimension

        # Convolution with SE block
        x = self.conv(x)
        x = self.se_block(x)  # Apply SE block

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = self.dropout1(self.fc1(x))
        x = self.dropout2(self.fc2(x))

        # Final output layer
        x = self.fc3(x)

        return x


class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super().__init__()
        self.average_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_keypoints * 2)
        self.relu = nn.Mish() if torch.__version__ >= "1.10.0" else nn.ReLU

    def forward(self, x):
        x = self.average_pool(x)
        x = self.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
