# Please see inception_net.py and flexible_inception_net.py. I also tried hourglass.py.

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import AuxiliaryClassifier, Resnet18SpatialTransformer, EnhancedAuxiliaryClassifier, SEBlock

class ConfigurableBaseLineModel(nn.Module):
    def __init__(self, num_keypoints, out_channels=[8, 16, 32], kernel_sizes=[3, 3, 3], spatial_transform=False, use_batch_norm=True, dropout_rate=0.0, use_residual=False, use_se=False, use_auxiliary=False):
        super().__init__()

        # Ensure that out_channels and kernel_sizes have the same length
        assert len(out_channels) == len(kernel_sizes), "out_channels and kernel_sizes must be of the same length"

        self.spatial_transform = spatial_transform
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.use_se = use_se  # Enable or disable SE blocks
        self.use_auxiliary = use_auxiliary  # Enable or disable auxiliary classifier

        if spatial_transform:
            self.stn = Resnet18SpatialTransformer()

        # Initialize convolutional, batch normalization, and SE blocks dynamically
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batch_norm else None  # Create batch norm layers only if needed
        self.se_blocks = nn.ModuleList() if use_se else None    # Create SE blocks only if needed
        in_channels = 1  # Starting with 1 channel for grayscale images
        for out_ch, kernel_size in zip(out_channels, kernel_sizes):
            self.convs.append(nn.Conv2d(in_channels, out_ch, kernel_size=kernel_size, padding=kernel_size // 2))
            if use_batch_norm:
                self.bns.append(nn.BatchNorm2d(out_ch))  # Add corresponding BatchNorm layer
            if use_se:
                self.se_blocks.append(SEBlock(out_ch))   # Add SE block for channel attention
            in_channels = out_ch

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        conv_output_size = self._get_conv_output_size(224)  # Assuming input size of 224x224
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_keypoints * 2)  # Output layer for keypoints

        # Dropout layer (apply dropout if dropout_rate > 0)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None

        # Auxiliary classifier setup
        if use_auxiliary:
            self.auxiliary_classifier = EnhancedAuxiliaryClassifier(in_channels=out_channels[-1], num_keypoints=num_keypoints)

    def _get_conv_output_size(self, input_size):
        # Calculate output size after convolutions and pooling
        size = input_size
        for conv in self.convs:
            size = (size + 2 * conv.padding[0] - (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1
            size = size // 2  # Each pooling layer halves the size
        return size * size * self.convs[-1].out_channels

    def forward(self, x, writer=None, global_step=None, logging=False):
        theta = None
        if self.spatial_transform:
            if logging and writer and global_step is not None:
                writer.add_image("Input_Image_Before_STN", x[0], global_step=global_step, dataformats="CHW")

            # Apply Spatial Transformer Network
            x, theta = self.stn(x, writer, global_step, logging)

            if logging and writer and global_step is not None:
                writer.add_image("Input_Image_After_STN", x[0], global_step=global_step, dataformats="CHW")

        # Pass through convolutional, pooling layers with optional batch normalization, SE, and residual connections
        for i, conv in enumerate(self.convs):
            residual = x  # Save input for residual connection
            x = conv(x)
            if self.use_batch_norm:
                x = self.bns[i](x)  # Apply batch normalization if enabled
            x = F.relu(x)

            if self.use_se:
                x = self.se_blocks[i](x)  # Apply SE Block if enabled

            # Apply residual connection if enabled and dimensions match
            if self.use_residual and residual.shape == x.shape:
                x = x + residual

            x = self.pool(x)

        # Auxiliary classifier output
        aux_output = self.auxiliary_classifier(x) if self.use_auxiliary else None

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activations and optional dropout
        x = F.relu(self.fc1(x))
        if self.dropout:  # Apply dropout if configured
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.dropout:
            x = self.dropout(x)

        # Output layer
        x = self.fc3(x)

        if logging and writer and global_step is not None:
            writer.add_histogram("FC_Output", x, global_step=global_step)

        return x, aux_output, theta  # Return the auxiliary output if use_auxiliary is True



class BaseLineModel(nn.Module):
    def __init__(self, num_keypoints, spatial_transform=False):
        super().__init__()

        self.spatial_transform = spatial_transform

        if spatial_transform == True:
            self.stn = Resnet18SpatialTransformer()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 grayscale channel -> 32 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 -> 64 filters
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64 -> 128 filters

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Adjust according to input size
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_keypoints * 2)  # Output layer for keypoints

    def forward(self, x, writer=None, global_step=None, logging=False):
        theta = None
        if self.spatial_transform == True:
            if logging and writer and global_step is not None:
                writer.add_image("Input_Image_Before_STN", x[0], global_step=global_step, dataformats="CHW")

            # Phase 1: Apply Spatial Transformer Network
            x, theta = self.stn(x, writer, global_step, logging)

            if logging and writer and global_step is not None:
                writer.add_image("Input_Image_After_STN", x[0], global_step=global_step, dataformats="CHW")

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer
        x = self.fc3(x)

        if logging and writer and global_step is not None:
            writer.add_histogram("FC_Output", x, global_step=global_step)

        return x, None
