import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Resnet18SpatialTransformer, EnhancedAuxiliaryClassifier

class InceptionModuleResidual(nn.Module):
    def __init__(self, in_channels, config, use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        ch1x1 = config['ch1x1']
        ch3x3_reduce = config['ch3x3_reduce']
        ch3x3 = config['ch3x3']
        ch5x5_reduce = config['ch5x5_reduce']
        ch5x5 = config['ch5x5']
        pool_proj = config['pool_proj']

        # 1x1 convolution branch
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.Mish(inplace=True)
        )

        # 1x1 convolution followed by 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(ch3x3_reduce),
            nn.ELU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.Mish(inplace=True)
        )

        # 1x1 convolution followed by 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(ch5x5_reduce),
            nn.Mish(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.Mish(inplace=True)
        )

        # 3x3 max pooling followed by 1x1 convolution branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.Mish(inplace=True)
        )

        # Calculate total output channels for concatenation
        self.out_channels = ch1x1 + ch3x3 + ch5x5 + pool_proj

        # Residual projection (1x1 convolution if input channels != output channels)
        if self.use_residual and in_channels != self.out_channels:
            self.residual_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):
        # Concatenate all branches
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        outputs = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], 1)

        # Add residual connection if enabled
        if self.use_residual:
            residual = self.residual_conv(x)
            outputs += residual  # Element-wise addition

        return outputs

class InceptionModule(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        ch1x1 = config['ch1x1']
        ch3x3_reduce = config['ch3x3_reduce']
        ch3x3 = config['ch3x3']
        ch5x5_reduce = config['ch5x5_reduce']
        ch5x5 = config['ch5x5']
        pool_proj = config['pool_proj']

        # 1x1 convolution branch
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
            nn.Mish(),
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

        # Calculate the total output channels
        self.out_channels = ch1x1 + ch3x3 + ch5x5 + pool_proj

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class FlexibleInceptionNet(nn.Module):
    def __init__(self, num_keypoints, inception_configs, use_spatial_transform=False, use_residual=False, use_aux=False):
        super().__init__()
        
        if use_aux:
            print("Using auxiliary classifier in InceptionNet")
            self.use_aux = use_aux
        
        if use_residual:
            print("Using residual connections in InceptionNet")

        self.spatial_transform = use_spatial_transform

        # Adding Spatial Transformer
        if self.spatial_transform:            
            self.stn = Resnet18SpatialTransformer()
            print(f"Using spatial transformer ({self.stn.__class__.__name__}) in InceptionNet")

        # Initial convolution layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.Mish()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.Mish()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build Inception modules dynamically based on configs
        self.inception_modules = nn.ModuleList()
        self.residual_convs = nn.ModuleList() if use_residual else None  # Convs for residual alignment
        self.auxiliary_classifiers = nn.ModuleList()
        self.auxiliary_indices = []  # Indices where auxiliary classifiers are applied
        in_channels = 192  # Starting channels after conv2

        for idx, config in enumerate(inception_configs):
            # Create Inception module
            if use_residual:
                module = InceptionModuleResidual(in_channels, config, use_residual=use_residual)
            else:
                module = InceptionModule(in_channels, config)
                
            self.inception_modules.append(module)

            # Update channels for the next module
            out_channels = module.out_channels

            # Add residual alignment convolution if needed
            if use_residual:
                if in_channels != out_channels:
                    # 1x1 convolution to match the channels if residual connection is applied
                    self.residual_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
                else:
                    self.residual_convs.append(nn.Identity())  # No alignment needed if channels match
            
            self.use_aux = use_aux
            
            # Handle auxiliary classifier
            if use_aux and 'aux' in config and config['aux']:
                aux_in_channels = config['aux_in_channels']
                aux_classifier = EnhancedAuxiliaryClassifier(aux_in_channels, num_keypoints)
                self.auxiliary_classifiers.append(aux_classifier)
                
                # The index where this auxiliary output should be computed
                self.auxiliary_indices.append(len(self.inception_modules) - 1)  # Current module index

            in_channels = out_channels  # Set for the next loop iteration

            # Add pooling layers at specified positions
            if 'pool' in config and config['pool']:
                self.inception_modules.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                # in_channels remain the same after pooling

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))        

        # Fully connected layer for regression
        self.fc = nn.Linear(in_channels * 6 * 6, num_keypoints * 2)                

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

        # Initial layers
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        # Inception modules with macro residuals
        residual_index = 0  # Tracks index in residual_convs
        aux_outputs = []
        aux_classifier_index = 0  # Tracks which auxiliary classifier to use

        for idx, module in enumerate(self.inception_modules):
            if isinstance(module, InceptionModuleResidual):
                residual = x  # Store input for residual

                # Forward through Inception module
                x = module(x)

                # Apply residual if use_residual is True
                if self.residual_convs:
                    aligned_residual = self.residual_convs[residual_index](residual)
                    x = x + aligned_residual  # Add residual connection
                    residual_index += 1

            elif isinstance(module, InceptionModule):
                x = module(x)                

            elif isinstance(module, nn.MaxPool2d):
                x = module(x)

            # Check if we need to compute an auxiliary output
            if self.use_aux and idx in self.auxiliary_indices and self.training:
                aux_classifier = self.auxiliary_classifiers[aux_classifier_index]                
                aux_output = aux_classifier(x)
                aux_outputs.append(aux_output)
                aux_classifier_index += 1

        # Adaptive pooling and flatten
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layer to predict keypoints
        x = self.fc(x)

        if logging and writer and global_step is not None:
            writer.add_histogram("FC_Output", x, global_step=global_step)

        return x, aux_outputs, theta

