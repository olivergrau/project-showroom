import torch
import torch.nn as nn
import torch.nn.functional as F
# Example for 64 channels with depth 4 and stacks = 2 
#
#           Input Image (Grayscale)
#                   |
#                   V
#           +--------------------+
#           |   Initial Conv     |
#           |  (Conv, ReLU, BN)  |
#           +--------------------+
#                   |
#                   V
#           +--------------------+
#           |  Spatial Transformer |
#           +--------------------+
#                   |
#                   V
#       +-------------------------+
#       |     Initial Features    |
#       | (64 channels, downsampled) |
#       +-------------------------+
#                   |
#                   V
#       +---------------------+
#       |     Stack 1         |
#       | (Hourglass Module 1)|
#       +---------------------+
#                   |
#                   V
# +---------------------------------------+
# |         Hourglass Module 1           |
# |                                       |
# |                                       |
# |     +-----------+          +--------+ |
# |     |   Down    |          |  Up    | |
# |     | Sampling  |          |Sampling| |
# |     +-----------+          +--------+ |
# |            |                   |      |
# |            V                   |      |
# |     +-----------+             |      |
# |     |    Down    |            |      |
# |     | Sampling 2 |            |      |
# |     +-----------+             |      |
# |            |                  |      |
# |            V                  |      |
# |     +-----------+             |      |
# |     |     ...   |           |      |
# |     +-----------+             |      |
# |            |                  |      |
# |            V                  |      |
# |      +------------+           |      |
# |      |   Bottom   |           |      |
# |      +------------+           |      |
# |             |                 |      |
# |             V                 |      |
# |      +-----------+            |      |
# |      |     Up     |           |      |
# |      | Sampling 2 |           |      |
# |      +-----------+            |      |
# |            |                  |      |
# |            V                  |      |
# |      +-----------+            |      |
# |      |    Up     |            |      |
# |      | Sampling 1|            |      |
# |      +-----------+            |      |
# |                 |             |      |
# +---------------------------------------+
#                   |
#                   V
#           +-------------------+
#           |   Intermediate    |
#           |    Prediction     |
#           |(Auxiliary Output) |
#           +-------------------+
#                   |
#                   V
#           +--------------------+
#           |     Merge Path     |
#           +--------------------+
#                   |
#                   V
#           +---------------------+
#           |     Stack 2         |
#           | (Hourglass Module 2)|
#           +---------------------+
#                   |
#                   V
# +---------------------------------------+
# |         Hourglass Module 2           |
# |                                       |
# |     +-----------+          +--------+ |
# |     |   Down    |          |  Up    | |
# |     | Sampling  |          |Sampling| |
# |     +-----------+          +--------+ |
# |            |                   |      |
# |            V                   |      |
# |     +-----------+             |      |
# |     |    Down    |            |      |
# |     | Sampling 2 |            |      |
# |     +-----------+             |      |
# |            |                  |      |
# |            V                  |      |
# |     +-----------+             |      |
# |     |     ...     |           |      |
# |     +-----------+             |      |
# |            |                  |      |
# |            V                  |      |
# |      +------------+           |      |
# |      |   Bottom   |           |      |
# |      +------------+           |      |
# |             |                 |      |
# |             V                 |      |
# |      +-----------+            |      |
# |      |     Up     |           |      |
# |      | Sampling 2 |           |      |
# |      +-----------+            |      |
# |            |                  |      |
# |            V                  |      |
# |      +-----------+            |      |
# |      |    Up     |            |      |
# |      | Sampling 1|            |      |
# |      +-----------+            |      |
# |                 |             |      |
# +---------------------------------------+
#               |
#               V
#       +--------------------+
#       | Final Output       |
#       +--------------------+
#               |
#               V
#       +-----------------------+
#       |    Global Avg Pooling |
#       +-----------------------+
#               |
#               V
#       +-----------------------+
#       |    Fully Connected    |
#       +-----------------------+
#               |
#               V
#       +---------------------+
#       |   Output Keypoints  |
#       +---------------------+

from modules import Resnet18SpatialTransformer, EnhancedAuxiliaryClassifier

# A Residual Block stabilizes gradients and enables the net to learn additional transformation besides simply adding
# the input to the output
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        mid_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Adjust the identity mapping if necessary
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


# Define Hourglass Module
class HourglassModule(nn.Module):
    def __init__(self, depth, channels):
        super().__init__()
        self.depth = depth
        self.channels = channels
        self.down_residual_blocks = nn.ModuleList([ResidualBlock(channels, channels) for _ in range(depth)])
        self.up_residual_blocks = nn.ModuleList([ResidualBlock(channels, channels) for _ in range(depth)])
        self.pool = nn.MaxPool2d(2, stride=2)

        if depth > 1:
            self.inner_hourglass = HourglassModule(depth - 1, channels)
        else:
            self.bottom_residual = ResidualBlock(channels, channels)

        self.bn_up = nn.ModuleList([nn.BatchNorm2d(channels) for _ in range(depth)])

    def forward(self, x):
        down_features = []
        current_size = x.shape[-1]  # Current spatial dimension size

        # Downsampling path
        for i in range(self.depth):
            x = self.down_residual_blocks[i](x)
            down_features.append(x)

            # Adaptive pooling to avoid reaching 0x0 dimensions
            if current_size > 1:  # Only apply pooling if spatial dimension > 1
                x = self.pool(x)
                current_size = x.shape[-1]  # Update spatial dimension size

        # Bottom (smallest scale)
        if self.depth > 1:
            x = self.inner_hourglass(x)
        else:
            x = self.bottom_residual(x)

        # Upsampling path
        # for i in reversed(range(self.depth)):
        #     # Ensure the upsampling matches the size of down_features[i]
        #     x = F.interpolate(x, size=down_features[i].shape[-2:], mode='nearest')
        #     x = x + down_features[i]
        #     x = self.up_residual_blocks[i](x)

        for i in reversed(range(self.depth)):
            x = F.interpolate(x, size=down_features[i].shape[-2:], mode='nearest')
            x = x + down_features[i]
            x = self.up_residual_blocks[i](x)
            x = self.bn_up[i](x)  # Add BatchNorm after each upsampled residual block
            
        return x

class StackedHourglass(nn.Module):
    def __init__(self, num_keypoints, num_channels=64, depth=4, stacks=2, spatial_transform=True, use_aux_classifier=False):
        super().__init__()
        self.spatial_transform = spatial_transform
        self.use_aux_classifier = use_aux_classifier

        if spatial_transform:
            self.stn = Resnet18SpatialTransformer()

        self.num_keypoints = num_keypoints
        self.stacks = stacks

        # Initial layers
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, num_channels // 2, kernel_size=7, stride=2, padding=3, bias=False), # 112x112
            nn.BatchNorm2d(num_channels // 2),
            nn.ReLU(inplace=False),
            ResidualBlock(num_channels // 2, num_channels),
            nn.BatchNorm2d(num_channels),
            nn.MaxPool2d(2, stride=2) # 56x56
        )

        # Stacked hourglass modules
        self.hourglasses = nn.ModuleList([
            HourglassModule(depth=depth, channels=num_channels) for _ in range(stacks) # 56x56
        ])

        # Intermediate output for each stack
        self.intermediate_outputs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_channels, num_keypoints * 2, kernel_size=1),
                nn.BatchNorm2d(num_keypoints * 2)
            ) for _ in range(stacks)
        ])

        self.merge_features = nn.ModuleList([
            nn.Conv2d(num_channels, num_channels, kernel_size=1) for _ in range(stacks - 1)
        ])
        self.merge_preds = nn.ModuleList([
            nn.Conv2d(num_keypoints * 2, num_channels, kernel_size=1) for _ in range(stacks - 1)
        ])

        # Define auxiliary classifiers if enabled
        if use_aux_classifier:
            self.auxiliary_classifiers = nn.ModuleList([
                EnhancedAuxiliaryClassifier(num_channels, num_keypoints) for _ in range(stacks - 1)
            ])

        # Adaptive pooling for final output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc = nn.Linear(num_keypoints * 2 * 6 * 6, num_keypoints * 2)
        
        # Apply weight initialization
        self.apply(self._init_weights)

    def forward(self, x, writer=None, global_step=None, logging=False):
        theta = None
        if self.spatial_transform:
            if logging and writer and global_step is not None:
                writer.add_image("Input_Image_Before_STN", x[0], global_step=global_step, dataformats="CHW")

            x, theta = self.stn(x, writer, global_step, logging)

            if logging and writer and global_step is not None:
                writer.add_image("Input_Image_After_STN", x[0], global_step=global_step, dataformats="CHW")

        # Initial convolution and downsampling
        x = self.initial_conv(x)
        
        # Phase 2: Stacked Hourglass Network
        outputs = []
        aux_outputs = [] if self.use_aux_classifier else None

        for i in range(self.stacks):
            hg = self.hourglasses[i](x)            
            preds = self.intermediate_outputs[i](hg)
            outputs.append(preds)

            # Compute auxiliary output if enabled and not in the last stack
            if self.use_aux_classifier and i < self.stacks - 1:
                aux_outputs.append(self.auxiliary_classifiers[i](hg))

            if i < self.stacks - 1:
                x = x + self.merge_features[i](hg) + self.merge_preds[i](preds)

        # Final output with global average pooling
        final_output = self.adaptive_pool(outputs[-1])
        final_output = final_output.view(final_output.size(0), -1)  # Flatten
        final_output = self.fc(final_output)  # Map to keypoints
        final_output = torch.tanh(final_output)  # Ensure output is in [-1, 1] range

        if logging and writer and global_step is not None:
            writer.add_histogram("FC_Output", final_output, global_step=global_step)

        return final_output, aux_outputs, theta

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m in self.intermediate_outputs:
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
