"""
Architecture optimization utilities for hardware-aware model optimization in medical imaging.

This module provides comprehensive implementations of modern neural network optimization
techniques specifically designed for clinical deployment scenarios. Focuses on reducing
computational overhead, memory usage, and inference latency while maintaining diagnostic
accuracy for the PneumoniaMNIST binary classification task.

Key optimization strategies:
    - Interpolation Removal: Eliminates computational overhead from resolution upscaling
    - Depthwise Separable Convolutions: Reduces parameters and FLOPs significantly
    - Grouped Convolutions: Parallel channel processing for improved efficiency
    - Inverted Residual Blocks: Mobile-optimized residual architectures
    - Low-Rank Factorization: Matrix decomposition for parameter reduction
    - Channel Optimization: Memory layout and activation optimizations
    - Parameter Sharing: Weight reuse across similar layer configurations
"""

import copy
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def create_optimized_model(base_model: nn.Module, optimizations: Dict[str, Any]) -> nn.Module:
    """
    Apply selected optimization strategies in order to create a clinically-optimized model.

    Args:
        base_model: Original ResNet model to optimize for clinical deployment
        optimizations: Dictionary specifying which optimizations to apply with parameters:
            - 'interpolation_removal': bool - Remove upscaling overhead (recommended: True)
            - 'depthwise_separable': bool - Apply depthwise separable convolutions
            - 'grouped_conv': bool - Use grouped convolutions for parallel processing
            - 'channel_optimization': bool - Optimize memory layout and activations
            - 'inverted_residuals': bool - Replace blocks with inverted residuals
            - 'lowrank_factorization': bool - Apply matrix factorization to linear layers
            - 'parameter_sharing': bool - Share weights between similar layers
            
    Returns:
        Optimized model with selected techniques applied, ready for clinical deployment
        
    Example:
        >>> base_model = create_baseline_model()
        >>> optimization_config = {
        ...     'interpolation_removal': True,
        ...     'depthwise_separable': True,
        ...     'channel_optimization': True
        ... }
        >>> optimized_model = create_optimized_model(base_model, optimization_config)
        >>> print("Clinical deployment model ready")
    """
    model = copy.deepcopy(base_model)
  
    print("Starting clinical model optimization pipeline...")
    
    # Define the optimization order by filling in this list
    # HINT: Consider which optimizations should be applied first and why
    # Think about: architectural changes → layer modifications → hardware opts → parameter opts
    optimization_order = [
        'interpolation_removal',
        'grouped_conv',
        #'depthwise_separable',
        #'inverted_residuals',
        'lowrank_factorization',
        'channel_optimization',
        #'parameter_sharing'
    ]
    
    # Optimization function mapping - connects optimization names to their implementation
    # IMPORTANT: Make sure to experiment with different input parameters for each optimization function, if performance is suboptimal
    optimization_functions = {
        'interpolation_removal': lambda m: apply_interpolation_removal_optimization(m),
        'depthwise_separable': lambda m: apply_depthwise_separable_optimization(m),
        'grouped_conv': lambda m: apply_grouped_convolution_optimization(m, groups=8, layer_names=["model.layer3", "model.layer4"]),
        'channel_optimization': lambda m: apply_channel_optimization(m),
        'inverted_residuals': lambda m: apply_inverted_residual_optimization(m),
        'lowrank_factorization': lambda m: apply_lowrank_factorization(m, layer_names=["inner.model.layer4"], rank_ratio=0.5, min_channels=64),
        'parameter_sharing': lambda m: apply_parameter_sharing(m)
    }
    
    # Smart iteration through the defined optimization order
    applied_optimizations = []
    for opt_name in optimization_order:
        # Check if this optimization is requested and available
        if optimizations.get(opt_name, False) and opt_name in optimization_functions:
            print(f"   Applying {opt_name.replace('_', ' ')} optimization...")
            try:
                # Apply the optimization using the mapped function
                model = optimization_functions[opt_name](model)
                applied_optimizations.append(opt_name)
            except Exception as e:
                print(f"   ERROR: {opt_name} optimization failed: {e}")
        elif opt_name not in optimization_functions:
            print(f"   WARNING: Unknown optimization: {opt_name}")
    
    # Report results
    if applied_optimizations:
        print(f"Applied optimizations in order: {' → '.join(applied_optimizations)}")
    else:
        print("No optimizations were applied")
        
    return model

# --------------------------------------
# INTERPOLATION REMOVAL (NATIVE RESOLUTION)
# --------------------------------------

def apply_interpolation_removal_optimization(model: nn.Module, native_size: int = 64) -> nn.Module:
    """
    Remove interpolation overhead by processing images at native resolution.

    This optimization bypasses the internal bilinear interpolation step in
    ResNetBaseline and directly feeds native-resolution inputs to the backbone.

    Args:
        model: ResNetBaseline model instance
        native_size: Native input resolution (64 for PneumoniaMNIST)

    Returns:
        Optimized model without interpolation overhead
    """
    import copy
    import torch.nn as nn

    optimized_model = copy.deepcopy(model)

    # Sanity check
    if not hasattr(optimized_model, "model"):
        raise ValueError(
            "Interpolation removal expects a ResNetBaseline-like model "
            "with an underlying `.model` attribute."
        )

    class NativeResolutionWrapper(nn.Module):
        def __init__(self, backbone, num_classes, input_size):
            super().__init__()
            self.model = backbone
            self.input_size = input_size
            self.target_size = input_size  # for metadata consistency
            self.architecture_name = "ResNet18-Native64"
            self.num_classes = num_classes

        def forward(self, x):
            # Assumes input already at native resolution
            return self.model(x)

    optimized_model = NativeResolutionWrapper(
        backbone=optimized_model.model,
        num_classes=optimized_model.num_classes,
        input_size=native_size,
    )

    print("INTERPOLATION REMOVAL applied:")
    print(f"   - Native input resolution: {native_size}x{native_size}")
    print("   - Bilinear interpolation bypassed")
    print("   - Backbone architecture unchanged")

    return optimized_model


# # TODO: Implement this optimization method, if selected in your optimization strategy
# def apply_interpolation_removal_optimization(model: nn.Module, native_size: int = 64) -> nn.Module:
#     """
#     Remove interpolation overhead by processing images at native resolution.
    
#     Args:
#         model: Model with interpolation capability (e.g., ResNetBaseline)
#         native_size: Native input resolution to process (64 for clinical deployment)
        
#     Returns:
#         Optimized model that processes at native resolution without interpolation

#     Note: 
#         In `data_loader.py`, we would also want to replace ImageNet stats with chest 
#         X-ray specific to check if accuracy improves, but you can skip this for simplicity 
#         as normalization affects accuracy/sensitivity and not operational efficiency.
        
#     Example:
#         >>> baseline_model = create_baseline_model()
#         >>> optimized_model = apply_interpolation_removal_optimization(baseline_model, 64)
#         >>> # Model now processes 64x64 images directly without upscaling
#     """
#     # Deep copy model to avoid modifying original
#     optimized_model = copy.deepcopy(model)

#     print(f"Applying native resolution optimization ({native_size}x{native_size})...")
    
#     # TODO: Update the existing model class to bypasses interpolation and processes images at native resolution.
#     # HINT: The ResNetBaseline model automatically interpolates input images from 64x64 to 224x224 
#     # before passing them to the underlying ResNet. One option is to create a wrapper that:
#     # 1. Stores the original model architecture and metadata
#     # 2. Updates the input_size attribute to reflect native processing  
#     # 3. In the forward pass, bypasses the interpolation step entirely
#     # 4. Directly calls the underlying ResNet model (model.model if it's a ResNetBaseline)
#     #
#     # See the ResNetBaseline.forward() method to understand how interpolation currently works.

#     # Add your code here

#     # Report optimization status and provide deployment guidance
#     print("INTERPOLATION REMOVAL completed.")
    
#     return optimized_model

# --------------------------------------
# DEPTHWISE SEPARABLE CONVOLUTION MODULES
# --------------------------------------

# TODO: Implement this optimization method, if selected in your optimization strategy
def apply_depthwise_separable_optimization(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
    min_channels: int = 16,
    preserve_residuals: bool = True
) -> nn.Module:
    """
    Convert suitable Conv2d layers to DepthwiseSeparableConv2d for clinical efficiency.
    
    Systematically replaces standard convolutions with depthwise separable alternatives
    to reduce computational cost and memory usage while preserving diagnostic accuracy.
    Essential for deploying medical imaging models on resource-constrained devices.
    
    Args:
        model: Input model to optimize for clinical deployment
        layer_names: Specific layer names to convert (None = convert all suitable layers)
        min_channels: Minimum input/output channels required for conversion
        preserve_residuals: Use residual-compatible configurations for ResNet models
        
    Returns:
        Optimized model with depthwise separable convolutions applied
        
    Note:
        Only converts layers that benefit from depthwise separation (kernel_size > 1,
        sufficient channels, not already grouped). Preserves ResNet compatibility by
        maintaining residual connection requirements.
        
    Example:
        >>> model = create_baseline_model()
        >>> optimized_model = apply_depthwise_separable_optimization(
        ...     model, min_channels=32
        ... )
        >>> # Suitable Conv2d layers now use depthwise separable convolutions
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    replacements = 0  # Track number of successful replacements

    print("Applying depthwise separable convolution optimization...")

    # TODO: Update the model to use depthwise separable convolution instead of convolution. 
    # HINT: To transform a conv2d into depthwise separable, you need to convolve each channel with its own kernel (groups=in_channels) for depthwise, 
    # and then combine information across channels processed by depthwise layer to define the pointwise layer.
    # Note that a conv2d block is also composed by activation and batchnorm in ResNet - Do you want to keep both, either, or none in?
    # Also, think about how the residuals are handled.
    # See https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/ for an intuitive explanation and code template.

    # Add your code here

    # Report optimization status
    if replacements > 0:
        print(f"DEPTHWISE SEPARABLE completed: Successfully applied to layers with {replacements} replacements")
    else:
        print("WARNING: DEPTHWISE SEPARABLE not applied: No suitable layers found for replacement")

    return optimized_model

# --------------------------------------
# GROUPED CONVOLUTION MODULES
# --------------------------------------

def apply_grouped_convolution_optimization(
    model: nn.Module,
    groups: int = 2,
    min_channels: int = 32,
    layer_names: Optional[List[str]] = None,
    do_depthwise: Optional[bool] = False,
) -> nn.Module:
    """
    Convert suitable Conv2d layers to grouped convolutions for parallel efficiency.

    Strategy (safe for ResNet-18):
      - Only convert 3x3 convolutions by default (kernel_size == 3)
      - Skip stem conv and downsample 1x1 convs
      - Apply only when in/out channels are divisible by groups
      - Weight transfer: keep within-group weights, drop cross-group mixing

    Args:
        model: Input model to optimize
        groups: Number of groups for grouped convolution (typically 2-8)
        min_channels: Minimum channels required for conversion
        layer_names: Specific layer name prefixes to convert (None = all suitable)
        do_depthwise: If True, use groups=in_channels for eligible layers (depthwise-like).
                      Not recommended for ResNet-18 unless very selective.

    Returns:
        Optimized model with grouped convolutions applied where safe.
    """
    optimized_model = copy.deepcopy(model)

    replacements = 0
    skipped = 0

    print(f"Applying grouped convolution optimization (groups={groups}, depthwise={do_depthwise})...")

    # Helper: get parent module by dotted name
    def _get_parent(root: nn.Module, full_name: str):
        parts = full_name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]

    # Decide whether this module name should be targeted
    def _name_is_selected(name: str) -> bool:
        if layer_names is None:
            return True
        return any(name.startswith(prefix) for prefix in layer_names)

    # Determine whether a conv is safe to convert
    def _is_convertible_conv(name: str, conv: nn.Conv2d) -> bool:
        # Skip 1x1 convs and stem conv
        k = conv.kernel_size
        if isinstance(k, tuple):
            kh, kw = k
        else:
            kh = kw = int(k)

        if kh != 3 or kw != 3:
            return False

        # Avoid the stem conv by name convention if present
        if name.endswith("conv1") and "layer" not in name:
            return False

        # Avoid very small channels
        if conv.in_channels < min_channels or conv.out_channels < min_channels:
            return False

        # Depthwise grouping option
        if do_depthwise:
            # True depthwise requires out_channels == in_channels (or a multiple, but classic depthwise equals)
            if conv.in_channels != conv.out_channels:
                return False
            return True

        # Standard grouped conv requires divisibility
        if conv.in_channels % groups != 0 or conv.out_channels % groups != 0:
            return False

        # Do not re-group already grouped convs
        if conv.groups != 1:
            return False

        return True

    # Convert one conv (with weight transfer)
    def _convert_conv(conv: nn.Conv2d, new_groups: int) -> nn.Conv2d:
        new_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=new_groups,
            bias=(conv.bias is not None),
            padding_mode=conv.padding_mode,
        )

        # Weight transfer:
        # Original weight: [out_c, in_c, kH, kW]
        # Grouped weight:  [out_c, in_c/new_groups, kH, kW]
        with torch.no_grad():
            out_c = conv.out_channels
            in_c = conv.in_channels
            out_per_g = out_c // new_groups
            in_per_g = in_c // new_groups

            w_old = conv.weight.detach()
            w_new = new_conv.weight.detach()

            # For each group, copy the matching in/out slices
            for g in range(new_groups):
                out_s = slice(g * out_per_g, (g + 1) * out_per_g)
                in_s = slice(g * in_per_g, (g + 1) * in_per_g)
                w_new[out_s, :, :, :] = w_old[out_s, in_s, :, :]

            new_conv.weight.copy_(w_new)

            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias.detach())

        return new_conv

    # Iterate named modules and replace where appropriate
    for name, module in list(optimized_model.named_modules()):
        if not isinstance(module, nn.Conv2d):
            continue
        if not _name_is_selected(name):
            continue
        if not _is_convertible_conv(name, module):
            continue

        # Determine grouping
        new_groups = module.in_channels if do_depthwise else groups

        # Extra safety checks
        if module.in_channels % new_groups != 0 or module.out_channels % new_groups != 0:
            skipped += 1
            continue

        parent, attr = _get_parent(optimized_model, name)
        try:
            setattr(parent, attr, _convert_conv(module, new_groups))
            replacements += 1
        except Exception:
            skipped += 1

    if replacements > 0:
        print(f"GROUPED CONV completed: Successfully applied to {replacements} layers. Skipped {skipped} layers.")
        print("DEPLOYMENT TIP: Grouped conv benefits are strongest when combined with channels_last and mixed precision.")
        if layer_names is not None:
            print(f"   Applied only to layer prefixes: {layer_names}")
    else:
        print("WARNING: GROUPED CONV not applied: No suitable layers found for replacement")

    return optimized_model

# # TODO: Implement this optimization method, if selected in your optimization strategy
# def apply_grouped_convolution_optimization(
#     model: nn.Module,
#     groups: int = 2,
#     min_channels: int = 32,
#     layer_names: Optional[List[str]] = None,
#     do_depthwise: Optional[bool] = False,
# ) -> nn.Module:
#     """
#     Convert suitable Conv2d layers to grouped convolutions for parallel efficiency.
    
#     Args:
#         model: Input model to optimize
#         groups: Number of groups for grouped convolution (typically 2-8)
#         min_channels: Minimum channels required for conversion
#         layer_names: Specific layers to convert (None = all suitable layers)
#         do_depthwise: Whether to apply depthwise grouping (groups=in_channels)
        
#     Returns:
#         Model with grouped convolutions applied for enhanced efficiency
        
#     Note:
#         Grouped convolutions can be highly efficient on certain hardware backends, 
#         especially when used with memory formats like channels_last and mixed precision (AMP)
        
#     Example:
#         >>> model = create_baseline_model()
#         >>> optimized_model = apply_grouped_convolution_optimization(
#         ...     model, groups=4, min_channels=64
#         ... )
#         >>> # Suitable layers now use 4-group parallel processing
#     """
#     # Deep copy model to avoid modifying original
#     optimized_model = copy.deepcopy(model)
#     # Track number of successful and skipped replacements
#     replacements = 0
#     skipped = 0

#     print(f"Applying grouped convolution optimization (groups={groups})...")

#     # TODO: Convert suitable Conv2d layers to grouped convolutions.
#     # HINT: Grouped convolution divides input channels into independent groups and applies separate 
#     # convolutions to each group. To make this happen, you need to ensure that the later is suitable for this transformation.
#     # See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html for how to use the group parameter.

#     # Add your code here

#     # Report optimization status and provide deployment tipes
#     if replacements > 0:
#         print(f"GROUPED CONV completed: Successfully applied to layers with {replacements} replacements. Skipped {skipped} layers.")
#         print("\nDEPLOYMENT TIP: For some hardware (like NVIDIA GPUs), grouped convolutions may require specific memory formats (channels_last) and mixed precision to achieve maximum throughput.")
#     else:
#         print("WARNING: GROUPED CONV not applied: No suitable layers found for replacement")

#     return optimized_model

# --------------------------------------
# INVERTED RESIDUAL BLOCKS
# --------------------------------------

# TODO: Implement this optimization method, if selected in your optimization strategy
def apply_inverted_residual_optimization(
    model: nn.Module,
    target_layers: Optional[List[str]] = None,
    expand_ratio: int = 6
) -> nn.Module:
    """
    Replace suitable blocks with mobile-optimized InvertedResidual blocks.

    Args:
        model: Original model for mobile optimization
        target_layers: Specific layer names to convert (None = auto-detect suitable blocks)
        expand_ratio: Channel expansion factor for inverted residuals (6 is optimal)
        
    Returns:
        Model with mobile-optimized inverted residual blocks
        
    Note:
        This optimization targets BasicBlock structures and converts them to mobile-friendly
        inverted residuals. Most effective for deployment on edge devices and mobile platforms
        common in point-of-care medical applications.
        
    Example:
        >>> model = create_baseline_model()
        >>> mobile_model = apply_inverted_residual_optimization(
        ...     model, expand_ratio=6
        ... )
        >>> # Suitable blocks now use mobile-optimized inverted residuals
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    replacements = 0  # Track number of successful replacements

    print(f"Applying mobile inverted residual optimization...")
    
    # TODO: Replaces suitable blocks in the model with InvertedResidual blocks.
    # HINT: Inverted residuals use an expand→depthwise→project pattern as used in MobileNetV2.
    # The "inverted" aspect means we expand channels first (unlike standard residuals that compress).
    # Architecture flow: input → [expand] → depthwise → project → [+residual]
    #
    # Check the MobileNetV2 code at https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py 
    # for a code template, and consider whether to use ReLU or ReLU6 and batchnorm.

    # Add your code here

    # Report optimization status
    if replacements > 0:
        print(f"INVERTED RESIDUALS completed: Successfully applied to layers with {replacements} replacements")
    else:
        print("WARNING: INVERTED RESIDUALS not applied: No suitable layers found for replacement")

    return optimized_model

# --------------------------------------
# LOW-RANK FACTORIZATION MODULES
# --------------------------------------

def apply_lowrank_factorization(
    model: nn.Module,
    layer_names=None,
    rank_ratio: float = 0.5,
    min_channels: int = 64,
):
    """
    Apply low-rank spatial factorization to selected Conv2d layers:
    Conv(kxk) -> Conv(kx1) + Conv(1xk)

    Designed for ResNet-18 late stages.

    Args:
        model: nn.Module
        layer_names: list of prefixes (e.g. ["model.layer4"])
        rank_ratio: fraction of channels used in intermediate rank
        min_channels: minimum channels to consider
    """
    import copy
    import torch.nn as nn

    optimized_model = copy.deepcopy(model)
    replaced = 0

    def _get_parent(root, name):
        parts = name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]

    for name, m in list(optimized_model.named_modules()):
        if not isinstance(m, nn.Conv2d):
            continue

        if layer_names and not any(name.startswith(p) for p in layer_names):
            continue

        # Skip unsuitable convs
        if m.kernel_size != (3, 3):
            continue
        if m.groups != 1:
            continue
        if m.in_channels < min_channels or m.out_channels < min_channels:
            continue

        # Determine rank
        rank_channels = int(m.out_channels * rank_ratio)
        rank_channels = max(rank_channels, m.out_channels // 4)

        # Build factorized conv
        conv_k1 = nn.Conv2d(
            m.in_channels,
            rank_channels,
            kernel_size=(3, 1),
            stride=m.stride,
            padding=(1, 0),
            bias=False,
        )
        conv_1k = nn.Conv2d(
            rank_channels,
            m.out_channels,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
            bias=(m.bias is not None),
        )

        factorized = nn.Sequential(conv_k1, conv_1k)

        # Optional: weight initialization (simple heuristic)
        with torch.no_grad():
            conv_k1.weight.zero_()
            conv_1k.weight.zero_()

        parent, attr = _get_parent(optimized_model, name)
        setattr(parent, attr, factorized)
        replaced += 1

    print(f"LOW-RANK FACTORIZATION applied to {replaced} Conv2d layers")

    return optimized_model

# TODO: Implement this optimization method, if selected in your optimization strategy
# def apply_lowrank_factorization(
#     model: nn.Module,
#     min_params: int = 10_000,
#     rank_ratio: float = 0.25
# ) -> nn.Module:
#     """
#     Apply low-rank factorization to large linear layers for parameter reduction.
    
#     Args:
#         model: Input model to optimize for clinical deployment
#         min_params: Minimum parameter count to consider for factorization
#         rank_ratio: Fraction of minimum dimension to use as factorization rank
    
#     Returns:
#         Model with low-rank factorized linear layers for reduced memory usage
        
#     Note:
#         Only factorizes layers with sufficient parameters to benefit from compression.
#         Rank selection balances compression ratio with accuracy preservation - lower
#         ranks provide more compression but may impact diagnostic performance.
        
#     Example:
#         >>> model = create_baseline_model()
#         >>> compressed_model = apply_lowrank_factorization(
#         ...     model, min_params=5000, rank_ratio=0.5
#         ... )
#         >>> # Large linear layers now use low-rank factorization
#     """
#     # Deep copy model to avoid modifying original
#     optimized_model = copy.deepcopy(model)
#     replacements = 0  # Track number of successful replacements

#     print("Applying low-rank factorization optimization...")

#     # TODO: Factorizes large linear layers into low-rank approximations.
#     # HINT: Low-rank factorization decomposes a large weight matrix W into two smaller matrices U and V
#     # such that W ≈ U @ V. This dramatically reduces parameters while maintaining representational capacity.
#     # Remember that higher rank = better approximation but less compression
#     #
#     # See https://arikpoz.github.io/posts/2025-04-29-low-rank-factorization-in-pytorch-compressing-neural-networks-with-linear-algebra/ 
#     # for explanation and code template, and consider how to initialize parameters with respect to the new rank.

#     # Add your code here

#     # Report optimization status
#     if replacements > 0:
#         print(f"LOW RANK FACTORIZATION completed: Successfully applied to layers with {replacements} replacements")
#     else:
#         print("WARNING: LOW RANK FACTORIZATION not applied: No suitable layers found for replacement")

#     return optimized_model

# --------------------------------------
# CHANNEL OPTIMIZATION FUNCTIONS
# --------------------------------------

def apply_channel_optimization(
    model: nn.Module,
    enable_channels_last: bool = True,
    enable_inplace_relu: bool = True
) -> nn.Module:
    """
    Apply channel-level optimizations for enhanced hardware efficiency.

    - channels_last: sets conv weights to NHWC-friendly memory format and wraps forward to
      make inputs channels_last contiguous.
    - inplace ReLU: reduces activation memory and can improve performance.

    Returns:
        Optimized model.
    """
    optimized_model = copy.deepcopy(model)

    # 1) In-place ReLU conversion
    if enable_inplace_relu:
        relu_replaced = 0
        for name, module in optimized_model.named_modules():
            if isinstance(module, nn.ReLU) and module.inplace is False:
                module.inplace = True
                relu_replaced += 1
        if relu_replaced > 0:
            print(f"   In-place ReLU enabled for {relu_replaced} ReLU layers.")
        else:
            print("   In-place ReLU: no changes (likely already inplace).")

    # 2) channels_last handling
    if enable_channels_last:
        # Convert weights / buffers to channels_last memory format
        optimized_model = optimized_model.to(memory_format=torch.channels_last)

        # Wrap model so input tensors are also channels_last contiguous
        class _ChannelsLastWrapper(nn.Module):
            def __init__(self, inner: nn.Module):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                # Ensure NHWC-friendly layout for conv kernels
                x = x.contiguous(memory_format=torch.channels_last)
                return self.inner(x)

        optimized_model = _ChannelsLastWrapper(optimized_model)
        print("   Channels-last memory format enabled (weights + input conversion).")

    print("CHANNEL OPTIMIZATION completed")
    return optimized_model


# TODO: Implement this optimization method, if selected in your optimization strategy
# def apply_channel_optimization(
#     model: nn.Module,
#     enable_channels_last: bool = True,
#     enable_inplace_relu: bool = True
# ) -> nn.Module:
#     """
#     Apply channel-level optimizations for enhanced hardware efficiency.

#     Implements memory layout and activation optimizations to improve hardware utilization
#     and reduce memory bandwidth requirements.

#     Args:
#         model: Model to optimize for hardware efficiency
#         enable_channels_last: E.g., you'd use NHWC memory layout for faster GPU convolutions
#         enable_inplace_relu: Convert ReLU layers to in-place for memory savings
    
#     Returns:
#         Hardware-optimized model with improved memory efficiency
        
#     Note:
#         The 'channels last' memory format can significantly improve convolution performance on certain hardware 
#         (e.g., modern GPUs with specialized cores) but requires input tensors to be converted...
        
#     Example:
#         >>> model = create_baseline_model()
#         >>> optimized_model = apply_channel_optimization(model)
#         >>> # Remember to convert inputs: input.to(memory_format=torch.channels_last)
#     """
#     # Deep copy model to avoid modifying original
#     optimized_model = copy.deepcopy(model)
    
#     print("Applying channel-level hardware optimizations...")
    
#     # TODO: Applies channel-level optimizations such as memory format changes
#     # and in-place ReLU conversions for better hardware efficiency.
#     # HINT: See https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html for a tutorial on channels last organization,
#     # and note how input needs to be handled for it. 
#     # Also, consider ensuring activations are in place by reviewing https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948/2 
#     # for more details.

#     # Add your code here

#     # Report optimization status
#     print("CHANNEL OPTIMIZATION completed")

#     return optimized_model

# --------------------------------------
# PARAMETER SHARING FUNCTIONS
# --------------------------------------

# TODO: Implement this optimization method, if selected in your optimization strategy
def apply_parameter_sharing(
    model: nn.Module,
    sharing_groups: Optional[List[List[str]]] = None,
    layer_types: Optional[List[Type[nn.Module]]] = None
) -> nn.Module:
    """
    Apply parameter sharing between layers to reduce memory and improve efficiency.

    Shares weight parameters between layers with identical shapes to reduce memory
    footprint and potentially improve generalization. 

    Args:
        model: Model to optimize through parameter sharing
        sharing_groups: Manual specification of layer groups to share parameters.
                       If None, automatically groups layers with identical weight shapes.
        layer_types: Types of layers to consider for parameter sharing 
                    (defaults to Conv2d for maximum impact)
    
    Returns:
        Memory-optimized model with parameter sharing applied
        
    Note:
        Parameter sharing can improve model generalization by enforcing weight
        consistency across similar layers. Most effective when applied to layers
        with identical computational roles and sufficient parameter count.
        
    Example:
        >>> model = create_baseline_model()
        >>> shared_model = apply_parameter_sharing(model)
        >>> # Layers with identical shapes now share parameters
    """    
    # Default to Conv2d layers (largest parameter count and memory footprint)
    if layer_types is None:
        layer_types = [nn.Conv2d]

    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    # Track number of sharing layers and shared parameters
    total_shared = 0
    total_parameters_shared = 0
    
    print("Applying parameter sharing optimization...")

    # TODO: Shares parameters between layers in specified groups to reduce memory and computation.
    # HINT: Parameter sharing involves assigning the same `nn.Parameter` instance to multiple layers
    #
    # See https://stackoverflow.com/questions/57929299/how-to-share-weights-between-modules-in-pytorch 
    # for some inspiration.

    # Add your code here
   
    # Report optimization status
    if total_shared > 0:
        print(f"PARAMETER SHARING completed - Successfully shared parameters for {total_shared} layers")
        print(f"   Total parameters shared: {total_parameters_shared:,}")
    else:
        print("WARNING: PARAMETER SHARING failed - No suitable layer groups found for optimization")
    
    return optimized_model