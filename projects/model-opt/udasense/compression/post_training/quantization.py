"""
udasense Project: Post-Training Quantization Module

This module provides utilities for applying post-training quantization to PyTorch models,
supporting both static and dynamic quantization methods.
"""

import os
import copy
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torch.ao.quantization import fuse_modules  # add this if not present

# Consider whether you want to quantize the whole model or parts of it only


class QuantizableMobileNetV3_Household(nn.Module):
    """Wrapper around the MobileNetV3_Household model that adds a fuse_model method.

    This class is intentionally thin: it reuses the already defined MobileNetV3_Household
    and only adds generic Conv+BN fusion logic where possible.
    """

    def __init__(self, original_model: nn.Module):
        super().__init__()
        # Keep a reference to the underlying model; this is the one that actually
        # contains all Conv / BN / activation layers.
        self.model = original_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Delegate to the underlying model implementation
        return self.model(x)

    def fuse_model(self) -> None:
        """
        Fuse Conv2d + BatchNorm2d (and optionally activation) patterns where possible.

        This is a generic pass: it looks for Conv + BN pairs inside Sequential
        containers. It does not try to be MobileNetV3 specific.
        """
        print("Fusing Conv+BN patterns in MobileNetV3_Household...")

        # Helper to fuse Conv+BN inside an nn.Sequential
        def _fuse_in_sequential(seq: nn.Sequential) -> None:
            prev_name, prev_module = None, None
            for name, module in seq.named_children():
                # Pattern: Conv2d -> BatchNorm2d
                if isinstance(prev_module, nn.Conv2d) and isinstance(module, nn.BatchNorm2d):
                    # Fuse in place inside the Sequential container
                    try:
                        fuse_modules(seq, [prev_name, name], inplace=True)
                        # After fusion, seq._modules[prev_name] holds the fused module
                    except Exception as e:
                        print(f"Warning: fusion of {prev_name} and {name} failed: {e}")
                prev_name, prev_module = name, module

        # Walk through the underlying model and fuse where appropriate
        for name, module in self.model.named_children():
            # Many backbones use Sequential containers for features / blocks
            if isinstance(module, nn.Sequential):
                _fuse_in_sequential(module)

        print("Layer fusion complete.")
        

def quantize_model(
    model: nn.Module,
    calibration_data_loader: Optional[DataLoader] = None,
    calibration_num_batches: Optional[int] = None,
    quantization_type: str = "dynamic",
    backend: str = "fbgemm",
) -> nn.Module:
    """Apply post-training quantization to a PyTorch model.
    
    Args:
        model: The original model to quantize
        calibration_data_loader: DataLoader for calibration data,
            required for static quantization
        calibration_num_batches: Number of batches to run calibration on
        quantization_type: Type of quantization to apply:
            - "dynamic": Dynamic quantization (weights are quantized, activations quantized during inference)
            - "static": Static quantization (weights and activations are pre-quantized)
        backend: Quantization backend, either "fbgemm" (x86) or "qnnpack" (ARM)
            
    Returns:
        Quantized model
        
    Raises:
        ValueError: If an unsupported backend or quantization type is specified,
                   or if static quantization is requested without calibration data
    """
    # Verify backend
    if backend not in ["fbgemm", "qnnpack"]:
        raise ValueError("Backend must be either 'fbgemm' (x86) or 'qnnpack' (ARM)")
    
    # Create a copy of the model for quantization
    model_to_quantize = copy.deepcopy(model)
    
    # Set model to evaluation mode
    model_to_quantize.eval()
    
    # NOTE: Feel free to not implement all quantization types
    # Apply quantization based on type
    if quantization_type.lower() == "dynamic":
        return _apply_dynamic_quantization(model_to_quantize)
    elif quantization_type.lower() == "static":
        if calibration_data_loader is None:
            raise ValueError("Static quantization requires a calibration_data_loader")
        return _apply_static_quantization(model_to_quantize, calibration_data_loader, calibration_num_batches, backend)
    else:
        raise ValueError(f"Unsupported quantization type: {quantization_type}")

# Remember to look at built-in pytorch functionalities whenever possible
def _apply_dynamic_quantization(
    model: nn.Module
) -> nn.Module:
    """Apply dynamic quantization to a model.
    
    Dynamic quantization quantizes weights ahead of time but quantizes activations
    dynamically during inference.
    
    Args:
        model: Model to quantize (in eval mode)
        
    Returns:
        Dynamically quantized model
    """
    print("Applying dynamic quantization...")

    # Select common layers for dynamic quantization
    # Dynamic quantization is typically effective on Linear / LSTM-like layers.
    qconfig_spec = {nn.Linear}

    # Use PyTorch built-in dynamic quantization helper
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        qconfig_spec=qconfig_spec,
        dtype=torch.qint8,
    )

    print("Dynamic quantization complete.")
    return quantized_model
                

# Remember to look at built-in pytorch functionalities whenever possible
# And that you first need to prepare the model for quantization, then apply calibration, and finally convert the model to quantized
def _apply_static_quantization(
    model: nn.Module,
    calibration_data_loader: DataLoader,
    calibration_num_batches: Optional[int] = None,
    backend: str = "fbgemm",
) -> nn.Module:
    """Apply static quantization to a model using provided calibration data.
    
    Static quantization quantizes both weights and activations ahead of time.
    
    Args:
        model: Model to quantize (in eval mode)
        calibration_data_loader: DataLoader for calibration data
        calibration_num_batches: Number of batches to use for calibration
        backend: Quantization backend, either "fbgemm" (x86) or "qnnpack" (ARM)
        
    Returns:
        Statically quantized model
    """
    print(f"Applying static quantization with backend = {backend}")

    if backend not in ["fbgemm", "qnnpack"]:
        raise ValueError("Backend must be either 'fbgemm' or 'qnnpack'")

    torch.backends.quantized.engine = backend

    # Quantization is CPU-only
    model_cpu = model.to("cpu").eval()

    print("Sum BatchNorm2d for original model: " + str(sum(isinstance(m, nn.BatchNorm2d) for m in model_cpu.modules())))

    # Use one batch as example input for FX
    example_inputs, _ = next(iter(calibration_data_loader))
    example_inputs = example_inputs.to("cpu")

    # Get default qconfig mapping for the backend
    qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping(backend)

    # Prepare model (insert observers)
    prepared_model = quantize_fx.prepare_fx(model_cpu, qconfig_mapping, example_inputs)

    print("Sum BatchNorm2d for prepared model: " + str(sum(isinstance(m, nn.BatchNorm2d) for m in prepared_model.modules())))
    print()

    print(type(prepared_model))
    # Should be something like: <class 'torch.fx.graph_module.GraphModule'>

    print(prepared_model.graph.print_tabular())

    # Decide how many batches to use for calibration
    if calibration_num_batches is None:
        calibration_num_batches = len(calibration_data_loader)

    print(f"Running calibration for {calibration_num_batches} batches...")
    prepared_model.eval()
    with torch.inference_mode():
        for i, (inputs, _) in enumerate(calibration_data_loader):
            if i >= calibration_num_batches:
                break
            inputs = inputs.to("cpu")
            _ = prepared_model(inputs)

    # Convert to quantized model
    print("Converting to quantized model...")
    quantized_model = quantize_fx.convert_fx(prepared_model)

    print("Static quantization complete.")
    return quantized_model