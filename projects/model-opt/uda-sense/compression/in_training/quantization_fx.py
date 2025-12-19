"""
udasense Project: Quantization-Aware Training Module

This module provides a quantizable MobileNetV3 model implementation for the household objects 
dataset, along with functions for quantization-aware training and model conversion.
"""

import copy
import os
import time
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from torchvision.models.quantization.mobilenetv3 import _mobilenet_v3_conf, _mobilenet_v3_model
from tqdm import tqdm

import torch.ao.quantization.quantize_fx as quantize_fx

try:
    from torch.ao.quantization import QConfigMapping
except ImportError:
    QConfigMapping = None  # older PyTorch, we handle this below

from utils.model import (
    save_model,
    train_single_epoch,
    validate_single_epoch,
)

class QuantizableMobileNetV3_Household(nn.Module):
    """FX-QAT-ready MobileNetV3 model for the household objects dataset.

    This wrapper creates a MobileNetV3-Small backbone with a modified classifier
    and keeps it in FP32. FX-based QAT will later trace and transform this model.

    Args:
        num_classes: Number of output classes (10 for household objects)
        dropout_rate: Dropout probability in the classifier
        quantize: Kept for API compatibility with the eager module.
                  In the FX variant this flag is ignored and the model is always float.
        pretrained: Whether to load ImageNet pretrained weights
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
        quantize: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()

        if quantize:
            # In FX mode we always start from a float model and let FX handle quantization.
            print(
                "[WARN][QuantizableMobileNetV3_Household-FX] "
                "`quantize=True` is ignored in FX mode; using a float backbone instead."
            )

        # Create a (float) MobileNetV3-Small configuration
        inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small")

        # Build the base model in float: FX-QAT will transform this later
        self.model = _mobilenet_v3_model(
            inverted_residual_setting=inverted_residual_setting,
            last_channel=last_channel,
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None,
            progress=True,
            quantize=False,  # important: always float for FX
        )

        # Adapt classifier to household objects dataset
        last_channel = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(last_channel, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Expects input of shape [B, C, H, W] with CIFAR-like resolution.
        Resizes to 224×224 as expected by MobileNetV3.
        """
        x = torch.nn.functional.interpolate(
            x, size=(224, 224), mode="bilinear", align_corners=False
        )
        return self.model(x)

    def fuse_model(self, is_qat: bool = False) -> "QuantizableMobileNetV3_Household":
        """Fuse Conv+BN(+Activation) in the underlying model if supported.

        For FX QAT this is not strictly required, because FX has its own
        graph-level fusion passes. We keep this method for API compatibility,
        and so that the same class can still be used in eager workflows if needed.
        """
        if hasattr(self.model, "fuse_model"):
            try:
                self.model.fuse_model(is_qat=is_qat)
            except TypeError:
                self.model.fuse_model()
        else:
            print(
                "[WARN][QuantizableMobileNetV3_Household-FX] "
                "Underlying model has no fuse_model(); skipping fusion."
            )

        return self



def _prepare_qat_model(model: nn.Module, backend: str = "fbgemm", example_inputs: Optional[torch.Tensor] = None) -> nn.Module:
    """Prepare model for FX-based quantization-aware training.

    This is the FX counterpart of the original eager-mode _prepare_qat_model.
    The signature is intentionally kept the same so that existing notebook code
    does not need to change.

    Steps:
      1. Set quantization backend (CPU only)
      2. Move model to CPU and put it in train mode
      3. Build a QAT qconfig mapping for FX
      4. Create example_inputs for FX tracing
      5. Call prepare_qat_fx to insert fake-quant and observers

    Args:
        model: Float model to prepare for QAT (will be replaced by an FX GraphModule)
        backend: Quantization backend ("fbgemm" or "qnnpack")
        example_inputs: Optional example inputs for FX tracing (if None, a default CIFAR-like input is used)

    Returns:
        FX QAT-prepared model (GraphModule), ready for QAT training.
    """
    if backend not in ["fbgemm", "qnnpack"]:
        raise ValueError("backend must be either 'fbgemm' or 'qnnpack'")

    # FX / QAT is CPU-only during preparation
    torch.backends.quantized.engine = backend
    model = model.cpu()
    model.train()  # QAT expects train mode

    # Build QAT qconfig mapping for FX
    try:
        # Newer PyTorch versions
        qconfig_mapping = tq.get_default_qat_qconfig_mapping(backend)
    except AttributeError:
        # Fallback for older versions: use global default_qat_qconfig
        default_qconfig = tq.get_default_qat_qconfig(backend)
        if QConfigMapping is not None:
            qconfig_mapping = QConfigMapping().set_global(default_qconfig)
        else:
            # Very old versions: prepare_qat_fx sometimes accepts plain qconfig
            qconfig_mapping = default_qconfig

    # FX needs an example input to trace the graph.
    # Our household model expects CIFAR-like images and internally resizes to 224×224,
    # so a 1×3×32×32 dummy batch is sufficient for tracing.
    # If no example_inputs are given, fall back to a dummy tensor
    if example_inputs is None:
        example_inputs = torch.randn(1, 3, 32, 32)

    print("Preparing model for FX-based QAT (prepare_qat_fx)...")
    prepared_model = quantize_fx.prepare_qat_fx(
        model,
        qconfig_mapping,
        example_inputs,
    )

    return prepared_model


def _convert_qat_model_to_quantized(model: nn.Module) -> nn.Module:
    """Convert an FX QAT model to a fully quantized model for inference.

    This is the FX counterpart of the original eager-mode helper.

    Steps:
      1. Move the model to CPU
      2. Set eval mode
      3. Call convert_fx to replace fake-quant / QAT modules with real quantized kernels

    Args:
        model: FX QAT-trained model (GraphModule) on which prepare_qat_fx was run

    Returns:
        Fully quantized model (still an nn.Module, but with quantized ops) ready for CPU inference.
    """
    # Conversion and quantized kernels are CPU-only
    model = model.cpu()
    model.eval()

    print("Converting FX QAT model to fully quantized model (convert_fx)...")
    quantized_model = quantize_fx.convert_fx(model)

    return quantized_model


def _freeze_bn_stats_in_module(module: nn.Module):
    """Freeze BatchNorm statistics in QAT modules, if supported.

    Many QAT modules (e.g., ConvBn2d, ConvBnReLU2d) implement a `freeze_bn_stats`
    method that stops updating running_mean / running_var. This helper finds such
    modules and calls it.
    """
    if hasattr(module, "freeze_bn_stats"):
        module.freeze_bn_stats()


def train_model_qat(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    training_config: Dict[str, Any],
    checkpoint_path: str,
    backend: str = "fbgemm",
) -> Tuple[nn.Module, Dict[str, Any], float, int]:
    """Train a model using FX-based quantization-aware training.

    This function mirrors the original eager-mode train_model_qat API, but
    internally uses FX (prepare_qat_fx / convert_fx) instead of eager
    prepare_qat / convert.

    Workflow:
      1. Optional initial FP32 training before QAT
      2. FX QAT activation and fine-tuning from epoch `qat_start_epoch`
      3. Observer disabling and batch norm freezing from `freeze_bn_epochs`
      4. Periodic evaluation of a converted quantized copy on CPU
      5. Final conversion of the best FX QAT model to a fully quantized model

    Args:
        model: Float PyTorch model (will be replaced by an FX GraphModule at QAT start)
        train_loader: Training data loader
        test_loader: Test data loader
        training_config: Dictionary containing training configuration:
            - num_epochs
            - criterion
            - optimizer
            - scheduler
            - patience
            - device
            - grad_clip_norm
            - freeze_bn_epochs
            - qat_start_epoch
        checkpoint_path: Path to save the best QAT model (FX QAT form)
        backend: Quantization backend ("fbgemm" for x86, "qnnpack" for ARM)

    Returns:
        Tuple of (quantized_model, training_stats, best_accuracy, best_epoch)
    """

    # ----------------------------
    # Step 1: Extract training configuration
    # ----------------------------
    num_epochs = training_config.get("num_epochs", 100)
    criterion = training_config.get("criterion")
    optimizer = training_config.get("optimizer")
    scheduler = training_config.get("scheduler")
    patience = training_config.get("patience", 5)
    device = training_config.get(
        "device",
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    grad_clip_norm = training_config.get("grad_clip_norm", None)

    # freeze_bn_epochs: epoch index at which we freeze BN and disable observers (if > 0)
    freeze_bn_epochs = training_config.get("freeze_bn_epochs", 0)
    # qat_start_epoch: epoch index at which we switch from pure FP32 to FX QAT
    qat_start_epoch = training_config.get("qat_start_epoch", 0)

    print(f"Training with FX-based QAT for {num_epochs} epochs")
    print(f"QAT start epoch: {qat_start_epoch}, freeze BN / observers at epoch: {freeze_bn_epochs}")
    print(f"QAT backend: {backend}")

    # Initial debug: pure FP32 performance before any QAT manipulation
    model.eval()
    with torch.no_grad():
        baseline_loss, baseline_acc = validate_single_epoch(
            model,
            test_loader,
            criterion,
            device,
            epoch=0,
            num_epochs=1,
        )
    print(
        f"DEBUG[QAT-FX]: pre-QAT baseline eval inside QAT loop: "
        f"loss={baseline_loss:.4f}, acc={baseline_acc:.2f}%"
    )
    model.train()

    # Move initial FP32 model to training device
    model.to(device)

    # Training statistics
    best_accuracy = 0.0
    best_epoch = 0
    training_stats: Dict[str, Any] = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "epoch_time": [],
        "lr": [],
    }

    early_stop_counter = 0
    qat_active = False  # flag: has QAT been activated (prepare_qat_fx called)?

    # ----------------------------
    # Step 2: FX QAT training loop
    # ----------------------------
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Activate FX QAT at the start of the QAT epoch
        if (not qat_active) and (epoch >= qat_start_epoch):
            print(f"\n=== Activating FX QAT at epoch {epoch + 1} ===")

            # 1) Take one real batch from the training loader as example_inputs for FX
            example_inputs, _ = next(iter(train_loader))
            example_inputs = example_inputs.to("cpu")

            # 2) Prepare model for FX QAT on CPU, then move back to training device
            model = model.cpu()
            model = _prepare_qat_model(
                model,
                backend=backend,
                example_inputs=example_inputs,
            )
            model.to(device)

            # 3) Optional debug: evaluate FX QAT-prepared model (still fake-quant) on CPU
            debug_model = copy.deepcopy(model).cpu()
            debug_model.eval()
            with torch.no_grad():
                qat_pre_loss, qat_pre_acc = validate_single_epoch(
                    debug_model,
                    test_loader,
                    criterion,
                    torch.device("cpu"),
                    epoch=epoch,
                    num_epochs=num_epochs,
                )
            print(
                "DEBUG[QAT-FX]: accuracy immediately after prepare_qat_fx, "
                f"before any QAT SGD: acc={qat_pre_acc:.2f}%"
            )

            qat_active = True

            # 4) Optimizer needs to see the updated parameters (FX GraphModule)
            if optimizer is not None:
                print("Re-binding optimizer parameters for FX QAT model...")
                params = [p for p in model.parameters() if p.requires_grad]
                if len(optimizer.param_groups) == 1:
                    optimizer.param_groups[0]["params"] = params
                else:
                    opt_cls = type(optimizer)
                    new_groups = []
                    for group in optimizer.param_groups:
                        group_copy = {k: v for k, v in group.items() if k != "params"}
                        group_copy["params"] = params
                        new_groups.append(group_copy)
                    optimizer = opt_cls(new_groups)

        # Make sure model is in train mode
        model.train()

        # ----------------------------
        # Training for one epoch
        # ----------------------------
        train_loss, train_accuracy = train_single_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip_norm=grad_clip_norm,
            epoch=epoch,
            num_epochs=num_epochs,
        )

        # Disable observers and freeze BN after sufficient QAT training
        if qat_active and (freeze_bn_epochs > 0) and (epoch >= freeze_bn_epochs):
            print(f"Disabling observers and freezing BN stats at epoch {epoch + 1}...")
            model.apply(tq.disable_observer)
            model.apply(_freeze_bn_stats_in_module)
            # Prevent repeating this block
            freeze_bn_epochs = -1

        # ----------------------------
        # Evaluation
        # ----------------------------
        if qat_active and (epoch >= qat_start_epoch):
            # Evaluate a quantized copy on CPU (via convert_fx)
            eval_model = copy.deepcopy(model).cpu()
            eval_model.eval()

            print("Converting FX QAT model copy to quantized model for evaluation...")
            quantized_model = _convert_qat_model_to_quantized(eval_model)

            test_loss, test_accuracy = validate_single_epoch(
                quantized_model,
                test_loader,
                criterion,
                torch.device("cpu"),
                epoch,
                num_epochs,
            )
        else:
            # Evaluate FP32 model
            test_loss, test_accuracy = validate_single_epoch(
                model,
                test_loader,
                criterion,
                device,
                epoch,
                num_epochs,
            )

        # ----------------------------
        # Scheduler step
        # ----------------------------
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

        # Record epoch time and LR
        epoch_time = time.time() - epoch_start_time
        lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0

        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, "
            f"LR: {lr:.6f}, Time: {epoch_time:.2f}s"
        )

        # ----------------------------
        # Best model tracking / early stopping
        # ----------------------------
        # We only care about quantized performance once FX QAT is active
        if qat_active and (epoch >= qat_start_epoch) and (test_accuracy > best_accuracy):
            print(f"New best FX QAT model! Saving... ({test_accuracy:.2f}%)")
            best_accuracy = test_accuracy
            best_epoch = epoch + 1

            # Ensure directory exists
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            # For FX QAT, save only the state_dict (full module cannot be pickled cleanly)
            torch.save(model.state_dict(), checkpoint_path + "best_qat_fx_model.pth")
            print(f"[QAT-FX] Saved best model state_dict to {checkpoint_path + 'best_qat_fx_model.pth'}")

            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}. No improvement for {patience} epochs.")
            break

        # Save statistics
        training_stats["epoch"].append(epoch + 1)
        training_stats["train_loss"].append(train_loss)
        training_stats["train_accuracy"].append(train_accuracy)
        training_stats["test_loss"].append(test_loss)
        training_stats["test_accuracy"].append(test_accuracy)
        training_stats["epoch_time"].append(epoch_time)
        training_stats["lr"].append(lr)

    print(f"Training completed. Best FX QAT accuracy: {best_accuracy:.2f}%")
    print(f"Best FX QAT model saved as '{checkpoint_path}' at epoch {best_epoch}")

    # ----------------------------
    # Step 3: Load best FX QAT model and convert to final quantized model
    # ----------------------------
    print("Converting best FX QAT model to fully quantized model for inference...")
    model.load_state_dict(torch.load(checkpoint_path + "best_qat_fx_model.pth", map_location="cpu"))
    model.eval()
    quantized_model = _convert_qat_model_to_quantized(model)

    return quantized_model, training_stats, best_accuracy, best_epoch


