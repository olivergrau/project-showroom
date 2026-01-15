"""
udasense Project: Quantization-Aware Training Module

This module provides a quantizable MobileNetV3 model implementation for the household objects 
dataset, along with functions for quantization-aware training and model conversion.
"""

import copy
import time
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from torchvision.models.quantization.mobilenetv3 import _mobilenet_v3_conf, _mobilenet_v3_model
from tqdm import tqdm

from utils.model import get_model_size, save_model, train_single_epoch, validate_single_epoch


class QuantizableMobileNetV3_Household(nn.Module):
    """Quantizable MobileNetV3 model for household objects dataset.
    
    This model is designed to be compatible with PyTorch's quantization features,
    including quantization-aware training (QAT).
    
    Attributes:
        model: The underlying MobileNetV3 model with a modified classifier
    """
    
    def __init__(
        self, 
        num_classes: int = 10, 
        dropout_rate: float = 0.2, 
        quantize: bool = False, 
        pretrained: bool = True
    ):
        """Initialize a quantizable MobileNetV3 model.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout probability in the classifier
            quantize: Whether to create a quantization-ready model
            pretrained: Whether to load ImageNet pretrained weights
        """
        super().__init__()
        
        # Create a quantizable MobileNetV3 Small
        inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small")
        self.model = _mobilenet_v3_model(
            inverted_residual_setting=inverted_residual_setting,
            last_channel=last_channel,
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None,
            progress=True,
            quantize=quantize,
        )
        
        # Modify the classifier for the household objects dataset
        last_channel = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(last_channel, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Output tensor of shape [B, num_classes]
        """
        # Resize the image to the format expected by MobileNetV3
        x = torch.nn.functional.interpolate(
            x, size=(224, 224), mode='bilinear', align_corners=False
        )
        return self.model(x)
    
    def fuse_model(self, is_qat: bool = False) -> 'QuantizableMobileNetV3_Household':
        """Fuse operations like Conv+BN+Activation for improved performance and QAT.

        Args:
            is_qat: Whether the fusion is for quantization-aware training.
                    Some backends may ignore this flag, but it is provided
                    for API compatibility.

        Returns:
            Self with fused operations.
        """
        # Delegate to the underlying quantizable MobileNetV3 implementation, if available
        if hasattr(self.model, "fuse_model"):
            # Some torchvision versions accept an is_qat flag, others do not.
            try:
                self.model.fuse_model(is_qat=is_qat)
            except TypeError:
                # Fallback if underlying implementation does not accept is_qat
                self.model.fuse_model()
        else:
            print("Warning: underlying MobileNetV3 model has no fuse_model method. Skipping fusion.")

        return self

def debug_qat_stages(
    model_fp32: nn.Module,
    test_loader,
    backend: str = "fbgemm",
    device: torch.device = torch.device("cpu")
):
    model_fp32.eval()
    model_fp32.to(device)

    # 1) Baseline FP32 accuracy
    acc1 = validate_single_epoch(model_fp32, test_loader,
                                 nn.CrossEntropyLoss(), device,
                                 epoch=0, num_epochs=1)[1]
    print(f"  → Accuracy FP32: {acc1:.2f}%")

    # Clone so we do not destroy the original
    m_fused = copy.deepcopy(model_fp32).cpu()

    # 2) After fuse_model only
    if hasattr(m_fused, "fuse_model"):
        try:
            m_fused.fuse_model(is_qat=False)
        except TypeError:
            m_fused.fuse_model()
    else:
        print("[WARNING] Model has no fuse_model. Skipping fusion stage.")
    
    m_fused.to(device)
    m_fused.eval()

    acc2 = validate_single_epoch(m_fused, test_loader,
                                 nn.CrossEntropyLoss(), device,
                                 epoch=0, num_epochs=1)[1]
    print(f"  → Accuracy after fusion only: {acc2:.2f}%")

    # 3) After prepare_qat (fake quant), still float
    torch.backends.quantized.engine = backend
    m_qat = copy.deepcopy(model_fp32).cpu()
    m_qat.eval()

    # fuse again for the QAT variant
    if hasattr(m_qat, "fuse_model"):
        try:
            m_qat.fuse_model(is_qat=True)
        except TypeError:
            m_qat.fuse_model()
    
    m_qat.train()  # prepare_qat expects model in train mode
    m_qat.qconfig = tq.get_default_qat_qconfig(backend)
    tq.prepare_qat(m_qat, inplace=True)

    m_qat.to(device)
    m_qat.eval()  # we just want inference

    acc3 = validate_single_epoch(m_qat, test_loader,
                                 nn.CrossEntropyLoss(), device,
                                 epoch=0, num_epochs=1)[1]
    print(f"  → Accuracy after prepare_qat (no convert): {acc3:.2f}%")

    # 4) After convert to fully quantized model
    m_quant = copy.deepcopy(m_qat).cpu()
    m_quant.eval()
    m_quant = tq.convert(m_quant, inplace=False)

    acc4 = validate_single_epoch(m_quant, test_loader,
                                 nn.CrossEntropyLoss(), torch.device("cpu"),
                                 epoch=0, num_epochs=1)[1]
    print(f"  → Accuracy after convert: {acc4:.2f}%")

    return acc1, acc2, acc3, acc4


def _prepare_qat_model(model: nn.Module, backend: str = "fbgemm") -> nn.Module:
    """Prepare model for quantization-aware training.

    This performs the necessary steps to convert a regular model into
    a QAT-ready model:
      1. Set quantization backend
      2. Fuse Conv+BN+Activation blocks
      3. Attach a QAT qconfig
      4. Call prepare_qat to insert fake quant and observers

    Args:
        model: Model to prepare for QAT (will be modified in-place)
        backend: Quantization backend ("fbgemm" or "qnnpack")

    Returns:
        Model prepared for QAT (same instance as input)
    """
    if backend not in ["fbgemm", "qnnpack"]:
        raise ValueError("backend must be either 'fbgemm' or 'qnnpack'")

    # Quantization backends are CPU-only; preparation is typically done on CPU
    torch.backends.quantized.engine = backend
    model.cpu()
    model.eval()

    # 1) Fuse modules (Conv+BN+Activation etc.) if supported
    if hasattr(model, "fuse_model"):
        print("Fusing model modules for QAT...")
        # For QAT, we can pass is_qat=True if supported
        try:
            model.fuse_model(is_qat=True)
        except TypeError:
            model.fuse_model()
    else:
        print("Warning: model has no fuse_model() method. Proceeding without fusion.")

    # 2) Attach QAT qconfig
    qat_qconfig = tq.get_default_qat_qconfig(backend)
    model.qconfig = qat_qconfig

    # # DEBUG-3: keep final classifier in FP32
    # # Our wrapper has self.model.classifier as the head
    # if hasattr(model, "model") and hasattr(model.model, "classifier"):
    #     print("DEBUG[QAT]: disabling qconfig for classifier head")
    #     model.model.classifier.qconfig = None

    # After preparation, the model is now QAT-ready. Caller should move it to the desired device.
    model.train()

    # 3) Prepare QAT (insert fake-quant and observers)
    print("Preparing model for QAT (inserting fake quantization modules)...")
    tq.prepare_qat(model, inplace=True)

    return model


def _convert_qat_model_to_quantized(model: nn.Module) -> nn.Module:
    """Convert a QAT model to a fully quantized model for inference.

    This function:
      1. Moves the model to CPU
      2. Sets the appropriate quantized backend
      3. Calls torch.ao.quantization.convert to replace fake-quant ops
         and QAT modules with real quantized kernels.

    Args:
        model: QAT-trained model

    Returns:
        Fully quantized model ready for CPU inference.
    """
    # Conversion and quantized kernels are CPU-only
    model.cpu()
    model.eval()

    # Use the currently configured backend (default: fbgemm)
    # If you want to enforce a specific backend, you can set it here.
    # For now, we respect the engine already set externally.
    print("Converting QAT model to fully quantized model...")
    quantized_model = tq.convert(model, inplace=False)

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
    """Train a model using quantization-aware training.

    This function implements the complete QAT workflow, including:
    1. Initial training before QAT
    2. QAT activation and fine-tuning
    3. Observer disabling and batch norm freezing
    4. Final conversion to a fully quantized model

    Args:
        model: PyTorch model (should support fuse_model method)
        train_loader: Training data loader
        test_loader: Test data loader
        training_config: Dictionary containing training configuration
        checkpoint_path: Path to save the best QAT model (in QAT form)
        backend: Quantization backend ("fbgemm" for x86, "qnnpack" for ARM)

    Returns:
        Tuple of (quantized_model, training_stats, best_accuracy, best_epoch)
    """

    # ----------------------------
    # Step 1: Extract training configuration
    # ----------------------------
    num_epochs = training_config.get('num_epochs', 100)
    criterion = training_config.get('criterion')
    optimizer = training_config.get('optimizer')
    scheduler = training_config.get('scheduler')
    patience = training_config.get('patience', 5)
    device = training_config.get(
        'device',
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    grad_clip_norm = training_config.get('grad_clip_norm', None)
    
    # freeze_bn_epochs: epoch index at which we freeze BN and disable observers (if > 0)
    freeze_bn_epochs = training_config.get('freeze_bn_epochs', 0)
    # qat_start_epoch: epoch index at which we switch from pure FP32 to QAT
    qat_start_epoch = training_config.get('qat_start_epoch', 0)

    print(f"Training with quantization-aware training for {num_epochs} epochs")
    print(f"QAT start epoch: {qat_start_epoch}, freeze BN / observers at epoch: {freeze_bn_epochs}")
    print(f"QAT backend: {backend}")

    # DEBUG-1: At top of train_model_qat, after model.to(device) and before the for-loop
    model.eval()
    with torch.no_grad():
        baseline_loss, baseline_acc = validate_single_epoch(
            model, test_loader, criterion, device, epoch=0, num_epochs=1
        )
    print(f"DEBUG[QAT]: pre-QAT baseline eval inside QAT loop: loss={baseline_loss:.4f}, acc={baseline_acc:.2f}%")
    model.train()

    # Move initial FP32 model to training device
    model.to(device)

    # Training statistics
    best_accuracy = 0.0
    best_epoch = 0
    training_stats = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "epoch_time": [],
        "lr": [],
    }

    early_stop_counter = 0
    qat_active = False  # flag: has QAT been activated (prepare_qat called)?

    # ----------------------------
    # Step 2: QAT training loop
    # ----------------------------
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Activate QAT at the start of the QAT epoch
        if (not qat_active) and (epoch >= qat_start_epoch):
            print(f"\n=== Activating QAT at epoch {epoch+1} ===")
            # Prepare model for QAT on CPU, then move back to training device
            model = model.cpu()
            model = _prepare_qat_model(model, backend=backend)
            model.to(device)

            # DEBUG-2: evaluate QAT-prepared float model on a copy, without any QAT training yet
            debug_model = copy.deepcopy(model).cpu()
            debug_model.eval()
            with torch.no_grad():
                qat_pre_loss, qat_pre_acc = validate_single_epoch(
                    debug_model, test_loader, criterion, torch.device("cpu"),
                    epoch=epoch, num_epochs=num_epochs
                )
            print(f"DEBUG[QAT]: accuracy immediately after prepare_qat, before any QAT SGD: acc={qat_pre_acc:.2f}%")

            qat_active = True

            # Optimizer needs to see the updated parameters (after prepare_qat)
            if optimizer is not None:
                print("Re-binding optimizer parameters for QAT model...")
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

        # Train for one epoch
        train_loss, train_accuracy = train_single_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip_norm=grad_clip_norm, epoch=epoch, num_epochs=num_epochs,
        )

        # Disable observers and freeze BN after sufficient QAT training
        if qat_active and (freeze_bn_epochs > 0) and (epoch >= freeze_bn_epochs):
            print(f"Disabling observers and freezing BN stats at epoch {epoch+1}...")
            model.apply(tq.disable_observer)
            #model.apply(tq.freeze_bn_stats)
            model.apply(_freeze_bn_stats_in_module)
            # Prevent repeating this block
            freeze_bn_epochs = -1

        # ----------------------------
        # Evaluation
        # ----------------------------
        if epoch >= qat_start_epoch and qat_active:
            # Evaluate a quantized copy on CPU
            eval_model = copy.deepcopy(model).cpu()
            eval_model.eval()

            print("Converting QAT model copy to quantized model for evaluation...")
            quantized_model = tq.convert(eval_model, inplace=False)

            test_loss, test_accuracy = validate_single_epoch(
                quantized_model, test_loader, criterion, torch.device("cpu"), epoch, num_epochs
            )
        else:
            # Evaluate FP32 model
            test_loss, test_accuracy = validate_single_epoch(
                model, test_loader, criterion, device, epoch, num_epochs
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
        lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 0.0

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, "
            f"LR: {lr:.6f}, Time: {epoch_time:.2f}s"
        )

        # ----------------------------
        # Best model tracking / early stopping
        # ----------------------------
        # We only care about quantized performance once QAT is active
        if qat_active and (epoch >= qat_start_epoch) and (test_accuracy > best_accuracy):
            print(f"New best QAT model! Saving... ({test_accuracy:.2f}%)")
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            save_model(model, checkpoint_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
            break

        # Save statistics
        training_stats["epoch"].append(epoch + 1)
        training_stats["train_loss"].append(train_loss)
        training_stats["train_accuracy"].append(train_accuracy)
        training_stats["test_loss"].append(test_loss)
        training_stats["test_accuracy"].append(test_accuracy)
        training_stats["epoch_time"].append(epoch_time)
        training_stats["lr"].append(lr)

    print(f"Training completed. Best QAT accuracy: {best_accuracy:.2f}%")
    print(f"Best QAT model saved as '{checkpoint_path}' at epoch {best_epoch}")

    # ----------------------------
    # Step 3: Load best QAT model and convert to final quantized model
    # ----------------------------
    print("Converting best QAT model to fully quantized model for inference...")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    quantized_model = _convert_qat_model_to_quantized(model)

    return quantized_model, training_stats, best_accuracy, best_epoch
