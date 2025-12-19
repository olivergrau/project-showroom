"""
Graph optimization utilities for PyTorch models.
Supports TorchScript and TorchFX optimizations with a unified API.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.experimental.optimization import (
    fuse,
    remove_dropout,
    optimize_for_inference,
)


def _ensure_device(device: torch.device | str) -> torch.device:
    if isinstance(device, str):
        return torch.device(device)
    return device


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_model(
    model: nn.Module,
    optimization_method: Literal["torchscript", "torch_fx"] = "torchscript",
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: torch.device = torch.device("cuda"),
    custom_options: Optional[Dict[str, Any]] = None,
) -> tuple[nn.Module, nn.Module]:
    """
    Optimize a model using TorchScript or TorchFX graph optimizations.

    This function is intentionally conservative:
    it keeps the original computation semantics and only applies safe
    inference-time transformations like Conv+BN fusion, dropout removal
    and JIT or FX graph canonicalization.

    Args:
        model:
            PyTorch model to optimize. The model is not modified in-place;
            an optimized copy is returned.
        optimization_method:
            One of "torchscript" or "torch_fx".
        input_shape:
            Shape of a single input example used for tracing, for example (1, 3, 32, 32).
        device:
            Device on which to place the model and dummy input for tracing.
        custom_options:
            Optional dictionary with method-specific options, for example:

            TorchScript:
                - "use_script" (bool, default: False)
                - "check_trace" (bool, default: True)

            FX:
                - "use_optimize_for_inference" (bool, default: True)
                - "run_fuse_pass" (bool, default: False)
                - "run_remove_dropout" (bool, default: False)

    Returns:
        Optimized model:
            torch.jit.ScriptModule for TorchScript,
            fx.GraphModule for FX.
    """
    if custom_options is None:
        custom_options = {}

    device = _ensure_device(device)

    # Work on a copy on the requested device
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(input_shape, device=device)

    if optimization_method == "torchscript":
        optimized, serializable = _optimize_with_torchscript(
            model=model,
            dummy_input=dummy_input,
            options=custom_options,
        )
    elif optimization_method == "torch_fx":
        optimized = _optimize_with_fx(
            model=model,
            dummy_input=dummy_input,
            options=custom_options,
            trace_device=device,
        )
    else:
        raise ValueError(
            f"Unsupported optimization_method '{optimization_method}'. "
            "Expected 'torchscript' or 'torch_fx'."
        )

    return optimized, serializable if optimization_method == "torchscript" else optimized


# ---------------------------------------------------------------------------
# TorchScript path
# ---------------------------------------------------------------------------

def _optimize_with_torchscript(
    model: nn.Module,
    dummy_input: torch.Tensor,
    options: Dict[str, Any],
) -> tuple[torch.jit.ScriptModule, torch.jit.ScriptModule]:
    """
    Optimize a model using TorchScript.

    The default is a traced graph with torch.jit.optimize_for_inference.
    Scripting can be enabled via the "use_script" option in options.
    """
    use_script: bool = bool(options.get("use_script", False))
    check_trace: bool = bool(options.get("check_trace", True))

    model.eval()

    with torch.no_grad():
        if use_script:
            try:
                scripted = torch.jit.script(model)
            except Exception as exc:
                warnings.warn(
                    f"torch.jit.script() failed, falling back to trace. Error: {exc}"
                )
                scripted = torch.jit.trace(model, dummy_input, check_trace=check_trace)
        else:
            scripted = torch.jit.trace(model, dummy_input, check_trace=check_trace)

        serializable = scripted
        optimized = torch.jit.optimize_for_inference(scripted)

    return optimized, serializable

# ---------------------------------------------------------------------------
# FX path
# ---------------------------------------------------------------------------

def _optimize_with_fx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    options: Dict[str, Any],
    trace_device: torch.device,
) -> fx.GraphModule:
    """
    Optimize a model using torch.fx graph transformations.

    By default this function:
      - traces the model,
      - optionally fuses Conv+BN(+ReLU),
      - optionally removes Dropout.

    We *do not* call optimize_for_inference by default, because it can
    convert tensors to MKLDNN layout, which breaks for some models
    (e.g. MobileNetV3 blocks with broadcasted elementwise ops) and does
    not move cleanly to CUDA.

    Options:
        - "run_fuse_pass" (bool, default: True)
        - "run_remove_dropout" (bool, default: True)
        - "use_optimize_for_inference" (bool, default: False)
          If set to True, we try optimize_for_inference but this may
          introduce MKLDNN tensors and has known limitations.
    """
    # Defaults: safe passes only
    run_fuse_pass: bool = bool(options.get("run_fuse_pass", True))
    run_remove_do: bool = bool(options.get("run_remove_dropout", True))
    use_opt_for_inference: bool = bool(options.get("use_optimize_for_inference", False))

    original_device = trace_device
    cpu_device = torch.device("cpu")

    # FX tracing is simpler on CPU
    model_cpu = model.to(cpu_device)
    model_cpu.eval()

    try:
        graph_module = fx.symbolic_trace(model_cpu)
    except Exception as exc:
        raise RuntimeError(f"FX symbolic_trace failed: {exc}") from exc

    # Safe manual passes
    if run_fuse_pass:
        try:
            graph_module = fuse(graph_module)
        except Exception as exc:
            warnings.warn(f"FX fuse() failed, continuing without fusion: {exc}")

    if run_remove_do:
        try:
            graph_module = remove_dropout(graph_module)
        except Exception as exc:
            warnings.warn(
                f"FX remove_dropout() failed, continuing without dropout removal: {exc}"
            )

    # Optional, risky pass: may introduce MKLDNN and break broadcasting
    if use_opt_for_inference:
        try:
            # Warning: may cause mkldnn_* errors for some models
            graph_module = optimize_for_inference(graph_module)
        except Exception as exc:
            warnings.warn(
                "optimize_for_inference() failed, returning FX graph without "
                f"that pass. Error was: {exc}"
            )

    # Move back to original device and keep as regular dense tensors
    graph_module.to(original_device)
    graph_module.eval()

    return graph_module


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_model_equivalence(
    original_model: nn.Module,
    optimized_model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: torch.device = torch.device("cpu"),
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> bool:
    """
    Verify that original and optimized models behave numerically the same.

    This performs a single forward pass on a fixed random input and compares
    the outputs with torch.allclose. For most classification models this
    is sufficient as a sanity check.

    Args:
        original_model:
            Original PyTorch model.
        optimized_model:
            Optimized model produced by optimize_model.
        input_shape:
            Shape of the random input used for comparison.
        device:
            Device on which to run the comparison.
        rtol:
            Relative tolerance for torch.allclose.
        atol:
            Absolute tolerance for torch.allclose.

    Returns:
        True if the outputs are numerically close within the tolerances,
        otherwise False.
    """
    device = _ensure_device(device)

    original_model = original_model.to(device)
    optimized_model = optimized_model.to(device)

    original_model.eval()
    optimized_model.eval()

    # Deterministic random input
    torch.manual_seed(0)
    input_tensor = torch.randn(input_shape, device=device)

    with torch.no_grad():
        orig_out = original_model(input_tensor)
        opt_out = optimized_model(input_tensor)

    # Many models return tuples, for example (output, aux_output).
    # We compare the first element in that case.
    if isinstance(orig_out, tuple):
        orig_out = orig_out[0]
    if isinstance(opt_out, tuple):
        opt_out = opt_out[0]

    if not isinstance(orig_out, torch.Tensor) or not isinstance(opt_out, torch.Tensor):
        raise TypeError(
            "verify_model_equivalence currently supports models that return "
            "a single Tensor or a tuple whose first element is a Tensor."
        )

    is_close = torch.allclose(orig_out, opt_out, rtol=rtol, atol=atol)

    if is_close:
        print("Original and optimized models produce equivalent outputs.")
    else:
        diff = torch.abs(orig_out - opt_out)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(
            f"Models differ beyond tolerance. "
            f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
        )

    return is_close
