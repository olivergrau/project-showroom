"""
Post-Training Pruning techniques for model compression.
Supports both structured and unstructured pruning without retraining.
"""
import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, Any, Optional, Tuple, Literal, List, Union, Callable

from utils.compression import calculate_sparsity, find_prunable_modules, is_pruned
from utils.model import count_parameters, MobileNetV3_Household

def prune_model(
    model: nn.Module,
    pruning_method: Literal["l1_unstructured", "random_unstructured", "ln_structured", "global_unstructured"] = "l1_unstructured",
    amount: Union[float, int] = 0.3,
    modules_to_prune: Optional[List[Tuple[nn.Module, str]]] = None,
    custom_pruning_fn: Optional[Callable] = None,
    n: Optional[int] = None,
    dim: Optional[int] = None
) -> nn.Module:
    """
    Apply post-training pruning to a model.
    
    Args:
        model: Model to prune
        pruning_method: Type of pruning ("l1_unstructured", "random_unstructured", "ln_structured", "global_unstructured")
        amount: Amount to prune (fraction or absolute number)
        modules_to_prune: Specific modules to prune (optional, default is all Conv2d and Linear layers)
        custom_pruning_fn: Custom pruning function (optional)
        n: Order of the norm for ln_structured pruning (default=1)
        dim: Dimension along which to prune for structured pruning (default=0 for Conv2d output channels, 1 for Linear input features)
        
    Returns:
        Pruned model
    """
    
    # For the user-specified modules, or find all prunable modules (Conv2d and Linear)
    if modules_to_prune is None:
        modules_to_prune = find_prunable_modules(model)
    
    print(f"Pruning {len(modules_to_prune)} modules with method: {pruning_method}, amount: {amount}")
    
    # 1. Print initial sparsity
    initial_sparsity = calculate_sparsity(model)
    print(f"Initial model sparsity: {initial_sparsity:.2f}%")

    if pruning_method == "l1_unstructured":
        _apply_unstructured_pruning(model, modules_to_prune, amount)
    
    elif pruning_method == "random_unstructured":
        _apply_random_unstructured_pruning(model, modules_to_prune, amount)
    
    elif pruning_method == "ln_structured":
        _apply_structured_pruning(model, modules_to_prune, amount, n, dim)
    
    elif pruning_method == "global_unstructured":
        _apply_global_pruning(model, modules_to_prune, amount)
    
    elif custom_pruning_fn is not None:
        # Apply custom pruning function
        custom_pruning_fn(model, modules_to_prune, amount)
        
    else:
        raise ValueError(f"Unsupported pruning method: {pruning_method}")
    
    # 3. Check that model is indeed pruned
    is_pruned(model)
    
    # 4. Print final sparsity
    final_sparsity = calculate_sparsity(model)
    print(f"Final model sparsity: {final_sparsity:.2f}%")
    
    # 5. Remove pruning reparameterization to make pruning permanent
    #    This removes `weight_orig` and `weight_mask` and keeps the pruned `weight`.
    for module, param_name in modules_to_prune:
        # Only attempt to remove if this parameter was pruned
        if hasattr(module, f"{param_name}_orig"):
            try:
                prune.remove(module, param_name)
            except ValueError:
                # Parameter was not pruned or already cleaned up
                pass
    
    return model


def _apply_unstructured_pruning(
    model: nn.Module,
    modules_to_prune: List[Tuple[nn.Module, str]],
    amount: Union[float, int]
) -> nn.Module:
    """
    Apply unstructured (element-wise) L1 pruning to model.
    
    Args:
        model: Model to prune
        modules_to_prune: Modules to prune
        amount: Amount to prune
        
    Returns:
        Pruned model
    """
    if not modules_to_prune:
        print("No modules_to_prune passed to _apply_unstructured_pruning.")
        return model

    # Apply L1 unstructured pruning and then remove reparameterization
    for module, param_name in modules_to_prune:
        # First, apply pruning (this creates weight_orig and weight_mask)
        prune.l1_unstructured(module, name=param_name, amount=amount)

    return model

def _apply_random_unstructured_pruning(
    model: nn.Module,
    modules_to_prune: List[Tuple[nn.Module, str]],
    amount: Union[float, int]
) -> nn.Module:
    """
    Apply random unstructured pruning to model.
    
    Args:
        model: Model to prune
        modules_to_prune: Modules to prune
        amount: Amount to prune
        
    Returns:
        Pruned model
    """
    if not modules_to_prune:
        print("No modules_to_prune passed to _apply_random_unstructured_pruning. Skipping.")
        return model

    for module, param_name in modules_to_prune:
        prune.random_unstructured(module, name=param_name, amount=amount)

    return model

def _apply_structured_pruning(
    model: nn.Module,
    modules_to_prune: List[Tuple[nn.Module, str]],
    amount: Union[float, int],
    n: int = 1,
    dim: Optional[int] = None
) -> nn.Module:
    """
    Apply structured (channel/filter) pruning to model.
    
    Args:
        model: Model to prune
        modules_to_prune: Modules to prune
        amount: Amount to prune
        n: Order of the norm (default=1 for L1 norm)
        dim: Dimension along which to prune (if None, use default based on layer type)
        
    Returns:
        Pruned model
    """
    if not modules_to_prune:
        print("No modules_to_prune passed to _apply_structured_pruning. Skipping.")
        return model

    if n is None:
        n = 1

    for module, param_name in modules_to_prune:
        # Choose default dim if not provided, based on layer type
        local_dim = dim
        if local_dim is None:
            if isinstance(module, nn.Conv2d):
                # Prune entire output channels (filters)
                local_dim = 0
            elif isinstance(module, nn.Linear):
                # Follow your comment: default = 1 for Linear input features
                local_dim = 1
            else:
                # Unsupported module type for structured pruning, skip
                continue

        prune.ln_structured(
            module,
            name=param_name,
            amount=amount,
            n=n,
            dim=local_dim,
        )

    return model

def _apply_global_pruning(
    model: nn.Module,
    modules_to_prune: List[Tuple[nn.Module, str]],
    amount: Union[float, int]
) -> nn.Module:
    """
    Apply global pruning to model.
    
    Args:
        model: Model to prune
        modules_to_prune: Modules to prune
        amount: Amount to prune
        
    Returns:
        Pruned model
    """
    if not modules_to_prune:
        print("No modules_to_prune passed to _apply_global_pruning. Skipping.")
        return model

    # Global pruning takes a list of (module, name) tuples directly
    prune.global_unstructured(
        parameters=modules_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    return model