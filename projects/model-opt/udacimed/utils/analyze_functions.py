## Implement logic for each single technique analysis

import torch
import torch.nn as nn
import inspect
import re

### Implement logic for each single technique analysis 

def analyze_mixed_precision_potential(detailed_results, timing_results, memory_results):
    """
    Analyze mixed precision (FP16) acceleration potential.

    Expects (best case):
      - detailed_results['operation_breakdown'] from profile_with_pytorch_profiler()
      - timing_results available either as:
          * detailed_results['timing_results']
          * detailed_results['timing']
          * global variable `timing_results` in the notebook scope
      - peak memory available either as:
          * detailed_results['memory_profile']['peak_memory_mb']
          * detailed_results['memory_results']['peak_memory_mb']
          * detailed_results['peak_memory_mb']
          * global variable `memory_results` in the notebook scope

    Notes:
      - Operation breakdown is time-percentage by category (sums ~100).
      - Eligible ops for FP16 acceleration are primarily conv + matmul/linear (Tensor Cores on NVIDIA).
      - Speedup is a heuristic and is capped to avoid inflated claims.
    """

    # -------- helpers (safe fetch) --------
    def _get(dct, path, default=None):
        cur = dct
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    # Operation breakdown: percentages of time per op category
    op_breakdown = detailed_results.get("operation_breakdown", {}) if isinstance(detailed_results, dict) else {}
    conv_pct = float(op_breakdown.get("convolution_ops", 0.0) or 0.0)
    mm_pct = float(op_breakdown.get("matrix_multiply_ops", 0.0) or 0.0)

    # Total % (usually ~100, but don’t assume)
    total_pct = float(sum(float(v) for v in op_breakdown.values())) if op_breakdown else 0.0
    if total_pct <= 0.0:
        # No profiler data: cannot estimate coverage, keep conservative defaults
        total_pct = 100.0

    eligible_pct = conv_pct + mm_pct
    coverage_percent = (eligible_pct / total_pct) * 100.0 if total_pct > 0 else 0.0

    # -------- speedup heuristic (conservative, capped) --------
    # Target range from prompt: ~1.8–2.5x for high coverage.
    # We interpolate smoothly:
    # - up to 50% coverage: linearly scale 1.0 -> 1.8
    # - above 50%: use the provided formula, capped at 2.5
    if coverage_percent <= 0:
        estimated_speedup = 1.0
    elif coverage_percent <= 50:
        estimated_speedup = 1.0 + (coverage_percent / 50.0) * 0.8  # 1.0 -> 1.8
    else:
        estimated_speedup = 1.8 + (coverage_percent - 50.0) * 0.014  # hint formula
        estimated_speedup = min(estimated_speedup, 2.5)

    # -------- baseline throughput (try multiple places, then globals) --------

    baseline_throughput = None
    if isinstance(timing_results, dict):
        baseline_throughput = timing_results.get("batch_throughput_samples_per_sec")

    if baseline_throughput is None:
        # notebook-global fallback (works in your current notebook layout)
        try:
            baseline_throughput = timing_results.get("batch_throughput_samples_per_sec")  # noqa: F821
        except Exception:
            baseline_throughput = 0.0

    baseline_throughput = float(baseline_throughput or 0.0)

    # -------- peak memory (try multiple places, then globals) --------
    peak_mem = (
        _get(detailed_results, ["memory_profile", "peak_memory_mb"])
        or _get(detailed_results, ["memory_results", "peak_memory_mb"])
        or detailed_results.get("peak_memory_mb") if isinstance(detailed_results, dict) else None
    )

    if peak_mem is None:
        try:
            peak_mem = memory_results.get("peak_memory_mb")  # noqa: F821
        except Exception:
            peak_mem = 0.0

    peak_mem = float(peak_mem or 0.0)
    estimated_memory_reduction_mb = peak_mem * 0.5  # FP32 -> FP16 storage ~ half

    # Throughput scales approximately with speedup in batch scenarios
    estimated_throughput = baseline_throughput * estimated_speedup if baseline_throughput > 0 else 0.0
    throughput_improvement_percent = (estimated_speedup - 1.0) * 100.0

    analysis = {
        "technique": "Mixed Precision (FP16)",
        # interpret "eligible ops" as eligible time share percentage points
        "mixed_precision_eligible_ops": eligible_pct,
        "mixed_precision_coverage_percent": coverage_percent,
        "estimated_speedup": estimated_speedup,
        "estimated_memory_reduction_mb": estimated_memory_reduction_mb,
        "avg_flop_reduction_percent": 0.0,  # FP16 does not reduce FLOPs, it speeds execution
        "estimated_throughput_samples_sec": estimated_throughput,
        "throughput_improvement_percent": throughput_improvement_percent,
        "sensitivity_risk": (
            "Low to moderate: FP16 inference typically preserves accuracy with proper autocast "
            "and FP32 accumulation, but numerical edge cases can shift logits slightly. "
            "Validate sensitivity at the chosen operating threshold after enabling FP16."
        ),
    }

    return analysis


def analyze_batch_processing_scenarios(model, mixed_precision_speedup, profiler, sample_images):
    """
    Correctly analyze batch scenarios based on profiling.py semantics.

    - Real-time: minimize single-sample latency (single_sample_ms)
    - Batch processing: maximize batch throughput (batch_throughput_samples_per_sec)
    """

    scenarios = {
        "real_time_diagnosis": {
            "optimal_batch_size": None,
            "current_latency_ms": None,
            "mixed_precision_latency_ms": None,
            "use_case": "Emergency diagnosis, single patient processing",
        },
        "batch_processing": {
            "optimal_batch_size": None,
            "current_throughput_samples_sec": None,
            "mixed_precision_throughput_samples_sec": None,
            "use_case": "Screening workflows, research processing",
        },
    }

    # Reasonable exploration set for T4
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    print("   Profiling multiple batch sizes...")
    batch_results = profiler.profile_multiple_batch_sizes(  # noqa: F821
        model, sample_images.shape, batch_sizes  # noqa: F821
    )

    # Collect valid entries
    rows = []
    for key, res in batch_results.items():
        if not isinstance(res, dict) or "error" in res:
            continue
        timing = res.get("timing", {})
        if not isinstance(timing, dict):
            continue

        try:
            bsz = int(key.split("_")[1])
        except Exception:
            continue

        single_ms = timing.get("single_sample_ms", timing.get("mean_ms", None))
        batch_tp = timing.get("batch_throughput_samples_per_sec", None)

        if single_ms is None or batch_tp is None:
            continue

        rows.append(
            {
                "batch_size": bsz,
                "single_sample_ms": float(single_ms),
                "batch_throughput_sps": float(batch_tp),
            }
        )

    if not rows:
        return batch_results, scenarios

    # Real-time optimal: lowest true single-sample latency
    rt = min(rows, key=lambda d: d["single_sample_ms"])
    scenarios["real_time_diagnosis"]["optimal_batch_size"] = rt["batch_size"]
    scenarios["real_time_diagnosis"]["current_latency_ms"] = rt["single_sample_ms"]
    scenarios["real_time_diagnosis"]["mixed_precision_latency_ms"] = rt["single_sample_ms"] / max(float(mixed_precision_speedup), 1e-6)

    # Batch optimal: highest true batch throughput
    bp = max(rows, key=lambda d: d["batch_throughput_sps"])
    scenarios["batch_processing"]["optimal_batch_size"] = bp["batch_size"]
    scenarios["batch_processing"]["current_throughput_samples_sec"] = bp["batch_throughput_sps"]
    scenarios["batch_processing"]["mixed_precision_throughput_samples_sec"] = bp["batch_throughput_sps"] * float(mixed_precision_speedup)

    return batch_results, scenarios


def analyze_grouped_conv_potential(
    model,
    sample_input_shape=(3, 64, 64),
    timing_results=None,
    groups=2,
    device=None,
):
    """
    Analyze which Conv2d layers could benefit from grouped convolutions (groups>1).

    Heuristics:
      - Only consider standard convs (module.groups == 1)
      - Only consider spatial convs (kernel_size > 1) because they dominate FLOPs in ResNet-18
      - Require divisibility: in_channels % groups == 0 and out_channels % groups == 0

    FLOPs approximation:
      MACs ≈ H_out * W_out * C_out * (C_in / groups) * kH * kW
      where MAC is multiply-accumulate count (some people multiply by 2 to call it FLOPs).
      For relative comparisons the factor 2 is irrelevant, so we keep MACs.
    """

    model.eval()

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Provide safe default timing_results
    baseline_throughput = None
    if timing_results is not None:
        baseline_throughput = timing_results.get("throughput_samples_per_sec", None)

    analysis = {
        "technique": "Grouped Convolutions",
        "groups": groups,
        "candidate_layers": [],
        "total_candidates": 0,
        "avg_flop_reduction_percent": 0.0,
        "avg_param_reduction_percent": 0.0,
        "estimated_speedup": 1.0,
        "estimated_memory_reduction_mb": 0.0,
        "estimated_throughput_samples_sec": baseline_throughput,
        "throughput_improvement_percent": 0.0,
        "sensitivity_risk": (
            "Moderate: grouped conv reduces cross-channel mixing. "
            "Low risk for small groups (2–4), higher risk if applied aggressively or early."
        ),
    }

    # --- 1) Gather conv output shapes with hooks (1 forward pass) ---
    conv_out_shapes = {}  # name -> (C_out, H_out, W_out)
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            # out shape: (N, C, H, W)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if hasattr(out, "shape") and len(out.shape) == 4:
                _, c, h, w = out.shape
                conv_out_shapes[name] = (int(c), int(h), int(w))
        return hook

    # Register hooks only on Conv2d
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Run one forward pass with dummy input
    with torch.no_grad():
        dummy = torch.zeros((1, *sample_input_shape), device=device)
        _ = model(dummy)

    for h in hooks:
        h.remove()

    # --- 2) Identify candidates and estimate savings ---
    total_base_params = 0
    total_grouped_params = 0
    total_base_macs = 0
    total_grouped_macs = 0

    cand_param_reductions = []
    cand_flop_reductions = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue

        kH, kW = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)

        # Only focus on spatial convs > 1x1 (dominant cost center in ResNet-18)
        if kH <= 1 and kW <= 1:
            continue

        # Must be standard conv to "convert" to grouped conv in this analysis
        if module.groups != 1:
            continue

        Cin = int(module.in_channels)
        Cout = int(module.out_channels)

        # Divisibility constraints for grouped conv
        if Cin % groups != 0 or Cout % groups != 0:
            continue

        # Need output shape to estimate MACs
        if name not in conv_out_shapes:
            continue
        Cout_out, Hout, Wout = conv_out_shapes[name]

        # Params (ignore bias for simplicity, but we can include it)
        has_bias = module.bias is not None
        base_params = Cout * Cin * kH * kW + (Cout if has_bias else 0)
        grouped_params = Cout * (Cin // groups) * kH * kW + (Cout if has_bias else 0)

        # MACs (bias ignored for MACs)
        base_macs = Hout * Wout * Cout * Cin * kH * kW
        grouped_macs = Hout * Wout * Cout * (Cin // groups) * kH * kW

        param_reduction = (base_params - grouped_params) / base_params
        flop_reduction = (base_macs - grouped_macs) / base_macs

        analysis["candidate_layers"].append({
            "name": name,
            "in_channels": Cin,
            "out_channels": Cout,
            "kernel_size": (kH, kW),
            "output_hw": (Hout, Wout),
            "base_params": int(base_params),
            "grouped_params": int(grouped_params),
            "param_reduction_percent": float(param_reduction * 100.0),
            "base_macs": int(base_macs),
            "grouped_macs": int(grouped_macs),
            "flop_reduction_percent": float(flop_reduction * 100.0),
        })

        total_base_params += base_params
        total_grouped_params += grouped_params
        total_base_macs += base_macs
        total_grouped_macs += grouped_macs

        cand_param_reductions.append(param_reduction)
        cand_flop_reductions.append(flop_reduction)

    analysis["total_candidates"] = len(analysis["candidate_layers"])

    if analysis["total_candidates"] == 0:
        # Nothing to do, return early
        return analysis

    # Average reductions across candidates (simple mean)
    analysis["avg_param_reduction_percent"] = float(sum(cand_param_reductions) / len(cand_param_reductions) * 100.0)
    analysis["avg_flop_reduction_percent"] = float(sum(cand_flop_reductions) / len(cand_flop_reductions) * 100.0)

    # Aggregate reductions across all candidates (weighted by actual size)
    total_param_reduction = (total_base_params - total_grouped_params) / total_base_params
    total_flop_reduction = (total_base_macs - total_grouped_macs) / total_base_macs

    # Memory reduction for params only (float32 assumption) -> MB
    # Note: This is only the candidate conv params, not the whole model.
    bytes_saved = (total_base_params - total_grouped_params) * 4
    analysis["estimated_memory_reduction_mb"] = float(bytes_saved / (1024 ** 2))

    # Estimated speedup: very rough.
    # Real speedup is often smaller due to memory bandwidth limits and kernel launch overhead.
    # Add a conservative efficiency factor.
    hardware_efficiency_factor = 0.7  # conservative: grouped ops sometimes underutilize GPU/CPU
    estimated_speedup = 1.0 / (1.0 - total_flop_reduction * hardware_efficiency_factor)
    analysis["estimated_speedup"] = float(estimated_speedup)

    if baseline_throughput is not None:
        est_tp = baseline_throughput * estimated_speedup
        analysis["estimated_throughput_samples_sec"] = float(est_tp)
        analysis["throughput_improvement_percent"] = float((estimated_speedup - 1.0) * 100.0)

    # Sort candidates by MAC savings (descending), so you see what matters
    analysis["candidate_layers"].sort(
        key=lambda d: (d["base_macs"] - d["grouped_macs"]),
        reverse=True
    )

    return analysis



def analyze_depthwise_separable_potential(
    model,
    sample_input_shape=(3, 64, 64),
    timing_results=None,
    device=None,
    min_channels=16,
    hardware_penalty=0.6,
    skip_stem=True,
    skip_downsample=True,
):
    """
    Analyze which Conv2d layers could benefit from depthwise separable convolutions.

    Depthwise-separable replacement for a standard k×k conv (k>1):
      Standard:
        params = Cin * Cout * k^2 (+ bias)
        MACs   = Hout * Wout * Cin * Cout * k^2
      Separable:
        depthwise k×k: params = Cin * k^2
                       MACs   = Hout * Wout * Cin * k^2
        pointwise 1×1: params = Cin * Cout
                       MACs   = Hout * Wout * Cin * Cout
        total params = Cin*k^2 + Cin*Cout (+ bias terms if used)
        total MACs   = Hout*Wout*(Cin*k^2 + Cin*Cout)

    Notes:
      - We focus on kernel_size > 1 convs and Cin >= min_channels to avoid fragile early layers.
      - Real speedup is often smaller than MAC reduction due to memory access and kernel launch overhead;
        we apply a conservative hardware_penalty (default 0.6).
      - The function uses forward hooks to obtain per-layer output H×W from one dummy forward pass.
    """
    model.eval()

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    baseline_throughput = None
    if timing_results is not None:
        baseline_throughput = timing_results.get("throughput_samples_per_sec", None)

    analysis = {
        "technique": "Depthwise Separable Convolutions",
        "candidate_layers": [],
        "total_candidates": 0,
        "avg_flop_reduction_percent": 0.0,
        "avg_param_reduction_percent": 0.0,
        "estimated_speedup": 1.0,
        "estimated_memory_reduction_mb": 0.0,
        "estimated_throughput_samples_sec": baseline_throughput,
        "throughput_improvement_percent": 0.0,
        "sensitivity_risk": (
            "Higher than grouped conv: depthwise step removes cross-channel spatial interactions. "
            "Highest risk in early layers and downsample blocks; safer in later stages (high channel counts)."
        ),
    }

    # --- 1) Capture output shapes (C_out, H_out, W_out) for each Conv2d via one forward pass ---
    conv_out_shapes = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            if hasattr(out, "shape") and len(out.shape) == 4:
                _, c, h, w = out.shape
                conv_out_shapes[name] = (int(c), int(h), int(w))
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        dummy = torch.zeros((1, *sample_input_shape), device=device)
        _ = model(dummy)

    for h in hooks:
        h.remove()

    # --- 2) Candidate scan + savings estimates ---
    total_base_params = 0
    total_sep_params = 0
    total_base_macs = 0
    total_sep_macs = 0

    cand_param_reductions = []
    cand_flop_reductions = []

    def _risk_label(name: str, Cin: int, stride, Hout: int, Wout: int) -> str:
        # Heuristics reflecting our conceptual discussion
        if skip_stem and (name == "conv1" or name.endswith(".conv1") and "layer" not in name):
            return "high"
        if skip_downsample and "downsample" in name:
            return "high"
        if Cin < 64:
            return "high"
        if stride is not None:
            if isinstance(stride, tuple):
                if stride[0] > 1 or stride[1] > 1:
                    return "high"
            else:
                if stride > 1:
                    return "high"
        if Hout * Wout >= 56 * 56:
            return "medium"
        if Cin >= 256:
            return "low"
        return "medium"

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue

        # optional skips for stem/downsample paths
        if skip_stem and (name == "conv1" or name.endswith(".conv1") and "layer" not in name):
            continue
        if skip_downsample and "downsample" in name:
            continue

        kH, kW = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)

        # Only spatial convs
        if kH <= 1 and kW <= 1:
            continue

        # Only standard convs as replacement candidates
        if module.groups != 1:
            continue

        Cin = int(module.in_channels)
        Cout = int(module.out_channels)

        # Avoid fragile tiny channel counts (per hint)
        if Cin < min_channels:
            continue

        # Need output shape for MAC estimation
        if name not in conv_out_shapes:
            continue
        _, Hout, Wout = conv_out_shapes[name]

        has_bias = module.bias is not None

        # Baseline params / MACs
        base_params = Cout * Cin * kH * kW + (Cout if has_bias else 0)
        base_macs = Hout * Wout * Cout * Cin * kH * kW

        # Depthwise-separable params / MACs
        # Depthwise: Cin * k^2 (typically Cout_dw = Cin)
        # Pointwise: Cin * Cout
        sep_params = (Cin * kH * kW) + (Cin * Cout) + (Cout if has_bias else 0)
        sep_macs = Hout * Wout * ((Cin * kH * kW) + (Cin * Cout))

        param_reduction = (base_params - sep_params) / base_params
        flop_reduction = (base_macs - sep_macs) / base_macs

        risk = _risk_label(name, Cin, module.stride, Hout, Wout)

        analysis["candidate_layers"].append({
            "name": name,
            "in_channels": Cin,
            "out_channels": Cout,
            "kernel_size": (kH, kW),
            "stride": module.stride,
            "output_hw": (Hout, Wout),
            "base_params": int(base_params),
            "separable_params": int(sep_params),
            "param_reduction_percent": float(param_reduction * 100.0),
            "base_macs": int(base_macs),
            "separable_macs": int(sep_macs),
            "flop_reduction_percent": float(flop_reduction * 100.0),
            "risk": risk,
        })

        total_base_params += base_params
        total_sep_params += sep_params
        total_base_macs += base_macs
        total_sep_macs += sep_macs

        cand_param_reductions.append(param_reduction)
        cand_flop_reductions.append(flop_reduction)

    analysis["total_candidates"] = len(analysis["candidate_layers"])
    if analysis["total_candidates"] == 0:
        return analysis

    # Average reductions across candidates (simple mean)
    analysis["avg_param_reduction_percent"] = float(sum(cand_param_reductions) / len(cand_param_reductions) * 100.0)
    analysis["avg_flop_reduction_percent"] = float(sum(cand_flop_reductions) / len(cand_flop_reductions) * 100.0)

    # Aggregate reduction ratio (weighted by actual totals)
    total_param_reduction = (total_base_params - total_sep_params) / total_base_params
    total_flop_reduction = (total_base_macs - total_sep_macs) / total_base_macs

    # Parameter memory saved (float32) in MB, for candidate conv params only
    bytes_saved = (total_base_params - total_sep_params) * 4
    analysis["estimated_memory_reduction_mb"] = float(bytes_saved / (1024 ** 2))

    # Theoretical speedup from MAC reduction, then apply conservative penalty
    theoretical_speedup = 1.0 / (1.0 - total_flop_reduction)
    final_speedup = 1.0 + (theoretical_speedup - 1.0) * hardware_penalty
    analysis["estimated_speedup"] = float(final_speedup)

    if baseline_throughput is not None:
        est_tp = baseline_throughput * final_speedup
        analysis["estimated_throughput_samples_sec"] = float(est_tp)
        analysis["throughput_improvement_percent"] = float((final_speedup - 1.0) * 100.0)

    # Sort by MAC savings descending (most impactful first)
    analysis["candidate_layers"].sort(
        key=lambda d: (d["base_macs"] - d["separable_macs"]),
        reverse=True
    )

    return analysis


def analyze_inverted_residuals_potential(
    model,
    timing_results=None,
    conv_impact=0.6,
    assumed_flop_reduction_percent=60.0,
    max_speedup=1.20,
):
    """
    Analyze potential for *inverted residual block* replacement (expand -> depthwise -> project).

    IMPORTANT (skeptical framing consistent with ResNet-18):
      - ResNet-18 BasicBlocks are NOT inverted residual blocks.
      - Any "potential" here is hypothetical: it assumes you would *replace* some ResNet-style blocks
        with MobileNetV2-style inverted residual blocks and then retrain/fine-tune.
      - FLOP reductions can be large per replaced block, but end-to-end speedups are modest and
        hardware-dependent, so we cap speedup conservatively.

    Candidate heuristic:
      - Identify BasicBlock-like modules that contain >=2 Conv2d layers with 3x3 kernels.
      - These are the natural places where you *could* swap in an inverted residual design.
      - We exclude downsample paths from "easy" candidates because shape changes increase risk.

    Metrics:
      - avg_flop_reduction_percent: use a rule-of-thumb (default 60%) per candidate (as notebook hint suggests)
      - speedup: Amdahl-style using conv_impact and a coverage factor, then capped (<= 1.2x)
      - memory reduction: set to 0.0 by default (uncertain), and explained in sensitivity_risk

    Returns:
      analysis dict
    """
    baseline_throughput = None
    if timing_results is not None:
        baseline_throughput = timing_results.get("throughput_samples_per_sec", None)

    analysis = {
        "technique": "Inverted Residual Blocks",
        "residual_candidates": [],
        "total_candidates": 0,
        "avg_flop_reduction_percent": 0.0,
        "estimated_speedup": 1.0,
        "estimated_memory_reduction_mb": 0.0,
        "estimated_throughput_samples_sec": baseline_throughput,
        "throughput_improvement_percent": 0.0,
        "sensitivity_risk": (
            "High: ResNet-18 blocks are not inverted residuals. Adopting inverted residual patterns "
            "requires architectural replacement and retraining. Per-block FLOP savings can be large, "
            "but end-to-end speedup is hardware dependent and often modest. Peak activation memory may "
            "increase during the expansion phase inside each block."
        ),
        "_assumptions": {
            "assumed_flop_reduction_percent_per_block": float(assumed_flop_reduction_percent),
            "conv_impact": float(conv_impact),
            "max_speedup_cap": float(max_speedup),
            "note": "Memory reduction is left as 0.0 because it can go either way (lower average, higher peak).",
        },
    }

    # --- Identify candidate "residual-like blocks" ---
    # We look for modules that contain at least two 3x3 convs, which is the ResNet BasicBlock pattern.
    # We'll also capture whether the module contains a 'downsample' submodule (higher risk).
    for name, module in model.named_modules():
        # Skip the top-level model itself; we want internal blocks
        if module is model:
            continue

        convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
        if len(convs) < 2:
            continue

        # Count 3x3 convs inside this module
        conv3x3 = [c for c in convs if tuple(c.kernel_size) == (3, 3)]
        if len(conv3x3) < 2:
            continue

        has_downsample = any("downsample" in child_name for child_name, _ in module.named_modules())
        # Basic metadata: use the first 3x3 conv as representative
        rep = conv3x3[0]
        Cin = int(rep.in_channels)
        Cout = int(rep.out_channels)
        stride = rep.stride

        risk = "high" if has_downsample or Cin < 64 else "medium"
        if Cin >= 256 and not has_downsample:
            risk = "medium"  # still not low, because it's a major structural swap

        analysis["residual_candidates"].append({
            "name": name,
            "num_convs_total": len(convs),
            "num_convs_3x3": len(conv3x3),
            "representative_in_channels": Cin,
            "representative_out_channels": Cout,
            "representative_stride": stride,
            "has_downsample": bool(has_downsample),
            "assumed_flop_reduction_percent": float(assumed_flop_reduction_percent),
            "risk": risk,
            "notes": (
                "This is a ResNet-style residual block (typically 2× 3x3 conv). "
                "Potential would come from replacing it with expand→depthwise→project, "
                "then retraining/fine-tuning."
            ),
        })

    analysis["total_candidates"] = len(analysis["residual_candidates"])
    if analysis["total_candidates"] == 0:
        return analysis

    # --- Aggregate metrics ---
    analysis["avg_flop_reduction_percent"] = float(assumed_flop_reduction_percent)

    # coverage factor per notebook hint: min(1.0, total_candidates / 8)
    coverage_factor = min(1.0, analysis["total_candidates"] / 8.0)

    # Estimated speedup (conservative):
    # Use the notebook's idea: speedup = 1 + (avg_reduction/100) * conv_impact * 2.5 * coverage_factor
    # Then cap to avoid unrealistic claims.
    raw_speedup = 1.0 + (assumed_flop_reduction_percent / 100.0) * conv_impact * 2.5 * coverage_factor
    estimated_speedup = min(raw_speedup, max_speedup)
    analysis["estimated_speedup"] = float(estimated_speedup)

    if baseline_throughput is not None:
        est_tp = baseline_throughput * estimated_speedup
        analysis["estimated_throughput_samples_sec"] = float(est_tp)
        analysis["throughput_improvement_percent"] = float((estimated_speedup - 1.0) * 100.0)

    # Order candidates: prefer later-stage blocks (higher channels), and no downsample, as "less risky"
    def _sort_key(c):
        return (
            c["has_downsample"],                 # False first
            -c["representative_in_channels"],    # higher channels first
            c["name"],
        )

    analysis["residual_candidates"].sort(key=_sort_key)

    return analysis


def analyze_lowrank_factorization_potential(
    model,
    batch_size=32,
    timing_results=None,
    rank_ratio=0.5,
    min_weight_elements=10_000,
    hardware_penalty=0.7,
    mm_impact=None,
):
    """
    Analyze linear layers that could benefit from low-rank factorization.

    We keep this intentionally conservative for ResNet-18:
      - In many ResNet-18 fine-tuning setups, the only Linear layer is the classifier head
        (often small, e.g., 512 -> 2), so improvements are usually negligible.
      - Low-rank factorization is not a toggle. It typically requires architectural replacement
        (two Linear layers) and fine-tuning to recover accuracy.

    Candidate criteria:
      - nn.Linear with in_features * out_features > min_weight_elements

    Per-layer estimate:
      Standard params:  in*out (+ bias)
      Factorized params (rank r): in*r + r*out (+ bias kept on the second layer)
      Standard MACs per batch: batch_size * in*out
      Factorized MACs per batch: batch_size * (in*r + r*out)

    Speedup estimate:
      - bounded by the fraction of runtime spent in matmuls (mm_impact). If unknown, default small.
      - apply a hardware_penalty because two GEMMs may not be faster than one, plus overhead.

    Args:
      model: torch.nn.Module
      batch_size: batch size for MAC estimation
      timing_results: optional dict with 'throughput_samples_per_sec'
      rank_ratio: fraction of min(in_features, out_features) used as rank (e.g., 0.5)
      min_weight_elements: threshold for in*out to consider layer "large"
      hardware_penalty: conservative factor applied to theoretical speedup delta (default 0.7)
      mm_impact: optional fraction [0..1] for how much inference time is dominated by matrix multiplies.
                 If None, defaults to 0.05 (small for ResNet-18).

    Returns:
      analysis dict
    """
    baseline_throughput = None
    if timing_results is not None:
        baseline_throughput = timing_results.get("throughput_samples_per_sec", None)

    if mm_impact is None:
        # In ResNet-18, matmul (Linear) is typically a tiny part of runtime.
        mm_impact = 0.05

    analysis = {
        "technique": "Linear Layer Low-Rank Factorization",
        "factorization_candidates": [],
        "total_candidates": 0,
        "avg_param_reduction_percent": 0.0,
        "avg_flop_reduction_percent": 0.0,
        "estimated_speedup": 1.0,
        "estimated_memory_reduction_mb": 0.0,
        "estimated_throughput_samples_sec": baseline_throughput,
        "throughput_improvement_percent": 0.0,
        "sensitivity_risk": (
            "Moderate–High: factorizing layers changes the model graph (two Linear ops instead of one) "
            "and typically requires fine-tuning to recover accuracy. Benefits are often negligible in "
            "ResNet-18 because linear layers are small."
        ),
        "_assumptions": {
            "rank_ratio": float(rank_ratio),
            "min_weight_elements": int(min_weight_elements),
            "hardware_penalty": float(hardware_penalty),
            "mm_impact": float(mm_impact),
            "batch_size": int(batch_size),
        },
    }

    cand_param_reductions = []
    cand_flop_reductions = []

    total_base_params = 0
    total_fact_params = 0
    total_base_macs = 0
    total_fact_macs = 0

    # --- 1) Find candidate Linear layers ---
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        in_f = int(module.in_features)
        out_f = int(module.out_features)
        weight_elems = in_f * out_f

        if weight_elems <= min_weight_elements:
            continue

        min_dim = min(in_f, out_f)
        # rank must be at least 1 and at most min_dim-1 to actually reduce
        r = int(min_dim * rank_ratio)
        r = max(1, min(r, min_dim - 1))

        has_bias = module.bias is not None

        # Params: standard vs factorized (bias typically kept on second layer)
        base_params = in_f * out_f + (out_f if has_bias else 0)
        fact_params = (in_f * r) + (r * out_f) + (out_f if has_bias else 0)

        # MACs (batch) - relative comparisons, factor 2 for FLOPs omitted
        base_macs = batch_size * in_f * out_f
        fact_macs = batch_size * ((in_f * r) + (r * out_f))

        param_reduction = (base_params - fact_params) / base_params
        flop_reduction = (base_macs - fact_macs) / base_macs

        analysis["factorization_candidates"].append({
            "name": name,
            "in_features": in_f,
            "out_features": out_f,
            "weight_elements": int(weight_elems),
            "chosen_rank": int(r),
            "base_params": int(base_params),
            "factorized_params": int(fact_params),
            "param_reduction_percent": float(param_reduction * 100.0),
            "base_macs_batch": int(base_macs),
            "factorized_macs_batch": int(fact_macs),
            "flop_reduction_percent": float(flop_reduction * 100.0),
            "notes": (
                "Replacement would require nn.Sequential(Linear(in->r), Linear(r->out)) "
                "and fine-tuning for accuracy recovery."
            ),
        })

        cand_param_reductions.append(param_reduction)
        cand_flop_reductions.append(flop_reduction)

        total_base_params += base_params
        total_fact_params += fact_params
        total_base_macs += base_macs
        total_fact_macs += fact_macs

    analysis["total_candidates"] = len(analysis["factorization_candidates"])
    if analysis["total_candidates"] == 0:
        # Likely outcome for ResNet-18 with small head: nothing meets threshold
        return analysis

    # --- 2) Aggregate metrics ---
    analysis["avg_param_reduction_percent"] = float(sum(cand_param_reductions) / len(cand_param_reductions) * 100.0)
    analysis["avg_flop_reduction_percent"] = float(sum(cand_flop_reductions) / len(cand_flop_reductions) * 100.0)

    # Parameter memory savings (FP32 assumption)
    bytes_saved = (total_base_params - total_fact_params) * 4
    analysis["estimated_memory_reduction_mb"] = float(bytes_saved / (1024 ** 2))

    # Theoretical speedup for matmul portion only, then apply penalty and bound by mm_impact.
    total_flop_reduction = (total_base_macs - total_fact_macs) / total_base_macs
    # Amdahl-like: only mm_impact fraction benefits, scaled by hardware_penalty
    # effective_improvement = mm_impact * hardware_penalty * total_flop_reduction
    effective_improvement = mm_impact * hardware_penalty * total_flop_reduction
    estimated_speedup = 1.0 / (1.0 - effective_improvement)
    analysis["estimated_speedup"] = float(estimated_speedup)

    if baseline_throughput is not None:
        est_tp = baseline_throughput * estimated_speedup
        analysis["estimated_throughput_samples_sec"] = float(est_tp)
        analysis["throughput_improvement_percent"] = float((estimated_speedup - 1.0) * 100.0)

    # Sort by MAC savings to show what matters
    analysis["factorization_candidates"].sort(
        key=lambda d: (d["base_macs_batch"] - d["factorized_macs_batch"]),
        reverse=True
    )

    return analysis

def analyze_conv_lowrank_factorization_potential(
    model,
    sample_input_shape=(3, 64, 64),
    timing_results=None,
    rank_ratio=0.5,
    min_channels=16,
    hardware_penalty=0.65,
    skip_stem=True,
    skip_downsample=True,
    factorization_variant="1x1_then_kxk",
):
    """
    Analyze Conv2d layers for *channel* low-rank factorization potential.

    Concept:
      Standard k×k conv: Conv2d(Cin -> Cout, k×k)
      Approximate with two convs using intermediate rank r:
        Variant A (common bottleneck): 1×1 Cin->r, then k×k r->Cout
        Variant B: k×k Cin->r, then 1×1 r->Cout

      This reflects low-rank structure in the (Cin*k^2) -> Cout mapping.

    Candidate selection:
      - Conv2d with kernel_size > 1 (usually 3×3)
      - module.groups == 1 (standard convs)
      - Cin >= min_channels
      - optionally skip stem conv1 and downsample paths

    Per-layer estimates (MACs use output Hout*Wout measured by hooks):
      Standard params:
        P_std = Cout * Cin * k^2 (+ bias)
      Standard MACs:
        M_std = Hout*Wout * Cout * Cin * k^2

      Factorized params / MACs depend on variant:

      Variant A: 1×1 Cin->r, then k×k r->Cout
        P_fact = (r*Cin*1) + (Cout*r*k^2) (+ bias on last conv if desired)
        M_fact = Hout*Wout * (r*Cin + Cout*r*k^2)

      Variant B: k×k Cin->r, then 1×1 r->Cout
        P_fact = (r*Cin*k^2) + (Cout*r*1) (+ bias on last conv if desired)
        M_fact = Hout*Wout * (r*Cin*k^2 + Cout*r)

    Speedup estimate:
      - Theoretical speedup from MAC reduction
      - Apply conservative penalty (hardware_penalty) because two convs can introduce overhead.

    Returns:
      analysis dict similar to your other architecture analyses.
    """
    model.eval()

    baseline_throughput = None
    if timing_results is not None:
        baseline_throughput = timing_results.get("throughput_samples_per_sec", None)

    analysis = {
        "technique": "Conv Low-Rank Factorization (Channel)",
        "rank_ratio": float(rank_ratio),
        "variant": factorization_variant,
        "candidate_layers": [],
        "total_candidates": 0,
        "avg_param_reduction_percent": 0.0,
        "avg_flop_reduction_percent": 0.0,
        "estimated_speedup": 1.0,
        "estimated_memory_reduction_mb": 0.0,
        "estimated_throughput_samples_sec": baseline_throughput,
        "throughput_improvement_percent": 0.0,
        "sensitivity_risk": (
            "Moderate–High: low-rank factorization changes the block structure (two convs instead of one) "
            "and typically requires fine-tuning to recover accuracy. Highest risk in early layers and "
            "downsample blocks; safer in later stages (high channel counts)."
        ),
        "_assumptions": {
            "min_channels": int(min_channels),
            "hardware_penalty": float(hardware_penalty),
            "skip_stem": bool(skip_stem),
            "skip_downsample": bool(skip_downsample),
        }
    }

    # --- 1) Capture output shapes for Conv2d layers via forward hooks ---
    device = None
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    conv_out_shapes = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            if hasattr(out, "shape") and len(out.shape) == 4:
                _, c, h, w = out.shape
                conv_out_shapes[name] = (int(c), int(h), int(w))
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        dummy = torch.zeros((1, *sample_input_shape), device=device)
        _ = model(dummy)

    for h in hooks:
        h.remove()

    # --- 2) Candidate scan + savings ---
    cand_param_reductions = []
    cand_flop_reductions = []

    total_base_params = 0
    total_fact_params = 0
    total_base_macs = 0
    total_fact_macs = 0

    def _is_stem(name: str) -> bool:
        # Conservative: treat the top-level 'conv1' as stem
        return name == "conv1"

    def _risk_label(name: str, Cin: int, stride, Hout: int, Wout: int) -> str:
        # Heuristic aligned with earlier discussions
        if "downsample" in name:
            return "high"
        if Cin < 64:
            return "high"
        if stride is not None:
            if isinstance(stride, tuple):
                if stride[0] > 1 or stride[1] > 1:
                    return "high"
            else:
                if stride > 1:
                    return "high"
        if Hout * Wout >= 56 * 56:
            return "medium"
        if Cin >= 256:
            return "low"
        return "medium"

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue

        if skip_stem and _is_stem(name):
            continue
        if skip_downsample and "downsample" in name:
            continue

        if module.groups != 1:
            continue

        kH, kW = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        if kH <= 1 and kW <= 1:
            continue

        Cin = int(module.in_channels)
        Cout = int(module.out_channels)

        if Cin < min_channels:
            continue

        if name not in conv_out_shapes:
            continue
        _, Hout, Wout = conv_out_shapes[name]

        has_bias = module.bias is not None

        # Choose rank r based on min(Cin*k^2, Cout) or min(Cin, Cout)?
        # For channel bottleneck, a common practical choice is r = int(min(Cin, Cout) * rank_ratio)
        r = int(min(Cin, Cout) * rank_ratio)
        r = max(1, min(r, min(Cin, Cout) - 1))

        # Standard params / MACs
        base_params = Cout * Cin * kH * kW + (Cout if has_bias else 0)
        base_macs = Hout * Wout * Cout * Cin * kH * kW

        # Factorized params / MACs
        if factorization_variant == "kxk_then_1x1":
            # k×k Cin->r, then 1×1 r->Cout
            # Bias typically on last layer only (keeps output bias count Cout)
            fact_params = (r * Cin * kH * kW) + (Cout * r) + (Cout if has_bias else 0)
            fact_macs = Hout * Wout * ((r * Cin * kH * kW) + (Cout * r))
        else:
            # 1×1 Cin->r, then k×k r->Cout  (default)
            fact_params = (r * Cin) + (Cout * r * kH * kW) + (Cout if has_bias else 0)
            fact_macs = Hout * Wout * ((r * Cin) + (Cout * r * kH * kW))

        # If factorization doesn't actually reduce, skip it
        if fact_params >= base_params or fact_macs >= base_macs:
            continue

        param_reduction = (base_params - fact_params) / base_params
        flop_reduction = (base_macs - fact_macs) / base_macs

        risk = _risk_label(name, Cin, module.stride, Hout, Wout)

        analysis["candidate_layers"].append({
            "name": name,
            "in_channels": Cin,
            "out_channels": Cout,
            "kernel_size": (kH, kW),
            "stride": module.stride,
            "output_hw": (Hout, Wout),
            "chosen_rank": int(r),
            "base_params": int(base_params),
            "factorized_params": int(fact_params),
            "param_reduction_percent": float(param_reduction * 100.0),
            "base_macs": int(base_macs),
            "factorized_macs": int(fact_macs),
            "flop_reduction_percent": float(flop_reduction * 100.0),
            "risk": risk,
            "notes": (
                f"Replace with two convs ({factorization_variant}), then fine-tune to recover accuracy."
            )
        })

        cand_param_reductions.append(param_reduction)
        cand_flop_reductions.append(flop_reduction)

        total_base_params += base_params
        total_fact_params += fact_params
        total_base_macs += base_macs
        total_fact_macs += fact_macs

    analysis["total_candidates"] = len(analysis["candidate_layers"])
    if analysis["total_candidates"] == 0:
        return analysis

    # --- 3) Aggregate metrics ---
    analysis["avg_param_reduction_percent"] = float(sum(cand_param_reductions) / len(cand_param_reductions) * 100.0)
    analysis["avg_flop_reduction_percent"] = float(sum(cand_flop_reductions) / len(cand_flop_reductions) * 100.0)

    # Parameter memory saved (FP32) in MB for candidate layers only
    bytes_saved = (total_base_params - total_fact_params) * 4
    analysis["estimated_memory_reduction_mb"] = float(bytes_saved / (1024 ** 2))

    # Theoretical speedup from MAC reduction, then conservative penalty on the *delta*
    total_flop_reduction = (total_base_macs - total_fact_macs) / total_base_macs
    theoretical_speedup = 1.0 / (1.0 - total_flop_reduction)
    final_speedup = 1.0 + (theoretical_speedup - 1.0) * hardware_penalty
    analysis["estimated_speedup"] = float(final_speedup)

    if baseline_throughput is not None:
        est_tp = baseline_throughput * final_speedup
        analysis["estimated_throughput_samples_sec"] = float(est_tp)
        analysis["throughput_improvement_percent"] = float((final_speedup - 1.0) * 100.0)

    # Sort by MAC savings descending
    analysis["candidate_layers"].sort(
        key=lambda d: (d["base_macs"] - d["factorized_macs"]),
        reverse=True
    )

    return analysis


def analyze_channel_organization_potential(model, timing_results=None):
    """
    Analyze channel organization optimizations for hardware efficiency.

    Focus:
      1) In-place ReLU opportunities (reduce temporary allocations / memory traffic)
      2) channels_last memory format potential (improve conv kernel efficiency on many GPUs)

    Notes:
      - No FLOP reduction (same math), just more efficient execution.
      - Real speedups are hardware/backend dependent. We provide conservative heuristic estimates.
      - This is an *analysis* function: it does not modify the model.

    Args:
      model: torch.nn.Module
      timing_results: optional dict with 'throughput_samples_per_sec'

    Returns:
      analysis dict
    """
    baseline_throughput = None
    if timing_results is not None:
        baseline_throughput = timing_results.get("throughput_samples_per_sec", None)

    # ---- 1) Count in-place opportunities (ReLU) ----
    relu_total = 0
    relu_not_inplace = 0
    relu_inplace = 0

    # Also count other potentially in-place activations, if present
    activations_total = 0
    activations_inplace_capable = 0

    for m in model.modules():
        if isinstance(m, nn.ReLU):
            relu_total += 1
            if getattr(m, "inplace", False):
                relu_inplace += 1
            else:
                relu_not_inplace += 1
        # Extendable: some activations (e.g., LeakyReLU, ELU) have inplace options
        if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SiLU)):
            activations_total += 1
            if hasattr(m, "inplace"):
                activations_inplace_capable += 1

    inplace_opportunities = {
        "relu_total": relu_total,
        "relu_inplace": relu_inplace,
        "relu_not_inplace": relu_not_inplace,
        "activation_layers_total": activations_total,
        "activation_layers_inplace_capable": activations_inplace_capable,
        "estimated_layers_convertible_to_inplace": relu_not_inplace,  # conservative
    }

    # ---- 2) Heuristic speedup estimate ----
    # channels_last: small, hardware-dependent uplift; use conservative 1.05–1.10 range.
    # We'll pick 1.07 as a default baseline estimate.
    base_channels_last_speedup = 1.07

    # in-place ReLU: small incremental improvements; heuristic from notebook hint.
    # speedup *= (1.0 + relu_count * 0.008) but clamp to avoid unrealistic growth.
    # Use only the *convertible* (non-inplace) ReLUs.
    relu_delta = 1.0 + (relu_not_inplace * 0.008)
    relu_delta = min(relu_delta, 1.15)  # hard clamp

    estimated_speedup = base_channels_last_speedup * relu_delta

    # ---- 3) Memory reduction estimate ----
    # This is runtime/working memory reduction from fewer intermediate allocations, not disk size.
    # Without runtime profiling, give 0 and explain via sensitivity_risk / notes.
    estimated_memory_reduction_mb = 0.0

    analysis = {
        "technique": "Channel Organization (channels_last + in-place ReLU)",
        "inplace_opportunities": inplace_opportunities,
        "estimated_speedup": float(estimated_speedup),
        "estimated_memory_reduction_mb": float(estimated_memory_reduction_mb),
        "avg_flop_reduction_percent": 0.0,
        "estimated_throughput_samples_sec": baseline_throughput,
        "throughput_improvement_percent": 0.0,
        "sensitivity_risk": (
            "Low: these optimizations preserve model math. channels_last performance gains are hardware/backend dependent. "
            "In-place ReLU is generally safe but must be validated if tensors are reused (autograd or shared references)."
        ),
        "_assumptions": {
            "base_channels_last_speedup": base_channels_last_speedup,
            "relu_inplace_multiplier_per_layer": 0.008,
            "relu_inplace_multiplier_clamp": 1.15,
            "note": "Memory reduction MB is left at 0.0 without runtime profiling; improvements are mainly fewer temporaries and better memory access patterns."
        }
    }

    if baseline_throughput is not None:
        est_tp = baseline_throughput * estimated_speedup
        analysis["estimated_throughput_samples_sec"] = float(est_tp)
        analysis["throughput_improvement_percent"] = float((estimated_speedup - 1.0) * 100.0)

    return analysis



def analyze_parameter_sharing_potential(
    model,
    timing_results=None,
    channel_threshold=0,
    max_sharing_cap=0.25,
    fp32_bytes_per_param=4,
):
    """
    Analyze potential for parameter sharing across similar Conv2d layers.

    Design goals (skeptical + notebook-friendly):
      - Treat parameter sharing primarily as *parameter memory* reduction.
      - Assume ~0 FLOP reduction (sharing does not remove convolution ops).
      - Any speedup is small and uncertain; we keep it conservative and capped.

    Grouping heuristic:
      - Group conv layers by "similar shape":
          same kernel_size, stride, groups
          and |Cin - Cin'| <= channel_threshold
          and |Cout - Cout'| <= channel_threshold
        If channel_threshold == 0, this becomes exact-match grouping by (Cin, Cout, k, stride, groups).

    Sharing potential:
      - For each group, assume we could share one "canonical" conv's weights and tie the rest.
      - Shareable params = sum(params of all but the largest (or first) layer in each group.
      - sharing_potential_percent = min(max_sharing_cap, shareable_params / total_params) * 100

    Speedup:
      - We keep FLOP reduction ~0.
      - Use the notebook's speedup form, but apply a 0.5x penalty (implementation complexity),
        and cap the total uplift (sharing rarely speeds inference meaningfully).
        speedup = 1 + sharing_potential * 0.4 * 0.5 = 1 + sharing_potential * 0.2
        cap at 1.10.

    Returns:
      analysis dict
    """
    baseline_throughput = None
    if timing_results is not None:
        baseline_throughput = timing_results.get("throughput_samples_per_sec", None)

    # Count total params (all) and conv params (informational)
    total_params = sum(p.numel() for p in model.parameters())
    conv_params = 0

    # Collect conv layers with metadata
    conv_layers = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            p = m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
            conv_params += p
            k = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
            s = m.stride if isinstance(m.stride, tuple) else (m.stride, m.stride)
            conv_layers.append({
                "name": name,
                "in_ch": int(m.in_channels),
                "out_ch": int(m.out_channels),
                "kernel": (int(k[0]), int(k[1])),
                "stride": (int(s[0]), int(s[1])),
                "groups": int(m.groups),
                "params": int(p),
            })

    # Helper: decide if two convs are "similar enough" to group
    def similar(a, b):
        if a["kernel"] != b["kernel"]:
            return False
        if a["stride"] != b["stride"]:
            return False
        if a["groups"] != b["groups"]:
            return False
        if abs(a["in_ch"] - b["in_ch"]) > channel_threshold:
            return False
        if abs(a["out_ch"] - b["out_ch"]) > channel_threshold:
            return False
        return True

    # Greedy grouping
    groups = []
    used = set()
    for i, layer in enumerate(conv_layers):
        if i in used:
            continue
        group = [layer]
        used.add(i)
        for j in range(i + 1, len(conv_layers)):
            if j in used:
                continue
            if similar(layer, conv_layers[j]):
                group.append(conv_layers[j])
                used.add(j)
        if len(group) > 1:
            groups.append(group)

    # Estimate shareable params: in each group, keep one canonical layer, tie the rest
    # We choose the largest-param conv in the group as canonical (conservative if threshold > 0).
    shareable_params = 0
    similar_layer_groups = []

    for g in groups:
        g_sorted = sorted(g, key=lambda d: d["params"], reverse=True)
        canonical = g_sorted[0]
        duplicates = g_sorted[1:]
        dup_params = sum(d["params"] for d in duplicates)
        shareable_params += dup_params

        similar_layer_groups.append({
            "signature": {
                "kernel": canonical["kernel"],
                "stride": canonical["stride"],
                "groups": canonical["groups"],
                "in_ch": canonical["in_ch"],
                "out_ch": canonical["out_ch"],
                "channel_threshold": channel_threshold,
            },
            "canonical_layer": canonical["name"],
            "tied_layers": [d["name"] for d in duplicates],
            "num_layers": len(g_sorted),
            "shareable_params": int(dup_params),
            "shareable_params_mb_fp32": float(dup_params * fp32_bytes_per_param / (1024 ** 2)),
            "notes": (
                "This assumes weights could be tied across these layers. In practice, tying across depth "
                "reduces specialization and usually requires retraining. BatchNorm is typically NOT shared."
            ),
        })

    # Sharing potential (as fraction of total params), capped
    sharing_fraction = (shareable_params / total_params) if total_params > 0 else 0.0
    sharing_fraction_capped = min(max_sharing_cap, sharing_fraction)

    # Conservative: parameter sharing does not reduce MACs, so keep FLOP reduction ~0
    avg_flop_reduction_percent = 0.0

    # Speedup estimate (very conservative + capped)
    estimated_speedup = 1.0 + sharing_fraction_capped * 0.4 * 0.5  # notebook form with 0.5x penalty
    estimated_speedup = min(estimated_speedup, 1.10)

    # Parameter memory saved (FP32) in MB
    estimated_memory_reduction_mb = float(shareable_params * fp32_bytes_per_param / (1024 ** 2))

    analysis = {
        "technique": "Parameter Sharing",
        "similar_layer_groups": similar_layer_groups,
        "sharing_potential_percent": float(sharing_fraction_capped * 100.0),
        "avg_flop_reduction_percent": float(avg_flop_reduction_percent),
        "estimated_speedup": float(estimated_speedup),
        "estimated_memory_reduction_mb": float(estimated_memory_reduction_mb),
        "estimated_throughput_samples_sec": baseline_throughput,
        "throughput_improvement_percent": 0.0,
        "sensitivity_risk": (
            "High (conceptual): parameter sharing is an architectural constraint, not a compute optimization. "
            "It reduces parameter memory but typically does not reduce FLOPs/MACs. Accuracy risk is non-trivial "
            "because blocks lose the ability to specialize across depth; usually requires retraining. "
            "Any runtime speedup is uncertain and likely small (cache effects), so estimates are conservative."
        ),
        "_assumptions": {
            "channel_threshold": int(channel_threshold),
            "max_sharing_cap": float(max_sharing_cap),
            "fp32_bytes_per_param": int(fp32_bytes_per_param),
            "conv_param_ratio_percent": float((conv_params / total_params) * 100.0) if total_params > 0 else 0.0,
            "total_params": int(total_params),
            "conv_params": int(conv_params),
            "shareable_params_raw": int(shareable_params),
            "note": "FLOP reduction is set to 0 by design; sharing primarily reduces parameter memory and constrains the hypothesis space.",
        },
    }

    if baseline_throughput is not None:
        est_tp = baseline_throughput * estimated_speedup
        analysis["estimated_throughput_samples_sec"] = float(est_tp)
        analysis["throughput_improvement_percent"] = float((estimated_speedup - 1.0) * 100.0)

    return analysis


def analyze_interpolation_removal_potential(model, sample_input_shape=(3, 64, 64), timing_results=None):
    """
    Analyze the potential for removing interpolation overhead by processing images at native resolution.

    Core idea:
      - If the model upsamples 64×64 -> 224×224 via F.interpolate, the spatial workload in the backbone
        scales ~ (224/64)^2 = 12.25×.
      - Theoretical FLOP reduction for the scalable portion is ~ 1 - 1/12.25 ≈ 91.8%.
      - Real speedup is bounded by Amdahl's law: only part of runtime scales with spatial resolution.

    We:
      1) Try to infer interpolation target size from model.forward() source code (best-effort).
      2) Compute interpolation_factor and theoretical FLOP reduction.
      3) Estimate speedup with Amdahl's law using conservative assumptions.

    Notes:
      - This is an estimate. Validate with actual profiling and retraining experiments.
      - We do not attempt to estimate exact activation memory in MB; instead we provide a conservative
        reduction estimate based on scaling, leaving MB as 0 if no baseline activation MB is available.
    """
    baseline_throughput = None
    if timing_results is not None:
        baseline_throughput = timing_results.get("throughput_samples_per_sec", None)

    analysis = {
        "technique": "Interpolation Removal (Native Resolution)",
        "interpolation_size": None,             # inferred below
        "original_image_size": sample_input_shape[-1],
        "avg_flop_reduction_percent": 0.0,
        "estimated_speedup": 1.0,
        "estimated_memory_reduction_mb": 0.0,
        "estimated_throughput_samples_sec": baseline_throughput,
        "throughput_improvement_percent": 0.0,
        "sensitivity_risk": (
            "Moderate: removing interpolation changes the effective input distribution and receptive-field usage. "
            "Requires retraining/validation at native resolution. If using pretrained ImageNet weights, risk is higher; "
            "when training from scratch, risk is often lower but still empirical."
        ),
    }

    target_size = 224  # default assumption

    analysis["interpolation_size"] = int(target_size)

    # --- 1) Compute interpolation factor and theoretical FLOP reduction ---
    native_size = int(sample_input_shape[-1])
    interpolation_factor = (target_size / native_size) ** 2  # (224/64)^2 = 12.25

    theoretical_flop_reduction = 1.0 - (1.0 / interpolation_factor)  # ~0.918
    analysis["avg_flop_reduction_percent"] = float(theoretical_flop_reduction * 100.0)

    # --- 2) Speedup estimate via Amdahl's Law ---
    # We need "fraction of runtime that scales with resolution". We do not have exact profiling here,
    # so we use a conservative model:
    #
    #   conv_coverage ~ 0.85  (resnet inference dominated by convs, but not 100%)
    #   efficiency_factor ~ 0.60 (real-world penalties, kernel launch overhead, bandwidth limits)
    #
    # The scalable fraction is f = conv_coverage * efficiency_factor.
    conv_coverage = 0.85
    efficiency_factor = 0.60
    scalable_fraction = conv_coverage * efficiency_factor
    fixed_fraction = 1.0 - scalable_fraction

    # Amdahl: speedup = 1 / (fixed + scalable / interpolation_factor)
    speedup = 1.0 / (fixed_fraction + (scalable_fraction / interpolation_factor))
    analysis["estimated_speedup"] = float(speedup)

    # Memory reduction (activation-dominated) scales similarly for the scalable portion.
    # Without a measured baseline MB, we keep MB at 0 but you can interpret the reduction factor.
    # If you *do* have a baseline activation MB estimate, multiply by theoretical_flop_reduction.
    analysis["estimated_memory_reduction_mb"] = 0.0

    if baseline_throughput is not None:
        est_tp = baseline_throughput * speedup
        analysis["estimated_throughput_samples_sec"] = float(est_tp)
        analysis["throughput_improvement_percent"] = float((speedup - 1.0) * 100.0)

    # Optional: include the internal assumptions for transparency
    analysis["_assumptions"] = {
        "native_size": native_size,
        "target_size": target_size,
        "interpolation_factor": float(interpolation_factor),
        "conv_coverage": conv_coverage,
        "efficiency_factor": efficiency_factor,
        "scalable_fraction": float(scalable_fraction),
    }

    return analysis
