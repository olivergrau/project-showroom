import os
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from evaluate import load as load_metric
from time import time
from pprint import pprint
import torch.nn as nn
import torch.nn.utils.prune as prune
import gc
from typing import Any, Dict, List, Optional, Sequence, Tuple
import copy

def _cuda_mem_snapshot_mb():
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
    }

def _cleanup_model(model):
    try:
        del model
    except Exception:
        print("Warning: failed to delete model cleanly")
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # Optional: helps reduce fragmentation in some setups
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

# ------------------------------------------------------------
# Speculative decoding: generation helper (separate from baseline)
# ------------------------------------------------------------
def generate_headline_speculative(
    target_model,
    draft_model,
    tokenizer,
    summary,
    generation_args,
    prompt_template,
    num_assistant_tokens: int,
):
    """
    Generate one headline using Hugging Face assisted/speculative decoding and measure latency.

    - Uses: target_model.generate(..., assistant_model=draft_model, num_assistant_tokens=K)
    - Keeps decoding only the new tokens
    - Performs CUDA sync around timing for meaningful GPU timings

    Returns:
      (headline: str, latency_s: float, n_new_tokens: int)
    """
    prompt_text = prompt_template.format(summary=summary)

    inputs = tokenizer(prompt_text, return_tensors="pt")

    # Place inputs on a sensible device (mirrors generate_headline)
    if hasattr(target_model, "device"):
        device = target_model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Build generation kwargs from the shared generation_args dict
    # IMPORTANT: do not mutate generation_args in-place
    gen_kwargs = copy.deepcopy(generation_args)
    gen_kwargs["assistant_model"] = draft_model
    gen_kwargs["num_assistant_tokens"] = int(num_assistant_tokens)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time()

    with torch.no_grad():
        output_ids = target_model.generate(**inputs, **gen_kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time() - t0

    # Decode only the generated continuation (new tokens)
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
    n_new_tokens = int(new_tokens.shape[-1])

    headline = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Defensive cleanup (same as baseline)
    headline = headline.split("\n")[0].strip()
    if headline.lower().startswith("headline:"):
        headline = headline[len("headline:"):].strip()

    return headline, latency, n_new_tokens


# ------------------------------------------------------------
# Speculative decoding: evaluation loop (mirrors evaluate_model)
# ------------------------------------------------------------
def evaluate_model_speculative(
    dataset,
    target_model,
    draft_model,
    tokenizer,
    generation_args,
    prompt_template,
    num_assistant_tokens: int,
    n: int = 20,
    warmup: int = 5,
):
    """
    Mirror of evaluate_model(), but uses generate_headline_speculative().

    Returns:
      metrics, results, latencies, n_new_tokens_list
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    n = min(n, len(dataset))
    warmup = min(warmup, len(dataset))

    # Warm-up (not recorded)
    for i in range(warmup):
        row = dataset[i]
        _pred, _lat, _ntok = generate_headline_speculative(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            summary=row["summary"],
            generation_args=generation_args,
            prompt_template=prompt_template,
            num_assistant_tokens=num_assistant_tokens,
        )

    results = []
    latencies = []
    n_new_tokens_list = []

    for i in range(n):
        row = dataset[i]
        summary = row["summary"]
        reference = row["headline"]

        prediction, latency, n_new_tokens = generate_headline_speculative(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            summary=summary,
            generation_args=generation_args,
            prompt_template=prompt_template,
            num_assistant_tokens=num_assistant_tokens,
        )

        results.append({
            "summary": summary,
            "reference": reference,
            "prediction": prediction
        })
        latencies.append(latency)
        n_new_tokens_list.append(n_new_tokens)

    metrics = report_metrics(results, latencies, n_new_tokens_list)
    return metrics, results, latencies, n_new_tokens_list


# ------------------------------------------------------------
# Section 7: Speculative decoding runner (aligned with others)
# ------------------------------------------------------------
def run_trials_speculative(
    ds,
    target_model,
    draft_model,
    tokenizer,
    prompt,
    generation_args=None,
    assistant_k_values=None,
    n=100,
    trials=10,
    warmup=5,
):
    """
    Run repeated trials for:
      - Baseline target-only generation (evaluate_model)
      - Speculative/assisted decoding (evaluate_model_speculative) for each K in assistant_k_values

    Uses the SAME gold dataset (ds), SAME prompt template, and SAME generation_args dict
    as the other sections. Only difference: speculative adds assistant_model + K.

    Returns a stats dict aligned in style with run_trials_kv_cache():
      - baseline_* arrays
      - speculative[K] arrays (nested dict keyed by K)
      - includes token-generation diagnostics
    """
    assert generation_args is not None, "generation_args must be provided"
    if assistant_k_values is None:
        assistant_k_values = [8]

    assistant_k_values = [int(k) for k in assistant_k_values]

    print(f"Starting speculative trials: n={n}, trials={trials}, warmup={warmup}, assistant_k_values={assistant_k_values}")

    # storage: baseline
    base_lat = []
    base_tps = []
    base_r1 = []
    base_rL = []
    base_avg_new = []
    base_total_new = []
    base_min_new = []
    base_max_new = []

    # storage: per-k speculative
    spec = {}
    for k in assistant_k_values:
        spec[k] = {
            "lat": [],
            "tps": [],
            "r1": [],
            "rL": [],
            "avg_new": [],
            "total_new": [],
            "min_new": [],
            "max_new": [],
        }

    for t in range(trials):
        print(f"\n=== Trial {t+1}/{trials} ===")
        # Alternate order to reduce drift / warm-run bias:
        # even trials: baseline first, then speculative Ks
        # odd trials:  speculative Ks first, then baseline
        if t % 2 == 0:
            print("  Order: baseline first, then speculative Ks")
            # --- baseline ---
            print("  Running baseline evaluation...")
            m, _, _, new_tokens_list = evaluate_model(ds, target_model, tokenizer, generation_args, prompt, n=n, warmup=warmup)
            base_lat.append(m["mean_latency_s"])
            base_tps.append(m["throughput_tokens_per_s"])
            base_r1.append(m["rouge1"])
            base_rL.append(m["rougeL"])

            new_arr = np.array(new_tokens_list, dtype=np.int64)
            base_avg_new.append(float(new_arr.mean()) if new_arr.size else 0.0)
            base_total_new.append(int(new_arr.sum()) if new_arr.size else 0)
            base_min_new.append(int(new_arr.min()) if new_arr.size else 0)
            base_max_new.append(int(new_arr.max()) if new_arr.size else 0)

            print(
                "  Baseline result:",
                f"mean_latency_s={m['mean_latency_s']:.4f},",
                f"throughput_tokens_per_s={m['throughput_tokens_per_s']:.1f},",
                f"rouge1={m['rouge1']:.4f},",
                f"rougeL={m['rougeL']:.4f},",
                f"new_tokens(mean/total/min/max)=({base_avg_new[-1]:.2f}/{base_total_new[-1]}/{base_min_new[-1]}/{base_max_new[-1]})",
            )
            # --- speculative for each k ---
            for idx, k in enumerate(assistant_k_values):
                # Only warm up on the first speculative run per trial
                effective_warmup = 0 # warmup if idx == 0 else 0

                print(
                    f"    Speculative K={k}: running evaluate_model_speculative "
                    f"(warmup={effective_warmup})..."
                )

                mk, _, _, newk = evaluate_model_speculative(
                    ds, target_model, draft_model, tokenizer,
                    generation_args, prompt,
                    num_assistant_tokens=k,
                    n=n,
                    warmup=effective_warmup,
                )

                spec[k]["lat"].append(mk["mean_latency_s"])
                spec[k]["tps"].append(mk["throughput_tokens_per_s"])
                spec[k]["r1"].append(mk["rouge1"])
                spec[k]["rL"].append(mk["rougeL"])

                newk_arr = np.array(newk, dtype=np.int64)
                spec[k]["avg_new"].append(float(newk_arr.mean()) if newk_arr.size else 0.0)
                spec[k]["total_new"].append(int(newk_arr.sum()) if newk_arr.size else 0)
                spec[k]["min_new"].append(int(newk_arr.min()) if newk_arr.size else 0)
                spec[k]["max_new"].append(int(newk_arr.max()) if newk_arr.size else 0)

                print(
                    f"    Speculative K={k} result:",
                    f"mean_latency_s={mk['mean_latency_s']:.4f},",
                    f"throughput_tokens_per_s={mk['throughput_tokens_per_s']:.1f},",
                    f"rouge1={mk['rouge1']:.4f},",
                    f"rougeL={mk['rougeL']:.4f},",
                    f"new_tokens(mean/total/min/max)=({spec[k]['avg_new'][-1]:.2f}/{spec[k]['total_new'][-1]}/{spec[k]['min_new'][-1]}/{spec[k]['max_new'][-1]})",
                )
        else:
            print("  Order: speculative Ks first, then baseline")
            # --- speculative first ---
            for idx, k in enumerate(assistant_k_values):
                # Only warm up on the first speculative run per trial
                effective_warmup = warmup if idx == 0 else 0

                print(
                    f"    Speculative K={k}: running evaluate_model_speculative "
                    f"(warmup={effective_warmup})..."
                )

                mk, _, _, newk = evaluate_model_speculative(
                    ds, target_model, draft_model, tokenizer,
                    generation_args, prompt,
                    num_assistant_tokens=k,
                    n=n,
                    warmup=effective_warmup,
                )
                spec[k]["lat"].append(mk["mean_latency_s"])
                spec[k]["tps"].append(mk["throughput_tokens_per_s"])
                spec[k]["r1"].append(mk["rouge1"])
                spec[k]["rL"].append(mk["rougeL"])

                newk_arr = np.array(newk, dtype=np.int64)
                spec[k]["avg_new"].append(float(newk_arr.mean()) if newk_arr.size else 0.0)
                spec[k]["total_new"].append(int(newk_arr.sum()) if newk_arr.size else 0)
                spec[k]["min_new"].append(int(newk_arr.min()) if newk_arr.size else 0)
                spec[k]["max_new"].append(int(newk_arr.max()) if newk_arr.size else 0)

                print(
                    f"    Speculative K={k} result:",
                    f"mean_latency_s={mk['mean_latency_s']:.4f},",
                    f"throughput_tokens_per_s={mk['throughput_tokens_per_s']:.1f},",
                    f"rouge1={mk['rouge1']:.4f},",
                    f"rougeL={mk['rougeL']:.4f},",
                    f"new_tokens(mean/total/min/max)=({spec[k]['avg_new'][-1]:.2f}/{spec[k]['total_new'][-1]}/{spec[k]['min_new'][-1]}/{spec[k]['max_new'][-1]})",
                )
            # --- baseline after ---
            print("  Running baseline evaluation...")
            m, _, _, new_tokens_list = evaluate_model(ds, target_model, tokenizer, generation_args, prompt, n=n, warmup=0)
            base_lat.append(m["mean_latency_s"])
            base_tps.append(m["throughput_tokens_per_s"])
            base_r1.append(m["rouge1"])
            base_rL.append(m["rougeL"])

            new_arr = np.array(new_tokens_list, dtype=np.int64)
            base_avg_new.append(float(new_arr.mean()) if new_arr.size else 0.0)
            base_total_new.append(int(new_arr.sum()) if new_arr.size else 0)
            base_min_new.append(int(new_arr.min()) if new_arr.size else 0)
            base_max_new.append(int(new_arr.max()) if new_arr.size else 0)

            print(
                "  Baseline result:",
                f"mean_latency_s={m['mean_latency_s']:.4f},",
                f"throughput_tokens_per_s={m['throughput_tokens_per_s']:.1f},",
                f"rouge1={m['rouge1']:.4f},",
                f"rougeL={m['rougeL']:.4f},",
                f"new_tokens(mean/total/min/max)=({base_avg_new[-1]:.2f}/{base_total_new[-1]}/{base_min_new[-1]}/{base_max_new[-1]})",
            )
    
    print("\n=== All trials completed ===")
    
    # Convert to numpy arrays for consistency with other run_trials_* functions
    base_lat = np.array(base_lat, dtype=np.float64)
    base_tps = np.array(base_tps, dtype=np.float64)
    base_r1  = np.array(base_r1, dtype=np.float64)
    base_rL  = np.array(base_rL, dtype=np.float64)

    base_avg_new   = np.array(base_avg_new, dtype=np.float64)
    base_total_new = np.array(base_total_new, dtype=np.int64)
    base_min_new   = np.array(base_min_new, dtype=np.int64)
    base_max_new   = np.array(base_max_new, dtype=np.int64)

    spec_out = {}
    for k in assistant_k_values:
        spec_out[k] = {
            "latency": np.array(spec[k]["lat"], dtype=np.float64),
            "throughput": np.array(spec[k]["tps"], dtype=np.float64),
            "rouge1": np.array(spec[k]["r1"], dtype=np.float64),
            "rougeL": np.array(spec[k]["rL"], dtype=np.float64),
            "avg_new_tokens_per_sample": np.array(spec[k]["avg_new"], dtype=np.float64),
            "total_new_tokens": np.array(spec[k]["total_new"], dtype=np.int64),
            "min_new_tokens_per_sample": np.array(spec[k]["min_new"], dtype=np.int64),
            "max_new_tokens_per_sample": np.array(spec[k]["max_new"], dtype=np.int64),
        }

    return {
        "baseline_latency": base_lat,
        "baseline_throughput": base_tps,
        "baseline_rouge1": base_r1,
        "baseline_rougeL": base_rL,
        "baseline_avg_new_tokens_per_sample": base_avg_new,
        "baseline_total_new_tokens": base_total_new,
        "baseline_min_new_tokens_per_sample": base_min_new,
        "baseline_max_new_tokens_per_sample": base_max_new,
        "speculative": spec_out,
        "assistant_k_values": assistant_k_values,
    }


def prune_model_weights(model: nn.Module, amount: float = 0.3, make_permanent: bool = True) -> nn.Module:
    """
    Applies L1 unstructured pruning to the *weight* tensors of all nn.Linear layers.

    Args:
        model: The model to prune (pruning is applied in-place).
        amount: Fraction of weights to prune in each Linear layer (0.0 - 1.0).
        make_permanent: If True, removes pruning reparameterization and leaves a real sparse tensor.

    Returns:
        The same model object, pruned.
    """
    assert 0.0 <= amount < 1.0, "amount must be in [0.0, 1.0)"

    pruned_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Apply L1 unstructured pruning to weights
            prune.l1_unstructured(module, name="weight", amount=amount)
            pruned_layers += 1

            # Optional: make pruning permanent by removing reparam (weight_orig + weight_mask)
            if make_permanent:
                prune.remove(module, "weight")

    print(f"Pruned {pruned_layers} nn.Linear layers with L1 unstructured amount={amount}.")
    return model

def compute_global_sparsity(model: nn.Module) -> float:
    """
    Computes global weight sparsity across all nn.Linear layers.
    """
    total = 0
    zeros = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            total += w.numel()
            zeros += (w == 0).sum().item()
    return (zeros / total) if total > 0 else 0.0


def run_trials_pruning(
    ds,
    model_name,
    prune_amount=0.3,
    prompt="",
    generation_args=None,
    n=100,
    trials=10,
    warmup=5,
):
    """
    Loads one baseline model, prunes it in-place, runs repeated trials, returns stats, then frees GPU memory.
    """
    assert generation_args is not None, "generation_args must be provided"

    # Clean start
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Load baseline model (FP16 baseline path)
    tokenizer, model, device = load_model(model_name, quantization_config=None)

    footprint_after_load = _cuda_mem_snapshot_mb()

    # Apply pruning
    prune_model_weights(model, amount=prune_amount, make_permanent=True)
    sparsity = compute_global_sparsity(model)
    print(f"Global Linear sparsity after pruning: {sparsity:.2%}")

    # Storage
    lat = []
    tps = []
    r1 = []
    rL = []
    peak_alloc = []
    peak_res = []

    # Token diagnostics
    avg_new_tokens = []
    total_new_tokens = []
    min_new_tokens = []
    max_new_tokens = []

    for i in range(trials):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        effective_warmup = 0
        if i == 0:
            effective_warmup = warmup
        
        m, _, _, new_tokens_list = evaluate_model(ds, model, tokenizer, generation_args, prompt, n=n, warmup=effective_warmup)

        lat.append(m["mean_latency_s"])
        tps.append(m["throughput_tokens_per_s"])
        r1.append(m["rouge1"])
        rL.append(m["rougeL"])

        new_arr = np.array(new_tokens_list, dtype=np.int64)
        if new_arr.size == 0:
            avg_new_tokens.append(0.0)
            total_new_tokens.append(0)
            min_new_tokens.append(0)
            max_new_tokens.append(0)
        else:
            avg_new_tokens.append(float(new_arr.mean()))
            total_new_tokens.append(int(new_arr.sum()))
            min_new_tokens.append(int(new_arr.min()))
            max_new_tokens.append(int(new_arr.max()))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_alloc.append(torch.cuda.max_memory_allocated() / (1024 ** 2))
            peak_res.append(torch.cuda.max_memory_reserved() / (1024 ** 2))
        else:
            peak_alloc.append(None)
            peak_res.append(None)

    # Convert to numpy
    lat = np.array(lat, dtype=np.float64)
    tps = np.array(tps, dtype=np.float64)
    r1  = np.array(r1, dtype=np.float64)
    rL  = np.array(rL, dtype=np.float64)

    avg_new_tokens   = np.array(avg_new_tokens, dtype=np.float64)
    total_new_tokens = np.array(total_new_tokens, dtype=np.int64)
    min_new_tokens   = np.array(min_new_tokens, dtype=np.int64)
    max_new_tokens   = np.array(max_new_tokens, dtype=np.int64)

    if torch.cuda.is_available():
        peak_alloc = np.array(peak_alloc, dtype=np.float64)
        peak_res   = np.array(peak_res, dtype=np.float64)

    # Print summary
    print(f"=== Pruned (L1 unstructured, amount={prune_amount}) ===")

    print("Latency (mean over trials):")
    print("  mean:", float(lat.mean()), "±", float(lat.std(ddof=1)))

    print("\nThroughput (tokens/s, mean over trials):")
    print("  mean:", float(tps.mean()), "±", float(tps.std(ddof=1)))

    print("\nROUGE-1 (mean over trials):")
    print("  mean:", float(r1.mean()), "±", float(r1.std(ddof=1)))

    print("\nROUGE-L (mean over trials):")
    print("  mean:", float(rL.mean()), "±", float(rL.std(ddof=1)))

    print("\nGenerated new tokens (per trial):")
    print("  avg tokens/sample:", float(avg_new_tokens.mean()), "±", float(avg_new_tokens.std(ddof=1)))
    print("  total tokens      :", int(total_new_tokens.mean()), "±", float(total_new_tokens.std(ddof=1)))
    print("  min tokens/sample :", int(min_new_tokens.min()))
    print("  max tokens/sample :", int(max_new_tokens.max()))

    if torch.cuda.is_available():
        print("\nGPU footprint after load (MB):", footprint_after_load)
        print("GPU peak allocated during eval (MB):", float(peak_alloc.mean()), "±", float(peak_alloc.std(ddof=1)))
        print("GPU peak reserved during eval (MB):  ", float(peak_res.mean()),   "±", float(peak_res.std(ddof=1)))

    stats = {
        "latency": lat,
        "throughput": tps,
        "rouge1": r1,
        "rougeL": rL,
        "avg_new_tokens_per_sample": avg_new_tokens,
        "total_new_tokens": total_new_tokens,
        "min_new_tokens_per_sample": min_new_tokens,
        "max_new_tokens_per_sample": max_new_tokens,
        "global_linear_sparsity": sparsity,
        "footprint_after_load_mb": footprint_after_load,
        "gpu_peak_alloc_mb": peak_alloc,
        "gpu_peak_reserved_mb": peak_res,
        "prune_amount": prune_amount,
    }

    # Cleanup
    del device
    del tokenizer
    _cleanup_model(model)

    return stats

def run_pairwise_trials_baseline_quantization(
    ds,
    baseline_model,
    quantized_model,
    tokenizer,
    prompt,
    generation_args,
    n=100,
    trials=10,
    warmup=5,
):
    """
    Repeated paired trials comparing baseline (FP16) vs quantized (e.g. 4-bit NF4) models.

    Measures per trial:
      - mean latency
      - throughput (tokens/s, based on actual generated tokens)
      - ROUGE-1 / ROUGE-L
      - GPU peak memory (MB), reset and measured per run

    Uses alternating execution order to reduce drift / warm-run bias.

    Notes on memory:
      - We reset CUDA peak stats right before each evaluate_model call.
      - We then read torch.cuda.max_memory_allocated() right after.
      - This is a "peak allocated during this run" proxy (not total reserved).
    """

    base_lat, quant_lat = [], []
    base_tps, quant_tps = [], []
    base_r1, quant_r1 = [], []
    base_rL, quant_rL = [], []
    base_mem_mb, quant_mem_mb = [], []

    def _eval_with_mem(model):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        m, _, _, _ = evaluate_model(
            ds, model, tokenizer, generation_args, prompt, n=n, warmup=warmup
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_mb = None
        return m, peak_mb

    for t in range(trials):
        if t % 2 == 0:
            base_m, base_peak = _eval_with_mem(baseline_model)
            quant_m, quant_peak = _eval_with_mem(quantized_model)
        else:
            quant_m, quant_peak = _eval_with_mem(quantized_model)
            base_m, base_peak = _eval_with_mem(baseline_model)

        base_lat.append(base_m["mean_latency_s"])
        quant_lat.append(quant_m["mean_latency_s"])

        base_tps.append(base_m["throughput_tokens_per_s"])
        quant_tps.append(quant_m["throughput_tokens_per_s"])

        base_r1.append(base_m["rouge1"])
        quant_r1.append(quant_m["rouge1"])

        base_rL.append(base_m["rougeL"])
        quant_rL.append(quant_m["rougeL"])

        base_mem_mb.append(base_peak)
        quant_mem_mb.append(quant_peak)

    # Convert to numpy arrays
    base_lat = np.array(base_lat, dtype=np.float64)
    quant_lat = np.array(quant_lat, dtype=np.float64)
    base_tps = np.array(base_tps, dtype=np.float64)
    quant_tps = np.array(quant_tps, dtype=np.float64)
    base_r1 = np.array(base_r1, dtype=np.float64)
    quant_r1 = np.array(quant_r1, dtype=np.float64)
    base_rL = np.array(base_rL, dtype=np.float64)
    quant_rL = np.array(quant_rL, dtype=np.float64)

    # Memory arrays (may contain None on CPU)
    if torch.cuda.is_available():
        base_mem_mb = np.array(base_mem_mb, dtype=np.float64)
        quant_mem_mb = np.array(quant_mem_mb, dtype=np.float64)
    else:
        base_mem_mb = np.array(base_mem_mb, dtype=object)
        quant_mem_mb = np.array(quant_mem_mb, dtype=object)

    # ---- Reporting ----
    print("Latency (mean over trials):")
    print("  baseline:", float(base_lat.mean()), "±", float(base_lat.std(ddof=1)))
    print("  quant   :", float(quant_lat.mean()), "±", float(quant_lat.std(ddof=1)))
    print("  delta   :", float((base_lat - quant_lat).mean()))

    print("\nThroughput (tokens/s, mean over trials):")
    print("  baseline:", float(base_tps.mean()), "±", float(base_tps.std(ddof=1)))
    print("  quant   :", float(quant_tps.mean()), "±", float(quant_tps.std(ddof=1)))
    print("  delta   :", float((quant_tps - base_tps).mean()))

    print("\nROUGE-1 (mean over trials):")
    print("  baseline:", float(base_r1.mean()), "±", float(base_r1.std(ddof=1)))
    print("  quant   :", float(quant_r1.mean()), "±", float(quant_r1.std(ddof=1)))
    print("  delta   :", float((quant_r1 - base_r1).mean()))

    print("\nROUGE-L (mean over trials):")
    print("  baseline:", float(base_rL.mean()), "±", float(base_rL.std(ddof=1)))
    print("  quant   :", float(quant_rL.mean()), "±", float(quant_rL.std(ddof=1)))
    print("  delta   :", float((quant_rL - base_rL).mean()))

    if torch.cuda.is_available():
        print("\nGPU Peak Memory (MB, max_memory_allocated, mean over trials):")
        print("  baseline:", float(base_mem_mb.mean()), "±", float(base_mem_mb.std(ddof=1)))
        print("  quant   :", float(quant_mem_mb.mean()), "±", float(quant_mem_mb.std(ddof=1)))
        print("  delta   :", float((base_mem_mb - quant_mem_mb).mean()))
    else:
        print("\nGPU Peak Memory: CUDA not available (skipping).")

    # Directional consistency counts
    print("\nQuantized model better latency in",
          int((quant_lat < base_lat).sum()), "of", trials, "trials.")
    print("Quantized model better throughput in",
          int((quant_tps > base_tps).sum()), "of", trials, "trials.")

    print("Quantized model higher ROUGE-1 in",
          int((quant_r1 > base_r1).sum()), "of", trials, "trials.")
    print("Quantized model higher ROUGE-L in",
          int((quant_rL > base_rL).sum()), "of", trials, "trials.")

    if torch.cuda.is_available():
        print("Quantized model lower peak memory in",
              int((quant_mem_mb < base_mem_mb).sum()), "of", trials, "trials.")

    return {
        "baseline_latency": base_lat,
        "quant_latency": quant_lat,
        "baseline_throughput": base_tps,
        "quant_throughput": quant_tps,
        "baseline_rouge1": base_r1,
        "quant_rouge1": quant_r1,
        "baseline_rougeL": base_rL,
        "quant_rougeL": quant_rL,
        "baseline_gpu_peak_mem_mb": base_mem_mb,
        "quant_gpu_peak_mem_mb": quant_mem_mb,
    }


def run_trials_kv_cache(ds, model, tokenizer, prompt, base_args, kv_args, n=100, trials=10, warmup=5):
    """
    Run repeated paired trials (baseline vs KV caching) and aggregate:
      - mean latency
      - throughput (based on actual generated tokens)
      - ROUGE-1 / ROUGE-L stability
      - generated token statistics (avg/total/min/max per trial)

    Alternating order cancels drift / warm-run bias and makes tiny KV-cache effects
    more interpretable in overhead-dominated regimes.
    """
    # Optional sanity asserts (safe to remove if you prefer)
    if "use_cache" in base_args:
        assert base_args["use_cache"] is False, "base_args should have use_cache=False"
    if "use_cache" in kv_args:
        assert kv_args["use_cache"] is True, "kv_args should have use_cache=True"

    base_lat = []
    kv_lat = []
    base_tps = []
    kv_tps = []
    base_r1 = []
    kv_r1 = []
    base_rL = []
    kv_rL = []

    # NEW: token stats per trial
    base_avg_new = []
    kv_avg_new = []
    base_total_new = []
    kv_total_new = []
    base_min_new = []
    kv_min_new = []
    base_max_new = []
    kv_max_new = []

    for t in range(trials):
        effective_warmup = 0
        if t == 0:
            effective_warmup = warmup

        # Alternate order to cancel drift / warm-run bias
        if t % 2 == 0:
            base_m, _, _, base_new_tokens = evaluate_model(ds, model, tokenizer, base_args, prompt, n=n, warmup=effective_warmup)
            kv_m,   _, _, kv_new_tokens   = evaluate_model(ds, model, tokenizer, kv_args,   prompt, n=n, warmup=effective_warmup)
        else:
            kv_m,   _, _, kv_new_tokens   = evaluate_model(ds, model, tokenizer, kv_args,   prompt, n=n, warmup=effective_warmup)
            base_m, _, _, base_new_tokens = evaluate_model(ds, model, tokenizer, base_args, prompt, n=n, warmup=effective_warmup)

        # Latency / throughput
        base_lat.append(base_m["mean_latency_s"])
        kv_lat.append(kv_m["mean_latency_s"])

        base_tps.append(base_m["throughput_tokens_per_s"])
        kv_tps.append(kv_m["throughput_tokens_per_s"])

        # Quality regression check
        base_r1.append(base_m["rouge1"])
        kv_r1.append(kv_m["rouge1"])
        base_rL.append(base_m["rougeL"])
        kv_rL.append(kv_m["rougeL"])

        # NEW: token summaries
        base_arr = np.array(base_new_tokens, dtype=np.int64)
        kv_arr   = np.array(kv_new_tokens,   dtype=np.int64)

        # Defensive: handle empty lists (shouldn't happen, but keeps function robust)
        if base_arr.size == 0 or kv_arr.size == 0:
            base_avg_new.append(0.0); kv_avg_new.append(0.0)
            base_total_new.append(0); kv_total_new.append(0)
            base_min_new.append(0); kv_min_new.append(0)
            base_max_new.append(0); kv_max_new.append(0)
        else:
            base_avg_new.append(float(base_arr.mean()))
            kv_avg_new.append(float(kv_arr.mean()))
            base_total_new.append(int(base_arr.sum()))
            kv_total_new.append(int(kv_arr.sum()))
            base_min_new.append(int(base_arr.min()))
            kv_min_new.append(int(kv_arr.min()))
            base_max_new.append(int(base_arr.max()))
            kv_max_new.append(int(kv_arr.max()))

    # Convert to numpy
    base_lat = np.array(base_lat, dtype=np.float64)
    kv_lat   = np.array(kv_lat,   dtype=np.float64)
    base_tps = np.array(base_tps, dtype=np.float64)
    kv_tps   = np.array(kv_tps,   dtype=np.float64)

    base_r1  = np.array(base_r1, dtype=np.float64)
    kv_r1    = np.array(kv_r1,   dtype=np.float64)
    base_rL  = np.array(base_rL, dtype=np.float64)
    kv_rL    = np.array(kv_rL,   dtype=np.float64)

    # NEW token arrays
    base_avg_new   = np.array(base_avg_new, dtype=np.float64)
    kv_avg_new     = np.array(kv_avg_new,   dtype=np.float64)
    base_total_new = np.array(base_total_new, dtype=np.int64)
    kv_total_new   = np.array(kv_total_new,   dtype=np.int64)
    base_min_new   = np.array(base_min_new, dtype=np.int64)
    kv_min_new     = np.array(kv_min_new,   dtype=np.int64)
    base_max_new   = np.array(base_max_new, dtype=np.int64)
    kv_max_new     = np.array(kv_max_new,   dtype=np.int64)

    print("Latency (mean over trials):")
    print("  baseline:", float(base_lat.mean()), "±", float(base_lat.std(ddof=1)))
    print("  kv      :", float(kv_lat.mean()),   "±", float(kv_lat.std(ddof=1)))
    print("  delta   :", float((base_lat - kv_lat).mean()))

    print("\nThroughput (tokens/s, mean over trials):")
    print("  baseline:", float(base_tps.mean()), "±", float(base_tps.std(ddof=1)))
    print("  kv      :", float(kv_tps.mean()),   "±", float(kv_tps.std(ddof=1)))
    print("  delta   :", float((kv_tps - base_tps).mean()))

    print("\nROUGE-1 (mean over trials):")
    print("  baseline:", float(base_r1.mean()), "±", float(base_r1.std(ddof=1)))
    print("  kv      :", float(kv_r1.mean()),   "±", float(kv_r1.std(ddof=1)))
    print("  delta   :", float((kv_r1 - base_r1).mean()))

    print("\nROUGE-L (mean over trials):")
    print("  baseline:", float(base_rL.mean()), "±", float(base_rL.std(ddof=1)))
    print("  kv      :", float(kv_rL.mean()),   "±", float(kv_rL.std(ddof=1)))
    print("  delta   :", float((kv_rL - base_rL).mean()))

    # NEW: token diagnostics printout
    print("\nGenerated new tokens (per trial):")
    print("  baseline avg tokens/sample:", float(base_avg_new.mean()), "±", float(base_avg_new.std(ddof=1)))
    print("  kv       avg tokens/sample:", float(kv_avg_new.mean()),   "±", float(kv_avg_new.std(ddof=1)))
    print("  delta avg tokens/sample    :", float((kv_avg_new - base_avg_new).mean()))

    print("  baseline total new tokens  :", int(base_total_new.mean()), "±", float(base_total_new.std(ddof=1)))
    print("  kv       total new tokens  :", int(kv_total_new.mean()),   "±", float(kv_total_new.std(ddof=1)))
    print("  delta total new tokens     :", float((kv_total_new - base_total_new).mean()))

    print("  baseline min tokens/sample :", int(base_min_new.min()), "max:", int(base_max_new.max()))
    print("  kv       min tokens/sample :", int(kv_min_new.min()),   "max:", int(kv_max_new.max()))

    # Directional consistency counts
    print("\nKV better latency in", int((kv_lat < base_lat).sum()), "of", trials, "trials.")
    print("KV better throughput in", int((kv_tps > base_tps).sum()), "of", trials, "trials.")

    # NEW: token consistency counts
    print("KV generated more total tokens in", int((kv_total_new > base_total_new).sum()), "of", trials, "trials.")
    print("KV generated fewer total tokens in", int((kv_total_new < base_total_new).sum()), "of", trials, "trials.")

    # Quality stability checks
    print("KV higher ROUGE-1 in", int((kv_r1 > base_r1).sum()), "of", trials, "trials.")
    print("KV higher ROUGE-L in", int((kv_rL > base_rL).sum()), "of", trials, "trials.")

    return {
        "baseline_latency": base_lat,
        "kv_latency": kv_lat,
        "baseline_throughput": base_tps,
        "kv_throughput": kv_tps,
        "baseline_rouge1": base_r1,
        "kv_rouge1": kv_r1,
        "baseline_rougeL": base_rL,
        "kv_rougeL": kv_rL,
        # NEW token stats
        "baseline_avg_new_tokens_per_sample": base_avg_new,
        "kv_avg_new_tokens_per_sample": kv_avg_new,
        "baseline_total_new_tokens": base_total_new,
        "kv_total_new_tokens": kv_total_new,
        "baseline_min_new_tokens_per_sample": base_min_new,
        "kv_min_new_tokens_per_sample": kv_min_new,
        "baseline_max_new_tokens_per_sample": base_max_new,
        "kv_max_new_tokens_per_sample": kv_max_new,
    }


def run_trials(
    ds,
    model_name,
    prompt="",
    generation_args=None,
    n=100,
    trials=10,
    warmup=5,
):
    """
    Loads ONE baseline model, runs repeated trials, returns stats, then frees GPU memory.

    Params:
      - use_kv_cache: controls generation_args['use_cache'] for this run
      - tokenizer: optionally pass a tokenizer to reuse (if None, load_model returns one)
      - torch_dtype: dtype for baseline load (FP16 default).
    """
    assert generation_args is not None, "generation_args must be provided"

    # Create a local copy so caller args are not mutated
    gen_args = dict(generation_args)

    # Clean start
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Load baseline model
    tokenizer, model, device = load_model(model_name, quantization_config=None)

    footprint_after_load = _cuda_mem_snapshot_mb()

    # Capture metrics
    lat = []
    tps = []
    r1 = []
    rL = []
    peak_alloc = []
    peak_res = []
    
    avg_new_tokens = []
    total_new_tokens = []
    min_new_tokens = []
    max_new_tokens = []

    for i in range(trials):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        effective_warmup = warmup if i == 0 else 0
        m, _, _, n_new_tokens_list = evaluate_model(ds, model, tokenizer, gen_args, prompt, n=n, warmup=effective_warmup)

        n_new_tokens_arr = np.array(n_new_tokens_list, dtype=np.int64)
        avg_new_tokens.append(float(n_new_tokens_arr.mean()) if len(n_new_tokens_arr) else 0.0)
        total_new_tokens.append(int(n_new_tokens_arr.sum()))
        min_new_tokens.append(int(n_new_tokens_arr.min()) if len(n_new_tokens_arr) else 0)
        max_new_tokens.append(int(n_new_tokens_arr.max()) if len(n_new_tokens_arr) else 0)

        lat.append(m["mean_latency_s"])
        tps.append(m["throughput_tokens_per_s"])
        r1.append(m["rouge1"])
        rL.append(m["rougeL"])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_alloc.append(torch.cuda.max_memory_allocated() / (1024 ** 2))
            peak_res.append(torch.cuda.max_memory_reserved() / (1024 ** 2))
        else:
            peak_alloc.append(None)
            peak_res.append(None)

    # Convert to numpy
    lat = np.array(lat, dtype=np.float64)
    tps = np.array(tps, dtype=np.float64)
    r1  = np.array(r1, dtype=np.float64)
    rL  = np.array(rL, dtype=np.float64)

    avg_new_tokens = np.array(avg_new_tokens, dtype=np.float64)
    total_new_tokens = np.array(total_new_tokens, dtype=np.int64)
    min_new_tokens = np.array(min_new_tokens, dtype=np.int64)
    max_new_tokens = np.array(max_new_tokens, dtype=np.int64)

    if torch.cuda.is_available():
        peak_alloc = np.array(peak_alloc, dtype=np.float64)
        peak_res   = np.array(peak_res, dtype=np.float64)

    # Print summary
    label = "Baseline (use_cache=True)" if gen_args.get("use_cache") else "Baseline (use_cache=False)"
    print(f"=== {label} ===")

    print("Latency (mean over trials):")
    print("  mean:", float(lat.mean()), "±", float(lat.std(ddof=1)))

    print("\nThroughput (tokens/s, mean over trials):")
    print("  mean:", float(tps.mean()), "±", float(tps.std(ddof=1)))

    print("\nROUGE-1 (mean over trials):")
    print("  mean:", float(r1.mean()), "±", float(r1.std(ddof=1)))

    print("\nROUGE-L (mean over trials):")
    print("  mean:", float(rL.mean()), "±", float(rL.std(ddof=1)))

    print("\nGenerated new tokens (per trial):")
    print("  avg tokens/sample:", float(avg_new_tokens.mean()), "±", float(avg_new_tokens.std(ddof=1)))
    print("  total tokens      :", int(total_new_tokens.mean()), "±", float(total_new_tokens.std(ddof=1)))
    print("  min tokens/sample :", int(min_new_tokens.min()))
    print("  max tokens/sample :", int(max_new_tokens.max()))

    if torch.cuda.is_available():
        print("\nGPU footprint after load (MB):", footprint_after_load)
        print("GPU peak allocated during eval (MB):", float(peak_alloc.mean()), "±", float(peak_alloc.std(ddof=1)))
        print("GPU peak reserved during eval (MB):  ", float(peak_res.mean()),   "±", float(peak_res.std(ddof=1)))

    # Return stats
    stats = {
        "latency": lat,
        "throughput": tps,
        "rouge1": r1,
        "rougeL": rL,
        "footprint_after_load_mb": footprint_after_load,
        "gpu_peak_alloc_mb": peak_alloc,
        "gpu_peak_reserved_mb": peak_res,
        "avg_new_tokens_per_sample": avg_new_tokens,
        "total_new_tokens": total_new_tokens,
        "min_new_tokens_per_sample": min_new_tokens,
        "max_new_tokens_per_sample": max_new_tokens,
    }

    # Cleanup
    del device
    del tokenizer
    _cleanup_model(model)

    return stats


def run_trials_nf4(
    ds,
    model_name,
    quant_config,
    prompt="",
    generation_args=None,
    n=100,
    trials=10,
    warmup=5,
):
    """
    Loads ONE quantized model (e.g. 4-bit NF4), runs repeated trials, returns stats, then frees GPU memory.
    """
    assert generation_args is not None, "generation_args must be provided"

    # Clean start
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Load quantized model
    tokenizer, model, _ = load_model(model_name, quantization_config=quant_config)

    footprint_after_load = _cuda_mem_snapshot_mb()

    # Storage
    lat = []
    tps = []
    r1 = []
    rL = []
    peak_alloc = []
    peak_res = []

    # token diagnostics
    avg_new_tokens = []
    total_new_tokens = []
    min_new_tokens = []
    max_new_tokens = []

    for i in range(trials):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        effective_warmup = warmup if i == 0 else 0
        m, _, _, new_tokens_list = evaluate_model(ds, model, tokenizer, generation_args, prompt, n=n, warmup=effective_warmup)

        lat.append(m["mean_latency_s"])
        tps.append(m["throughput_tokens_per_s"])
        r1.append(m["rouge1"])
        rL.append(m["rougeL"])

        # token summary
        new_arr = np.array(new_tokens_list, dtype=np.int64)
        if new_arr.size == 0:
            avg_new_tokens.append(0.0)
            total_new_tokens.append(0)
            min_new_tokens.append(0)
            max_new_tokens.append(0)
        else:
            avg_new_tokens.append(float(new_arr.mean()))
            total_new_tokens.append(int(new_arr.sum()))
            min_new_tokens.append(int(new_arr.min()))
            max_new_tokens.append(int(new_arr.max()))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_alloc.append(torch.cuda.max_memory_allocated() / (1024 ** 2))
            peak_res.append(torch.cuda.max_memory_reserved() / (1024 ** 2))
        else:
            peak_alloc.append(None)
            peak_res.append(None)

    # Convert to numpy
    lat = np.array(lat, dtype=np.float64)
    tps = np.array(tps, dtype=np.float64)
    r1  = np.array(r1, dtype=np.float64)
    rL  = np.array(rL, dtype=np.float64)

    avg_new_tokens   = np.array(avg_new_tokens, dtype=np.float64)
    total_new_tokens = np.array(total_new_tokens, dtype=np.int64)
    min_new_tokens   = np.array(min_new_tokens, dtype=np.int64)
    max_new_tokens   = np.array(max_new_tokens, dtype=np.int64)

    if torch.cuda.is_available():
        peak_alloc = np.array(peak_alloc, dtype=np.float64)
        peak_res   = np.array(peak_res, dtype=np.float64)

    # Print summary
    print("=== Quantized (4-bit) ===")

    print("Latency (mean over trials):")
    print("  mean:", float(lat.mean()), "±", float(lat.std(ddof=1)))

    print("\nThroughput (tokens/s, mean over trials):")
    print("  mean:", float(tps.mean()), "±", float(tps.std(ddof=1)))

    print("\nROUGE-1 (mean over trials):")
    print("  mean:", float(r1.mean()), "±", float(r1.std(ddof=1)))

    print("\nROUGE-L (mean over trials):")
    print("  mean:", float(rL.mean()), "±", float(rL.std(ddof=1)))

    print("\nGenerated new tokens (per trial):")
    print("  avg tokens/sample:", float(avg_new_tokens.mean()), "±", float(avg_new_tokens.std(ddof=1)))
    print("  total tokens      :", int(total_new_tokens.mean()), "±", float(total_new_tokens.std(ddof=1)))
    print("  min tokens/sample :", int(min_new_tokens.min()))
    print("  max tokens/sample :", int(max_new_tokens.max()))

    if torch.cuda.is_available():
        print("\nGPU footprint after load (MB):", footprint_after_load)
        print("GPU peak allocated during eval (MB):", float(peak_alloc.mean()), "±", float(peak_alloc.std(ddof=1)))
        print("GPU peak reserved during eval (MB):  ", float(peak_res.mean()),   "±", float(peak_res.std(ddof=1)))

    stats = {
        "latency": lat,
        "throughput": tps,
        "rouge1": r1,
        "rougeL": rL,
        "avg_new_tokens_per_sample": avg_new_tokens,
        "total_new_tokens": total_new_tokens,
        "min_new_tokens_per_sample": min_new_tokens,
        "max_new_tokens_per_sample": max_new_tokens,
        "footprint_after_load_mb": footprint_after_load,
        "gpu_peak_alloc_mb": peak_alloc,
        "gpu_peak_reserved_mb": peak_res,
        "quant_config": str(quant_config),
    }

    _cleanup_model(model)
    return stats


def run_pairwise_trials_baseline_quantization(
    ds,
    baseline_model,
    quantized_model,
    tokenizer,
    prompt,
    generation_args,
    n=100,
    trials=10,
    warmup=5,
):
    """
    Repeated paired trials comparing baseline (FP16) vs quantized (e.g. 4-bit NF4) models.

    Measures per trial:
      - mean latency
      - throughput (tokens/s, based on actual generated tokens)
      - ROUGE-1 / ROUGE-L
      - GPU peak memory (MB), reset and measured per run

    Uses alternating execution order to reduce drift / warm-run bias.

    Notes on memory:
      - We reset CUDA peak stats right before each evaluate_model call.
      - We then read torch.cuda.max_memory_allocated() right after.
      - This is a "peak allocated during this run" proxy (not total reserved).
    """

    base_lat, quant_lat = [], []
    base_tps, quant_tps = [], []
    base_r1, quant_r1 = [], []
    base_rL, quant_rL = [], []
    base_mem_mb, quant_mem_mb = [], []

    def _eval_with_mem(model):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        m, _, _, _ = evaluate_model(
            ds, model, tokenizer, generation_args, prompt, n=n, warmup=warmup
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_mb = None
        return m, peak_mb

    for t in range(trials):
        if t % 2 == 0:
            base_m, base_peak = _eval_with_mem(baseline_model)
            quant_m, quant_peak = _eval_with_mem(quantized_model)
        else:
            quant_m, quant_peak = _eval_with_mem(quantized_model)
            base_m, base_peak = _eval_with_mem(baseline_model)

        base_lat.append(base_m["mean_latency_s"])
        quant_lat.append(quant_m["mean_latency_s"])

        base_tps.append(base_m["throughput_tokens_per_s"])
        quant_tps.append(quant_m["throughput_tokens_per_s"])

        base_r1.append(base_m["rouge1"])
        quant_r1.append(quant_m["rouge1"])

        base_rL.append(base_m["rougeL"])
        quant_rL.append(quant_m["rougeL"])

        base_mem_mb.append(base_peak)
        quant_mem_mb.append(quant_peak)

    # Convert to numpy arrays
    base_lat = np.array(base_lat, dtype=np.float64)
    quant_lat = np.array(quant_lat, dtype=np.float64)
    base_tps = np.array(base_tps, dtype=np.float64)
    quant_tps = np.array(quant_tps, dtype=np.float64)
    base_r1 = np.array(base_r1, dtype=np.float64)
    quant_r1 = np.array(quant_r1, dtype=np.float64)
    base_rL = np.array(base_rL, dtype=np.float64)
    quant_rL = np.array(quant_rL, dtype=np.float64)

    # Memory arrays (may contain None on CPU)
    if torch.cuda.is_available():
        base_mem_mb = np.array(base_mem_mb, dtype=np.float64)
        quant_mem_mb = np.array(quant_mem_mb, dtype=np.float64)
    else:
        base_mem_mb = np.array(base_mem_mb, dtype=object)
        quant_mem_mb = np.array(quant_mem_mb, dtype=object)

    # ---- Reporting ----
    print("Latency (mean over trials):")
    print("  baseline:", float(base_lat.mean()), "±", float(base_lat.std(ddof=1)))
    print("  quant   :", float(quant_lat.mean()), "±", float(quant_lat.std(ddof=1)))
    print("  delta   :", float((base_lat - quant_lat).mean()))

    print("\nThroughput (tokens/s, mean over trials):")
    print("  baseline:", float(base_tps.mean()), "±", float(base_tps.std(ddof=1)))
    print("  quant   :", float(quant_tps.mean()), "±", float(quant_tps.std(ddof=1)))
    print("  delta   :", float((quant_tps - base_tps).mean()))

    print("\nROUGE-1 (mean over trials):")
    print("  baseline:", float(base_r1.mean()), "±", float(base_r1.std(ddof=1)))
    print("  quant   :", float(quant_r1.mean()), "±", float(quant_r1.std(ddof=1)))
    print("  delta   :", float((quant_r1 - base_r1).mean()))

    print("\nROUGE-L (mean over trials):")
    print("  baseline:", float(base_rL.mean()), "±", float(base_rL.std(ddof=1)))
    print("  quant   :", float(quant_rL.mean()), "±", float(quant_rL.std(ddof=1)))
    print("  delta   :", float((quant_rL - base_rL).mean()))

    if torch.cuda.is_available():
        print("\nGPU Peak Memory (MB, max_memory_allocated, mean over trials):")
        print("  baseline:", float(base_mem_mb.mean()), "±", float(base_mem_mb.std(ddof=1)))
        print("  quant   :", float(quant_mem_mb.mean()), "±", float(quant_mem_mb.std(ddof=1)))
        print("  delta   :", float((base_mem_mb - quant_mem_mb).mean()))
    else:
        print("\nGPU Peak Memory: CUDA not available (skipping).")

    # Directional consistency counts
    print("\nQuantized model better latency in",
          int((quant_lat < base_lat).sum()), "of", trials, "trials.")
    print("Quantized model better throughput in",
          int((quant_tps > base_tps).sum()), "of", trials, "trials.")

    print("Quantized model higher ROUGE-1 in",
          int((quant_r1 > base_r1).sum()), "of", trials, "trials.")
    print("Quantized model higher ROUGE-L in",
          int((quant_rL > base_rL).sum()), "of", trials, "trials.")

    if torch.cuda.is_available():
        print("Quantized model lower peak memory in",
              int((quant_mem_mb < base_mem_mb).sum()), "of", trials, "trials.")

    return {
        "baseline_latency": base_lat,
        "quant_latency": quant_lat,
        "baseline_throughput": base_tps,
        "quant_throughput": quant_tps,
        "baseline_rouge1": base_r1,
        "quant_rouge1": quant_r1,
        "baseline_rougeL": base_rL,
        "quant_rougeL": quant_rL,
        "baseline_gpu_peak_mem_mb": base_mem_mb,
        "quant_gpu_peak_mem_mb": quant_mem_mb,
    }


def load_model(model_name, quantization_config=None, device_map=None):
    """
    Load tokenizer + model for generation.

    Args:
        model_name: Hugging Face model id
        quantization_config: BitsAndBytesConfig or None
        device_map: None (single device) or "auto" (tensor parallelism) etc.

    Returns:
        (tokenizer, model, device)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Many decoder-only LLMs don't define pad_token by default.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # For decoder-only generation it's usually better to left-pad if batching.
    tokenizer.padding_side = "left"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model_kwargs = {}

    if quantization_config is not None:
        # With bitsandbytes quantization, device_map is strongly recommended.
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = device_map or "auto"
    else:
        # Standard FP16 on GPU, FP32 on CPU for safety.
        if use_cuda:
            model_kwargs["torch_dtype"] = torch.float16

        # If device_map is not used, we'll move model manually.
        if device_map is not None:
            model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Manual device placement for the non-device_map case
    if "device_map" not in model_kwargs:
        model = model.to(device)

    model.eval()
    return tokenizer, model, device

def generate_headline(model, tokenizer, summary, generation_args, prompt_template):
    """
    Generate one headline and measure latency.

    Improvements vs previous version:
    - Returns actual number of generated tokens (n_new_tokens)
    - Keeps decoding of only the new tokens
    - Performs CUDA sync around timing for meaningful GPU timings
    """
    prompt_text = prompt_template.format(summary=summary)

    inputs = tokenizer(prompt_text, return_tensors="pt")

    # Place inputs on a sensible device:
    # - For typical single-device models: model.device works
    # - For device_map models: inputs can usually go to cuda:0 if available
    if hasattr(model, "device"):
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Accurate timing (GPU)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time()

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_args)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time() - t0

    # Decode only the generated continuation (new tokens)
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
    n_new_tokens = int(new_tokens.shape[-1])

    headline = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Defensive cleanup
    headline = headline.split("\n")[0].strip()
    if headline.lower().startswith("headline:"):
        headline = headline[len("headline:"):].strip()

    return headline, latency, n_new_tokens


def report_metrics(results, latencies, n_new_tokens_list):
    """
    Compute and print metrics:
      - Mean / P95 / P99 latency
      - True throughput based on ACTUAL generated tokens (not max_new_tokens)
      - ROUGE-1 and ROUGE-L
      - GPU peak memory (if available)
      - Avg generated tokens/sample (diagnostic)

    Note:
      This fixes the previous "throughput_tokens_per_s_approx" issue where we
      assumed max_new_tokens for every sample even when generation stopped early.
    """
    latencies = np.array(latencies, dtype=np.float64)
    n_new_tokens_arr = np.array(n_new_tokens_list, dtype=np.int64)

    mean_latency = float(latencies.mean()) if len(latencies) else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if len(latencies) else 0.0
    p99_latency = float(np.percentile(latencies, 99)) if len(latencies) else 0.0

    total_latency = float(latencies.sum())
    total_generated_tokens = int(n_new_tokens_arr.sum())
    throughput = float(total_generated_tokens / total_latency) if total_latency > 0 else 0.0

    avg_new_tokens = float(n_new_tokens_arr.mean()) if len(n_new_tokens_arr) else 0.0
    min_new_tokens = int(n_new_tokens_arr.min()) if len(n_new_tokens_arr) else 0
    max_new_tokens = int(n_new_tokens_arr.max()) if len(n_new_tokens_arr) else 0

    rouge = load_metric("rouge")
    predictions = [r["prediction"] for r in results]
    references = [r["reference"] for r in results]
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    if torch.cuda.is_available():
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        peak_mem_mb = peak_mem_bytes / (1024 ** 2)
    else:
        peak_mem_mb = None

    metrics = {
        "mean_latency_s": mean_latency,
        "p95_latency_s": p95_latency,
        "p99_latency_s": p99_latency,
        # true throughput, based on actual number of generated tokens
        "throughput_tokens_per_s": throughput,
        "rouge1": rouge_scores.get("rouge1", None),
        "rougeL": rouge_scores.get("rougeL", None),
        "gpu_peak_mem_mb": peak_mem_mb,
        "n_samples": int(len(latencies)),
        # diagnostics that explain why KV caching may look similar
        "avg_generated_new_tokens": avg_new_tokens,
        "min_generated_new_tokens": min_new_tokens,
        "max_generated_new_tokens": max_new_tokens,
        "total_generated_new_tokens": total_generated_tokens,
    }

    return metrics


def evaluate_model(dataset, model, tokenizer, generation_args, prompt_template, n=20, warmup=5):
    """
    Run generation on n samples and collect results + latencies.

    Improvements vs previous version:
    - Warm-up runs (not recorded) to reduce first-run and allocator noise
    - Tracks actual generated tokens per sample
    - Uses report_metrics() that computes true throughput

    Args:
        dataset: HF Dataset (not DatasetDict)
        model/tokenizer: HF model + tokenizer
        generation_args: passed to model.generate()
        prompt_template: prompt with {summary}
        n: number of measured samples
        warmup: number of warmup samples (not recorded)
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    n = min(n, len(dataset))
    warmup = min(warmup, len(dataset))

    # Warm-up (not recorded)
    for i in range(warmup):
        row = dataset[i]
        _pred, _lat, _ntok = generate_headline(
            model=model,
            tokenizer=tokenizer,
            summary=row["summary"],
            generation_args=generation_args,
            prompt_template=prompt_template,
        )

    results = []
    latencies = []
    n_new_tokens_list = []

    for i in range(n):
        row = dataset[i]
        summary = row["summary"]
        reference = row["headline"]

        prediction, latency, n_new_tokens = generate_headline(
            model=model,
            tokenizer=tokenizer,
            summary=summary,
            generation_args=generation_args,
            prompt_template=prompt_template,
        )

        results.append({
            "summary": summary,
            "reference": reference,
            "prediction": prediction
        })
        latencies.append(latency)
        n_new_tokens_list.append(n_new_tokens)

    metrics = report_metrics(results, latencies, n_new_tokens_list)
    return metrics, results, latencies, n_new_tokens_list
