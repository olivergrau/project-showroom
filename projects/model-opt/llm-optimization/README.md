# LLM Headline Generation

**LLM Inference Optimization for Headline Generation**

This repository contains a hands-on experimental project that explores how different **LLM inference optimization techniques** behave in practice, using a concrete and measurable task:
**generating short news headlines from article summaries**.

The goal is not to chase synthetic speedups, but to understand **what actually works**, **why it works**, and **where trade-offs appear** when optimizing real autoregressive inference.

---

## Project Motivation

Headline generation is deliberately chosen as the target task because it is:

* Autoregressive and latency-sensitive
* Short-output dominated, which stresses overheads
* Easy to evaluate qualitatively and quantitatively
* Small enough to allow repeated, controlled benchmarking

This makes it a surprisingly strict testbed. Many optimization techniques look impressive on paper but show limited benefit, or even regressions, once applied to short-generation workloads.

---

## What This Repository Contains

### 1. A reproducible benchmark harness

A unified evaluation framework that measures:

* Mean latency per request
* Throughput in tokens per second
* ROUGE-1 and ROUGE-L as quality proxies
* Generated token statistics (mean, min, max)
* Optional GPU memory diagnostics

Special care is taken to avoid common benchmarking pitfalls such as token-count bias or warm-run ordering effects.

---

### 2. A clear baseline

A single-model headline generation pipeline using a LLaMA-family model, with:

* Fixed prompts
* Controlled generation parameters
* A fixed evaluation subset

All optimizations are compared against this baseline under identical conditions.

---

### 3. Implemented optimization techniques

Each technique is implemented explicitly and benchmarked independently:

* **KV cache**
  Evaluates the real impact of caching past key-value states for short autoregressive outputs.

* **Quantization (NF4 via bitsandbytes)**
  Focuses on inference-time memory and compute trade-offs, not just model size.

* **Unstructured pruning (L1)**
  Applied to linear layers, with sparsity measured and made permanent for inference.

* **Speculative decoding**
  Includes both realistic headline-length benchmarks and stress tests with long generations to expose where speculative decoding actually helps.

* **Tensor Parallelism (TP)**
  Multi-GPU inference via DeepSpeed, with correctness and collective behavior explicitly addressed.

* **Pipeline Parallelism (PP)**
  A manual pipeline-parallel greedy decoding loop to illustrate the complexity of autoregressive PP beyond standard training setups.

---

### 4. Engineering-first analysis

The notebook and scripts explicitly discuss:

* Why some techniques underperform for short outputs
* Why output token behavior matters more than headline quality alone
* Why “faster” does not always mean “better” in real systems
* Where distributed inference adds complexity without guaranteed benefit

The emphasis is on **understanding**, not marketing.

---

## Repository Structure

```text
.
├── LLM-Headline-Generation.ipynb   # Jupyter notebook with full experiment narrative
├── codebase/benchmark.py               # Benchmark harness and evaluation utilities
├── codebase/run_llama_tp.py             # Tensor Parallel inference runner (DeepSpeed)
├── codebase/run_llama_pp.py             # Pipeline Parallel inference runner
└── README.md                   # You are here
```

---

## Getting Started

### Requirements

* Python 3.10+
* CUDA-capable GPU strongly recommended
* PyTorch
* Hugging Face Transformers
* bitsandbytes
* DeepSpeed (for TP and PP experiments)

A Colab-friendly setup is documented in the notebook export, but local multi-GPU runs are fully supported.

---

### The baseline

The simplest entry point is the notebook itself: LLM-Headline-Generation.ipynb
This runs the baseline model and selected optimizations sequentially.

---

### Running distributed inference

Tensor Parallel (example with 2 GPUs):

```bash
deepspeed --num_gpus 2 run_llama_tp.py \
  --benchmark_json gold_prompts.json \
  --benchmark_n 25 \
  --max_new_tokens 16
```

Pipeline Parallel:

```bash
deepspeed --num_gpus 2 run_llama_pp.py \
  --benchmark_json gold_prompts.json \
  --benchmark_n 25 \
  --max_new_tokens 16
```

These scripts are intentionally explicit and verbose to make data flow and synchronization visible.

---

## Important Design Choices

### Why an *Instruct* model is used

Although the rubric nominally mentions base models, this project uses an **instruction-tuned variant** for the core experiments.

Reason: base LLMs tend to continue generation until hard token limits are reached, which destroys output-length variance and invalidates latency and throughput comparisons for short tasks like headlines.

This is not a stylistic preference, it is a measurement correctness decision.

---

### Why headline generation is harder than it looks

Many inference optimizations shine on long generations. Headlines are short. That exposes:

* Fixed overheads
* Synchronization costs
* Draft-model verification penalties
* Diminishing returns from caching

If an optimization survives here, it is likely robust.

---

## Who This Project Is For

This repository is ideal for:

* ML engineers who want to *see* inference trade-offs, not just read about them
* Practitioners evaluating whether optimizations are worth production complexity
* Students who want a realistic, non-toy optimization benchmark
* Anyone skeptical of benchmark claims without context

It is probably **not** ideal if you are only looking for maximal leaderboard numbers.

---

## Final Note

This project intentionally favors **clarity over cleverness** and **measurement honesty over hype**.
Feel free to fork it, break it, extend it, or rerun everything with your own models and prompts.

That is the point.

---
