# Project Instructions: GPT-2 Model Optimization & Deployment

## Project Context

In this project, you will act as an ML Engineer responsible for optimizing a GPT-2 language model for efficient inference in a resource-constrained environment. You will apply techniques like attention head pruning, fine-tuning, INT8 post-training quantization, and performance analysis to create a lightweight version of GPT-2 that can run on a CPU with limited memory. This project mimics real-world tasks performed by AI/ML engineers at companies deploying large models on edge devices or cloud services.

---

## Files & Setup

Work for this project is completed in the provided `project.ipynb` Jupyter notebook. The notebook includes all the code scaffolding, explanations, and clearly marked `TODO` sections where you will implement your logic.

### What you will need:
- GPU: for pruning + fine-tuning (FP16)
- CPU: for quantization + evaluation
- Tokenizer and model: `gpt2`

---

## Development Strategy

We recommend completing the project in four structured phases:

---

### Part 1: Load and Profile the Base GPT-2 Model (FP16)

- ✅ Load the `gpt2` model using Hugging Face.
- ✅ Convert it to FP16.
- ✅ Profile baseline latency using `torch.profiler`.
- ✅ Measure perplexity and VRAM usage.

💡 This step helps you understand how the original model performs before any optimization.

---

### Part 2: Prune Attention Heads

- ✅ Implement the `prune_attention_heads()` function.
- ✅ Prune 20% of heads per transformer layer.
- ✅ Recast the model to FP16 after pruning.
- ✅ Measure and compare latency, perplexity, and GPU VRAM again.

💡 This simulates structured model compression by reducing compute-heavy components.

---

### Part 3: Fine-tune the Pruned Model

- ✅ Use a small custom dataset (or a sample from Hugging Face).
- ✅ Prepare your tokenizer with proper padding settings.
- ✅ Tokenize your dataset using `tokenize_dataset()`.
- ✅ Set up `TrainingArguments` and use Hugging Face's `Trainer`.
- ✅ Fine-tune for 1–5 epochs.

💡 Fine-tuning helps the model recover performance lost from pruning.

---

### Part 4: Post-Training Quantization (INT8 CPU)

- ✅ Use `INCQuantizer` to quantize the fine-tuned model.
- ✅ Filter and tokenize the calibration dataset.
- ✅ Export the model in OpenVINO IR format.
- ✅ Measure performance on CPU:
  - Inference latency
  - RAM usage
  - Perplexity

💡 This step prepares the model for low-latency CPU-only inference using INT8 precision.

---

### ⚠️ Hardware reality check (T4 GPU users)

If you’re working in an environment with an NVIDIA T4 GPU, don’t expect the INT8-quantized model (CPU, OpenVINO) to be faster than the FP16/FP32 model running on the T4 GPU. INT8 post-training quantization in this project targets CPU-only deployment and efficiency.
For fair comparisons, compare performance within the same device:

- Parts 1–3: GPU baselines (FP16/FP32 on T4)

- Part 4: CPU baselines (FP32/FP16 on CPU) vs INT8 on CPU

### 🔎 Why INT8 on CPU can be slower than FP16/FP32 on a T4 GPU

- Different devices, different strengths. A T4 has thousands of CUDA cores and Tensor Cores optimized for large matrix multiplies and high memory bandwidth. A typical CPU has far fewer vector lanes and much lower memory bandwidth, so even with INT8 it can’t match GPU parallelism.

- Specialized GPU hardware. T4 Tensor Cores accelerate FP16 (and INT8 on GPU). In this project, INT8 runs on CPU via OpenVINO—so you’re not tapping the T4’s Tensor Cores at all, while the FP16/FP32 baselines do use the GPU.

- Operator coverage & de/req quant overhead. Not every op/layer is quantized. Frameworks insert dequantize → (float op) → requantize steps where needed, which adds latency and erodes INT8 gains.

- Autoregressive decoding is small-batch & memory-bound. GPT-2 generates token-by-token (batch≈1). That limits CPU vector utilization, and the KV cache reads/writes make the workload memory-bound—bandwidth where GPUs excel.

- CPU ISA matters. Big INT8 gains rely on AVX512-VNNI/AMX. On CPUs without these, performance uplift over FP32 can be modest.

- Graph breaks & dynamic shapes. Varying sequence lengths and dynamic control flow can prevent kernel fusion, increasing framework overhead on CPU.

Bottom line: INT8 quantization here targets CPU efficiency and portability, not beating a T4 GPU. Compare CPU INT8 vs CPU FP32/FP16 to show the real benefit of quantization.

---

## Visualizations & Flame Graphs

- ✅ Use `matplotlib` to plot:
  - Latency per optimization stage
  - Perplexity per stage
  - VRAM/RAM usage
- ✅ Export `torch.profiler` traces for flame graph inspection via TensorBoard.

💡 These help you understand the tradeoffs between model size, speed, and quality.

---

## 🧩 Troubleshooting Tips

- For RAM issues, cast models to FP16 or use `low_cpu_mem_usage=True`
- Use `torch.cuda.empty_cache()` and `del model` between steps if memory is tight
- If using your local environment, install packages from the `requirements.txt` provided