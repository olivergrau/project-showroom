# Implementation plan for Notebook: 03_pipeline.py

## 1. Overall goals and design

We want:

* A reusable `OptimizationPipeline` that can chain multiple compression steps.
* At least one **concrete pipeline** that uses **two effective methods**:

  * Dynamic post training quantization on the classifier.
  * TorchScript graph optimization on CPU.

Important constraints:

* Use the **same evaluation utilities** as `02_compression.py` so all metrics are comparable:

  * `evaluate_optimized_model`
  * `compare_optimized_model_to_baseline`
* Make `visualize_results` work by providing `self.results['steps']` with the expected structure:

  ```python
  {
      "step_name": str,
      "metrics": {
          "size": {...},
          "timing": {...},
          "accuracy": {...}
      },
      "comparison": {...}  # optional, but useful
  }
  ```

---

## 2. Standardize the “step” interface

Define a clear contract for a pipeline step function.

### 2.1 Step function signature

Each helper like `apply_dynamic_quantization` and `apply_graph_optimization` should follow a common signature, for example:

```python
def apply_dynamic_quantization(
    model,
    train_loader,
    test_loader,
    class_names,
    input_size,
    device,
    experiment_root,
    **kwargs,
):
    """
    Returns:
        optimized_model
        metrics_dict   # same schema as baseline_metrics
        comparison_dict
        experiment_name  # string used for folder naming
    """
```

The same pattern for `apply_graph_optimization`:

```python
def apply_graph_optimization(
    model,
    train_loader,
    test_loader,
    class_names,
    input_size,
    device,
    experiment_root,
    **kwargs,
):
    ...
```

Where:

* `experiment_root` is something like `f"./results/pipeline/{pipeline_name}"` or a subfolder inside that, so each step has a unique subdirectory.
* `metrics_dict` should be exactly what `visualize_results` expects:

  ```python
  {
    "size": {
      "model_size_mb": float,
      "total_params": int
    },
    "timing": {
      "cpu": {"avg_time_ms": float, ...},
      "cuda": {"avg_time_ms": float, ...}  # if available
    },
    "accuracy": {
      "top1_acc": float,
      ...
    }
  }
  ```

We have two options:

1. Call `evaluate_optimized_model` and then **read its metrics JSON**.
2. Or call the lower level utilities directly (`measure_model_size`, `measure_inference_time`, `measure_accuracy`) inside the step function.

Plan: keep it simple and consistent. Use **the same pattern** that `02_compression.py` uses for other experiments. If `evaluate_optimized_model` already writes a metrics file with this schema, we can load that, then return it.

---

## 3. Implementation plan for `OptimizationPipeline.run`

`run` currently does nothing inside the loop. The plan:

### 3.1 Use a context dictionary

To avoid passing twenty arguments around, prepare a small context dict in `run`:

```python
context = {
    "train_loader": self.train_loader,
    "test_loader": self.test_loader,
    "class_names": self.class_names,
    "input_size": self.input_size,
    "device": device,
    "experiment_root": self.results_dir,
}
```

### 3.2 Loop over steps

For each step in `self.steps`:

1. Retrieve:

   * `step_name = step["name"]`
   * `step_fn = step["fn"]`
   * `step_kwargs = step["kwargs"]`
2. Print a small header for the step.
3. Call the function with:

   ```python
   optimized_model, metrics, comparison, experiment_name = step_fn(
       current_model,
       **context,
       **step_kwargs,
   )
   ```
4. Append to `step_results` a dict like:

   ```python
   step_results.append({
       "step_name": step_name,
       "experiment_name": experiment_name,
       "metrics": metrics,
       "comparison": comparison,
   })
   ```
5. Set `current_model = optimized_model`.

### 3.3 Saving the final model

At the end of `run`:

* Store `self.optimized_model = current_model`.

* Use `file_extension` to decide how to save:

  * If `file_extension == "pt"`:

    * Assume `current_model` is a **TorchScript** object and call `torch.jit.save`.
  * Otherwise:

    * Assume `current_model` is a normal `nn.Module` and call `save_model` from the UdaSensety utilities.

* Set `final_metrics` and `final_comparison` in `self.results`:

  * Either copy them from the last step in `step_results`.
  * Or re evaluate `current_model` one more time with an experiment name like `"final"`.

* Write `self.results` to `pipeline_metrics.json`.

Conceptual structure:

```python
self.results = {
    "pipeline_name": self.name,
    "steps": step_results,
    "final_metrics": step_results[-1]["metrics"],
    "final_comparison": step_results[-1]["comparison"],
}
```

If you want to be conservative, you can re evaluate the final model and treat that as the authoritative final metrics.

---

## 4. Implement the step helpers

We only need the **two helpful ones** for now. The others can stay as explicit stubs.

### 4.1 `apply_dynamic_quantization`

Goal: reuse what worked in `02_compression.py`, but integrate it into the pipeline.

Conceptual steps:

1. Decide experiment name:

   ```python
   experiment_name = os.path.join(experiment_root, "dynamic_quantization")
   ```

   or include a sequence number, for example `"01_dynamic_quantization"` to help ordering.

2. Move model to CPU, because dynamic quantization is CPU focused.

3. Either:

   * Use `quantize_model` directly as in `02_compression.py` with `quantization_type="dynamic"`.
   * Or reuse `apply_post_training_quantization` with `quantization_type="dynamic"`, but be careful with its **internal experiment paths**, or override them via parameters if that is possible.

4. Evaluate:

   * Call `evaluate_optimized_model` with:

     * `optimized_model`
     * `test_loader`
     * `experiment_name`
     * `class_names`
     * `input_size`
     * `device=torch.device('cpu')`
   * Call `compare_optimized_model_to_baseline` with:

     * `baseline_model` (from context)
     * `optimized_model`
     * `experiment_name`
     * `test_loader`
     * etc.

5. Load metrics JSON written by `evaluate_optimized_model` from `./results/{experiment_name}/metrics.json` and return it as `metrics_dict`.

Return values:

```python
return optimized_model, metrics_dict, comparison_dict, experiment_name
```

Possible pitfall:

* We combine dynamic quantization with TorchScript later. There is a small risk that some quantized layers do not like tracing. If this fails, we still have a valid pipeline with **TorchScript only**, and a second pipeline with **dynamic quantization only**.

So the plan is robust: implement both, then empirically choose which combination satisfies UdaSensety’s constraints.

---

### 4.2 `apply_graph_optimization` (TorchScript CPU)

Goal: apply the same TorchScript pipeline that already worked in `02_compression.py`.

Conceptual steps:

1. Decide experiment name:

   ```python
   experiment_name = os.path.join(experiment_root, "torchscript_cpu")
   ```

2. Prepare model for tracing:

   * Ensure `model.eval()`.
   * Move to CPU.

3. Create an example input:

   * Use `input_size` to build a dummy tensor, for example `1 x C x H x W`.

4. Trace the model and optimize for inference:

   ```python
   traced = torch.jit.trace(model, example_input)
   optimized = torch.jit.optimize_for_inference(traced)
   ```

5. Evaluate:

   * Call `evaluate_optimized_model` with the **TorchScript model** in CPU mode.
   * Call `compare_optimized_model_to_baseline` as before.

6. Load metrics JSON and return:

   ```python
   return optimized, metrics_dict, comparison_dict, experiment_name
   ```

Think about saving:

* Inside the step function you can also save the intermediate TorchScript model to `./models/pipeline/{pipeline_name}/torchscript_cpu/model.pt` for debugging.
* The final saving in `run` will save the last model again into `model.{file_extension}`.

---

### 4.3 Other helpers

For completeness, keep explicit stubs that make the intent obvious:

```python
def apply_post_training_pruning(...):
    raise NotImplementedError("Not used in the final pipeline; pruning did not yield useful results on MobileNetV3.")

def apply_knowledge_distillation(...):
    raise NotImplementedError("Optional extension; KD was explored in notebook 2 but not part of the main pipeline.")

def apply_in_training_pruning(...):
    raise NotImplementedError("In training pruning is out of scope for this pipeline implementation.")

def apply_in_training_quantization(...):
    raise NotImplementedError("QAT is already known to fail the accuracy constraint for this architecture.")
```

So the reader sees that their absence is intentional, not an oversight.

---

## 5. Defining concrete pipelines

Finally, decide how to use `add_step`.

### 5.1 Pipeline 1: TorchScript only

This is your safe, high performing solution.

```python
pipeline1 = OptimizationPipeline(..., name="p1_torchscript_only")
pipeline1.add_step(
    "TorchScript CPU",
    apply_graph_optimization,
    # optional kwargs like:
    # ts_backend="cpu"
)

optimized_model_p1 = pipeline1.run(
    device=torch.device("cpu"),
    file_extension="pt",
)
pipeline1.visualize_results(baseline_metrics, device=torch.device("cpu"))
```

This pipeline mirrors the single best technique from notebook 2 and should already meet all three targets.

### 5.2 Pipeline 2: Dynamic quantization + TorchScript

Second pipeline that combines two methods. Two variants:

1. **Dynamic quantization first, then TorchScript**:

   ```python
   pipeline2 = OptimizationPipeline(..., name="p2_dynq_then_ts")
   pipeline2.add_step("Dynamic Quantization", apply_dynamic_quantization)
   pipeline2.add_step("TorchScript CPU", apply_graph_optimization)
   ```

2. Or the reverse order, if we encounter technical issues.

Empirically we can see:

* Does combining both still preserve accuracy within 5 percentage points?
* Do we get additional size or latency benefits over TorchScript alone?

If the combined version misbehaves, we still have pipeline 1 as a rock solid solution and can describe pipeline 2 as an additional experiment in the report.

---

## 6. Final consistency checks

Before coding, keep these in mind:

* Ensure **all paths** (models and results) are under `./models/pipeline/{pipeline_name}` and `./results/pipeline/{pipeline_name}` so they do not conflict with experiments from notebook 2.
* Ensure the **metrics schema** returned by the step functions matches what `visualize_results` expects.
* Make sure `device` is handled consistently:

  * For our final evaluation we only really care about **CPU** metrics, since that is what UdaSensety measures.
* Keep the baseline model untouched:

  * Always work on a deep copy or freshly loaded version when you quantize or trace.

---

If you want, next step we can take this plan and translate it into concrete code, step by step, starting with:

1. Implementing the common step function interface.
2. Implementing `apply_dynamic_quantization`.
3. Implementing `apply_graph_optimization`.
4. Wiring `OptimizationPipeline.run`.
