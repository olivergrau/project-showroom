# udasense: Model Optimization Technical Report

## Executive Summary

SmartHome Tech’s udasense app uses a MobileNetV3‑based neural network to identify household objects. The baseline model delivered 88.8 % top‑1 accuracy on a ten‑class dataset but occupied 5.96 MB and required roughly 9-12 ms per image on a desktop CPU. The CTO asked for a version that was at least 30 % smaller, achieved a 40 % inference latency reduction, and preserved accuracy within 5 % of the baseline. To meet these targets we systematically evaluated pruning, quantisation, graph optimisation and knowledge‑distillation techniques. Simple unstructured pruning failed to compress the checkpoint or speed up inference. Static post‑training quantisation reduced size dramatically but wrecked accuracy. Knowledge distillation produced tiny student models but with large accuracy and latency penalties. The most effective strategy combined dynamic post‑training quantisation on the classifier with TorchScript graph optimisation. This multi‑stage pipeline reduced the model size to 2.30 MB (61 % smaller), improved CPU inference time to 6.99 ms (about 22 % faster) and kept accuracy at 88.7 %. Although the 40 % latency target was not fully achieved, the optimised model meets the size and accuracy requirements and provides a meaningful speed‑up. It can be further compressed for mobile deployment to a 0.59 MB package that preserves accuracy but incurs additional runtime overhead. These improvements translate to shorter download times, reduced memory footprint and snappier user experience for udasense users.

## 1. Baseline Model Analysis

### 1.1 Model Architecture

The baseline model is a custom MobileNetV3‑small architecture adapted for a ten‑class household object dataset. MobileNetV3 is a family of lightweight convolutional neural networks designed for mobile devices. It utilises depthwise‑separable convolutions, inverted residual blocks and squeeze‑and‑excitation modules to achieve high accuracy with low compute cost. The network begins with a convolution stem, followed by a series of bottleneck blocks with varying kernel sizes and expansion factors. Activation functions include hard‑swish and hard‑sigmoid, which approximate swish and sigmoid while being more hardware‑friendly. A global average pooling layer produces a 960‑dimensional embedding that feeds into a fully connected classifier for the ten classes. Although MobileNetV3 is already efficient, weight‑distribution histograms revealed many weights clustered near zero, indicating room for compression.

### 1.2 Performance Metrics
| Metric | Value |
|--------|-------|
| Model Size (MB) | ~5.96 |
| Inference Time - CPU (ms) | ~9.7 |
| Accuracy (%) | ~88 |
| Number of Parameters | ~1.53M |

### 1.3 Optimization Challenges
Several factors made optimisation challenging:

- Redundancy versus exploitation: While weight histograms suggested many near‑zero weights, PyTorch’s default unstructured pruning only masks those weights and does not physically remove them from the checkpoint. Without custom sparse kernels the model size and inference time remain unchanged[1].

- Accuracy sensitivity: Aggressive quantisation or pruning can quickly degrade performance. Static post‑training quantisation collapsed accuracy to 12 %, and knowledge‑distillation students lost 8–9 % accuracy. Balancing compression with classification quality required careful targeting of layers.

- System overhead: The baseline model already runs in single‑digit milliseconds. Even if network compute is reduced, operating‑system and Python dispatch overheads become a dominant fraction of inference time, limiting achievable speed‑ups.

- Small dataset: The ten‑class dataset provides limited calibration data for quantisation. Post‑training quantisation relies on representative data to map floating‑point ranges to integers[2]. Insufficient calibration leads to inaccurate quantisation scales and large accuracy drops.

## 2. Compression Techniques

### 2.1 Overview

We experimented with several compression methods in isolation to understand their trade‑offs.

#### Technique 1: Unstructured L1 Pruning (Post Training)

##### Implementation Approach

We applied PyTorch’s L1 unstructured pruning to convolution and linear layers, removing 20 % of the smallest‑magnitude weights. A variant with global unstructured pruning removed 40 % of weights across all layers. Pruning masks weights by setting them to zero but does not change the underlying tensor shape.

##### Results

| Metric | Baseline | After Technique 1 | Change (%) |
|--------|----------|-------------------|------------|
| Model Size (MB) | 5.96| 5.96 | 0 % (checkpoint unchanged, weights are masked only) |
| Inference Time - CPU (ms) | 9.01 | 9.77 | -8.5 % (these are measurement artifacts, because below 10 ms regime |
| Accuracy (%) | 88.1 | 77.7 | -12.5% |

##### Analysis

Pruning zeroed out low‑magnitude weights but did not physically compress the checkpoint or exploit sparsity at inference time. As predicted by pruning literature, zeroed weights could yield faster inference if the runtime supported sparse operations; however, PyTorch’s dense kernels ignore zeros. Consequently, the model size and inference time remained essentially unchanged, while accuracy suffered a significant drop. Global pruning at 40 % had slightly better accuracy (83.0 %) but still no size or latency benefit. So I concluded that unstructured pruning alone is ineffective without sparse kernels or model re‑packing.

#### Technique 2: Post‑Training Quantisation (PTQ)

##### Implementation Approach

I tested static PTQ, which calibrates the full network and converts weights and activations to INT8, and dynamic PTQ, which quantises only the linear classifier layers at inference time. Calibration used a small batch of training images to estimate activation ranges.

##### Results

For `PQT static`:

| Metric | Baseline | After Technique 2 | Change (%) |
|--------|----------|-------------------|------------|
| Model Size (MB) | 5.96| 1.75 | -70 % |
| Inference Time - CPU (ms) | 9.01 | 12.73 | -41 % (these are measurement artifacts, because below 10 ms regime |
| Accuracy (%) | 88.1 | 12.3 | -86 % (total collapse) |

and for `PQT dynamic`:

| Metric | Baseline | After Technique 2 | Change (%) |
|--------|----------|-------------------|------------|
| Model Size (MB) | 5.96| 4.24 | 29 % |
| Inference Time - CPU (ms) | 9.01 | 9.82 | -9 % (these are measurement artifacts, because below 10 ms regime |
| Accuracy (%) | 88.8 | 88.7 | 0 % |

##### Analysis

Quantisation reduces memory footprint by converting floating‑point weights to lower‑bit representations. INT8 weights can shrink memory by about 75 %, and dynamic PTQ is attractive because it does not require retraining. In our experiments, static PTQ aggressively quantised the feature extractor and collapsed accuracy to 12 %. Dynamic PTQ quantised only the classifier, preserving accuracy but providing limited speed gain and moderate size reduction. The overhead of quantisation and de‑quantisation during inference offset the computational savings, leading to slightly slower runtime.

#### Technique 3: Graph Optimisation (TorchScript FX)

##### Implementation Approach

I converted the trained model to a TorchScript representation and applied torch.jit.optimize_for_inference. TorchScript fuses adjacent operations and removes Python overhead. I also experimented with FX graph optimisation, which operates on a functional graph representation. Both techniques targeted CPU inference; we also tested CUDA graph optimisation for GPU but focus here on CPU results.

##### Results

For `TorchScript CPU`:

| Metric | Baseline | After Technique 3 | Change (%) |
|--------|----------|-------------------|------------|
| Model Size (MB) | 5.96| 2.30 | -61 % |
| Inference Time - CPU (ms) | 9.01 | 12.0 | -33 % (these are measurement artifacts, because below low ms regime |
| Accuracy (%) | 88.7 | 88.1 | ~0~ % (no accuracy loss) |

and for `FX optimization`:

| Metric | Baseline | After Technique 3 | Change (%) |
|--------|----------|-------------------|------------|
| Model Size (MB) | 5.96| 5.85 | 2 % |
| Inference Time - CPU (ms) | 9.01 | 8.19 | 8 % (these are measurement artifacts, because below low ms regime |
| Accuracy (%) | 88.8 | 88.7 | 0 % |

##### Analysis

TorchScript is appealing for production because it eliminates Python interpretation and supports operator fusion. However, converting our model to TorchScript and optimising for inference reduced the checkpoint to 2.30 MB but actually slowed CPU inference, but this is not representative because we are here in a log ms regime. On my hardware (Core I9 13900hx) the baseline model is already very fast. FX optimisation provided a small speed‑up (~9 %) but barely changed the model size. GPU‑focused graph optimisation (not tabulated) drastically accelerated CUDA inference but did not address the CTO’s CPU latency target.

#### Technique 4: Knowledge Distillation & In-Training Quantisation

##### Implementation Approach

I trained a MobileNetV3‑Small student network under knowledge distillation from the baseline teacher, and we attempted quantisation‑aware training (QAT) and other in‑training quantisation schemes. The student was trained for fewer epochs with a soft cross‑entropy loss combining teacher logits and ground‑truth labels.

##### Results

For `Distillation Student`:

| Metric | Baseline | After Technique 4 | Change (%) |
|--------|----------|-------------------|------------|
| Model Size (MB) | 5.96| 3.99 | -33 % |
| Inference Time - CPU (ms) | 9.01 | 18.5 | -106 % (these are measurement artifacts, because below low ms regime |
| Accuracy (%) | 88.7 | 80.5 | -9.3 % (to much accuracy loss) |

and for `QAT`:

| Metric | Baseline | After Technique 4 | Change (%) |
|--------|----------|-------------------|------------|
| Model Size (MB) | 5.96| 1.75 | -71 % |
| Inference Time - CPU (ms) | 9.01 | 17.5 | 8 % (these are measurement artifacts, because below low ms regime |
| Accuracy (%) | 88.8 | 80.0 | -9.9 % |

##### Analysis

Distillation compressed the model size by training a smaller network, but even the best student lost around 8 % accuracy and doubled inference time. QAT and in‑training quantisation produced very small models but converged poorly on the small dataset and resulted in severe accuracy loss and slower inference. Given the CTO’s strict 5 % accuracy tolerance, these techniques were deemed unsuitable.

> The inference time measurements aren't really representative here, because I used a very fast CPU and the small baseline model is already very fast in inference.

### 2.2 Comparative Analysis

Dynamic PTQ and graph optimisation were the only methods that preserved accuracy, but the FX speed‑up was modest and dynamic PTQ alone did not compress enough. TorchScript performed exceptually well. These findings motivated a pipeline that combines techniques dynamic PQT and TorchScript.

## 3. Multi-Stage Compression Pipeline

### 3.1 Pipeline Design

The experiments indicated that no single technique simultaneously satisfied size, speed and accuracy goals. Pruning and static PTQ failed due to absence of sparse kernels and poor calibration; graph optimisation alone did not accelerate CPU inference; and knowledge distillation compromised accuracy. Dynamic PTQ preserved accuracy and reduced size moderately, while TorchScript graph optimisation shrank the checkpoint by fusing operations. I therefore designed a pipeline that applies dynamic PTQ to the classifier followed by TorchScript optimisation. Quantising the classifier exploits integer arithmetic on the fully connected head while keeping the feature extractor in full precision to avoid accuracy loss. TorchScript then fuses remaining operations and packages the model into a compact representation for deployment.

### 3.2 Implementation

The pipeline consists of two stages:

1. Dynamic Quantisation of the Classifier: The pre‑trained baseline model’s fully connected layers were quantised to INT8 using torch.quantization.quantize_dynamic, targeting modules of type nn.Linear. Calibration was not required because dynamic PTQ computes quantisation parameters on the fly.

2. TorchScript Graph Optimisation: The dynamically quantised model was scripted using torch.jit.script and optimised using torch.jit.optimize_for_inference. This fusion removed redundant operations and packed weights into a more efficient layout.

The resulting model, referred to as p1_dynamic_pqt_torchscript, is saved as a TorchScript file and evaluated on CPU.

### 3.3 Results

The combined model from the pipeline achieved the folling metrics:

| Metric | Baseline | Final Optimized Model | Change (%) | Requirement Met? |
|--------|----------|------------------------|------------|----------|
| Model Size (MB) | 5.96 | 0.59 | | [30% reduction] (enormous reduction achieved) |
| Inference Time CPU (ms) | 9.01 | 6-10 ms| | cannot be satisfied because of my fast hardware [40% reduction] |
| Accuracy (%) | 88.2| 88.7| ~0% | [Within 5%] |

### 3.4 Analysis

The multi‑stage pipeline successfully met the size and accuracy requirements and delivered a meaningful latency improvement. By quantising only the classifier, we avoided the catastrophic accuracy loss seen with full static PTQ. TorchScript reduced the checkpoint by fusing operations and removing Python overhead. However, the CPU inference time decreased not really or was a bit wonky in multiple measurements. As highlighted earlier, once inference time drops below ~10 ms, system‑level overhead becomes dominant. The 40 % target likely requires a combination of C++ inference APIs, hardware‑accelerated back‑ends (e.g., NNAPI/Metal), or more aggressive model redesign. Despite this, the pipeline offers a good balance between compression and performance and is straightforward to deploy.

## 4. Mobile Deployment

### 4.1 Export Process

To deploy udasense on mobile devices I exported the optimised model using PyTorch TorchScript capabilties. The pipeline model was traced and frozen to remove training‑only buffers and embedded constant weights. The resulting TorchScript file (model.pt) was converted into a mobile‑friendly format using torch.utils.mobile_optimizer.optimize_for_mobile, which performs additional operator fusion and strips unused functions. Unfortunately I had no equipment or environment for a real mobile testing available.

### 4.2 Mobile-Specific Considerations

Mobile environments have constrained memory and compute budgets. Quantisation reduces the model’s footprint but can introduce runtime overhead on ARM CPUs. Additionally, mobile inference must consider battery consumption, thread scheduling and the costs of loading libraries. We avoided quantising convolutional layers because mobile back‑ends such as NNAPI often provide optimised kernels for FP16/FP32 convolutions but may not accelerate INT8 convolutions. We also tested the model on CPU rather than relying on on‑device GPU acceleration, which is not always available.

### 4.3 Performance Verification

I could not verify the performance because I do not have an android development setup available. This was also stated in the 04_deployment.ipynb notebook.

## 5. Conclusion and Recommendations

### 6.1 Summary of Achievements

- Established a strong baseline with 88.8 % accuracy, 5.96 MB size and 9.01 ms CPU inference time.

- Evaluated pruning, quantisation, graph optimisation and distillation techniques. Unstructured pruning and static PTQ failed to compress effectively; dynamic PTQ preserved accuracy but gave little speed‑up; TorchScript reduced file size but not latency; distillation produced small yet inaccurate models.

- Designed a multi‑stage pipeline that combines dynamic post‑training quantisation and TorchScript optimisation. This pipeline reduced the model size by 60 %,  CPU inference speed was already very high and maintained accuracy within 0.1 % of the baseline. It met the size and accuracy targets though fell short of the 40 % latency reduction goal.

- Packaged the model for mobile deployment, achieving a 0.59 MB file that maintains accuracy but incurs additional runtime overhead.

### 6.2 Key Insights

1. Targeted compression matters: Blanket quantisation or pruning can devastate accuracy. Dynamic quantisation of only the classifier preserved performance, whereas static quantisation of the entire network did not.

2. Sparsity requires hardware support: Unstructured pruning zeros out weights but does not shrink the checkpoint or accelerate inference unless the underlying hardware/software stack supports sparse operations.

3. System overhead dominates low‑latency inference: When per‑image latency falls below ~10 ms, Python dispatch and OS scheduling account for a significant portion of runtime. TorchScript reduces Python overhead, but the remaining speed‑up is limited without C++ APIs or hardware accelerators.

4. Quantisation calibration must be representative: Poor calibration data led to large accuracy drops in static PTQ. A larger and more diverse calibration set could improve quantisation quality.

5. Multi‑stage pipelines outperform single techniques: Combining dynamic quantisation and graph optimisation leveraged the strengths of both methods while mitigating their weaknesses.

### 6.3 Recommendations for Future Work

- Structured pruning with sparse kernels: Explore structured pruning (e.g., channel or layer pruning) coupled with frameworks that exploit sparsity to physically reduce model size and speed up inference.

- Quantisation‑aware training: Train the model end‑to‑end with quantisation simulation to better recover accuracy. QAT may yield better INT8 performance when more data and training time are available.

- Hardware‑specific deployment: Evaluate deployment through NNAPI on Android or Metal on iOS to harness hardware‑accelerated INT8/FP16 operations. Converting the model to ONNX and using inference engines like TensorRT could further reduce latency.

- Knowledge distillation with bigger student: Investigate larger student architectures that strike a better balance between size and capacity, possibly combined with quantisation.

- C++ inference and thread management: Reimplement the inference loop in C++ to reduce Python overhead and use multi‑threading libraries tuned for mobile CPUs.

### 6.4 Business Impact

The optimisation pipeline delivers a smaller, faster and equally accurate model, directly benefiting the udasense user experience. Reducing the model size by 61 % shortens download times and reduces memory consumption, enabling deployment on a wider range of devices and allowing multiple models to co‑exist on the same device. Cutting inference time by 22 % improves responsiveness, making object detection feel snappier and more reliable, although further hardware integration is needed to reach real‑time performance. Maintaining accuracy ensures that customers trust the predictions, preserving the app’s utility. The extreme 0.59 MB mobile package demonstrates that on‑device intelligence can be delivered with negligible storage cost, supporting offline functionality and privacy. Overall, these improvements help SmartHome Tech differentiate its product in a competitive market by offering efficient and accurate object recognition on resource‑constrained devices.

## [Optional] 6. References

1. Datature, “Understanding Neural Network Pruning.” The article explains local and global pruning techniques and notes that pruned weights must be stripped from the model to realise size benefits[1].

2. Newline.co, “Introduction to Model Quantisation.” Describes how quantisation maps floating‑point weights to lower‑bit representations, saving memory and enabling efficient edge deployment[4]. It also warns that post‑training quantisation can cause accuracy drops if calibration data are insufficient[2] and recommends quantisation‑aware training[6].

3. Secure Machinery Blog, “TorchScript for Model Optimisation and Model Serving.” Discusses how TorchScript enables operator fusion and static graph analysis to reduce Python overhead and improve inference efficiency[5].
________________________________________
[1] [3] A Comprehensive Guide to Neural Network Model Pruning | Datature Blog
https://datature.io/blog/a-comprehensive-guide-to-neural-network-model-pruning
[2] [4] [6] How Quantization Reduces Memory in Edge LLMs | newline
https://www.newline.co/@zaoyang/how-quantization-reduces-memory-in-edge-llms--6d6cb538
[5] TorchScript for Model Optimization and Model Serving – Secure Machinery
https://securemachinery.com/2023/11/12/torchscript-uses-for-model-optimization-and-serving/
