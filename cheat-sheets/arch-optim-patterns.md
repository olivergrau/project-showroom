\# Cheat Sheet



\## Common Architectural Optimization Patterns for Neural Networks



---



\## Page 1 – Channel and Convolution Structure Optimizations



\### 1. Depthwise Separable Convolution



\*\*Core idea\*\*

Split spatial filtering and channel mixing into two steps:



1\. Depthwise convolution: spatial filtering per channel

2\. Pointwise convolution (1×1): channel mixing



\*\*Why it exists\*\*



\* Standard convolution mixes spatial and channel dimensions at once

\* Depthwise separable conv factorizes this into cheaper operations



\*\*Computational effect\*\*



\* Reduces FLOPs roughly by

&nbsp; \[

&nbsp; \\frac{1}{C\_{out}} + \\frac{1}{k^2}

&nbsp; ]

\* Massive savings for large channel counts



\*\*Where it shines\*\*



\* Mobile and embedded inference

\* Vision tasks with strong spatial locality

\* Architectures designed around it (MobileNet family)



\*\*Where it hurts\*\*



\* Early layers with few channels

\* Tasks requiring strong cross-channel spatial interactions

\* Naive drop-in replacement for standard convs



\*\*Typical usage\*\*



\* Depthwise + pointwise block

\* Almost always paired with batch norm and nonlinearity

\* Often combined with squeeze-excitation



---



\### 2. Grouped Convolution



\*\*Core idea\*\*

Split channels into groups, apply convolution independently per group.



\*\*Why it exists\*\*



\* Trade-off between full connectivity and depthwise isolation

\* Control cross-channel interaction granularity



\*\*Computational effect\*\*



\* FLOPs reduced by factor ≈ number of groups

\* Parameter count reduced proportionally



\*\*Where it shines\*\*



\* Medium to large channel counts

\* When features naturally cluster into subspaces

\* GPUs that handle grouped kernels efficiently



\*\*Where it hurts\*\*



\* Small models

\* CPUs with poor cache behavior

\* Arbitrary grouping without architectural intent



\*\*Special cases\*\*



\* groups = 1 → standard convolution

\* groups = in\_channels → depthwise convolution



---



\### 3. Channel Bottlenecks (1×1 Convolutions)



\*\*Core idea\*\*

Reduce channel dimensionality before expensive operations, then expand.



\*\*Why it exists\*\*



\* Most compute in CNNs comes from channel dimensions

\* Many features are redundant



\*\*Computational effect\*\*



\* Cheap projection → expensive operation → cheap expansion

\* Enables deep networks at manageable cost



\*\*Where it shines\*\*



\* Deep CNNs (ResNet, EfficientNet)

\* Models with high channel counts

\* When paired with residual connections



\*\*Where it hurts\*\*



\* Extremely small models

\* If bottleneck dimension is too aggressive



\*\*Key insight\*\*

This is a \*\*low-rank approximation in channel space\*\*, expressed architecturally.



---



\## Page 2 – Low-Rank and Factorization Patterns



\### 4. Spatial Factorization (k×k → k×1 + 1×k)



\*\*Core idea\*\*

Replace large square kernels with two separable kernels.



\*\*Why it exists\*\*



\* Large kernels are expensive

\* Spatial correlations often separable



\*\*Computational effect\*\*



\* FLOPs reduced from (k^2) to (2k)



\*\*Where it shines\*\*



\* Large receptive fields (k ≥ 5)

\* Vision tasks with smooth spatial structure



\*\*Where it hurts\*\*



\* Small kernels (3×3)

\* Highly anisotropic patterns



\*\*Typical usage\*\*



\* Inception-style architectures

\* Sometimes used selectively, not everywhere



---



\### 5. Low-Rank Linear Factorization



\*\*Core idea\*\*

Approximate a large linear transformation with two smaller ones.



\*\*Why it exists\*\*



\* Weight matrices often have low effective rank

\* Redundancy in learned representations



\*\*Computational effect\*\*



\* Parameters reduced from (N×M) to (N×r + r×M)



\*\*Where it shines\*\*



\* Large fully connected layers

\* Attention projections

\* MLP heads



\*\*Where it hurts\*\*



\* When true rank is high

\* Without fine-tuning or retraining



\*\*Important distinction\*\*



\* Architectural low-rank: built into the model

\* Post-training low-rank: factorizing trained weights



---



\## Page 3 – Connectivity and Information Flow



\### 6. Residual Connections



\*\*Core idea\*\*

Learn a residual function instead of a full transformation.



\*\*Why it exists\*\*



\* Mitigates vanishing gradients

\* Makes deep networks trainable



\*\*Computational effect\*\*



\* No direct FLOP reduction

\* Enables \*effective\* depth without degradation



\*\*Where it shines\*\*



\* Deep networks

\* Any optimization-constrained architecture



\*\*Where it hurts\*\*



\* Very shallow models

\* When overused without necessity



\*\*Optimization relevance\*\*

Residuals are \*\*enablers\*\* of aggressive compression and bottlenecks.



---



\### 7. Skip Connections with Feature Reuse (DenseNet-style)



\*\*Core idea\*\*

Reuse features explicitly instead of recomputing them.



\*\*Why it exists\*\*



\* Encourage feature sharing

\* Reduce redundant computation



\*\*Computational effect\*\*



\* Fewer filters per layer

\* More memory traffic



\*\*Where it shines\*\*



\* When memory is cheap

\* When features are highly reusable



\*\*Where it hurts\*\*



\* Mobile deployment

\* Memory-bound systems



---



\### 8. Attention as Conditional Computation



\*\*Core idea\*\*

Allocate computation dynamically based on relevance.



\*\*Why it exists\*\*



\* Not all features matter equally

\* Conditional routing is more efficient than uniform processing



\*\*Computational effect\*\*



\* Often increases FLOPs

\* Improves \*effective\* efficiency



\*\*Examples\*\*



\* Squeeze-and-Excitation

\* Channel attention

\* Spatial attention



\*\*Key insight\*\*

Attention is often an \*\*efficiency amplifier\*\*, not an efficiency primitive.



---



\## Page 4 – Sparsity and Conditional Execution



\### 9. Structured Sparsity (Channels, Filters, Blocks)



\*\*Core idea\*\*

Remove entire computational units instead of individual weights.



\*\*Why it exists\*\*



\* Hardware benefits from structured removal

\* Unstructured sparsity often does not speed up inference



\*\*Computational effect\*\*



\* Real latency reduction

\* Smaller memory footprint



\*\*Where it shines\*\*



\* Post-training optimization

\* Deployment-driven pipelines



\*\*Where it hurts\*\*



\* Early training

\* When applied blindly



---



\### 10. Mixture of Experts (MoE)



\*\*Core idea\*\*

Route inputs to a subset of expert subnetworks.



\*\*Why it exists\*\*



\* Conditional computation

\* Massive capacity with bounded cost



\*\*Computational effect\*\*



\* FLOPs per sample reduced

\* Total parameter count increased



\*\*Where it shines\*\*



\* Large models

\* Non-uniform data distributions



\*\*Where it hurts\*\*



\* Small models

\* Real-time constraints

\* Training instability



---



\## Page 5 – Architectural vs Post-Training Optimizations



\### 11. Architectural Efficiency vs Compression



\*\*Architectural patterns\*\*



\* Depthwise conv

\* Bottlenecks

\* Grouped conv

\* Factorization



These:



\* change representational capacity

\* require training adaptation



\*\*Post-training patterns\*\*



\* Pruning

\* Quantization

\* Low-rank decomposition



These:



\* exploit redundancy already learned

\* are often safer late in the pipeline



\*\*Rule of thumb\*\*

Architectural efficiency is \*\*design-time bias\*\*.

Compression is \*\*evidence-based optimization\*\*.



---



\## Page 6 – Practical Design Heuristics



\### When to use efficient architectural patterns upfront



\* Hard deployment constraints

\* Known task complexity

\* Proven architecture families

\* Mobile-first design



\### When not to



\* Research and exploration

\* Unknown data regimes

\* Early prototyping

\* When accuracy ceiling is critical



\### A healthy workflow



1\. Train expressive baseline

2\. Analyze redundancy

3\. Introduce structural constraints

4\. Fine-tune or distill

5\. Quantize and prune



---



\## Final mental model



Every architectural optimization pattern is a \*\*bet\*\*:



\* a bet that some structure is unnecessary

\* a bet that hardware will reward the constraint

\* a bet that training can adapt



Good engineers place bets \*\*after observing the system\*\*, not before.



---



\## Decision tree for architectural efficiency patterns



\### Step 0: What is your goal?



\*\*A. I need a model that trains well and hits max accuracy.\*\*

Start with standard convs / standard attention, keep things simple, optimize later.



\*\*B. I have hard deployment constraints (latency, memory, battery).\*\*

Design for efficiency from day one, but align the pattern with the target runtime.



---



\### Step 1: Where is your bottleneck?



\*\*1) Inference is compute-bound (FLOPs dominate).\*\*

You want to reduce FLOPs or use kernels that map better to hardware.



\*\*2) Inference is memory-bound (bandwidth, cache misses, weight reads dominate).\*\*

You want to reduce activation sizes, reduce memory traffic, improve locality.



\*\*3) Inference is overhead-bound (framework, dispatch, tiny models, Python, runtime).\*\*

Architecture changes often won’t move the needle much. Focus on export/runtime (TorchScript, ONNX, compiler), batching, and I/O.



---



\### Step 2: What kind of model is it?



\#### Branch A: CNN-heavy model (vision, audio, small/medium nets)



\*\*A1. Do you target mobile or edge CPUs/NPUs?\*\*



\* Yes → Prefer \*\*depthwise separable conv\*\* blocks, \*\*bottlenecks\*\*, maybe \*\*SE\*\*.

\* No → Prefer \*\*bottlenecks\*\*, and consider \*\*grouped conv\*\* only if kernels are efficient on your target.



\*\*A2. Are you allowed to redesign blocks, or only “tweak” a baseline?\*\*



\* Redesign allowed → Use proven families (MobileNet/EfficientNet style blocks) instead of inventing your own mix.

\* Only tweak → Use safer patterns: \*\*1×1 bottlenecks\*\*, occasional \*\*k×1 + 1×k\*\* factorization for large kernels.



\*\*A3. Is accuracy fragile / dataset small / task subtle?\*\*



\* Yes → Avoid aggressive depthwise/grouping everywhere. Use them selectively, or rely more on post-training compression later.

\* No → You can be more aggressive.



\#### Branch B: Transformer-heavy model (NLP, recsys, vision transformers)



\*\*B1. Is attention the main compute hog?\*\*



\* Yes → Look at \*\*attention variants\*\* or reducing sequence length first, then consider low-rank projections.

\* No, MLP dominates → Consider \*\*MLP factorization\*\* (low-rank), \*\*gated MLP variants\*\*, bottlenecked feed-forward.



\*\*B2. Do you need massive capacity but bounded compute?\*\*



\* Yes → \*\*Mixture of Experts (MoE)\*\* is the canonical conditional-compute pattern.

\* No → Keep dense for simplicity.



---



\### Step 3: Pick patterns based on “risk budget”



\*\*Low risk (usually safe):\*\*



\* Residual connections

\* Bottlenecking with 1×1 convs (CNN) or reduced FFN dimension (Transformers)

\* Spatial factorization for big kernels (k≥5)



\*\*Medium risk (needs validation on your hardware and task):\*\*



\* Grouped convolution

\* Depthwise separable convolution (if not using a known MobileNet-like block design)

\* SE / attention modules (can improve accuracy but may hurt latency)



\*\*High risk (architecture and training complexity):\*\*



\* MoE (routing, load balancing, stability)

\* Aggressive low-rank constraints everywhere

\* Heavy feature reuse designs that increase memory traffic



---



\### Step 4: The final sanity check (do not skip)



Before you commit:



1\. Does your target runtime have optimized kernels for this pattern?

2\. Does it reduce real latency, not only FLOPs?

3\. Does it preserve the inductive biases your task needs?



If any answer is “not sure”, treat the pattern as experimental, not a default.



---



\## Mapping patterns to real architectures



\### 1) ResNet (v1/v2)



\*\*Core patterns\*\*



\* Residual connections (the whole point)

\* Bottleneck blocks in deeper variants (ResNet-50/101/152)



\*\*What it’s betting on\*\*



\* Optimization stability matters more than clever operators

\* “Deep + residual” is a safe way to gain accuracy



\*\*Where efficiency comes from\*\*



\* Bottlenecks reduce compute while enabling depth



---



\### 2) MobileNet (v1/v2/v3)



\*\*Core patterns\*\*



\* Depthwise separable conv (v1)

\* Inverted residual + linear bottleneck (v2/v3)

\* Squeeze-and-Excitation (especially v3)

\* Sometimes hard-swish, careful block engineering



\*\*What it’s betting on\*\*



\* Spatial filtering per channel is “good enough” most of the time

\* Channel mixing can be delayed to 1×1 layers

\* Mobile runtimes reward depthwise kernels



\*\*Why you shouldn’t copy it partially\*\*



\* These blocks are co-designed. Randomly sprinkling depthwise conv into a generic CNN often disappoints.



---



\### 3) ShuffleNet (v1/v2)



\*\*Core patterns\*\*



\* Grouped conv (heavily)

\* Channel shuffle (to fix the “no cross-group mixing” problem)

\* Pointwise group conv tricks



\*\*What it’s betting on\*\*



\* You can get away with groups if you force cross-group mixing via shuffling

\* Very hardware-dependent: great when kernels are good, awkward when they are not



---



\### 4) EfficientNet



\*\*Core patterns\*\*



\* MBConv blocks (MobileNetV2-style inverted residuals)

\* SE blocks

\* Compound scaling (depth/width/resolution scaling rules)



\*\*What it’s betting on\*\*



\* Carefully scaled MobileNet-like blocks give strong accuracy per compute

\* SE improves representational efficiency enough to justify cost



---



\### 5) Inception (v2/v3) and Xception



\*\*Inception\*\*



\* Spatial factorization: k×k → k×1 + 1×k

\* Multi-branch designs (different receptive fields)

\* Lots of 1×1 bottlenecks



\*\*Xception\*\*



\* Depthwise separable conv taken seriously (often described as “extreme Inception”)



\*\*What they’re betting on\*\*



\* Spatial factorization captures structure with less compute

\* Branching gives expressive receptive fields without huge kernels



---



\### 6) DenseNet



\*\*Core patterns\*\*



\* Feature reuse via dense skip connections



\*\*What it’s betting on\*\*



\* Reusing features avoids redundant learning

\* Works well when compute is the bottleneck, but can become memory-traffic heavy



---



\### 7) Vision Transformers (ViT), DeiT, Swin



\*\*ViT/DeiT\*\*



\* Standard attention + MLP blocks, not “operator-efficient”

\* Efficiency comes from scaling rules and training tricks, not depthwise/grouped conv



\*\*Swin\*\*



\* Windowed attention (reduces quadratic attention cost)

\* Hierarchical representation (like CNN stages)



\*\*What they’re betting on\*\*



\* Attention gives great accuracy, but needs structural tricks (windowing, hierarchy) to be efficient



---



\### 8) Transformers in NLP (BERT/GPT-style)



\*\*Core patterns\*\*



\* Residual connections + LayerNorm

\* Big FFNs (often the compute hog)

\* Efficiency variants: grouped-query attention (GQA), multi-query attention (MQA) in some modern families



\*\*What they’re betting on\*\*



\* Dense transformations are stable and expressive

\* Efficiency is often achieved by attention variants or distillation, not by “depthwise-style” operators



---



\### 9) MoE Transformers (Switch Transformer, Mixtral-style families)



\*\*Core patterns\*\*



\* Mixture of Experts (conditional compute)

\* Sparse routing, load balancing losses



\*\*What they’re betting on\*\*



\* You can increase capacity dramatically without paying full compute per token

\* Complexity is worth it at large scale



---



\## If you want a crisp takeaway



\* \*\*CNN efficiency\*\* is mostly about \*conv structure\*: depthwise, groups, bottlenecks, factorization.

\* \*\*Transformer efficiency\*\* is mostly about \*sequence structure and routing\*: windowing, attention variants, MoE, and slimming FFNs.

