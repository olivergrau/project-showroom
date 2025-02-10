| Activation Function  | Formula | Range | Properties | Pros | Cons | Use Cases |
|---------------------|---------|-------|------------|------|------|-----------|
| **Sigmoid** | $ \sigma(x) = \frac{1}{1 + e^{-x}} $ | (0,1) | Non-linear, S-shaped | Used for probabilities, smooth gradient | Vanishing gradient, not zero-centered | Binary classification, logistic regression |
| **Tanh (Hyperbolic Tangent)** | $ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $ | (-1,1) | Non-linear, zero-centered | Helps with symmetry, better than Sigmoid | Still suffers from vanishing gradients | RNNs, some classification tasks |
| **ReLU (Rectified Linear Unit)** | $ f(x) = \max(0, x) $ | [0, ∞) | Linear for $ x > 0 $, non-linear otherwise | Simple, computationally efficient, avoids vanishing gradient | Dying ReLU problem (neurons stuck at 0) | CNNs, Deep Networks |
| **Leaky ReLU** | $ f(x) = x $ if $ x > 0 $, else $ \alpha x $ | (-∞, ∞) | Small negative slope for $ x < 0 $ | Fixes dying ReLU issue | Still unbounded | CNNs, Deep Networks |
| **Parametric ReLU (PReLU)** | $ f(x) = x $ if $ x > 0 $, else $ \alpha x $ (where $ \alpha $ is learned) | (-∞, ∞) | Learnable slope parameter | Adaptive to data | Can lead to overfitting | CNNs, deeper networks |
| **ELU (Exponential Linear Unit)** | $ f(x) = x $ if $ x > 0 $, else $ \alpha (e^x - 1) $ | (-∞, ∞) | Smooth, non-linear | Avoids vanishing gradient, negative outputs help learning | Slightly more expensive computation | Deep learning architectures |
| **Swish** | $ f(x) = x \cdot \sigma(x) $ | (-∞, ∞) | Non-monotonic, smooth | Outperforms ReLU in some deep networks | More complex computation | Google’s EfficientNet, deep networks |
| **Softmax** | $ \sigma(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} $ | (0,1) | Normalized exponential function | Converts logits to probabilities | Susceptible to large values affecting output | Multi-class classification |
| **GELU (Gaussian Error Linear Unit)** | $ f(x) = 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3))) $ | (-∞, ∞) | Smooth, differentiable | Used in transformers, better than ReLU in some cases | More computationally expensive | Transformers (BERT, GPT) |
