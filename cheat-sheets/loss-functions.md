| Loss Function            | Type                | Use Case                                  | Formula (Simplified) | Notes |
|--------------------------|---------------------|-------------------------------------------|----------------------|-------|
| **Mean Squared Error (MSE)** | Regression        | Continuous targets                        | $ \frac{1}{n} \sum (y - \hat{y})^2 $ | Penalizes large errors more |
| **Mean Absolute Error (MAE)** | Regression        | Continuous targets                        | $ \frac{1}{n} \sum \|y - \hat{y}\| $ | Less sensitive to outliers |
| **Huber Loss**           | Regression        | Continuous targets with robustness        | Mix of MSE and MAE   | Less sensitive to outliers than MSE |
| **Log-Cosh Loss**        | Regression        | Similar to Huber Loss                     | $ \sum \log(\cosh(y - \hat{y})) $ | Smooth alternative to MAE |
| **Binary Cross-Entropy** | Classification (Binary) | Binary classification (0/1 labels)       | $ - \frac{1}{n} \sum [y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})] $ | Used in logistic regression |
| **Categorical Cross-Entropy** | Classification (Multi-Class) | Multi-class classification               | $ - \sum y_i \log(\hat{y}_i) $ | Used for one-hot encoded labels |
| **Sparse Categorical Cross-Entropy** | Classification (Multi-Class) | Multi-class classification (integer labels) | $ - \sum y_i \log(\hat{y}_i) $ | Used when labels are integers instead of one-hot |
| **Kullback-Leibler Divergence (KL-Divergence)** | Probability Distributions | Comparing probability distributions       | $ \sum p(x) \log \frac{p(x)}{q(x)} $ | Measures divergence between two distributions |
| **Hinge Loss**           | Classification (Binary) | Support Vector Machines (SVM)            | $ \sum \max(0, 1 - y \cdot \hat{y}) $ | Used for margin-based classifiers |
| **Squared Hinge Loss**   | Classification (Binary) | SVM with quadratic penalty               | $ \sum (\max(0, 1 - y \cdot \hat{y}))^2 $ | Similar to hinge loss but squared |
| **Contrastive Loss**     | Metric Learning    | Siamese networks, similarity learning    | $ (1 - y) d^2 + y \max(0, m - d)^2 $ | Used in face verification and similarity tasks |
| **Triplet Loss**         | Metric Learning    | Learning embeddings                      | $ \max(0, d(a, p) - d(a, n) + margin) $ | Used in deep metric learning |
| **CTC Loss (Connectionist Temporal Classification)** | Sequence Learning | Speech recognition, handwriting recognition | Special alignment-based loss | Used when input and output lengths vary |
