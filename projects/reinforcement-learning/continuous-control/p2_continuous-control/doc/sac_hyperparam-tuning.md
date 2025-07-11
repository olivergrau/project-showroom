# **📜 Soft Actor-Critic (SAC) Hyperparameters Cheat Sheet 🚀**
This cheat sheet provides a **comprehensive overview** of **SAC hyperparameters** and their effects on **loss curves, alpha, log probabilities, actor loss, critic loss, and Q-values**.

---

## **1️⃣ Learning Rates**
| **Hyperparameter** | **Effect** | **Observations in Metrics** | **Fix if Wrong** |
|-------------------|-----------|-----------------------------|-----------------|
| **`lr_actor`** *(Actor Learning Rate)* | Controls how fast the policy (actor) updates. | **Too high** → Actor loss unstable. <br> **Too low** → Slow convergence. | **If unstable:** Lower it (e.g., `1e-3 → 3e-4`). <br> **If learning is too slow:** Increase slightly. |
| **`lr_critic`** *(Critic Learning Rate)* | Affects how quickly the critic learns Q-values. | **Too high** → Q-values oscillate, loss unstable. <br> **Too low** → Q-values increase slowly, critic overfits. | If **critic loss is unstable**, lower it (`1e-3 → 3e-4`). |
| **`lr_alpha`** *(Temperature Learning Rate)* | Controls how fast entropy (α) adapts. | **Too high** → Alpha changes too quickly, unstable log prob. <br> **Too low** → Alpha updates too slowly. | Lower to `1e-4` if α is changing too fast. |

---

## **2️⃣ Entropy & Exploration (α - Temperature Parameter)**
| **Hyperparameter** | **Effect** | **Observations in Metrics** | **Fix if Wrong** |
|-------------------|-----------|-----------------------------|-----------------|
| **`target_entropy`** | Encourages exploration. Set to `-action_dim` by default. | **If too high** → The agent explores too much, slow convergence. <br> **If too low** → The agent becomes deterministic too quickly. | Try `-0.5 * action_dim` to reduce excessive randomness. |
| **`alpha` (if fixed)** | Controls trade-off between exploration & exploitation. | **High α** → More exploration, higher log_probs. <br> **Low α** → Less exploration, overfits to exploitation. | If **log_probs are too high**, lower α. If **eval rewards are poor**, increase α. |
| **`log_alpha`** *(learned α parameter)* | If α is learned, `log_alpha` determines its update. | **If decreasing too much** → Check `target_entropy`. <br> **If increasing too much** → Training might be too stochastic. | Clamp `log_alpha` to prevent extreme changes. |

🔹 **How it affects metrics:**
- **Increasing α** → **Higher log_probs**, encourages exploration.
- **Decreasing α** → **Lower log_probs**, reduces randomness.
- **Wrong α** → Actor loss oscillates, critic loss diverges.

---

## **3️⃣ Critic & Q-Value Stability**
| **Hyperparameter** | **Effect** | **Observations in Metrics** | **Fix if Wrong** |
|-------------------|-----------|-----------------------------|-----------------|
| **`gamma` (Discount Factor)** | Balances short-term & long-term rewards. | **High (0.99)** → Stable learning but slow updates. <br> **Low (0.9)** → Faster updates but less stability. | Use `0.99` unless rewards are highly delayed. |
| **`tau` (Soft Update Rate)** | Controls how quickly target Q-networks update. | **High (0.01)** → Target Q-values update fast, might diverge. <br> **Low (0.005)** → Smoother learning, more stability. | Try `0.005` for stability, `0.01` for faster adaptation. |
| **`critic_loss` (MSE Loss on Q-values)** | Indicates how well Q-values approximate true returns. | **If loss is too high** → Q-values might be overestimated. <br> **If loss is too low** → Critic might not generalize well. | Check reward scaling and batch size. |

🔹 **How it affects metrics:**
- **Q-values too high?** → Reduce `lr_critic`, adjust reward scaling.
- **Q-values too low?** → Increase `lr_critic`, improve critic training.
- **Critic loss unstable?** → Adjust `tau`, lower learning rate.

---

## **4️⃣ Training Efficiency**
| **Hyperparameter** | **Effect** | **Observations in Metrics** | **Fix if Wrong** |
|-------------------|-----------|-----------------------------|-----------------|
| **`batch_size`** | Number of samples per update. | **Small batch (128)** → Noisy updates. <br> **Large batch (512)** → Smoother training, slower updates. | Use `256-512` for stability. |
| **`replay_buffer_size`** | Stores past transitions for off-policy training. | **Too small** → Overfitting, poor generalization. <br> **Too large** → Old transitions hurt learning. | Keep `100,000 - 1,000,000` for SAC. |
| **`env_steps_per_update`** | How often updates occur relative to env steps. | **If too high** → Delayed updates, policy lags behind. <br> **If too low** → Wastes computation, slow training. | Use `50` (common for SAC). |
| **`updates_per_block`** | Number of updates per training step. | **If too high** → Overfitting, policy updates too fast. <br> **If too low** → Training is too slow. | Use `10-20` per update cycle. |

🔹 **How it affects metrics:**
- **If training is too slow** → Increase `updates_per_block`, `batch_size`.
- **If overfitting occurs** → Reduce `updates_per_block`, use `target_entropy`.

---

## **5️⃣ Actor-Critic Synchronization**
| **Hyperparameter** | **Effect** | **Observations in Metrics** | **Fix if Wrong** |
|-------------------|-----------|-----------------------------|-----------------|
| **`soft_update()` (`tau`)** | Controls the target Q-network updates. | **If too low** → Q-values lag behind, slow training. <br> **If too high** → Unstable Q-values. | `tau = 0.005` is standard. |
| **`gradient_clipping`** | Prevents exploding gradients. | **If too low** → Gradients explode, loss spikes. <br> **If too high** → Training slows down. | Clip at `5.0` if needed. |

🔹 **How it affects metrics:**
- **Diverging Q-values?** → Reduce `tau`, clip gradients.
- **Slow Q-updates?** → Increase `tau`.

---

## **6️⃣ Reward Scaling**
| **Hyperparameter** | **Effect** | **Observations in Metrics** | **Fix if Wrong** |
|-------------------|-----------|-----------------------------|-----------------|
| **`reward_scaling_factor`** | Scales rewards before training. | **If too high** → Exploding Q-values, high critic loss. <br> **If too low** → Small updates, slow learning. | Use `1-10` for SAC. |
| **`normalize_rewards`** | Ensures consistent reward values. | **If not normalized** → Q-values might be inconsistent. | Normalize rewards if they vary significantly. |

🔹 **How it affects metrics:**
- **Q-values too high?** → Reduce `reward_scaling_factor`.
- **Q-values too low?** → Increase `reward_scaling_factor`.

---

## **🔑 Final SAC Hyperparameter Recommendations**
| **Hyperparameter** | **Recommended Value** |
|-------------------|----------------------|
| **Actor LR (`lr_actor`)** | `3e-4` |
| **Critic LR (`lr_critic`)** | `3e-4` |
| **Alpha LR (`lr_alpha`)** | `1e-4` |
| **Target Entropy (`target_entropy`)** | `-action_dim` or `-0.5 * action_dim` |
| **Tau (`tau`)** | `0.005` |
| **Gamma (`gamma`)** | `0.99` |
| **Batch Size (`batch_size`)** | `256-512` |
| **Replay Buffer Size** | `100,000 - 1,000,000` |
| **Updates per Block** | `10-20` |
| **Env Steps per Update** | `50` |

---

## **🚀 Summary**
- **Too much exploration?** → Lower `target_entropy`, reduce `alpha`.
- **Unstable Q-values?** → Reduce `tau`, lower `lr_critic`.
- **Slow learning?** → Increase `updates_per_block`, batch size.
- **Poor evaluation performance?** → Train with deterministic actions sometimes.