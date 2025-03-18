### **Mathematical Foundations of Soft Actor-Critic (SAC)**

Soft Actor-Critic (**SAC**) is an **off-policy, model-free reinforcement learning algorithm** that extends traditional Actor-Critic methods by incorporating an **entropy maximization term**. This encourages the policy to explore more diverse actions, leading to improved stability and sample efficiency.

---

## **1️⃣ SAC Optimization Objectives**
Unlike traditional RL algorithms that maximize only the expected reward, SAC **maximizes both the reward and entropy**:

\[
J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \Big[ r(s_t, a_t) + \alpha \mathbb{H}(\pi(\cdot | s_t)) \Big]
\]

where:
- \( \pi(a|s) \) is the **stochastic policy** (actor).
- \( Q(s, a) \) is the **critic function** (expected return).
- \( \alpha \) is the **temperature parameter**, controlling the balance between reward and exploration.
- \( \mathbb{H}(\pi(\cdot | s_t)) \) is the **entropy**:
  \[
  \mathbb{H}(\pi(\cdot | s_t)) = -\mathbb{E}_{a \sim \pi} [\log \pi(a | s)]
  \]

### **Why Include Entropy?**
- **Encourages exploration**: Prevents premature convergence to suboptimal policies.
- **Improves stability**: Helps avoid deterministic policy collapse (common in DDPG/TD3).

---

## **2️⃣ Critic Update (Q-Function Learning)**
SAC learns **two Q-functions** to mitigate **overestimation bias**:

\[
y = r + \gamma \left( \min(Q_{\theta_1}(s', a'), Q_{\theta_2}(s', a')) - \alpha \log \pi(a' | s') \right)
\]

where:
- \( Q_{\theta_1}, Q_{\theta_2} \) are the two critics.
- \( a' \sim \pi(\cdot | s') \) is the next action sampled from the policy.
- The **minimum Q-value** helps reduce overestimation.
- The **entropy term** \( - \alpha \log \pi(a' | s') \) regularizes learning.

**Loss function for each critic**:

\[
J_Q(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \Big( Q_{\theta}(s, a) - y \Big)^2 \right]
\]

where \( \mathcal{D} \) is the **replay buffer**.

---

## **3️⃣ Actor Update (Policy Improvement)**
The policy \( \pi_{\phi}(a | s) \) is **trained to maximize expected Q-values while also maximizing entropy**:

\[
J_{\pi}(\phi) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\phi} \left[ \alpha \log \pi_{\phi}(a | s) - Q_{\theta}(s, a) \right]
\]

This encourages the policy to:
- **Select actions with high expected returns** \( (Q(s, a)) \).
- **Maintain high entropy** (to keep exploring).

**Gradient Update for the Actor**:
\[
\nabla_{\phi} J_{\pi} = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_{\phi}} \left[ \nabla_{\phi} \alpha \log \pi_{\phi}(a | s) - \nabla_{\phi} Q_{\theta}(s, a) \right]
\]

---

## **4️⃣ Temperature Parameter Update (\(\alpha\) Tuning)**
The **temperature parameter** \( \alpha \) controls the trade-off between:
- **Exploration** (higher entropy)
- **Exploitation** (maximizing Q-values)

SAC can **automatically adjust \(\alpha\)** by minimizing:

\[
J(\alpha) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi} \left[ -\alpha \big( \log \pi(a | s) + \mathcal{H} \big) \right]
\]

where:
- \( \mathcal{H} \) is the **target entropy** (a predefined constant, e.g., \( -|A| \) for continuous actions).
- **Adaptive entropy tuning** adjusts \( \alpha \) so the policy maintains a desired level of randomness.

Gradient update:

\[
\nabla_{\alpha} J(\alpha) = \mathbb{E}_{s, a} \left[ -\log \pi(a | s) - \mathcal{H} \right]
\]

---

## **5️⃣ Target Networks & Polyak Averaging**
Like TD3, SAC **uses target networks** for stability:

\[
\theta_{\text{target}, i} \leftarrow \tau \theta_i + (1 - \tau) \theta_{\text{target}, i}
\]

where:
- \( \tau \) is a small update coefficient (e.g., \( 0.005 \)).
- This prevents drastic updates, stabilizing training.

---

## **6️⃣ Full SAC Algorithm Summary**
### **1. Sample a batch of transitions from replay buffer**
\[
(s, a, r, s') \sim \mathcal{D}
\]

### **2. Compute the Target Q-Value**
\[
y = r + \gamma \left( \min(Q_{\theta_1}(s', a'), Q_{\theta_2}(s', a')) - \alpha \log \pi(a' | s') \right)
\]

### **3. Critic Update (Minimize Bellman Error)**
\[
J_Q(\theta) = \mathbb{E} \left[ \Big( Q_{\theta}(s, a) - y \Big)^2 \right]
\]

### **4. Actor Update (Maximize Expected Return + Entropy)**
\[
J_{\pi}(\phi) = \mathbb{E} \left[ \alpha \log \pi_{\phi}(a | s) - Q_{\theta}(s, a) \right]
\]

### **5. Temperature Parameter Update (Optional)**
\[
J(\alpha) = \mathbb{E} \left[ -\alpha \big( \log \pi(a | s) + \mathcal{H} \big) \right]
\]

### **6. Polyak Update of Target Networks**
\[
\theta_{\text{target}, i} \leftarrow \tau \theta_i + (1 - \tau) \theta_{\text{target}, i}
\]

---

## **7️⃣ Key Differences Between SAC and Other RL Algorithms**
| Feature          | SAC | DDPG | TD3 |
|-----------------|-----|------|-----|
| **Policy Type** | Stochastic | Deterministic | Deterministic |
| **Q-Function**  | 2 Critics | 1 Critic | 2 Critics (like SAC) |
| **Exploration** | Entropy regularization (no noise needed) | OU noise / Gaussian noise | Gaussian noise |
| **Update Stability** | More stable (entropy & dual critics) | Prone to overestimation | Fixes overestimation bias |
| **Training Efficiency** | More sample-efficient | Less sample-efficient | Slightly better than DDPG |

🚀 **TL;DR**:
- SAC **balances exploration & exploitation naturally**.
- Uses **two critics** to **avoid overestimation bias**.
- Uses **entropy regularization** instead of noise-based exploration.
- **More stable** than DDPG/TD3 but **more computationally expensive**.