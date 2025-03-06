### **TD3 vs. SAC: Key Differences and Concepts**

Both **Twin Delayed Deep Deterministic Policy Gradient (TD3)** and **Soft Actor-Critic (SAC)** are designed for **continuous control tasks** but differ fundamentally in their approach to **exploration, stability, and optimization**.

---

### **1. TD3 vs. SAC: Conceptual Differences**

| Feature | **TD3** (Your Current Model) | **SAC** (Planned Model) |
|---------|----------------|----------------|
| **Policy Type** | Deterministic | Stochastic |
| **Action Selection** | Directly outputs an action \( a = \pi(s) \) | Samples an action from a probability distribution \( a \sim \pi(a|s) \) |
| **Exploration** | Uses action noise (Gaussian noise or clipped noise) | Uses entropy regularization to promote exploration |
| **Critic Networks** | Two Q-value critics (chooses the min Q-value) | Two Q-value critics + entropy term |
| **Actor Updates** | Delayed updates, using the **min Q-value** for stability | Uses the **soft Q-value**, optimizing an **entropy-regularized objective** |
| **Training Stability** | More stable than DDPG due to twin critics but can struggle in sparse rewards | More stable due to **entropy maximization**, making it better for sparse rewards |
| **Computational Cost** | Lower | Higher (stochastic sampling, entropy tuning) |

The **main reason SAC is better for your environment** is that it explicitly encourages **diverse exploration** by maximizing the **entropy of the policy**. This ensures the agent does not converge to a suboptimal deterministic strategy too quickly.

---

## **2. Understanding Soft Actor-Critic (SAC)**

SAC is a **stochastic off-policy reinforcement learning** algorithm that extends **TD3/DDPG** with an **entropy regularization term**. This encourages **diverse** action selection, leading to better exploration.

### **Mathematical Formulation of SAC**
SAC maximizes a **modified objective** compared to traditional RL algorithms:

\[
J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right]
\]

Where:
- \( r(s_t, a_t) \) is the environment reward.
- \( \mathcal{H}(\pi(\cdot | s_t)) \) is the **entropy** of the policy.
- \( \alpha \) is the **temperature parameter**, controlling the tradeoff between **exploration** (high entropy) and **exploitation** (low entropy).

---

### **3. SAC Components**

SAC consists of **three networks**:
1. **Actor Network (Policy)** \( \pi_\theta(a | s) \) → Stochastic policy that **samples** actions instead of outputting deterministic ones.
2. **Two Critic Networks** \( Q_{\phi_1}(s, a) \), \( Q_{\phi_2}(s, a) \) → Similar to TD3, uses **two Q-values** and selects the **minimum** for stability.
3. **Temperature Parameter \( \alpha \)** → Controls the balance between exploration (high \( \alpha \)) and exploitation (low \( \alpha \)).

---

### **4. SAC Policy Update (Entropy-Regularized Policy Gradient)**

Instead of directly maximizing expected return like in DDPG/TD3, SAC **maximizes both reward and entropy**.

The policy update is done via **maximum entropy reinforcement learning**:

\[
J_{\pi} = \mathbb{E}_{s_t \sim \mathcal{D}, a_t \sim \pi} \left[ \alpha \log \pi(a_t | s_t) - Q(s_t, a_t) \right]
\]

Where:
- The first term **maximizes entropy** (encourages diverse actions).
- The second term **minimizes expected Q-value** (finds high-value actions).

The gradient update for the policy is:

\[
\nabla_\theta J_\pi = \mathbb{E}_{s_t \sim \mathcal{D}, a_t \sim \pi} \left[ \alpha \nabla_\theta \log \pi_\theta(a_t | s_t) - \nabla_\theta Q(s_t, a_t) \right]
\]

This **stochastic policy gradient** prevents premature convergence to suboptimal policies.

---

### **5. Q-Function Update in SAC**
Like TD3, SAC uses two Q-functions to reduce overestimation bias:

\[
J_Q(\phi) = \mathbb{E} \left[ (Q_{\phi}(s, a) - (r + \gamma (1 - d) \hat{V}(s')))^2 \right]
\]

The target value \( \hat{V}(s') \) incorporates entropy:

\[
\hat{V}(s') = \mathbb{E}_{a' \sim \pi} \left[ \min (Q_{\phi_1}(s', a'), Q_{\phi_2}(s', a')) - \alpha \log \pi(a' | s') \right]
\]

Unlike TD3, **SAC does not use target policy smoothing** but relies on entropy regularization instead.

---

### **6. Temperature Parameter \( \alpha \) (Entropy Tradeoff)**
The temperature \( \alpha \) controls exploration:

- **High \( \alpha \)** → Encourages more exploration (diverse actions).
- **Low \( \alpha \)** → Leads to more exploitation (choosing best actions).

SAC can **automatically tune \( \alpha \)** by minimizing:

\[
J(\alpha) = \mathbb{E}_{a_t \sim \pi} [-\alpha (\log \pi(a_t | s_t) + H_0)]
\]

Where \( H_0 \) is a target entropy (a hyperparameter). This adaptive approach **prevents manual tuning** and ensures exploration is neither too high nor too low.

---

## **6. Summary: Why SAC is Better for Your Task**
✅ **Handles Sparse Rewards Well** → Encourages exploration through entropy.  
✅ **Stochastic Policy Helps Learning** → Avoids premature convergence to bad policies.  
✅ **More Stable Than TD3/DDPG** → Uses entropy regularization instead of action noise.  
✅ **Automatically Adjusts Exploration** → Adaptive \( \alpha \) tuning removes the need for manual tuning.  