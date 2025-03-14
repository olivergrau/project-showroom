## Deep Deterministic Policy Gradient (DDPG) algorithm:

---

### 1. Actor and Critic Functions

- **Actor (Policy) Function:**  
  A deterministic policy parameterized by \( \theta^\pi \):
  \[
  a = \pi(s|\theta^\pi)
  \]
  
- **Critic (Q-value) Function:**  
  An action-value function parameterized by \( \theta^Q \):
  \[
  Q(s,a|\theta^Q) \approx \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t \,\middle|\, s_0=s, a_0=a\right]
  \]

---

### 2. Critic Update (Temporal-Difference Learning)

- **TD Target:**  
  Using target networks (denoted with a prime) to stabilize learning, the target for a sampled transition \((s, a, r, s')\) is:
  \[
  y = r + \gamma \, Q'\big(s', \pi'(s'|\theta^{\pi'}) \mid \theta^{Q'}\big)
  \]
  
- **Loss Function for the Critic:**  
  The mean-squared error (MSE) loss over a minibatch \(\mathcal{B}\) of transitions:
  \[
  L(\theta^Q) = \frac{1}{|\mathcal{B}|} \sum_{(s,a,r,s') \in \mathcal{B}} \left( Q(s,a|\theta^Q) - y \right)^2
  \]
  The critic parameters \( \theta^Q \) are updated by minimizing this loss.

---

### 3. Actor Update (Deterministic Policy Gradient)

- **Deterministic Policy Gradient:**  
  The objective is to maximize the expected return:
  \[
  J(\theta^\pi) = \mathbb{E}_{s \sim \mathcal{D}}\left[ Q\big(s, \pi(s|\theta^\pi)\,|\,\theta^Q\big) \right]
  \]
  
- **Gradient of the Objective:**  
  Using the chain rule, the gradient with respect to the actor parameters is given by:
  \[
  \nabla_{\theta^\pi} J \approx \mathbb{E}_{s \sim \mathcal{D}}\left[ \nabla_a Q(s,a|\theta^Q) \big|_{a=\pi(s)} \, \nabla_{\theta^\pi} \pi(s|\theta^\pi) \right]
  \]
  The actor parameters \( \theta^\pi \) are updated by performing gradient ascent on \( J(\theta^\pi) \).

---

### 4. Target Network Updates (Soft Updates)

- **Polyak Averaging:**  
  The target networks are slowly updated to track the learned networks. For the critic and actor respectively:
  \[
  \theta^{Q'} \leftarrow \tau \, \theta^Q + (1 - \tau) \, \theta^{Q'}
  \]
  \[
  \theta^{\pi'} \leftarrow \tau \, \theta^\pi + (1 - \tau) \, \theta^{\pi'}
  \]
  where \( \tau \ll 1 \) is a small constant (e.g., \(0.005\) or \(0.003\)).

---

### 5. Exploration

- **Adding Noise:**  
  Since the actor is deterministic, exploration is introduced externally (for example, using an Ornstein–Uhlenbeck process):
  \[
  a_{\text{explore}} = \pi(s|\theta^\pi) + \mathcal{N}
  \]
  where \( \mathcal{N} \) is temporally correlated noise.

---

### Summary of the DDPG Update Cycle

1. **Sample a Minibatch:**  
   Draw transitions \((s, a, r, s')\) from the replay buffer \( \mathcal{D} \).

2. **Critic Update:**  
   - Compute the target:
     \[
     y = r + \gamma \, Q'\big(s', \pi'(s'|\theta^{\pi'}) \mid \theta^{Q'}\big)
     \]
   - Minimize the loss:
     \[
     L(\theta^Q) = \frac{1}{|\mathcal{B}|} \sum \left( Q(s,a|\theta^Q) - y \right)^2
     \]

3. **Actor Update:**  
   - Update the policy by ascending the gradient:
     \[
     \nabla_{\theta^\pi} J \approx \mathbb{E}\left[ \nabla_a Q(s,a|\theta^Q) \big|_{a=\pi(s)} \nabla_{\theta^\pi} \pi(s|\theta^\pi) \right]
     \]

4. **Soft Target Updates:**  
   Update the target networks with:
   \[
   \theta^{Q'} \leftarrow \tau \, \theta^Q + (1 - \tau) \, \theta^{Q'}, \quad \theta^{\pi'} \leftarrow \tau \, \theta^\pi + (1 - \tau) \, \theta^{\pi'}
   \]

The code

```python
actor_loss = -self.critic(state, self.actor(state)).mean()
self.actor_optimizer.zero_grad()
actor_loss.backward()
```

implements the actor update as follows:

1. **Objective Transformation:**  
   Mathematically, the actor aims to maximize the expected Q-value:
   \[
   J(\theta^\pi) = \mathbb{E}_{s \sim \mathcal{D}}[\,Q(s, \pi(s|\theta^\pi))\,].
   \]
   In practice, since most optimization libraries perform minimization, we minimize the negative of this objective:
   \[
   L(\theta^\pi) = -\mathbb{E}_{s \sim \mathcal{D}}[\,Q(s, \pi(s|\theta^\pi))\,].
   \]

2. **Batch Approximation:**  
   The expectation is approximated by averaging over a minibatch of states. The expression 
   ```python
   self.critic(state, self.actor(state)).mean()
   ```
   computes the average Q-value for the current policy over the batch. Negating it produces the loss:
   \[
   L(\theta^\pi) \approx -\frac{1}{N}\sum_{i=1}^{N} Q\big(s_i, \pi(s_i|\theta^\pi)\big).
   \]

3. **Gradient Calculation and Update:**  
   Calling `actor_loss.backward()` computes the gradient of this loss with respect to the actor parameters \(\theta^\pi\). Then the optimizer uses these gradients to update \(\theta^\pi\) via gradient descent, which in turn is equivalent to performing gradient ascent on \(J(\theta^\pi)\).

Thus, the code is a direct implementation of minimizing the negative expected Q-value, aligning with the mathematical formulation of the deterministic policy gradient.