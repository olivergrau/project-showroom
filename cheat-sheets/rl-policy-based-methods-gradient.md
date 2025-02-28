# Reinforcement Learning Cheat Sheet: Policy-Based Methods with Gradient Methods

## Introduction
- Unlike **black-box** optimization methods, gradient-based methods optimize policies **using gradients**.
- Use **stochastic gradient ascent** to maximize expected rewards.

## Differentiation from Non-Gradient Methods
| Method | Update Type | Example Algorithms |
|--------|------------|--------------------|
| **Non-Gradient Methods** | Black-box optimization | Hill Climbing, Cross-Entropy |
| **Gradient-Based Methods** | Uses policy gradients | REINFORCE, PPO |

## REINFORCE Algorithm (Monte Carlo Policy Gradient)
- Directly optimizes policy parameters \( \theta \) using the **policy gradient theorem**:
  \[
  \nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ G_t \nabla_\theta \log \pi_\theta (a_t | s_t) \right]
  \]
- **Update rule**:
  \[
  \theta \leftarrow \theta + \alpha G_t \nabla_\theta \log \pi_\theta (a_t | s_t)
  \]
- **Pseudo Algorithm**:
```python
initialize policy parameters θ randomly
for each episode:
    generate trajectory (s_0, a_0, r_1, ..., s_T)
    compute returns G_t for each time step
    update policy parameters: θ ← θ + α G_t ∇_θ log π_θ(a_t | s_t)
```

## Proximal Policy Optimization (PPO)
- **Addresses instability** in policy gradient updates.
- **Key Improvements**:
  - **Noise Reduction**: Stabilizes updates.
  - **Credit Assignment**: Uses **advantage estimation**:
    \[
    A(s, a) = Q(s, a) - V(s)
    \]
  - **Importance Sampling**: Adjusts probability ratio:
    \[
    r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}
    \]
  - **Surrogate Function**:
    \[
    L(\theta) = \mathbb{E} \left[ r_t(\theta) A(s, a) \right]
    \]
  - **Clipping Policy**: Prevents large updates:
    \[
    L(\theta) = \mathbb{E} \left[ \min( r_t(\theta) A, clip(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A ) \right]
    \]

### PPO Algorithm:
```python
initialize policy parameters θ, value function V
for each iteration:
    collect trajectories using π_θ
    compute advantage estimates A(s, a)
    optimize policy using clipped surrogate loss
    update value function V
```

## Summary
- **REINFORCE**: Basic policy gradient method, high variance.
- **PPO**: More stable updates with **importance sampling** and **clipping**.

