# Reinforcement Learning Cheat Sheet: Deep Q-Networks (DQN)

## Introduction
- **Deep Q-Networks (DQN)** extend Q-learning to handle **high-dimensional state spaces**.
- Uses a **neural network** as a function approximator for Q-values.
- Introduced by **DeepMind** in 2015 for playing Atari games.

## From RL to Deep RL
- **Traditional RL** uses tabular Q-learning or linear function approximation.
- **Deep RL** leverages **deep neural networks** to approximate value functions and policies.

## Deep Q-Networks
- A **neural network** is trained to approximate the Q-function:
  \[
  Q(s, a; \theta) \approx Q^*(s, a)
  \]
- The network takes state **s** as input and outputs Q-values for all actions.

## Experience Replay
- Stores past experiences **(s, a, r, s')** in a **replay buffer**.
- Randomly samples mini-batches from the buffer to **break correlation** between samples.
- **Benefits**:
  - Stabilizes learning.
  - Increases sample efficiency.

## Fixed Q-Targets
- Uses a **target network** with fixed weights to compute Q-targets:
  \[
  y_t = r + \gamma \max_{a'} Q(s', a'; \theta^-)
  \]
- The target network is updated periodically to reduce instability.

## Deep Q-Learning Algorithm
```python
initialize Q-network Q(s, a; θ) with random weights
initialize target network Q_target with weights θ_target = θ
initialize experience replay memory D

for each episode:
    initialize state s
    for each step in episode:
        choose action a using ε-greedy policy
        execute action a, observe reward r and new state s'
        store (s, a, r, s') in memory D
        sample mini-batch from D
        compute target: y = r + γ * max Q_target(s', a'; θ_target)
        update Q-network by minimizing loss: L = (y - Q(s, a; θ))^2
        periodically update Q_target ← Q
```

## Deep Q-Learning Improvements

### Double DQN
- Addresses **overestimation bias** in Q-learning.
- Instead of:
  \[
  y = r + \gamma \max_{a'} Q(s', a')
  \]
  It uses:
  \[
  y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)
  \]
- Uses **one network for action selection** and **another for value estimation**.

### Prioritized Experience Replay
- Instead of sampling uniformly from replay memory, **prioritizes important transitions**.
- **Prioritization probability**:
  \[
  P(i) \propto |TD\_error_i|^\alpha
  \]
- **Helps the agent learn more efficiently**.

### Dueling DQN
- Separates Q-value estimation into **value function** and **advantage function**:
  \[
  Q(s, a) = V(s) + A(s, a) - \max_{a'} A(s, a')
  \]
- **Benefits**:
  - Better state representation learning.
  - Reduces variance.

### Rainbow DQN
- Combines multiple improvements:
  - **Double DQN**
  - **Dueling DQN**
  - **Prioritized Experience Replay**
  - **Multi-step learning**
  - **Noisy networks**
  - **Distributional RL**

