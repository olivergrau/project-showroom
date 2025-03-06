# TD3 Multiprocessing RL System - Analysis and Next Steps

## 1. Project Overview

This project implements a multiprocessing reinforcement learning system based on the **Twin Delayed Deep Deterministic Policy Gradient (TD3)** algorithm to solve the **Unity Reacher** environment.

The implementation is structured into multiple parallel processes:

- **Main Process (`training.py`)**:
  - Manages all processes and inter-process communication.
  - Defines hyperparameters and starts workers.
  - Ensures graceful shutdown upon completion or interruption.

- **Environment Worker (`env_worker.py`)**:
  - Runs the Unity environment in training mode (headless).
  - Collects experiences and feeds them into the replay buffer.
  - Dynamically updates policy weights when received from the training worker.
  - Logs training-related metrics.

- **Evaluation Worker (`eval_worker.py`)**:
  - Runs the Unity environment for evaluation (no exploration noise).
  - Computes moving average rewards over a sliding window to determine when the environment is solved.
  - Sends termination signals if the agent solves the task.
  - Logs evaluation metrics.

- **Training Worker (`train_worker.py`)**:
  - Samples from the replay buffer to train the TD3 agent.
  - Updates environment and evaluation workers with new policy weights.
  - Checks for evaluation worker's "solved" signals.
  - Logs training losses and Q-values.

## 2. TD3 Implementation

- **Actor Network (`actor.py`)**:
  - Fully connected layers with ReLU activations.
  - Final `tanh` activation to constrain actions to [-1,1].

- **Critic Networks (`critic.py`)**:
  - Two critics taking state-action pairs as input.
  - Uses orthogonal weight initialization for stability.

- **TD3 Algorithm Features**:
  - Target policy smoothing (policy noise with clipping).
  - Delayed actor updates to reduce variance.
  - Soft target updates with Polyak averaging.
  - Optional reward normalization.

## 3. Replay Buffer Implementation

- **Replay Wrapper (`replay_buffer.py`)**:
  - Handles asynchronous data collection and sampling.
  - Uses **UniformReplay** (no prioritization).

- **Replay Proxy (`replay_proxy.py`)**:
  - Provides an interface for worker processes to interact with the replay buffer.

## 4. Observations & Issues in Training

Despite implementing TD3 correctly, training did not succeed in achieving stable learning:

- **Negative Q-values**: Actor loss remained positive, indicating instability.
- **Reward Scaling / Normalization**: Helped get positive Q-values, but they remained small.
- **Sparse Rewards**: The Reacher environment provides only **positive and sparse** rewards, making TD3 potentially unsuitable.

## 5. Plan to Switch to SAC

### Why Switch?
TD3 relies on deterministic policy updates, which may not be ideal for environments with sparse rewards. **Soft Actor-Critic (SAC)**, in contrast, is an entropy-regularized RL algorithm that encourages exploration and handles sparse rewards better.

### Next Steps:
- **Implement SAC** as a replacement for TD3.
- Modify `agent.py` to support a stochastic actor (policy network).
- Adjust the replay buffer to handle entropy-based updates.
- Introduce **automatic entropy tuning** to optimize exploration.
- Train and evaluate the SAC agent under the same multiprocessing framework.

---

**Conclusion**: TD3 did not achieve stable training on the Unity Reacher task. Switching to **SAC** should improve learning performance, particularly in handling **sparse rewards** and encouraging more effective exploration.
