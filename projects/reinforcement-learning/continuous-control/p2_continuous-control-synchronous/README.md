# Project 2: Continuous Control (20 Agents)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

![Trained Agent][image1]

## Introduction

This repository contains my solution for the **20-agent** version of the Unity Reacher environment. In this version, **20 identical agents** run in parallel, each controlling its own arm. After each episode we:

1. Sum each agent’s undiscounted rewards to get 20 scores  
2. Average those 20 scores to produce a single **episode score**  

The environment is “solved” when the **mean** of those episode scores, taken over **100 consecutive episodes**, reaches **+30**.

---

## Environment

- **Version**: Twenty (20) Agents  
- **Linux build**: [Reacher_Linux.zip](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)  

Unzip into `p2_continuous-control/` so that your directory looks like:
```
p2_continuous-control/
├── Reacher_Linux/…
├── README.md
├── Continuous_Control.ipynb ← The notebook
└── codebase/  ← solution code
```

---

## Solution Overview

The implementation is organized as follows:

- **`env.py`** – a Unity‐environment wrapper (`BootstrappedEnvironment`) with:
  - train vs. eval mode  
  - optional reward shaping hooks  
  - reset‐failure retry logic  

- **`agent.py`** – `DDPGAgent` class:
  - Actor & Critic networks + targets  
  - Ornstein–Uhlenbeck exploration noise  
  - `act()`, `step()`, `learn()`, `soft_update()`  

- **`replay_buffer.py`** – uniform replay buffer (without replacement), storing `(state, action, reward, next_state, done)`.

- **`network definitions`** (`actor.py`, `critic.py`):
  - **Actor**: FC(33→256)–ReLU–FC(256→128)–ReLU–FC(128→4)–Tanh  
  - **Critic**: FC(state→256)–ReLU + concat(action) → FC(256+4→128)–ReLU → FC(128→1)

- **`train_ddpg.py`** or integrated in the notebook – training driver:
  1. Builds `TrainingConfig` (hyperparameters)  
  2. Initializes environment, agent, buffer, logging  
  3. Loops episodes & steps, calls `agent.step()`, logs to TensorBoard  
  4. Saves best models when the running average ≥30  

- **`evaluate.py`** or evaluation script (in notebook):
  - Loads a checkpoint  
  - Runs **100 episodes** in **inference mode** (no noise)  
  - Records the **average score per episode** for plotting

---

## Algorithm: Deep Deterministic Policy Gradient (DDPG)

1. **Actor–Critic**  
   - **Actor** \(\mu(s)\): deterministic policy network  
   - **Critic** \(Q(s,a)\): value estimator  

2. **Off-policy** replay & target networks  
   - Soft‐updates with \(\tau\):  
     \[
       \theta_{\text{target}}\leftarrow \tau\,\theta_{\text{local}} + (1-\tau)\,\theta_{\text{target}}
     \]

3. **Exploration**  
   - Ornstein–Uhlenbeck noise (µ=0, θ=0.15, σ=0.2) added during training  

---

## Hyperparameters

| Parameter            | Value      |
|----------------------|-----------:|
| **Agents**           |            |
| num_agents           | 20         |
| state_size           | 33         |
| action_size          | 4          |
| **Training**         |            |
| episodes             | 500        |
| max_steps/episode    | 1 000      |
| batch_size           | 128        |
| learn_after          | 0          |
| **Optimizer**        |            |
| lr_actor             | 1 × 10⁻⁴   |
| lr_critic            | 1 × 10⁻⁴   |
| gamma (discount)     | 0.99       |
| tau (soft update)    | 1 × 10⁻³   |
| **Exploration**      |            |
| use_ou_noise         | True       |
| ou_theta             | 0.15       |
| ou_sigma             | 0.20       |
| **Replay Buffer**    |            |
| capacity             | 100 000    |
| replacement          | False      |
| **Extras**           |            |
| Reward Shaping       | ✓          |
| Warmup Random actions| ✓          |
| Noise Decay          | ✓          |
| Zero‐reward filtering| ✓          |

---

## Additional Features

- **Reward Shaping**  
  – Custom `VelocityRewardShaper` to boost/penalize based on target‐velocity alignment.  

- **Warm-up Phase**  
  – First N steps use random actions to pre‐fill the buffer.  

- **Noise Scheduling**  
  – Optional exponential decay of OU‐noise scale over time.  

- **Zero-Reward Filtering**  
  – Annealed probability to skip storing transitions with zero reward.

---

## Results

- **Solved** in **32 episodes** (running average over 20 agents ≥ 30).  
- **Evaluation** (100 episodes, no noise):  
  - **Mean score** > 30 consistently  
  - **Low variance** across episodes → stable policy  

---

## Usage
Either in notebook or:

### 1. Train (via command line)
```bash
python train_ddpg.py # and configure everything before in the file
```

### 2. Evaluate (via commandline)
```bash
python evaluate.py # and configure everything before in the file
```