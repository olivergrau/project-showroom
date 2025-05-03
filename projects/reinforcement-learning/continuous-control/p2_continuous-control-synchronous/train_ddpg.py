import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Protocol, Tuple
from codebase.ddpg.runner import TrainingRunner

# For older numpy versions
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64

# Scheduling interface
class UpdateScheduler(Protocol):
    def interval(self, total_steps: int) -> int:
        ...

@dataclass
class StandardUpdateScheduler:
    def interval(self, total_steps: int) -> int:
        return 1

@dataclass
class DynamicUpdateScheduler:
    switch_step: int
    early_interval: int
    late_interval: int

    def interval(self, total_steps: int) -> int:
        return self.early_interval if total_steps < self.switch_step else self.late_interval

@dataclass
class TrainingConfig:
    seed = 0

    # --- Environment and agent configuration ---
    num_agents: int = 20               # Number of parallel agents (e.g., 20 for Unity Reacher)
    state_size: int = 33              # Dimension of the state space
    action_size: int = 4              # Dimension of the action space
    gamma: float = 0.99               # Discount factor for future rewards

    # --- Training schedule ---
    episodes: int = 500              # Total number of training episodes
    max_steps: int = 1000             # Max steps per episode
    batch_size: int = 128             # Number of samples per learning update
    learn_after: int = 0           # Delay learning until this many total env steps
    sampling_warmup: int = 0      # Use random actions for the first N steps

    # --- Learning rates and optimization ---
    lr_actor: float = 1e-4            # Learning rate for the actor network
    lr_critic: float = 1e-4           # Learning rate for the critic network
    critic_weight_decay = 0.0        # L2 regularization for critic optimizer
    critic_clipnorm: float = None      # Max norm for critic gradient clipping

    # --- Target network soft updates ---
    tau: float = 1e-3                # Polyak averaging factor for target networks

    # --- Exploration (OU noise) ---
    use_ou_noise: bool = True         # Whether to use Ornstein-Uhlenbeck noise
    ou_theta: float = 0.15            # OU noise: theta (mean reversion)
    ou_sigma: float = 0.2             # OU noise: sigma (volatility)
    noise_decay: bool = False      # Noise decay factor (for exploration)
    init_noise: float = 1.0           # Initial scaling factor for exploration noise
    min_noise: float = 0.05           # Minimum noise after decay  
    use_dynamic_noise: bool = False  # Use dynamic noise scaling
    dynamic_noise_distance_threshold: float = 0.5  # Distance threshold for dynamic noise scaling

    # --- Evaluation ---
    eval_freq: int = 10               # Evaluate every N episodes
    eval_eps: int = 1                 # Number of evaluation episodes per evaluation step
    eval_thresh: float = 30.0         # Consider environment solved at this average reward
    eval_warmup: int = 20             # Start evaluation only after this many episodes

    # --- Replay buffer ---
    replay_capacity: int = 100_000  # Total capacity of the replay buffer (attention: must be integer!)
    
    # Probability to retain a 0-reward transition
    zero_reward_filtering: bool = False
    zero_reward_prob_start: float = 0.2
    zero_reward_prob_end: float = 0.01
    zero_reward_prob_anneal_steps: int = 150_000

    # --- Replay buffer prioritization (PER) ---
    use_prioritized_replay: bool = False  # Use prioritized replay (instead of uniform)
    per_beta_start: float = 0.4  # initial beta
    per_beta_end: float = 1.0    # final beta (fully corrects sampling bias)
    per_beta_anneal_steps: int = 500_000  # how many steps to anneal over (you can tune this)

    # --- Reward preprocessing and shaping ---
    use_state_norm: bool = False     # Normalize states before feeding to actor/critic
    use_reward_scaling: bool = False # Multiply rewards by reward_scale
    reward_scale: float = 1.0        # Reward scaling factor (if enabled)
    reward_shaping_fn: Optional[callable] = None # lambda state, reward, total_steps: shaper(state, reward, total_steps)
    use_shaped_reward_only_steps: int = 0 # 20000  # or 5000, depending on your experiments

    # --- Learning schedule ---
    scheduler: UpdateScheduler = field(default_factory=lambda: StandardUpdateScheduler()) #field(default_factory=lambda: DynamicUpdateScheduler(10000, 10, 1))
    updates_per_block: int = 1       # Number of updates per learning trigger

    # --- Other ---
    worker_id: int = 1               # Unity worker ID (avoid conflicts when launching multiple environments)
    load_weights: Optional[str] = None  # Path to pretrained weights to resume training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_freq: int = 100              # Log every N episodes

def train(**kwargs) -> float:
    cfg = TrainingConfig(**kwargs)
    runner = TrainingRunner(cfg)
    return runner.run()

if __name__ == "__main__":
    print("Starting DDPG Training...")
    result = train()
    print(f"Done. Result: {result:.2f}")
