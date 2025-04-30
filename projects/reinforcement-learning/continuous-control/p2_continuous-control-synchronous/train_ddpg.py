import os
import time
import numpy as np
import torch
import traceback
from dataclasses import dataclass, field
from typing import Optional, Protocol
from torch.utils.tensorboard import SummaryWriter
from codebase.ddpg.agent import DDPGAgent
from codebase.ddpg.env import BootstrappedEnvironment
from codebase.ddpg.eval import evaluate
from codebase.replay.replay_buffer import PrioritizedReplay, UniformReplay
from codebase.utils.normalizer import RunningNormalizer
from codebase.utils.early_stopping import EarlyStopping
from optuna.exceptions import TrialPruned

# For older numpy versions
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64

LOG_FREQ = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_batch_to_tensor(batch, device, beta=0.4, use_priorities=False):
    """
    Converts a batch of transitions to torch tensors.
    """
    state = torch.as_tensor(batch.state, dtype=torch.float32, device=device)
    action = torch.as_tensor(batch.action, dtype=torch.float32, device=device)
    reward = torch.as_tensor(batch.reward, dtype=torch.float32, device=device)
    next_state = torch.as_tensor(batch.next_state, dtype=torch.float32, device=device)
    mask = torch.as_tensor(batch.mask, dtype=torch.float32, device=device)

    Transition = type(batch) # Remember batch is a namedtuple
    
    if use_priorities:
        sampling_prob = torch.as_tensor(batch.sampling_prob, dtype=torch.float32, device=device)
        idx = torch.as_tensor(batch.idx, dtype=torch.int64, device=device)

        # Compute importance sampling weights
        weights = (1.0 / (sampling_prob + 1e-8))  # avoid div by zero
        weights /= weights.max()  # normalize to 1
        weights = weights.pow(beta)

        Transition = Transition(state, action, reward, next_state, mask, sampling_prob, idx)

    else:
        Transition = Transition(state, action, reward, next_state, mask)
    
    return (Transition, weights) if use_priorities else Transition

# Scheduling interface
class UpdateScheduler(Protocol):
    def interval(self, total_steps: int) -> int:
        ...

@dataclass
class DynamicUpdateScheduler:
    switch_step: int
    early_interval: int
    late_interval: int

    def interval(self, total_steps: int) -> int:
        return self.early_interval if total_steps < self.switch_step else self.late_interval

@dataclass
class TrainingConfig:
    # --- Environment and agent configuration ---
    num_agents: int = 20               # Number of parallel agents (e.g., 20 for Unity Reacher)
    state_size: int = 33              # Dimension of the state space
    action_size: int = 4              # Dimension of the action space
    gamma: float = 0.99               # Discount factor for future rewards

    # --- Training schedule ---
    episodes: int = 1000              # Total number of training episodes
    max_steps: int = 1000             # Max steps per episode
    batch_size: int = 256             # Number of samples per learning update
    learn_after: int = 15000           # Delay learning until this many total env steps
    sampling_warmup: int = 20000      # Use random actions for the first N steps

    # --- Learning rates and optimization ---
    lr_actor: float = 1e-3            # Learning rate for the actor network
    lr_critic: float = 1e-3           # Learning rate for the critic network
    critic_weight_decay = 1e-4        # L2 regularization for critic optimizer
    critic_clipnorm: float = 1.0      # Max norm for critic gradient clipping

    # --- Target network soft updates ---
    tau: float = 0.005                # Polyak averaging factor for target networks

    # --- Exploration (OU noise) ---
    use_ou_noise: bool = True         # Whether to use Ornstein-Uhlenbeck noise
    ou_theta: float = 0.15            # OU noise: theta (mean reversion)
    ou_sigma: float = 0.2             # OU noise: sigma (volatility)
    init_noise: float = 1.0           # Initial scaling factor for exploration noise
    min_noise: float = 0.05           # Minimum noise after decay  

    # --- Evaluation ---
    eval_freq: int = 20               # Evaluate every N episodes
    eval_eps: int = 1                 # Number of evaluation episodes per evaluation step
    eval_thresh: float = 30.0         # Consider environment solved at this average reward
    eval_warmup: int = 50             # Start evaluation only after this many episodes

    # --- Replay buffer ---
    replay_capacity: int = 1_000_000  # Total capacity of the replay buffer
    
    # Probability to retain a 0-reward transition
    zero_reward_prob_start: float = 0.0
    zero_reward_prob_end: float = 0.2
    zero_reward_prob_anneal_steps: int = 200_000

    # --- Replay buffer prioritization (PER) ---
    use_prioritized_replay: bool = False  # Use prioritized replay (instead of uniform)
    per_beta_start: float = 0.4  # initial beta
    per_beta_end: float = 1.0    # final beta (fully corrects sampling bias)
    per_beta_anneal_steps: int = 500_000  # how many steps to anneal over (you can tune this)

    # --- Reward preprocessing ---
    use_state_norm: bool = False     # Normalize states before feeding to actor/critic
    use_reward_scaling: bool = False # Multiply rewards by reward_scale
    reward_scale: float = 1.0        # Reward scaling factor (if enabled)
    use_reward_norm: bool = False    # Normalize rewards (Welford's algorithm)

    # --- Learning schedule ---
    scheduler: UpdateScheduler = field(default_factory=lambda: DynamicUpdateScheduler(40000, 10, 1))
    updates_per_block: int = 1       # Number of updates per learning trigger

    # --- Other ---
    worker_id: int = 1               # Unity worker ID (avoid conflicts when launching multiple environments)
    load_weights: Optional[str] = None  # Path to pretrained weights to resume training
    trial: Optional[object] = None       # Optional Optuna trial object for hyperparameter tuning/pruning

class TrainingRunner:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.writer = SummaryWriter(log_dir=os.path.join("runs/train_ddpg", time.strftime("%Y-%m-%d_%H-%M-%S")))
        self.early_stop = EarlyStopping(patience=20, min_delta=0.01, verbose=True,
                                        zero_patience=10, actor_patience=8, actor_min_delta=0.005)
        self.reward_stats = {"mean": 0.0, "var": 1.0, "count": 0}
        self._build_components()

    def _build_components(self):
        cfg = self.cfg
                
        # Environment
        self.env = BootstrappedEnvironment(
            exe_path="Reacher_Linux/Reacher.x86_64",
            worker_id=cfg.worker_id,
            use_graphics=False
        )

        # Agent
        self.agent = DDPGAgent(
            state_size=cfg.state_size,
            action_size=cfg.action_size,
            lr_actor=cfg.lr_actor,
            lr_critic=cfg.lr_critic,
            gamma=cfg.gamma,
            tau=cfg.tau,
            use_ou_noise=cfg.use_ou_noise,
            ou_noise_theta=cfg.ou_theta,
            ou_noise_sigma=cfg.ou_sigma,
            critic_weight_decay=cfg.critic_weight_decay,
            critic_clip_norm=cfg.critic_clipnorm,
            use_prioritized_replay=cfg.use_prioritized_replay,
            use_batch_norm=False
        )

        if cfg.load_weights:
            data = torch.load(cfg.load_weights, map_location=device)
            for net in ['actor', 'actor_target', 'critic', 'critic_target']:
                getattr(self.agent, net).load_state_dict(data[net])
            cfg.sampling_warmup = 0
        
        # Replay buffer
        replay_args = dict(memory_size=cfg.replay_capacity,
                           batch_size=cfg.batch_size,
                           discount=cfg.gamma, n_step=1, history_length=1)
        self.buffer = PrioritizedReplay(**replay_args) if cfg.use_prioritized_replay else UniformReplay(**replay_args)
        
        # State normalizer
        self.norm = RunningNormalizer((cfg.state_size,), momentum=0.001) if cfg.use_state_norm else None
        
        # Noise scheduler
        self.noise_lambda = -np.log(cfg.min_noise / cfg.init_noise) / cfg.episodes

    def run(self) -> float:
        avg_reward = 0.0
        total_steps = 0
        train_iterations = 0

        try:
            with self.env as env:
                for ep in range(1, self.cfg.episodes + 1):
                    noise_scale = max(self.cfg.min_noise,
                                      self.cfg.init_noise * np.exp(-self.noise_lambda * ep))
                    self.writer.add_scalar("Env/NoiseScale", noise_scale, ep)

                    ep_reward, total_steps, train_iterations = self._run_episode(
                        env, ep, total_steps, train_iterations, noise_scale
                    )
                    self.writer.add_scalar("Episode/Reward", ep_reward, ep)
                    print(f"Episode {ep} | Reward {ep_reward:.2f} | Steps {total_steps} | Updates {train_iterations}")

                    # Evaluation block
                    if ep % self.cfg.eval_freq == 0 and ep > self.cfg.eval_warmup:
                        print(f"Evaluating agent at episode {ep}...")
                        try:
                            avg_eval_reward, solved = evaluate(
                                self.agent, env,
                                normalizer=self.norm,
                                episodes=self.cfg.eval_eps,
                                threshold=self.cfg.eval_thresh
                            )
                            self.writer.add_scalar("Eval/AverageReward", avg_eval_reward, ep)
                            print(f"Eval at ep {ep}: avg reward = {avg_eval_reward:.2f}")

                            # Save weights
                            save_dir = os.path.join("saved_weights", "train_ddpg")
                            os.makedirs(save_dir, exist_ok=True)
                            weights = extract_agent_weights(self.agent)
                            fname = f"ddpg_weights_ep_{ep}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pth"
                            torch.save(weights, os.path.join(save_dir, fname))
                            print(f"Saved weights: {fname}")

                            # Optuna pruning
                            if self.cfg.trial is not None:
                                self.cfg.trial.report(-avg_eval_reward, ep)
                                if self.cfg.trial.should_prune():
                                    print(f"Trial pruned at episode {ep}")
                                    raise TrialPruned()

                            if solved:
                                print("Environment solved! Stopping training.")
                                break

                            if self.early_stop.stepAvgReward(avg_eval_reward):
                                print("Early stopping triggered by eval reward. Stopping training.")
                                break

                        except Exception as eval_e:
                            print(f"[Training] Evaluation failed at episode {ep}: {eval_e}")

        except Exception as e:
            print(f"[Training] Fatal error: {e}")
            traceback.print_exc()

        finally:
            self.writer.close()
            self.env.close()
            print("[Training] Completed. Unity closed.")

        return avg_reward

    # Modified _run_episode method with per-agent transition handling for sparse reward learning
    def _run_episode(self, env, episode: int, total_steps: int, train_iters: int, noise_scale: float):
        cfg = self.cfg
        state = env.reset(train_mode=True)
        self.agent.reset_noise()
        ep_reward = 0.0
        steps_since_update = 0

        for step in range(cfg.max_steps):
            total_steps += 1
            steps_since_update += 1

            norm_state = self.norm.normalize(state) if self.norm else state
            if total_steps < cfg.sampling_warmup:
                action = np.random.uniform(-1, 1, (cfg.num_agents, cfg.action_size))
            else:
                action = self.agent.act(norm_state, noise=noise_scale)

            next_state, reward, dones = env.step(action)

            if self.norm:
                self.norm.update(state)

            # Ensure rewards are NumPy array
            reward = np.array(reward, dtype=np.float64)

            # Clip rewards per agent
            reward = np.clip(reward, -1.0, 1.0)

            # Optionally scale
            if cfg.use_reward_scaling:
                reward *= cfg.reward_scale

            # Optional reward normalization
            if cfg.use_reward_norm:
                n = reward.size
                rs = self.reward_stats
                old_c = rs["count"]
                new_c = old_c + n
                new_mean = (old_c * rs["mean"] + reward.sum()) / new_c
                new_var = ((old_c * (rs["var"] + rs["mean"]**2) + (reward**2).sum()) / new_c) - new_mean**2
                rs.update(mean=new_mean, var=(new_var if new_var > 0 else 1.0), count=new_c)
                reward = (reward - rs["mean"]) / (np.sqrt(rs["var"])+1e-8)

            # Sum mean reward for logging
            ep_reward += np.mean(reward)
            mask = np.array([0 if d else 1 for d in dones], dtype=np.float32)

            anneal_frac = min(1.0, total_steps / cfg.zero_reward_prob_anneal_steps)
            zero_reward_prob = (
                cfg.zero_reward_prob_start
                + anneal_frac * (cfg.zero_reward_prob_end - cfg.zero_reward_prob_start)
            )

            # --- Feed each agent's transition independently ---
            for i in range(cfg.num_agents):
                keep = True
                if np.abs(reward[i]) < 1e-6 and np.random.rand() > zero_reward_prob:
                    keep = False
                if keep:
                    self.buffer.feed({
                        'state': [state[i]],            # wrap into list
                        'action': [action[i]],           # wrap into list
                        'reward': [reward[i]],           # wrap into list
                        'next_state': [next_state[i]],   # wrap into list
                        'mask': [mask[i]],               # wrap into list
                    })

            # --- Trigger learning ---
            interval = cfg.scheduler.interval(total_steps)
            if total_steps >= cfg.learn_after and self.buffer.size() >= cfg.batch_size and steps_since_update >= interval:
                train_iters = self._learn_block(train_iters, total_steps)
                steps_since_update = 0

            # --- Logging ---
            if step % LOG_FREQ == 0:
                self.writer.add_scalar("ReplayBuffer/Size", self.buffer.size(), train_iters)
                self.writer.add_scalar("ReplayBuffer/ZeroRewardProb", zero_reward_prob, train_iters)

                if self.buffer.size() > 0:
                    # Extract rewards from the buffer
                    rewards = np.array(self.buffer.reward[:self.buffer.size()])

                    reward_mean = np.mean(rewards)
                    reward_std = np.std(rewards)

                    self.writer.add_scalar("ReplayBuffer/Reward/Mean", reward_mean, train_iters)
                    self.writer.add_scalar("ReplayBuffer/Reward/Std", reward_std, train_iters)

                    self.writer.add_histogram("ReplayBuffer/RewardDistribution", rewards, train_iters)

            state = next_state
            if all(dones):
                break

        return ep_reward, total_steps, train_iters

    def _learn_block(self, train_iters: int, total_steps: int) -> int:
        cfg = self.cfg
        
        for _ in range(cfg.updates_per_block):
            batch = self.buffer.sample()

            idxs = getattr(batch, 'idx', None)
            
            if cfg.use_prioritized_replay:
                # Linear annealing of beta
                anneal_fraction = min(1.0, total_steps / cfg.per_beta_anneal_steps)
                beta = cfg.per_beta_start + anneal_fraction * (cfg.per_beta_end - cfg.per_beta_start)

                self.writer.add_scalar("Debug/PER/Beta", beta, train_iters)

                batch, is_weights = convert_batch_to_tensor(batch, device=self.agent.device, use_priorities=True, beta=beta)
            else:
                batch = convert_batch_to_tensor(batch, device=self.agent.device, use_priorities=False)
                is_weights = None
            
            metrics = self.agent.learn(batch, is_weights)
            
            if train_iters % LOG_FREQ == 0:
                self.writer.add_scalar("Loss/Critic", metrics["critic_loss"], train_iters)
                self.writer.add_scalar("Loss/Actor", metrics["actor_loss"], train_iters)
                self.writer.add_scalar("Q-values/Current", metrics["current_Q_mean"], train_iters)
                self.writer.add_scalar("Q-values/Target", metrics["target_Q_mean"], train_iters)
                self.writer.add_scalar("Training/batch_reward_mean", metrics["batch_reward_mean"], train_iters)
                self.writer.add_scalar("Training/batch_reward_std", metrics["batch_reward_std"], train_iters)
                self.writer.add_scalar("Training/batch_reward_max", metrics["batch_reward_max"], train_iters)
                self.writer.add_scalar("Training/batch_reward_min", metrics["batch_reward_min"], train_iters)
                self.writer.add_scalar("Training/TD_Error/Mean", metrics["td_error_mean"], train_iters)
                self.writer.add_scalar("Training/TD_Error/Std", metrics["td_error_std"], train_iters)

                # ✅ Additional debug metrics:
                self.writer.add_scalar("Debug/StateMean", metrics["state_mean"], train_iters)
                self.writer.add_scalar("Debug/StateStd", metrics["state_std"], train_iters)
                self.writer.add_scalar("Debug/ActionMean", metrics["action_mean"], train_iters)
                self.writer.add_scalar("Debug/ActionStd", metrics["action_std"], train_iters)
                self.writer.add_scalar("Debug/TargetQStd", metrics["target_Q_std"], train_iters)
                self.writer.add_scalar("Debug/CurrentQMin", metrics["current_Q_min"], train_iters)
                self.writer.add_scalar("Debug/CurrentQMax", metrics["current_Q_max"], train_iters)
                
            train_iters += 1
            
            if idxs is not None:
                eps = 1e-6
                alpha = 0.6  # You can expose this as a hyperparameter if desired

                updates = [(i, (float(td) + eps) ** alpha) for i, td in zip(idxs, metrics['td_error'])]
                self.buffer.update_priorities(updates)
                self.writer.add_scalar("ReplayBuffer/Priorities/Mean", np.mean([u[1] for u in updates]), train_iters)
                
                priorities = np.array([float(td) for td in metrics["td_error"]])
                self.writer.add_histogram("ReplayBuffer/Priorities/TD_Error", priorities, train_iters)

                self.writer.add_scalar("IS_Weights/Mean", is_weights.mean(), train_iters)

        return train_iters


def extract_agent_weights(agent):
    return {
        "actor": {k: v.cpu() for k, v in agent.actor.state_dict().items()},
        "actor_target": {k: v.cpu() for k, v in agent.actor_target.state_dict().items()},
        "critic": {k: v.cpu() for k, v in agent.critic.state_dict().items()},
        "critic_target": {k: v.cpu() for k, v in agent.critic_target.state_dict().items()},
    }


def train(**kwargs) -> float:
    cfg = TrainingConfig(**kwargs)
    runner = TrainingRunner(cfg)
    return runner.run()

if __name__ == "__main__":
    print("Starting DDPG Training...")
    result = train()
    print(f"Done. Result: {result:.2f}")
