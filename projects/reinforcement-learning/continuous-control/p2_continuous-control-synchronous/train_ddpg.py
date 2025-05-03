import os
import time
import numpy as np
import torch
import traceback
from dataclasses import dataclass, field
from typing import Optional, Protocol, Tuple
from torch.utils.tensorboard import SummaryWriter
from codebase.ddpg.agent import DDPGAgent
from codebase.ddpg.env import BootstrappedEnvironment
from codebase.ddpg.eval import evaluate
from codebase.replay.replay import ReplayBuffer
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

class VelocityRewardShaper:
    def __init__(self, 
                 alpha: float = 1.0, 
                 momentum: float = 0.99, 
                 initial_baseline: float = 0.05,
                 decay_alpha_from: Optional[float] = None,
                 decay_alpha_to:   Optional[float] = None,
                 decay_alpha_until: Optional[int]  = None,
                 env_positive_boost: float = 0.3,
                 slowdown_weight:    float = 0.1,
                 slowdown_dist_thr:  float = 0.3,
                 proximity_weight:   float = 0.2):
        """
        alpha               – initial blending weight for shaping
        decay_alpha_from    – start value for alpha decay
        decay_alpha_to      – final value for alpha decay
        decay_alpha_until   – number of steps over which to decay alpha
        env_positive_boost  – bonus shaping when env_reward > 0
        slowdown_weight     – penalty for moving fast when near the goal
        slowdown_dist_thr   – distance threshold under which to start slowing
        proximity_weight    – bonus proportional to (1 - normalized distance)
        """
        self.alpha = alpha
        self.alpha_init = decay_alpha_from or alpha
        self.alpha_target = decay_alpha_to or alpha
        self.alpha_decay_until = decay_alpha_until

        self.momentum = momentum
        self.baseline = initial_baseline

        self.env_positive_boost = env_positive_boost
        self.slowdown_weight    = slowdown_weight
        self.slowdown_dist_thr  = slowdown_dist_thr
        self.proximity_weight   = proximity_weight

        # for logging
        self.last_goal_distance = None

    def _update_alpha(self, total_steps: int):
        if self.alpha_decay_until and total_steps < self.alpha_decay_until:
            frac = total_steps / self.alpha_decay_until
            self.alpha = (
                self.alpha_target +
                (self.alpha_init - self.alpha_target) * (1.0 - frac)
            )
        elif self.alpha_decay_until:
            self.alpha = self.alpha_target

    def __call__(self,
                 next_state: np.ndarray,
                 reward:     np.ndarray,
                 total_steps: Optional[int] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
        # 0) decay alpha if configured
        if total_steps is not None:
            self._update_alpha(total_steps)

        # 1) extract velocity & goal-vector
        v      = next_state[:, 23:26]
        goal   = next_state[:, 26:29]
        v_norm = np.linalg.norm(v, axis=1) + 1e-8
        g_norm = np.linalg.norm(goal, axis=1) + 1e-8
        cos_sim = np.sum(v * goal, axis=1) / (v_norm * g_norm)

        # log for distance diagnostics
        self.last_goal_distance = g_norm

        # 2) base shaping: encourage movement toward goal
        shaped = cos_sim * v_norm

        # 3) boost when env reward > 0 (partial or full goal contact)
        shaped += (reward > 0).astype(np.float64) * self.env_positive_boost

        # 4) slowdown penalty: discourage high speed when near the goal
        near_goal = (g_norm < self.slowdown_dist_thr).astype(np.float64)
        slowdown_penalty = v_norm * near_goal * self.slowdown_weight
        shaped -= slowdown_penalty

        # 5) proximity bonus: reward being closer to the goal
        #    (we normalize distance by a rough max of 1.0)
        proximity_term = (1.0 - np.clip(g_norm, 0.0, 1.0))
        shaped += proximity_term * self.proximity_weight

        # 6) update moving baseline from env reward
        reward_max = np.max(reward) + 1e-6
        self.baseline = (
            self.momentum * self.baseline +
            (1 - self.momentum) * reward_max
        )

        # 7) scale shaped to match baseline & current alpha
        shaped_peak = np.max(np.abs(shaped)) + 1e-6
        scaled_shaped = self.alpha * shaped / shaped_peak * self.baseline

        # return combined (env + shaped) and the shaping term
        return reward + scaled_shaped, scaled_shaped, g_norm


shaper = VelocityRewardShaper(momentum=0.995, initial_baseline=0.05, 
                              alpha=3.5, decay_alpha_from=3.5, 
                              decay_alpha_to=0.5, decay_alpha_until=50000,
                              env_positive_boost=1.0,
                              slowdown_weight = 0.1,
                              slowdown_dist_thr = 0.3,
                              proximity_weight = 0.2)

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
            use_graphics=False,
            reward_shaping_fn=cfg.reward_shaping_fn,
        )

        # Agent
        self.agent = DDPGAgent(
            num_agents=cfg.num_agents,
            state_size=cfg.state_size,
            action_size=cfg.action_size,
            critic_input_size=256,
            critic_hidden_size=128,
            actor_input_size=256,
            actor_hidden_size=128,
            lr_actor=cfg.lr_actor,
            lr_critic=cfg.lr_critic,
            gamma=cfg.gamma,
            tau=cfg.tau,
            use_ou_noise=cfg.use_ou_noise,
            ou_noise_theta=cfg.ou_theta,
            ou_noise_sigma=cfg.ou_sigma,
            critic_weight_decay=cfg.critic_weight_decay,
            critic_clip_norm=cfg.critic_clipnorm
        )

        if cfg.load_weights:
            data = torch.load(cfg.load_weights, map_location=device)
            for net in ['actor', 'actor_target', 'critic', 'critic_target']:
                getattr(self.agent, net).load_state_dict(data[net])
            cfg.sampling_warmup = 0
        
        # Replay buffer
        # replay_args = dict(memory_size=cfg.replay_capacity,
        #                    batch_size=cfg.batch_size,
        #                    discount=cfg.gamma, n_step=1, history_length=1)
        
        #self.buffer = PrioritizedReplay(**replay_args) if cfg.use_prioritized_replay else UniformReplay(**replay_args)
        self.buffer = ReplayBuffer(
            action_size=cfg.action_size,
            buffer_size=cfg.replay_capacity,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            device=self.agent.device,
        )

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
                ## TRAINING LOOP ##
                for ep in range(1, self.cfg.episodes + 1):                    
                    ep_reward, total_steps, train_iterations, metrics, ep_env_reward, ep_shaped_reward, goal_hit_steps = self._run_episode(
                        env, ep, total_steps, train_iterations
                    )

                    self.writer.add_scalar("Episode/Reward", ep_reward, ep)
                    self.writer.add_histogram('Reward/PerAgentRewards', ep_reward, ep)

                    if ep_shaped_reward is None:
                        ep_shaped_reward = 0.0

                    reward_stats = f"Reward/Env {ep_env_reward:.4f} | " \
                        f"Reward/Shaped {ep_shaped_reward:.4f} | "
                        
                    if metrics:
                        print(
                            f"Episode {ep} | Reward {ep_reward:.2f} | Steps {total_steps} | Updates {train_iterations} | Goal Hit Steps: {goal_hit_steps} | Actor Loss {metrics['actor_loss']:.4f} | Critic Loss {metrics['critic_loss']:.8f} | {reward_stats}"
                        )
                    else:
                        print(f"Episode {ep} | Reward {ep_reward:.2f} | Steps {total_steps} | Updates {train_iterations} | Goal Hit Steps: {goal_hit_steps} | {reward_stats}")

                    ## EVALUATION ##
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
    
    def _run_episode(self, env, episode: int, total_steps: int, train_iters: int ): #, noise_scale: float):
        cfg = self.cfg
        state = env.reset(train_mode=True)
        self.agent.reset_noise()
        
        ## DYNAMIC NOISE ##

        # Precompute initial goal distance from the reset state
        # (so even the first action can use distance-aware noise)
        g_norm = np.linalg.norm(state[:, 26:29], axis=1)  # shape: (num_agents,)
        mean_dist = g_norm.mean()

        # Compute the default (episode-based) noise scale
        default_noise = max(
            cfg.min_noise,
            cfg.init_noise * np.exp(-self.noise_lambda * episode)
        ) if cfg.noise_decay else cfg.init_noise

        # If we're already 'within threshold', override with distance-based noise
        threshold = cfg.dynamic_noise_distance_threshold
        if cfg.use_dynamic_noise and mean_dist < threshold:
            noise_scale = cfg.min_noise + (cfg.init_noise - cfg.min_noise) * (mean_dist / threshold)
        else:
            noise_scale = default_noise # fixed noise scale

        ## EPISODE ACCUMULATORS ##
        ep_reward = 0.0
        ep_env_reward = 0.0
        ep_shaped_reward = 0.0
        steps_since_update = 0
        
        goal_hit_steps = 0     # total hits across all steps and agents
        
        episode_actor_losses = []
        episode_critic_losses = []

        ## EPISODE LOOP ##
        for step in range(cfg.max_steps):
            total_steps += 1
            steps_since_update += 1

            norm_state = self.norm.normalize(state) if self.norm else state

            if total_steps < cfg.sampling_warmup:
                action = np.random.uniform(-1, 1, (cfg.num_agents, cfg.action_size))
            else:
                action, noise = self.agent.act(norm_state, eval=False, noise_scale=noise_scale)
            
            if noise is not None and step % LOG_FREQ == 0 and self.writer is not None:
                self.writer.add_scalar('Noise/Mean', np.mean(noise), total_steps)
                self.writer.add_scalar('Noise/Std', np.std(noise), total_steps)
                self.writer.add_scalar('Noise/Min', np.min(noise), total_steps)
                self.writer.add_scalar('Noise/Max', np.max(noise), total_steps)
                self.writer.add_histogram('Noise/Histogram', noise, total_steps)

            next_state, reward_info, dones = env.step(action)

            if self.norm:
                self.norm.update(state)

            reward = np.array(reward_info[0], dtype=np.float32)            
            env_reward = np.array(reward_info[1], dtype=np.float32)
            shaped_reward = np.array(reward_info[2], dtype=np.float32) if reward_info[2] is not None else None     
            
            g_norm = np.array(reward_info[3], dtype=np.float32) if len(reward_info) > 3 else None

            # — COUNT HOW MANY AGENTS GOT env_reward == 0.1 THIS STEP
            hits = np.isclose(env_reward, 0.1, atol=1e-6)
            goal_hit_steps += hits.sum()

            if cfg.reward_shaping_fn is not None and total_steps < cfg.use_shaped_reward_only_steps:
                reward = shaped_reward

            if total_steps == cfg.use_shaped_reward_only_steps:
                print("[INFO] Switching from shaped-only rewards to shaped + environment rewards.")

            if cfg.use_reward_scaling:
                reward *= cfg.reward_scale

            ep_reward += np.mean(reward) # episode reward is the mean from all agents
            ep_env_reward += np.mean(env_reward)

            if shaped_reward is not None:
                ep_shaped_reward += np.mean(shaped_reward) if not cfg.use_reward_scaling else np.mean(shaped_reward * cfg.reward_scale) 
            else:
                ep_shaped_reward = None

            mask = np.array([0 if d else 1 for d in dones], dtype=np.float32)

            anneal_frac = min(1.0, total_steps / cfg.zero_reward_prob_anneal_steps)
            zero_reward_prob = cfg.zero_reward_prob_start + anneal_frac * (cfg.zero_reward_prob_end - cfg.zero_reward_prob_start)

            for i in range(cfg.num_agents):
                keep = True
                if np.abs(reward[i]) < 1e-6 and np.random.rand() > zero_reward_prob:
                    keep = False
                
                if not cfg.zero_reward_filtering:
                    keep = True

                if keep:                    
                    self.buffer.add(
                        state[i],
                        action[i],
                        reward[i],
                        next_state[i],
                        mask[i]
                    )

            ## LEARN BLOCK ##
            interval = cfg.scheduler.interval(total_steps)
            if total_steps >= cfg.learn_after and self.buffer.size() >= cfg.batch_size and steps_since_update >= interval:
                train_iters, metrics = self._learn_block(train_iters, total_steps)
                steps_since_update = 0

                episode_actor_losses.append(metrics['actor_loss'])
                episode_critic_losses.append(metrics['critic_loss'])

            ## LOGGING BLOCK ##
            if step % LOG_FREQ == 0:                
                self.writer.add_scalar("ReplayBuffer/Size", self.buffer.size(), train_iters)
                self.writer.add_scalar("ReplayBuffer/ZeroRewardProb", zero_reward_prob, train_iters)
                
                self.writer.add_scalar("Noise/Scale", noise_scale, train_iters)

                if shaped_reward is not None:
                    mean_dist = g_norm.mean()
                    self.writer.add_scalar("GoalZone/Distance/Mean", mean_dist, total_steps)
                    self.writer.add_scalar("GoalZone/Distance/Max", np.max(mean_dist), total_steps)
                    self.writer.add_scalar("GoalZone/Distance/Min", np.min(mean_dist), total_steps)
                    
                    shaping_stats = {
                        "Reward/combined_mean": np.mean(reward),
                        "Reward/env_mean": np.mean(env_reward),
                        "Reward/shaped_mean": np.mean(shaped_reward),
                        "Reward/shaped_max": np.max(shaped_reward),
                        "Reward/shaped_min": np.min(shaped_reward),
                    }
                else:
                    shaping_stats = {                        
                        "Reward/env_mean": np.mean(reward),
                    }

                for key, value in shaping_stats.items():
                    self.writer.add_scalar(key, value, train_iters)

            state = next_state

            # Recompute noise_scale for the *next* action based on new distance
            mean_dist = g_norm.mean()
            
            if cfg.use_dynamic_noise and mean_dist < threshold:
                noise_scale = cfg.min_noise + (cfg.init_noise - cfg.min_noise) * (mean_dist / threshold)
            else:
                noise_scale = default_noise
            
            if all(dones):
                break
        
        ## AT END OF EPISODE: log how many agent-steps hit the goal zone —
        self.writer.add_scalar("GoalZone/StepsHitCount", goal_hit_steps, episode)
        if episode_actor_losses:
            mean_actor_loss = np.mean(episode_actor_losses)
            self.writer.add_scalar("Episode/Loss/Actor_Mean", mean_actor_loss, episode)            

        if episode_critic_losses:
            mean_critic_loss = np.mean(episode_critic_losses)
            self.writer.add_scalar("Episode/Loss/Critic_Mean", mean_critic_loss, episode)            
        
        loss_metrics = {
            "actor_loss": mean_actor_loss if episode_actor_losses else 0.0,
            "critic_loss": mean_critic_loss if episode_critic_losses else 0.0,
        }

        return ep_reward, total_steps, train_iters, loss_metrics, ep_env_reward, ep_shaped_reward, goal_hit_steps


    def _learn_block(self, train_iters: int, total_steps: int) -> int:
        cfg = self.cfg
        
        for _ in range(cfg.updates_per_block):
            batch = self.buffer.sample() # sample a batch from the replay buffer
           
            metrics = self.agent.learn(batch)
            
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
                self.writer.add_scalar("Debug/StateMin", metrics["state_min"], train_iters)
                self.writer.add_scalar("Debug/StateMax", metrics["state_max"], train_iters)
                self.writer.add_scalar("Debug/ActionMean", metrics["action_mean"], train_iters)
                self.writer.add_scalar("Debug/ActionStd", metrics["action_std"], train_iters)
                self.writer.add_scalar("Debug/ActionMin", metrics["action_min"], train_iters)
                self.writer.add_scalar("Debug/ActionMax", metrics["action_max"], train_iters)

                self.writer.add_scalar("Debug/TargetQStd", metrics["target_Q_std"], train_iters)
                self.writer.add_scalar("Debug/CurrentQMin", metrics["current_Q_min"], train_iters)
                self.writer.add_scalar("Debug/CurrentQMax", metrics["current_Q_max"], train_iters)
                
            train_iters += 1
            
        return train_iters, metrics


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
