import traceback
import numpy as np
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter
from codebase.experience.replay import MultiAgentReplayBuffer
from codebase.maddpg.env import BootstrappedEnvironment
from codebase.maddpg.maddpg_agent import MADDPGAgent
from tqdm.notebook import tqdm

from codebase.utils.running_norm import RunningNormalizer

LOG_FREQ = 100

class Trainer:
    def __init__(self,
                 env_path: str,
                 maddpg_agent: MADDPGAgent,
                 replay_buffer: MultiAgentReplayBuffer,
                 num_agents: int = 2,
                 obs_size: int = 8,
                 action_size: int = 2,
                 max_steps: int = 1000,
                 batch_size: int = 128,
                 train_every: int = 1,
                 warmup_steps: int = 1000,
                 log_dir: str = None,
                 worker_id: int = 0,
                 reward_shaping_fn=None,
                 use_state_norm: bool = False,
                 early_training_steps: int = 5000,
                 neg_sampling_ratio: float = 0.2):
        
        # print out hyperparameters nicely
        print(f"[Trainer] Hyperparameters:")
        print(f"  - env_path: {env_path}")
        print(f"  - num_agents: {num_agents}")
        print(f"  - obs_size: {obs_size}")
        print(f"  - action_size: {action_size}")
        print(f"  - max_steps: {max_steps}")
        print(f"  - batch_size: {batch_size}")
        print(f"  - train_every: {train_every}")
        print(f"  - warmup_steps: {warmup_steps}")
        print(f"  - log_dir: {log_dir}")
        print(f"  - worker_id: {worker_id}")
        
        # ——— Environment ———
        self.env = None
        self.env_path = env_path
        self.worker_id = worker_id
        self.reward_shaping_fn = reward_shaping_fn
        self.use_state_norm = use_state_norm
        self.early_training_steps = early_training_steps
        # during warmup, only keep this fraction of zero/negative transitions
        self.neg_sampling_ratio  = neg_sampling_ratio


        # ——— Agent & Buffer ———
        self.agent  = maddpg_agent
        self.buffer = replay_buffer

        # ——— Hyperparameters ———
        self.num_agents   = num_agents
        self.obs_size     = obs_size
        self.action_size  = action_size
        self.max_steps    = max_steps
        self.batch_size   = batch_size
        self.train_every  = train_every
        self.warmup_steps = warmup_steps

        # ——— Logging ———
        self.writer      = SummaryWriter(log_dir) if log_dir else SummaryWriter()
        self.total_steps = 0

        # State normalizer
        self.norm = RunningNormalizer((obs_size,), momentum=0.01) if use_state_norm else None

    def _train_offline(self, n_updates: int):
        """Offline: repeatedly sample & learn from the pre-filled buffer."""
        offline_metrics = []
        for upd in tqdm(range(1, n_updates+1), desc="Offline training"):
            if len(self.buffer) < self.batch_size:
                continue

            samples = self.buffer.sample()
            self.agent.step(samples)

            if upd % LOG_FREQ == 0:
                self._log_training_step(offline=True, step=upd)
                offline_metrics.append([logs[-1] for logs in self.agent.training_logs])

        if self.writer:
            self.writer.close()
        
        return offline_metrics

    def train(self, n_episodes: int = 2000, offline: bool = False):
        if offline:
            return self._train_offline(n_updates=n_episodes * self.max_steps)

        scores_deque  = deque(maxlen=100)
        all_scores    = []
        total_updates = 0

        try:
            with BootstrappedEnvironment(
                exe_path=self.env_path, 
                use_graphics=False, 
                worker_id=self.worker_id, 
                reward_shaping_fn=self.reward_shaping_fn) as env:
                
                self.env = env
                for ep in tqdm(range(1, n_episodes+1), desc="Training"): # for episode = 1 to M do
                    states = self.env.reset(train_mode=True) # Receive initial state x
                    self.agent.reset()
                    episode_scores = np.zeros(self.num_agents, np.float32)
                    episode_raw_scores = np.zeros(self.num_agents, np.float32)

                    for t in range(self.max_steps):                        
                        # 1) Select action on normalized states (after warmup)
                        if self.norm is not None and self.total_steps > self.warmup_steps:
                            preproc_states = self.norm.normalize(states)
                        else:
                            preproc_states = states
      
                        # —— 2) During warmup: override with random or Gaussian actions —— 
                        if self.total_steps <= self.warmup_steps:                            
                            # Gaussian noise around 0 (uncomment if you prefer)
                            sigma_warmup = 0.5
                            actions = [
                                np.clip(np.random.normal(0.0, sigma_warmup, size=self.action_size), -1.0, 1.0)
                                for _ in range(self.num_agents)
                            ]
                            noises = [None] * self.num_agents

                        else:
                            # —— 3) After warmup: use your learned policy + noise —— 
                            actions, noises = self.agent.act(preproc_states, eval=False)

                        self._log_noise(noises)
                            
                        # 3) step environment
                        next_states, reward_info, dones = self.env.step(actions)
                        #rewards = reward_info[0]
                        
                        rewards = reward_info[0] # potentially shaped
                        env_rewards = reward_info[1]

                        episode_scores += rewards                       
                        episode_raw_scores += env_rewards

                        # 4) selective negative‐reward filtering during warmup
                        store = False
                                            
                        # always keep any transition with a positive reward                    
                        if np.any(rewards > 0):
                            store = True
                        else:
                            # if we’re past warmup, keep all
                            if self.total_steps >= self.early_training_steps:
                                store = True
                            else:
                                # only keep a small fraction of zero/negative
                                if np.random.rand() < self.neg_sampling_ratio:
                                    store = True

                        if store:
                            self.buffer.add(
                                states,
                                actions,
                                rewards.tolist(),
                                next_states,
                                list(dones)
                            )

                        # 5) log buffer snapshot occasionally
                        if self.total_steps % LOG_FREQ == 0 and len(self.buffer.memory) > 0:
                            self._log_buffer(self.total_steps)

                        # 6) train if ready
                        if (len(self.buffer) >= self.batch_size
                            and self.total_steps > self.warmup_steps
                            and self.total_steps % self.train_every == 0):

                            samples = self.buffer.sample()
                            states_b, actions_b, rewards_b, next_states_b, dones_b = samples

                            # Update normalizer on large batch and normalize for learning
                            if self.norm is not None:
                                self.norm.update(states_b)
                                
                                states_b = self.norm.normalize(states_b)
                                next_states_b = self.norm.normalize(next_states_b)
                            
                                if self.total_steps % LOG_FREQ == 0:
                                    self._log_normalizer(self.total_steps)

                                    for a in range(self.num_agents):
                                        # log normalized states
                                        for j in range(states_b.shape[2]):  # obs_size
                                            self.writer.add_scalar(f"StatePreProc/agent_{a}/dim_{j}/mean", states_b[:, a, j].mean(), self.total_steps)
                                
                            self.agent.step((states_b, actions_b, rewards_b, next_states_b, dones_b))
                            total_updates += 1

                            if self.total_steps % LOG_FREQ == 0:
                                self._log_training_step(offline=False, step=self.total_steps)

                        states = next_states

                        self.total_steps += 1

                        if any(dones): # if all or any doesn't matter because of cooperative setting
                            break

                    # 6) end-of-episode logging
                    ma = self._log_episode(ep, episode_scores, episode_raw_scores, scores_deque, total_updates)
                    all_scores.append(np.sum(episode_scores)) # sum because of team setting (cooperative agents)

                    if ma > 0.6:
                        print(f"[Training] Early stopping at episode {ep} with score {ma:.3f}")
                        break

        except Exception as e:
            print(f"[Training] Fatal error: {e}")
            traceback.print_exc()
        finally:
            if self.writer:
                self.writer.close()
            
            print("[Training] Completed. Unity closed.")

        return all_scores

    # ——— Logging helpers ———

    def _log_normalizer(self, step: int):
        """Log the Distribution of per-feature mean & variance."""
        if self.norm is None or self.writer is None:
            return

        # pull out numpy arrays
        mean = self.norm.mean
        var  = self.norm.var
        if isinstance(mean, torch.Tensor):
            mean = mean.detach().cpu().numpy()
            var  = var .detach().cpu().numpy()

        # log as histograms
        self.writer.add_histogram("Normalizer/Mean", mean, step)
        self.writer.add_histogram("Normalizer/Var",  var,  step)

        # optionally also log summary scalars
        self.writer.add_scalar("Normalizer/Mean_min", mean.min(), step)
        self.writer.add_scalar("Normalizer/Mean_max", mean.max(), step)
        self.writer.add_scalar("Normalizer/Var_min",  var.min(),  step)
        self.writer.add_scalar("Normalizer/Var_max",  var.max(),  step)


    def _log_noise(self, noises):
        """Log per-agent noise stats."""
        if self.writer is None or self.total_steps % LOG_FREQ != 0 or noises is None:
            return
        
        for i, noise in enumerate(noises):
            if noise is None:
                continue
            
            self.writer.add_scalar(f"Noise/Agent{i}_Mean", noise.mean(), self.total_steps)
            self.writer.add_scalar(f"Noise/Agent{i}_Std",  noise.std(),  self.total_steps)
            self.writer.add_scalar(f"Noise/Agent{i}_Min",  noise.min(),  self.total_steps)
            self.writer.add_scalar(f"Noise/Agent{i}_Max",  noise.max(),  self.total_steps)
            self.writer.add_histogram(f"Noise/Agent{i}_Hist", noise, self.total_steps)

    def _log_buffer(self, step):
        """Log summaries of the current contents of the replay buffer."""
        if self.writer is None:
            return

        self.writer.add_scalar("Buffer/Size", len(self.buffer), step)

        for a in range(self.num_agents):
            # rewards
            rewards = np.array([e.rewards[a] for e in self.buffer.memory], np.float32)
            self.writer.add_scalar(f"Buffer_Reward/Mean_agent_{a}", rewards.mean(), step)
            self.writer.add_scalar(f"Buffer_Reward/Std_agent_{a}",  rewards.std(),  step)
            self.writer.add_scalar(f"Buffer_Reward/Min_agent_{a}",  rewards.min(),  step)
            self.writer.add_scalar(f"Buffer_Reward/Max_agent_{a}",  rewards.max(),  step)

            # actions
            acts = np.vstack([e.actions[a] for e in self.buffer.memory])
            self.writer.add_scalar(f"Buffer_Action/Mean_agent_{a}", acts.mean(), step)
            self.writer.add_scalar(f"Buffer_Action/Std_agent_{a}",  acts.std(),  step)
            self.writer.add_scalar(f"Buffer_Action/Min_agent_{a}",  acts.min(),  step)
            self.writer.add_scalar(f"Buffer_Action/Max_agent_{a}",  acts.max(),  step)
                        
            # Add per-dimension histogram
            for dim in range(acts.shape[1]):
                self.writer.add_histogram(
                    tag=f"Buffer_Action/agent_{a}/dim_{dim}",
                    values=acts[:, dim],
                    global_step=step
                )

            # states
            sts = np.vstack([e.states[a] for e in self.buffer.memory])
            self.writer.add_scalar(f"Buffer_State/Mean_agent_{a}", sts.mean(), step)
            self.writer.add_scalar(f"Buffer_State/Std_agent_{a}",  sts.std(),  step)
            self.writer.add_scalar(f"Buffer_State/Min_agent_{a}",  sts.min(),  step)
            self.writer.add_scalar(f"Buffer_State/Max_agent_{a}",  sts.max(),  step)

            # next_states
            nxt = np.vstack([e.next_states[a] for e in self.buffer.memory])
            self.writer.add_scalar(f"Buffer_State/Next_Mean_agent_{a}", nxt.mean(), step)
            self.writer.add_scalar(f"Buffer_State/Next_Std_agent_{a}",  nxt.std(),  step)
            self.writer.add_scalar(f"Buffer_State/Next_Min_agent_{a}",  nxt.min(),  step)
            self.writer.add_scalar(f"Buffer_State/Next_Max_agent_{a}",  nxt.max(),  step)


    def _log_training_step(self, offline: bool, step: int):
        """Log the most recent per-agent learning metrics."""
        tag = "offline_" if offline else ""
        
        if self.writer is None:
            return
        
        for i, logs in enumerate(self.agent.training_logs):
            if not logs:
                continue
            
            last = logs[-1]
            self.writer.add_scalar(f"Loss/Actor/{tag}agent_{i}",  last['actor_loss'],   step)
            self.writer.add_scalar(f"Loss/Critic/{tag}agent_{i}", last['critic_loss'],  step)
            self.writer.add_scalar(f"Q/Expected/{tag}agent_{i}",  last['q_expected'],   step)
            self.writer.add_scalar(f"Q/Target/{tag}agent_{i}",    last['q_target'],     step)
            self.writer.add_scalar(f"Grad/ActorNorm/{tag}agent_{i}", last['grad_norm'],  step)
            
            if 'dqda_mean' in last:
                self.writer.add_scalar(f"dQ_da/{tag}agent_{i}", last['dqda_mean'], step)

            actions = last['actions']
            for j in range(actions.shape[1]):
                self.writer.add_histogram(
                    tag=f"TrainingPredAction/{tag}agent_{i}/dim_{j}",
                    values=actions[:, j],
                    global_step=step
                )

    def _log_episode(self, ep, ep_scores, ep_raw_scores, scores_deque, total_updates):
        """Log end-of-episode and moving averages."""
        # per-agent episodic reward
        if self.writer is not None:
            for i, s in enumerate(ep_scores):
                self.writer.add_scalar(f"Reward/agent_{i}/EpisodeSum", s, ep)
        
        if self.writer is not None:
            self.writer.add_scalar("Reward/Episode/Mean", ep_scores.mean(), ep)
            self.writer.add_scalar("Reward/Episode/Std",  ep_scores.std(),  ep)
            self.writer.add_scalar("Reward/Episode/Min",  ep_scores.min(),  ep)
            self.writer.add_scalar("Reward/Episode/Max",  ep_scores.max(),  ep)

            self.writer.add_scalar("Reward/Raw/Episode/Mean", ep_raw_scores.mean(), ep)
            self.writer.add_scalar("Reward/Raw/Episode/Std",  ep_raw_scores.std(),  ep)
            self.writer.add_scalar("Reward/Raw/Episode/Min",  ep_raw_scores.min(),  ep)
            self.writer.add_scalar("Reward/Raw/Episode/Max",  ep_raw_scores.max(),  ep)

        # moving average over last 100
        scores_deque.append(ep_raw_scores.sum())
        ma = np.mean(scores_deque)
        
        if self.writer is not None:
            self.writer.add_scalar("Score/TeamSum_MovingAvg100", ma, ep)

        # update progress bar
        if ep % 25 == 0:
            tqdm.write(f"Episode {ep} | Last Score: {np.sum(ep_raw_scores):.2f} ({ep_raw_scores[0]:.2f}/{ep_raw_scores[1]:.2f}) | Moving Average {ma:.3f} | Steps {self.total_steps} | Updates {total_updates}")

        return ma
