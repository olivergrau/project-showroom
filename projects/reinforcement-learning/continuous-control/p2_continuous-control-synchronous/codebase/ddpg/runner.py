import numpy as np
import os
import time
import numpy as np
import torch
import traceback

from torch.utils.tensorboard import SummaryWriter
from codebase.ddpg.agent import DDPGAgent
from codebase.ddpg.env import BootstrappedEnvironment
from codebase.ddpg.eval import evaluate
from codebase.replay.replay import ReplayBuffer
from codebase.utils.normalizer import RunningNormalizer
from codebase.utils.early_stopping import EarlyStopping

class TrainingRunner:
    def __init__(self, cfg):
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
            data = torch.load(cfg.load_weights, map_location=cfg.device)
            for net in ['actor', 'actor_target', 'critic', 'critic_target']:
                getattr(self.agent, net).load_state_dict(data[net])
            cfg.sampling_warmup = 0
        
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
            
            if noise is not None and step % cfg.log_freq == 0 and self.writer is not None:
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
            if step % cfg.log_freq == 0:                
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
            
            if train_iters % cfg.log_freq == 0:
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