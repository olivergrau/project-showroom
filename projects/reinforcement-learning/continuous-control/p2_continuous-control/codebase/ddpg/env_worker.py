# env_worker.py
import os
import signal
import time
import torch
from unityagents import UnityEnvironment
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from codebase.ddpg.agent import DDPGAgent
from codebase.utils.normalizer import RunningNormalizer  # your RunningNormalizer class

DEBUG = False
LOG_FREQ = 100  # Log TensorBoard metrics every LOG_FREQ steps

def env_worker(
        env_conn,
        eval_conn, 
        replay_process, 
        stop_flag, 
        unity_exe_path="Reacher_Linux/Reacher.x86_64", 
        gamma=0.99,
        lr_actor=5e-3,
        lr_critic=5e-3, 
        exploration_noise=0.15,
        exploration_noise_decay=0.9999,
        reward_scaling_factor=10.0,
        throttle_env_by=0.0,
        use_ou_noise=False,
        log_dir=None):
    """
    Environment process:
      - Interacts with the Unity environment.
      - Uses a RunningNormalizer to update and normalize state observations.
      - Feeds normalized transitions into the replay buffer.
      - Sends updated normalizer parameters periodically so that other workers can remain in sync.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print(f"EnvWorker: gamma={gamma}, lr_actor={lr_actor}, lr_critic={lr_critic}, exploration_noise={exploration_noise}, exploration_noise_decay={exploration_noise_decay}, reward_scaling_factor={reward_scaling_factor}")
    print("[EnvWorker] Starting environment worker...")
    
    # Create a log directory for environment metrics
    if log_dir is None:
        env_log_dir = os.path.join("runs", "env")
    else:
        env_log_dir = os.path.join(log_dir, "env")
    
    env_writer = SummaryWriter(log_dir=env_log_dir)

    # Create the Unity environment
    env = UnityEnvironment(file_name=unity_exe_path, no_graphics=True, worker_id=1)
    brain_name = env.brain_names[0]
    
    # Reinitialize environment if necessary
    max_retries = 5
    retry_delay = 2  # seconds
    for attempt in range(max_retries):
        try:
            env_info = env.reset(train_mode=True)[brain_name]
            print(f"[EnvWorker] Successfully reset Unity environment on attempt {attempt + 1}")
            break
        except KeyError as e:
            print(f"[EnvWorker] Error on env.reset: {e}. Attempt {attempt+1}/{max_retries}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    else:
        raise RuntimeError("[EnvWorker] Failed to reset Unity environment after maximum retries.")

    # Signal readiness to parent process
    env_conn.send({"command": "ready", "worker": "env"})

    # Instantiate the agent that will act in the environment
    agent = DDPGAgent(
        state_size=33, 
        action_size=4, 
        gamma=gamma, 
        label="EnvWorker", 
        lr_actor=lr_actor, 
        lr_critic=lr_critic,
        use_ou_noise=use_ou_noise)

    # Instantiate RunningNormalizer (assume state dimension is 33)
    state_normalizer = RunningNormalizer(shape=(33,), momentum=0.001)
    
    decay_interval = 2000
    normalizer_update_interval = 200  # update normalizer params every 200 steps

    step_counter = 0
    episode_rewards = np.zeros(len(env_info.agents))  # one per agent

    try:
        env_info = env.reset(train_mode=True)[brain_name]
        raw_states = env_info.vector_observations  # raw states from environment
        
        # Update normalizer and normalize initial state.
        state_normalizer.update(raw_states)
        normalized_states = state_normalizer.normalize(raw_states)

        start_time = time.time()
        
        while not stop_flag.is_set(): # a step
            # Process incoming messages (e.g., weight updates, stop command)
            if env_conn.poll(0.001):
                message = env_conn.recv()
                if isinstance(message, dict):
                    if message.get("command") == "update_weights":
                        new_weights = message["weights"]
                        load_agent_weights(agent, new_weights)
                        
                        if DEBUG:
                            print("[EnvWorker] Updated agent weights.")

                    elif message.get("command") == "stop":
                        print("[EnvWorker] Received stop command. Terminating environment loop.")
                        break

                    # (Optionally, process normalizer updates from another worker here)
                elif isinstance(message, str) and message == "stop":
                    print("[EnvWorker] Received stop command. Terminating environment loop.")
                    break
            
            # Select actions using the normalized state.
            actions = agent.act(normalized_states, noise=exploration_noise)
            actions = np.clip(actions, -1, 1)
            
            # Step the environment.
            env_info = env.step(actions)[brain_name]
            raw_next_states = env_info.vector_observations  # raw next states
            rewards = env_info.rewards
            dones = env_info.local_done
            
            # Scale rewards.
            if reward_scaling_factor is not None and reward_scaling_factor != 1.0:
                scaled_rewards = [r * reward_scaling_factor for r in rewards]
            else:
                scaled_rewards = rewards

            # Update normalizer with the new raw next states, then normalize.
            state_normalizer.update(raw_next_states)
            normalized_next_states = state_normalizer.normalize(raw_next_states)
            
            avg_step_reward = np.mean(scaled_rewards)

            # Log step reward.
            if step_counter % LOG_FREQ == 0:                
                env_writer.add_scalar("Env/Step_Mean_Reward", avg_step_reward, step_counter)

            step_counter += 1
            
            # Accumulate rewards for each agent for episodic reward logging
            episode_rewards += np.array(scaled_rewards)
            
            # Check and log episode completion per agent.
            done_mask = np.array(dones, dtype=bool)
            if done_mask.any():
                for i, done in enumerate(done_mask):
                    if done:
                        env_writer.add_scalar("Env/Episode_Reward", episode_rewards[i], step_counter)
                        episode_rewards[i] = 0.0
            
            if step_counter % 1000 == 0 and agent.use_ou_noise == True:
                agent.reset_noise()

            if DEBUG and step_counter % LOG_FREQ == 0:
                print(f"[EnvWorker] Step {step_counter}, Avg. Step Reward: {avg_step_reward:.6f}")

            # Feed the normalized transition into the replay buffer. (one transition for data from 20 agents)
            replay_process.feed({
                'state': normalized_states,
                'action': actions,
                'reward': scaled_rewards,
                'next_state': normalized_next_states,
                'mask': [0 if d else 1 for d in dones]
            })
            
            # Update current state.
            normalized_states = normalized_next_states

            # Decay exploration noise.
            if exploration_noise_decay is not None and step_counter % decay_interval == 0:
                exploration_noise *= exploration_noise_decay
                exploration_noise = max(exploration_noise, 0.01)

                env_writer.add_scalar("Env/Exploration_Noise", exploration_noise, step_counter)
                
                if DEBUG and step_counter % 1000 == 0:
                    print(f"[EnvWorker] Exploration noise decayed to: {exploration_noise:.6f}")
            
            # Periodically send updated normalizer parameters to other workers.
            if step_counter % normalizer_update_interval == 0:
                norm_params = {
                    "mean": state_normalizer.mean.tolist(),
                    "var": state_normalizer.var.tolist(),
                    "count": state_normalizer.count
                }
                eval_conn.send({"command": "update_normalizer", "normalizer": norm_params})
            
            # Optionally log every 10 steps:
            if step_counter % 10 == 0:
                now = time.time()
                elapsed = now - start_time
                steps_per_second = 10 / elapsed
                
                if DEBUG:
                    print(f"[EnvWorker] Steps per second: {steps_per_second:.2f}")

                # Send the measurement to the train_worker:
                env_conn.send({"command": "step_rate", "steps_per_sec": steps_per_second})

                start_time = now
            
            if throttle_env_by > 0:
                time.sleep(throttle_env_by)
            
        print("[EnvWorker] Environment worker terminating...")
        print()
    
    finally:
        env.close()
        env_writer.close()
        print("[EnvWorker] Unity Environment closed.")

def load_agent_weights(agent, new_weights):
    """
    Helper function to load new weights into the environment worker's agent.
    Checks each network's state dict for consistency before loading the weights.
    """
    def check_state_dict_consistency(state_dict, module_name):
        for key, param in state_dict.items():
            if torch.isnan(param).any():
                print(f"[EnvWorker] Warning: {module_name} weight '{key}' contains NaN values!")
            if torch.sum(torch.abs(param)) == 0:
                print(f"[EnvWorker] Warning: {module_name} weight '{key}' is entirely zero!")
                
    if "actor" in new_weights:
        check_state_dict_consistency(new_weights["actor"], "Actor")
        old_weights = agent.actor.state_dict()
        agent.actor.load_state_dict(new_weights["actor"])
        
        if DEBUG:
            for key in old_weights.keys():
                old_norm = torch.norm(old_weights[key])
                new_norm = torch.norm(new_weights["actor"][key])
                print(f"[EnvWorker] Actor '{key}': old norm = {old_norm.item():.4f}, new norm = {new_norm.item():.4f}")
                if not torch.equal(old_weights[key].cpu(), new_weights["actor"][key]):
                    print(f"[EnvWorker] Actor weight updated: {key}")
    else:
        print("[EnvWorker] No actor weights found in the provided weights dictionary.")
        
    if "actor_target" in new_weights:
        check_state_dict_consistency(new_weights["actor_target"], "Actor Target")
        agent.actor_target.load_state_dict(new_weights["actor_target"])
        
    if "critic" in new_weights:
        check_state_dict_consistency(new_weights["critic"], "Critic")
        agent.critic.load_state_dict(new_weights["critic"])
        
    if "critic_target" in new_weights:
        check_state_dict_consistency(new_weights["critic_target"], "Critic Target")
        agent.critic_target.load_state_dict(new_weights["critic_target"])

    if DEBUG:
        print("[EnvWorker] Successfully loaded new agent weights.")
