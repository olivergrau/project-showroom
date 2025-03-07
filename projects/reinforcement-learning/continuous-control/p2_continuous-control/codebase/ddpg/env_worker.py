# env_worker.py
import os
import signal
import time
import torch
from unityagents import UnityEnvironment
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from codebase.ddpg.agent import DDPGAgent

DEBUG = False

def env_worker(
        env_conn, 
        replay_process, 
        stop_flag, 
        unity_exe_path="Reacher_Linux/Reacher.x86_64", 
        gamma=0.99,
        lr_actor=5e-3,
        lr_critic=5e-3, 
        exploration_noise=0.15,
        exploration_noise_decay=0.9999,
        reward_scaling_factor=10.0,
        log_dir=None):
    """
    Environment process: 
      - Steps the Unity environment using the current policy 
      - Feeds experience into the replay buffer
      - Listens for updated weights to refresh the policy
      - Logs reward metrics via SummaryWriter to monitor the reward signal.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print(f"EnvWorker: gamma={gamma}, lr_actor={lr_actor}, lr_critic={lr_critic}, exploration_noise={exploration_noise}, exploration_noise_decay={exploration_noise_decay}, reward_scaling_factor={reward_scaling_factor}")
    print("[EnvWorker] Starting environment worker...")
    
    # Create a separate log directory for environment metrics
    if log_dir is None:
        env_log_dir = os.path.join("runs", "env")
    else:
        env_log_dir = os.path.join(log_dir, "env")
    
    env_writer = SummaryWriter(log_dir=env_log_dir)

    # Create the environment in training mode
    env = UnityEnvironment(file_name=unity_exe_path, no_graphics=True, worker_id=1)
    brain_name = env.brain_names[0]
    
    # Reinitialize the environment if reset fails
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

    # Send a ready signal to the parent process via the pipe
    env_conn.send({"command": "ready", "worker": "env"})

    # Instantiate the agent that will act in the environment
    agent = DDPGAgent(
        state_size=33, action_size=4, gamma=gamma, label="EnvWorker", lr_actor=lr_actor, lr_critic=lr_critic)
    
    decay_intervall = 2000

    try:
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        num_agents = len(env_info.agents)
        
        # Initialize counters for logging
        step_counter = 0
        episode_rewards = np.zeros(num_agents)  # Cumulative rewards for each agent
        
        while not stop_flag.is_set():
            # Process any incoming messages (e.g. weight updates or stop command)
            if env_conn.poll(0.001):
                message = env_conn.recv()
                if isinstance(message, dict):
                    if message.get("command") == "update_weights":
                        new_weights = message["weights"]
                        load_agent_weights(agent, new_weights)
                        print("[EnvWorker] Updated agent weights.")
                    elif message.get("command") == "stop":
                        print("[EnvWorker] Received stop command. Terminating environment loop.")
                        break
                elif isinstance(message, str):
                    if message == "stop":
                        print("[EnvWorker] Received stop command. Terminating environment loop.")
                        break
            
            # Select actions for all agents using the current policy (with exploration noise)            
            actions = agent.act(states, noise=exploration_noise)  # shape: (num_agents, action_size)
            actions = np.clip(actions, -1, 1)
            
            # Step the environment
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done            

            # Scale rewards
            scaled_rewards = [r * reward_scaling_factor for r in rewards]

            # --- Logging Reward Metrics ---
            avg_step_reward = np.mean(scaled_rewards)
            env_writer.add_scalar("Env/Step_Mean_Reward", avg_step_reward, step_counter)
                        
            step_counter += 1

            # Accumulate rewards for each agent for episodic reward logging
            episode_rewards += np.array(scaled_rewards)

            # Check for episode termination for each agent
            done_mask = np.array(dones, dtype=bool)
            if done_mask.any():
                for i, done in enumerate(done_mask):
                    if done:
                        env_writer.add_scalar("Env/Episode_Reward", episode_rewards[i], step_counter)
                        episode_rewards[i] = 0.0            

            # Send experience data to replay buffer
            replay_process.feed({
                'state': states,
                'action': actions,
                'reward': scaled_rewards,
                'next_state': next_states,
                'mask': [0 if d else 1 for d in dones]
            })
            
            # Move on to the next states
            states = next_states

            # Decay exploration noise for the next action selection
            if exploration_noise_decay is not None:
                if step_counter % decay_intervall == 0:
                    exploration_noise *= exploration_noise_decay                
                    exploration_noise = max(exploration_noise, 0.01)  # Ensure noise doesn't decay below 0.01
                    env_writer.add_scalar("Env/Exploration_Noise", exploration_noise, step_counter)
                    if DEBUG and step_counter % 1000 == 0:
                        print(f"[EnvWorker] Exploration noise decayed to: {exploration_noise:.6f}")
            
        print("[EnvWorker] Environment worker terminating...")
    
    finally:
        env.close()
        env_writer.close()
        print("[EnvWorker] Unity Environment closed.")

import torch

def load_agent_weights(agent, new_weights):
    """
    Helper function to load new weights into the environment worker's agent.
    Checks each network's state dict for consistency (e.g., no NaN values and not entirely zero)
    before loading the weights.
    """
    def check_state_dict_consistency(state_dict, module_name):
        for key, param in state_dict.items():
            if torch.isnan(param).any():
                print(f"[EnvWorker] Warning: {module_name} weight '{key}' contains NaN values!")
            if torch.sum(torch.abs(param)) == 0:
                print(f"[EnvWorker] Warning: {module_name} weight '{key}' is entirely zero!")

    # Load actor weights
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
        
    # Load actor_target weights
    if "actor_target" in new_weights:
        check_state_dict_consistency(new_weights["actor_target"], "Actor Target")
        agent.actor_target.load_state_dict(new_weights["actor_target"])
        
    # Load critic weights (single critic for DDPG)
    if "critic" in new_weights:
        check_state_dict_consistency(new_weights["critic"], "Critic")
        agent.critic.load_state_dict(new_weights["critic"])
        
    # Load critic_target weights (single critic target for DDPG)
    if "critic_target" in new_weights:
        check_state_dict_consistency(new_weights["critic_target"], "Critic Target")
        agent.critic_target.load_state_dict(new_weights["critic_target"])

    print("[EnvWorker] Successfully loaded new agent weights.")
