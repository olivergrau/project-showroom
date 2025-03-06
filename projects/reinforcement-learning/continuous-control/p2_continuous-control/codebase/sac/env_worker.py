# sac_env_worker.py
import os
import signal
import time
import torch
from unityagents import UnityEnvironment
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from codebase.sac.agent import SACAgent  # Import the SAC agent instead of TD3Agent

DEBUG = False

def env_worker(
        env_conn, 
        replay_process, 
        stop_flag, 
        unity_exe_path="Reacher_Linux/Reacher.x86_64", 
        gamma=0.99,
        lr_actor=1e-3,
        lr_critic=1e-3, 
        reward_scaling_factor=10.0,
        log_dir=None):
    """
    Environment process for SAC:
      - Steps the Unity environment using the current SAC policy.
      - Feeds experience into the replay buffer.
      - Listens for updated weights to refresh the policy.
      - Logs reward metrics via Tensorboard.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print(f"EnvWorker: gamma={gamma}, lr_actor={lr_actor}, lr_critic={lr_critic}, reward_scaling_factor={reward_scaling_factor}")

    print("[EnvWorker] Starting environment worker...")
    
    # Create a log directory for environment metrics.
    if log_dir is None:
        env_log_dir = os.path.join("runs", "env")
    else:
        env_log_dir = os.path.join(log_dir, "env")
    
    env_writer = SummaryWriter(log_dir=env_log_dir)

    # Create the Unity environment in training mode.
    env = UnityEnvironment(file_name=unity_exe_path, no_graphics=True, worker_id=1)
    brain_name = env.brain_names[0]
    
    # Attempt to reset the environment (with retries)
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

    # Signal readiness to the parent process.
    env_conn.send({"command": "ready", "worker": "env"})

    # Instantiate the SAC agent.
    agent = SACAgent(
        state_size=33, action_size=4, gamma=gamma, label="EnvWorker", lr_actor=lr_actor, lr_critic=lr_critic)
    
    try:
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        num_agents = len(env_info.agents)
        
        # Initialize counters for logging.
        step_counter = 0
        episode_rewards = np.zeros(num_agents)  # Cumulative rewards for each agent
        
        while not stop_flag.is_set():
            # Check for incoming messages (e.g. weight updates or stop command).
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
            
            # Use the SAC agent's sampling method (which includes its own stochasticity)
            actions = agent.act(states, evaluate=False)
            actions = np.clip(actions, -1, 1)
            
            # Step the environment.
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done            

            # Scale rewards.
            scaled_rewards = [r * reward_scaling_factor for r in rewards]

            # --- Logging Reward Metrics ---
            avg_step_reward = np.mean(scaled_rewards)
            env_writer.add_scalar("Env/Step_Mean_Reward", avg_step_reward, step_counter)
                        
            step_counter += 1

            # Accumulate rewards for episodic logging.
            episode_rewards += np.array(scaled_rewards)
            done_mask = np.array(dones, dtype=bool)
            if done_mask.any():
                for i, done in enumerate(done_mask):
                    if done:
                        env_writer.add_scalar("Env/Episode_Reward", episode_rewards[i], step_counter)
                        episode_rewards[i] = 0.0            

            # Send experience to replay buffer.
            replay_process.feed({
                'state': states,
                'action': actions,
                'reward': scaled_rewards,
                'next_state': next_states,
                'mask': [0 if d else 1 for d in dones]
            })
            
            # Update the current state.
            states = next_states

        print("[EnvWorker] Environment worker terminating...")
    
    finally:
        env.close()
        env_writer.close()
        print("[EnvWorker] Unity Environment closed.")

def load_agent_weights(agent, new_weights):
    """
    Helper function to load new weights into the SAC agent used by the environment worker.
    It checks the keys for the actor, critic, and critic target.
    """
    def check_state_dict_consistency(state_dict, module_name):
        for key, param in state_dict.items():
            if torch.isnan(param).any():
                print(f"[EnvWorker] Warning: {module_name} weight '{key}' contains NaN values!")
            if torch.sum(torch.abs(param)) == 0:
                print(f"[EnvWorker] Warning: {module_name} weight '{key}' is entirely zero!")

    # Update actor weights.
    if "actor" in new_weights:
        check_state_dict_consistency(new_weights["actor"], "Actor")
        agent.actor.load_state_dict(new_weights["actor"])
        if DEBUG:
            print("[EnvWorker] Actor weights updated.")
    else:
        print("[EnvWorker] No actor weights found in the provided weights dictionary.")
    
    # Update critic weights.
    if "critic" in new_weights:
        check_state_dict_consistency(new_weights["critic"], "Critic")
        agent.critic.load_state_dict(new_weights["critic"])
        if DEBUG:
            print("[EnvWorker] Critic weights updated.")
    else:
        print("[EnvWorker] No critic weights found in the provided weights dictionary.")
        
    # Update critic target weights.
    if "critic_target" in new_weights:
        check_state_dict_consistency(new_weights["critic_target"], "Critic Target")
        agent.critic_target.load_state_dict(new_weights["critic_target"])
        if DEBUG:
            print("[EnvWorker] Critic target weights updated.")
    else:
        print("[EnvWorker] No critic target weights found in the provided weights dictionary.")
    
    # Optionally, update temperature parameter if provided.
    if "log_alpha" in new_weights:
        agent.log_alpha.data.copy_(new_weights["log_alpha"].data)
        if DEBUG:
            print("[EnvWorker] log_alpha updated.")
    
    print("[EnvWorker] Successfully loaded new agent weights/parameters.")