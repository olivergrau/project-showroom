import os
import signal
import threading
import time
import torch
from unityagents import UnityEnvironment
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from codebase.ddpg.agent import DDPGAgent
from codebase.utils.normalizer import RunningNormalizer

DEBUG = False
LOG_FREQ = 100  # Log TensorBoard metrics every LOG_FREQ steps

def message_listener(train_conn, state_normalizer, agent, stop_flag):
    """
    Dedicated listener thread that continuously checks for messages from the train worker.
    It handles both normalizer updates and weight update messages.
    """
    while not stop_flag.is_set():
        try:
            if train_conn.poll(0.001):
                message = train_conn.recv()
                if isinstance(message, dict):
                    command = message.get("command")
                    if state_normalizer is not None and command == "update_normalizer":
                        norm_params = message.get("normalizer")

                        # Update the local normalizer parameters
                        state_normalizer.mean = np.array(norm_params["mean"], dtype=np.float32)
                        state_normalizer.var = np.array(norm_params["var"], dtype=np.float32)
                        state_normalizer.count = norm_params["count"]
                        
                        if DEBUG:
                            print("[EnvWorker] Updated normalizer parameters.")

                    elif command == "update_weights":
                        new_weights = message.get("weights")
                        load_agent_weights(agent, new_weights)
                        
                        # Send acknowledgment back to the train worker
                        train_conn.send({"command": "ack_update", "worker": "env"})
                        
                        if DEBUG:
                            print("[EnvWorker] Updated agent weights and sent ack.")
                    
                    elif command == "stop":
                        stop_flag.set()
                        break
        except EOFError:
            print("[EnvWorker] Message listener: EOFError occurred. Exiting.")
            break
        
        time.sleep(0.001)

def env_worker(
        train_conn,
        eval_conn, 
        main_conn,
        replay_process, 
        stop_flag, 
        unity_exe_path="Reacher_Linux/Reacher.x86_64",
        use_state_norm=False,
        # Actor        
        actor_input_size=400,      # Actor input layer size
        actor_hidden_size=300,     # Actor hidden layer size        
        # Critic
        critic_input_size=400,     # Critic input layer size
        critic_hidden_size=300,    # Critic hidden layer size
        # Noise
        use_ou_noise=False,
        ou_noise_theta=0.15,       # Noise theta parameter
        ou_noise_sigma=0.2,        # Noise sigma parameter  
        start_noise_scaling_factor=1.0,           # Initial noise scaling factor
        min_noise_scaling_factor=0.05,     # Minimum noise scaling factor after decay
        noise_decay_rate=0.99,     # Decay rate per episode for noise scaling factor        
        throttle_by=0.0,  
        log_dir=None):
    """
    Environment process:
      - Interacts with the Unity environment.
      - Uses a RunningNormalizer to update and normalize state observations.
      - Feeds raw transitions into the replay buffer.
      - Receives update messages from the train worker via a dedicated listener thread.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # print out parameters in nice format
    print("[EnvWorker] Environment worker parameters:")
    print(f"  - Unity Executable: {unity_exe_path}")
    print(f"  - Use State Normalization: {use_state_norm}")
    print(f"  - Actor Input Size: {actor_input_size}")
    print(f"  - Actor Hidden Size: {actor_hidden_size}")
    print(f"  - Critic Input Size: {critic_input_size}")
    print(f"  - Critic Hidden Size: {critic_hidden_size}")
    print(f"  - Use OU Noise: {use_ou_noise}")
    print(f"  - OU Noise Theta: {ou_noise_theta}")
    print(f"  - OU Noise Sigma: {ou_noise_sigma}")
    print(f"  - Exploration Start Noise Scaling Factor: {start_noise_scaling_factor}")
    print(f"  - Min Noise Scaling Factor: {min_noise_scaling_factor}")
    print(f"  - Noise Decay Rate: {noise_decay_rate}")
    print(f"  - Throttle By: {throttle_by}")
    print(f"  - Log Directory: {log_dir}")
    print()
    
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
            print()
            break
        except KeyError as e:
            print(f"[EnvWorker] Error on env.reset: {e}. Attempt {attempt+1}/{max_retries}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    else:
        raise RuntimeError("[EnvWorker] Failed to reset Unity environment after maximum retries.")

    # Signal readiness to parent process
    main_conn.send({"command": "ready", "worker": "env"})

    # Instantiate the agent that will act in the environment
    agent = DDPGAgent(
        state_size=33,
        action_size=4,
        lr_actor=1e-3, # not really needed when only act is used
        lr_critic=1e-3, # not really needed when only act is used
        use_ou_noise=use_ou_noise,
        ou_noise_theta=ou_noise_theta,
        ou_noise_sigma=ou_noise_sigma,
        actor_input_size=actor_input_size,
        actor_hidden_size=actor_hidden_size,
        critic_input_size=critic_input_size,
        critic_hidden_size=critic_hidden_size,
        label="EnvWorker"
    )

    # Instantiate RunningNormalizer (assume state dimension is 33)
    if use_state_norm:
        state_normalizer = RunningNormalizer(shape=(33,), momentum=0.001)
    else:
        state_normalizer = None

    noise_decay_interval = 2000    
    step_counter = 0
    episode_rewards = np.zeros(len(env_info.agents))  # one per agent

    # Variables for reporting steps per minute every 30 seconds
    last_report_time = time.time()
    last_report_steps = 0

    # Start the dedicated listener thread for messages from train worker
    listener_thread = threading.Thread(target=message_listener, args=(train_conn, state_normalizer, agent, stop_flag))
    listener_thread.daemon = True
    listener_thread.start()
    
    noise_scaling_factor = start_noise_scaling_factor

    try:
        env_info = env.reset(train_mode=True)[brain_name]
        raw_states = env_info.vector_observations  # raw states from environment
        
        # Normalize initial state for action selection
        if use_state_norm:
            preprocessed_states = state_normalizer.normalize(raw_states)
        else:
            preprocessed_states = raw_states

        start_time = time.time()
        
        while not stop_flag.is_set():
            if main_conn.poll():
                message = main_conn.recv()
                if isinstance(message, dict) and message.get("command") == "stop":
                    print("[EnvWorker] Received stop command from main process.")
                    stop_flag.set()
                    break

            # Select actions using the normalized state
            actions = agent.act(preprocessed_states, noise_scaling_factor=noise_scaling_factor)
            actions = np.clip(actions, -1, 1)
            
            # Step the environment
            env_info = env.step(actions)[brain_name]
            raw_next_states = env_info.vector_observations  # raw next states
            rewards = env_info.rewards
            dones = env_info.local_done
            
            if use_state_norm:
                preprocessed_next_states = state_normalizer.normalize(raw_next_states)
            else:
                preprocessed_next_states = raw_next_states

            avg_step_reward = np.mean(rewards)

            if step_counter % LOG_FREQ == 0:
                env_writer.add_scalar("Env/Step_Mean_Reward", avg_step_reward, step_counter)

            step_counter += 1
            
            # Accumulate rewards for episodic logging.
            episode_rewards += np.array(rewards)
            done_mask = np.array(dones, dtype=bool)
            if done_mask.any():
                for i, done in enumerate(done_mask):
                    if done:
                        env_writer.add_scalar("Env/Episode_Reward", episode_rewards[i], step_counter)
                        episode_rewards[i] = 0.0
            
            if step_counter % 1000 == 0 and agent.use_ou_noise:
                agent.reset_noise()

            if DEBUG and step_counter % LOG_FREQ == 0:
                print(f"[EnvWorker] Step {step_counter}, Avg. Step Reward: {avg_step_reward:.6f}")

            # --------- Now, update the replay buffer with the new transitions ---------
            # Only feed this transition if:
            #   - The reward vector is not all zeros, OR
            #   - It is all zeros but with 10% probability (to keep some zero-reward transitions)
            reward_array = np.array(rewards, dtype=np.float64)
            if np.all(np.abs(reward_array) < 1e-6):
                if np.random.rand() < 0.01:
                    # With x% probability, store the zero-reward transition
                    replay_process.feed({
                        'state': preprocessed_states,
                        'action': actions,
                        'reward': rewards,
                        'next_state': preprocessed_next_states,
                        'mask': done_mask
                    })
                else:
                    # Otherwise, skip feeding this transition
                    pass
            else:
                # If at least one agent got a nonzero reward, store the transition
                replay_process.feed({
                    'state': preprocessed_states,
                    'action': actions,
                    'reward': rewards,
                    'next_state': preprocessed_next_states,
                    'mask': done_mask
                })

            # Update current state.
            preprocessed_states = preprocessed_next_states

            if noise_decay_rate is not None and step_counter % noise_decay_interval == 0:
                noise_scaling_factor *= noise_decay_rate
                noise_scaling_factor = max(noise_scaling_factor, min_noise_scaling_factor)
                env_writer.add_scalar("Env/Exploration_Noise_Scaling_Factor", noise_scaling_factor, step_counter)
                
                if DEBUG and step_counter % 1000 == 0:
                    print(f"[EnvWorker] Exploration noise scaling factor decayed to: {noise_scaling_factor:.6f}")
            
            # Report steps per minute every 30 seconds.
            current_time = time.time()
            if current_time - last_report_time >= 30:
                steps_in_interval = step_counter - last_report_steps
                steps_per_second = steps_in_interval / (current_time - last_report_time)
                
                print(f"[EnvWorker] Steps per second: {steps_per_second:.2f}")
                
                last_report_time = current_time
                last_report_steps = step_counter

            # Optionally log every 100 steps: send step rate to train worker.
            if step_counter % 100 == 0:
                now = time.time()
                elapsed = now - start_time
                steps_per_second = 100 / elapsed
                
                if DEBUG:
                    print(f"[EnvWorker] Steps per second: {steps_per_second:.2f}")
                
                train_conn.send({"command": "step_rate", "steps_per_sec": steps_per_second})
                start_time = now

            if throttle_by > 0.0:
                time.sleep(throttle_by)
            
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
        #check_state_dict_consistency(new_weights["actor"], "Actor")
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
        #check_state_dict_consistency(new_weights["actor_target"], "Actor Target")
        agent.actor_target.load_state_dict(new_weights["actor_target"])
        
    if "critic" in new_weights:
        #check_state_dict_consistency(new_weights["critic"], "Critic")
        agent.critic.load_state_dict(new_weights["critic"])
        
    if "critic_target" in new_weights:
        #check_state_dict_consistency(new_weights["critic_target"], "Critic Target")
        agent.critic_target.load_state_dict(new_weights["critic_target"])
    
    if DEBUG:
        print("[EnvWorker] Successfully loaded new agent weights.")
