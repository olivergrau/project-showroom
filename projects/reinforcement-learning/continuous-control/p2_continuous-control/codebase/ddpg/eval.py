import os
import time
import numpy as np
import torch
import signal
import threading
from unityagents import UnityEnvironment
from torch.utils.tensorboard import SummaryWriter

from codebase.ddpg.agent import DDPGAgent
from codebase.utils.normalizer import RunningNormalizer

DEBUG = False
LOG_FREQ = 10  # Log TensorBoard metrics every LOG_FREQ episodes

def message_listener(train_conn, state_normalizer, agent, stop_flag):
    """
    Unified listener thread for the eval worker that continuously checks for messages 
    from the train worker. It handles both normalizer updates and weight updates.
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
                            print("[EvalWorker] Updated normalizer parameters.")
                    
                    elif command == "update_weights":
                        new_weights = message.get("weights")
                        load_agent_weights(agent, new_weights)
                        
                        # Send acknowledgment back to the train worker
                        train_conn.send({"command": "ack_update", "worker": "eval"})
                        
                        if DEBUG:
                            print("[EvalWorker] Updated agent weights and sent ack.")
                    
                    elif command == "stop":
                        stop_flag.set()
                        break
        except EOFError:
            print("[EvalWorker] Message listener: EOFError occurred. Exiting.")
            break
        
        time.sleep(0.001)

def eval_worker(
    train_conn,
    env_conn,
    main_conn,
    stop_flag,
    unity_exe_path="Reacher_Linux/Reacher.x86_64",
    reward_threshold=30.0,
    use_state_norm=False,
    actor_input_size = 256,
    actor_hidden_size = 256,
    critic_input_size = 256,
    critic_hidden_size = 256,
    log_dir=None,
    window_size=100
):
    # Ignore SIGINT in this worker
    signal.signal(signal.SIGINT, signal.SIG_IGN)
        
    print(f"EvalWorker: use_state_norm={use_state_norm}, reward_threshold={reward_threshold}, window_size={window_size}")
    print("[EvalWorker] Starting evaluation worker...")
    print()
    
    # Create a separate log directory for evaluation metrics
    if log_dir is None:
        eval_log_dir = os.path.join("runs", "eval")
    else:
        eval_log_dir = os.path.join(log_dir, "eval")
    
    eval_writer = SummaryWriter(log_dir=eval_log_dir)

    env = UnityEnvironment(file_name=unity_exe_path, no_graphics=True, worker_id=2)
    brain_name = env.brain_names[0]

    # Reinitialize the environment if reset fails
    max_retries = 5
    retry_delay = 2  # seconds
    for attempt in range(max_retries):
        try:
            _ = env.reset(train_mode=True)[brain_name]
            print(f"[EvalWorker] Successfully reset Unity environment on attempt {attempt + 1}")
            print()
            break
        except KeyError as e:            
            print(f"[EvalWorker] Error on env.reset: {e}. Attempt {attempt+1}/{max_retries}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    else:
        raise RuntimeError("[EvalWorker] Failed to reset Unity environment after maximum retries.")
    
    # Signal readiness
    main_conn.send({"command": "ready", "worker": "eval"})

    # Instantiate the agent
    agent = DDPGAgent(
        state_size=33, action_size=4, label="EvalWorker", 
        actor_input_size=actor_input_size, actor_hidden_size=actor_hidden_size,
        critic_input_size=critic_input_size, critic_hidden_size=critic_hidden_size
    )
    
    # Create a local RunningNormalizer instance (assume state dimension 33)
    if use_state_norm:
        state_normalizer = RunningNormalizer(shape=(33,), momentum=0.001)
    else:
        state_normalizer = None

    episode_rewards = []  # For sliding window average
    episode_count = 0

    # Start the unified listener thread for messages from train worker.
    listener_thread = threading.Thread(target=message_listener, args=(train_conn, state_normalizer, agent, stop_flag))
    listener_thread.daemon = True
    listener_thread.start()

    try:
        while not stop_flag.is_set():
            if main_conn.poll():
                message = main_conn.recv()
                if isinstance(message, dict) and message.get("command") == "stop":
                    print("[EvalWorker] Received stop command from main process.")
                    stop_flag.set()
                    break        

            # Evaluate one episode
            ep_reward = evaluate_one_episode(env, brain_name, agent, state_normalizer)
            episode_count += 1
            episode_rewards.append(ep_reward)

            # Log individual episode reward to TensorBoard (log every episode)
            eval_writer.add_scalar("Eval/Episode_Reward", ep_reward, episode_count)
            
            if DEBUG:
                print(f"[EvalWorker] Episode {episode_count}: Reward = {ep_reward:.2f}")

            # Compute and log moving average reward every LOG_FREQ episodes
            if episode_count % LOG_FREQ == 0:
                if len(episode_rewards) >= window_size:
                    recent_avg = np.mean(episode_rewards[-window_size:])
                else:
                    recent_avg = np.mean(episode_rewards)
                
                eval_writer.add_scalar("Eval/Recent_Avg_Reward", recent_avg, episode_count)
                                
                if len(episode_rewards) >= window_size:
                    print(f"[EvalWorker] Recent average (last {window_size} episodes): {recent_avg:.2f}")
                else:
                    print(f"[EvalWorker] Current average (over {episode_count} episodes): {recent_avg:.2f}")
            
                # Log state_normalizer statistics
                if use_state_norm:
                    eval_writer.add_scalar("Normalizer/Mean_Avg", np.mean(state_normalizer.mean), episode_count)
                    eval_writer.add_scalar("Normalizer/Var_Avg", np.mean(state_normalizer.var), episode_count)
                    eval_writer.add_scalar("Normalizer/Count", state_normalizer.count, episode_count)
                    
                # Check if solved and send stop signal if needed
                reward_threshold_scaled = 30.0
                if recent_avg >= reward_threshold_scaled:                    
                    print("[EvalWorker] Environment solved! Sending stop signal to train_worker.")
                    stop_flag.set()
                    train_conn.send({"command": "solved", "avg_reward": recent_avg})
    finally:
        env.close()
        eval_writer.close()        
        print("[EvalWorker] Evaluation environment closed.")

def evaluate_one_episode(env, brain_name, agent, state_normalizer):
    """
    Runs one episode in the environment using the agent's current policy (without exploration noise).
    Applies normalization to the raw states using the provided RunningNormalizer.
    """
    env_info = env.reset(train_mode=True)[brain_name]
    raw_states = env_info.vector_observations  # raw state from environment
    
    # Normalize the initial state using the current normalizer    
    states = state_normalizer.normalize(raw_states) if state_normalizer is not None else raw_states
    
    num_agents = len(env_info.agents)
    done = [False] * num_agents
    total_rewards = np.zeros(num_agents)
    
    while not all(done):
        actions = agent.act(states, noise_scaling_factor=0.0)  # No exploration noise during evaluation
        env_info = env.step(actions)[brain_name]
        rewards = env_info.rewards
        done = env_info.local_done
        next_states = env_info.vector_observations
        
        # Normalize the new state
        states = state_normalizer.normalize(next_states) if state_normalizer is not None else next_states

        rewards = np.asarray(rewards)
        total_rewards += np.array(rewards)
    
    return np.mean(total_rewards)

def load_agent_weights(agent, new_weights):
    """
    Helper function to load new weights into the evaluation worker's agent.
    Checks each network's state dict for consistency before loading the weights.
    """
    def check_state_dict_consistency(state_dict, module_name):
        for key, param in state_dict.items():
            if torch.isnan(param).any():
                if DEBUG:
                    print(f"[EvalWorker] Warning: {module_name} weight '{key}' contains NaN values!")
            if torch.sum(torch.abs(param)) == 0:
                if DEBUG:
                    print(f"[EvalWorker] Warning: {module_name} weight '{key}' is entirely zero!")

    if "actor" in new_weights:
        if DEBUG:
            check_state_dict_consistency(new_weights["actor"], "Actor")
        agent.actor.load_state_dict(new_weights["actor"])        
    else:
        print("[EvalWorker] No actor weights found in the provided weights dictionary.")
        
    if "actor_target" in new_weights:
        if DEBUG:
            check_state_dict_consistency(new_weights["actor_target"], "Actor Target")
        agent.actor_target.load_state_dict(new_weights["actor_target"])
        
    if "critic" in new_weights:
        if DEBUG:
            check_state_dict_consistency(new_weights["critic"], "Critic")
        agent.critic.load_state_dict(new_weights["critic"])
        
    if "critic_target" in new_weights:
        if DEBUG:
            check_state_dict_consistency(new_weights["critic_target"], "Critic Target")
        agent.critic_target.load_state_dict(new_weights["critic_target"])

    if DEBUG:
        print("[EvalWorker] Successfully loaded new agent weights.")
