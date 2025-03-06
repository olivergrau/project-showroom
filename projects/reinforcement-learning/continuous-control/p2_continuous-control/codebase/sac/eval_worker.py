# sac_eval_worker.py
import os
import time
import numpy as np
import torch
import signal
from unityagents import UnityEnvironment
from torch.utils.tensorboard import SummaryWriter

from codebase.sac.agent import SACAgent

DEBUG = False

def eval_worker(
    eval_conn,
    stop_flag,
    unity_exe_path="Reacher_Linux/Reacher.x86_64",
    reward_threshold=30.0,
    gamma=0.99,
    lr_actor=1e-3,
    lr_critic=1e-3,
    reward_scaling_factor=10.0,
    log_dir=None,
    window_size=100
):
    # Ignore SIGINT in this worker
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    print(f"EvalWorker: reward_threshold={reward_threshold}, gamma={gamma}, lr_actor={lr_actor}, "
          f"lr_critic={lr_critic}, reward_scaling_factor={reward_scaling_factor}, window_size={window_size}")
    
    print("[EvalWorker] Starting evaluation worker...")

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
            break
        except KeyError as e:
            print(f"[EvalWorker] Error on env.reset: {e}. Attempt {attempt+1}/{max_retries}. "
                  f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    else:
        raise RuntimeError("[EvalWorker] Failed to reset Unity environment after maximum retries.")
    
    # Signal readiness
    eval_conn.send({"command": "ready", "worker": "eval"})

    # Instantiate the SAC agent
    agent = SACAgent(
        state_size=33, action_size=4, gamma=gamma, lr_actor=lr_actor, lr_critic=lr_critic, label="EvalWorker")
    
    episode_rewards = []  # For sliding window average
    episode_count = 0

    try:
        while not stop_flag.is_set():
            # Process any pending messages (non-blocking)
            while eval_conn.poll():
                message = eval_conn.recv()
                if isinstance(message, dict):
                    if message.get("command") == "update_weights":
                        load_agent_weights(agent, message["weights"])
                        print("[EvalWorker] Updated agent weights.")
                    elif message.get("command") == "stop":
                        print("[EvalWorker] Received stop command.")
                        stop_flag.set()
                        break

            # Evaluate one episode using deterministic actions
            ep_reward = evaluate_one_episode(env, brain_name, agent, reward_scaling_factor)
            episode_count += 1
            episode_rewards.append(ep_reward)

            # Log episode reward and average reward
            eval_writer.add_scalar("Eval/Episode_Reward", ep_reward, episode_count)
            print(f"[EvalWorker] Episode {episode_count}: Reward = {ep_reward:.2f}")

            if len(episode_rewards) >= window_size:
                recent_avg = np.mean(episode_rewards[-window_size:])
                eval_writer.add_scalar("Eval/Recent_Avg_Reward", recent_avg, episode_count)
                print(f"[EvalWorker] Recent average (last {window_size} episodes): {recent_avg:.2f}")
                
                # Scale threshold to match reward scaling factor
                reward_threshold = 30.0 * reward_scaling_factor
                if recent_avg >= reward_threshold:
                    print("[EvalWorker] Environment solved! Sending stop signal to train_worker.")
                    stop_flag.set()
                    eval_conn.send({"command": "solved", "avg_reward": recent_avg})
            else:
                current_avg = np.mean(episode_rewards)
                eval_writer.add_scalar("Eval/Recent_Avg_Reward", current_avg, episode_count)
                print(f"[EvalWorker] Current average (over {episode_count} episodes): {current_avg:.2f}")

    finally:
        env.close()
        eval_writer.close()
        print("[EvalWorker] Evaluation environment closed.")

def evaluate_one_episode(env, brain_name, agent, reward_scaling_factor):
    """
    Runs one episode in the environment using the agent's current policy (deterministic action)
    and returns the average reward across all agents.
    """
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations  # shape: (num_agents, state_size)
    num_agents = len(env_info.agents)
    done = [False] * num_agents
    total_rewards = np.zeros(num_agents)
    
    while not all(done):
        # Use deterministic action for evaluation.
        actions = agent.act(states, evaluate=True)

        env_info = env.step(actions)[brain_name]
        rewards = env_info.rewards
        done = env_info.local_done
        states = env_info.vector_observations

        # Scale rewards before accumulating
        scaled_rewards = [r * reward_scaling_factor for r in rewards]
        total_rewards += np.array(scaled_rewards)
    
    return np.mean(total_rewards)

def load_agent_weights(agent, new_weights):
    """
    Helper function to load new weights into the evaluation worker's SAC agent.
    It checks for consistency and updates the following keys:
      - "actor"
      - "critic"
      - "critic_target"
      - Optionally "log_alpha"
    """
    def check_state_dict_consistency(state_dict, module_name):
        for key, param in state_dict.items():
            if torch.isnan(param).any():
                print(f"[EvalWorker] Warning: {module_name} weight '{key}' contains NaN values!")
            if torch.sum(torch.abs(param)) == 0:
                print(f"[EvalWorker] Warning: {module_name} weight '{key}' is entirely zero!")

    if "actor" in new_weights:
        check_state_dict_consistency(new_weights["actor"], "Actor")
        old_weights = agent.actor.state_dict()
        agent.actor.load_state_dict(new_weights["actor"])
        
        if DEBUG:
            for key in old_weights.keys():
                old_norm = torch.norm(old_weights[key])
                new_norm = torch.norm(new_weights["actor"][key])
                print(f"[EvalWorker] Actor '{key}': old norm = {old_norm.item():.4f}, new norm = {new_norm.item():.4f}")
                if not torch.equal(old_weights[key].cpu(), new_weights["actor"][key]):
                    print(f"[EvalWorker] Actor weight updated: {key}")
    else:
        print("[EvalWorker] No actor weights found in the provided weights dictionary.")
    
    if "critic" in new_weights:
        check_state_dict_consistency(new_weights["critic"], "Critic")
        agent.critic.load_state_dict(new_weights["critic"])
    else:
        print("[EvalWorker] No critic weights found in the provided weights dictionary.")
        
    if "critic_target" in new_weights:
        check_state_dict_consistency(new_weights["critic_target"], "Critic Target")
        agent.critic_target.load_state_dict(new_weights["critic_target"])
    else:
        print("[EvalWorker] No critic target weights found in the provided weights dictionary.")
    
    if "log_alpha" in new_weights:
        agent.log_alpha.data.copy_(new_weights["log_alpha"].data)
        print("[EvalWorker] log_alpha updated.")
    
    print("[EvalWorker] Successfully loaded new agent weights/parameters.")
