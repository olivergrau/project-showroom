import time
import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import signal
from codebase.td3.agent import TD3Agent

DEBUG = False
LOG_FREQ = 100  # Log every 100 iterations to TensorBoard

def train_worker(
        replay_process, 
        stop_flag, 
        env_conn, 
        eval_conn, 
        gamma=0.99, 
        lr_actor=5e-3,
        lr_critic=5e-3, 
        upd_w_frequency=10,  
        use_reward_normalization=False,
        log_dir=None):
    """
    A training process that:
      1) Instantiates a TD3Agent to learn from replay buffer samples.
      2) Periodically updates the environment worker with new policy weights.
      3) Periodically updates the evaluation worker with new policy weights and requests an evaluation.
      4) Stops if the evaluation worker signals the environment is solved, or if stop_flag is set externally.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    print("Training process started")
    
    # Create a log directory for training metrics
    if log_dir is None:
        train_log_dir = os.path.join("runs", "train")
    else:
        train_log_dir = os.path.join(log_dir, "train")
    
    train_writer = SummaryWriter(log_dir=train_log_dir)

    # 1) Instantiate the DDPGAgent
    agent = TD3Agent(
        state_size=33, 
        action_size=4, 
        gamma=gamma, 
        lr_critic=lr_critic, 
        lr_actor=lr_actor, 
        use_reward_normalization=use_reward_normalization,        
        label="TrainWorker")
    
    # Allow some time for the environment worker to start and populate the replay buffer
    if DEBUG:
        print("Waiting for replay buffer to fill...")
    
    time.sleep(0.5)
    
    iteration = 0
    start_time = time.time()    
    sample_times = []  # List to accumulate sampling times

    # Main training loop
    while not stop_flag.is_set():
        iteration += 1        
        
        # 2) Sample a batch of transitions from the replay buffer
        transitions = None
        while True:
            start_time_sampling = time.time() 
            transitions = replay_process.sample()  # Returns a namedtuple, e.g. Transition(*arrays)
            now = time.time()
            elapsed_sampling = now - start_time_sampling
            sample_times.append(elapsed_sampling)

            if iteration % LOG_FREQ == 0:
                avg_sampling_time = sum(sample_times) / len(sample_times)
                train_writer.add_scalar("Sampling/AverageTime", avg_sampling_time, iteration)
                if DEBUG:
                    print(f"[TrainWorker] Average sampling time over last {LOG_FREQ} iterations: {avg_sampling_time:.4f} seconds")
                
                sample_times = []  # Reset list

            if DEBUG:    
                print(f"[TrainWorker] Sampled batch of transitions in {elapsed_sampling:.4f} seconds")

            if transitions is None:
                if DEBUG:
                    print("Insufficient data in sample, waiting for more data...")
                time.sleep(1)
                continue

            # Check if the batch is nonzero.
            state_sum = transitions.state.abs().sum().item()
            action_sum = transitions.action.abs().sum().item()
            reward_sum = transitions.reward.abs().sum().item()
            next_state_sum = transitions.next_state.abs().sum().item()

            if state_sum == 0 and action_sum == 0 and reward_sum == 0 and next_state_sum == 0:
                if DEBUG:
                    print("Batch data is all zero, waiting for more data...")
                time.sleep(1)
            else:
                break
        
        # 3) Perform a learning step with the DDPG agent
        metrics = agent.learn(transitions)
        
        if DEBUG:
            print(f"[TrainWorker] Completed training iteration {iteration}")
        
        # Log training metrics only every LOG_FREQ iterations
        if iteration % LOG_FREQ == 0:
            train_writer.add_scalar("Loss/Critic", metrics["critic_loss"], iteration)
            train_writer.add_scalar("Loss/Critic1", metrics["critic1_loss"], iteration)
            train_writer.add_scalar("Loss/Critic2", metrics["critic2_loss"], iteration)

            if metrics["actor_loss"] is not None:
                train_writer.add_scalar("Loss/Actor", metrics["actor_loss"], iteration)
            
            train_writer.add_scalar("Q-values/Mean_Q1", metrics["current_Q1_mean"], iteration)
            train_writer.add_scalar("Q-values/Mean_Q2", metrics["current_Q2_mean"], iteration)
            train_writer.add_scalar("Target_Q_Mean", metrics["target_Q_mean"], iteration)
            
        # 4) Synchronous weight update: Broadcast weights and wait for acks
        if iteration % upd_w_frequency == 0:
            new_weights = extract_agent_weights(agent)
            
            if DEBUG:
                for name, sd in new_weights.items():
                    for key, tensor in sd.items():
                        norm = torch.norm(tensor).item()
                        print(f"[TrainWorker] {name} - {key} norm: {norm:.4f}")
                
                print(f"[TrainWorker] Broadcasting weights at iteration {iteration}")
            
            # Send update command to both workers
            eval_conn.send({"command": "update_weights", "weights": new_weights})
            env_conn.send({"command": "update_weights", "weights": new_weights})
            
            # Wait for environment worker ack with a longer timeout
            env_ack_timeout = 2.0  # seconds
            start_wait = time.time()
            
            env_ack_received = False
            while not env_ack_received and (time.time() - start_wait < env_ack_timeout):
                if env_conn.poll(0.01):
                    msg = env_conn.recv()
                    
                    if isinstance(msg, dict) and msg.get("command") == "ack_update" and msg.get("worker") == "env":
                        env_ack_received = True
                        
                        if DEBUG:
                            print("[TrainWorker] Received ack from env_worker.")
                
                time.sleep(0.005)
            
            if not env_ack_received:
                print("[TrainWorker] Warning: Env worker did not ack in time.")
            
            # Wait for evaluation worker ack, but only briefly
            eval_ack_timeout = 1.0  # seconds
            start_wait_eval = time.time()

            eval_ack_received = False
            while not eval_ack_received and (time.time() - start_wait_eval < eval_ack_timeout):
                if eval_conn.poll(0.01):
                    msg = eval_conn.recv()
                    
                    if isinstance(msg, dict) and msg.get("command") == "ack_update" and msg.get("worker") == "eval":
                        eval_ack_received = True
                        
                        if DEBUG:
                            print("[TrainWorker] Received ack from eval_worker.")
                
                time.sleep(0.005)
            
            if not eval_ack_received:
                print("[TrainWorker] Warning: Eval worker did not ack in time, proceeding without its ack.")
        
        # Non-blocking check for "solved" message from eval_worker
        while eval_conn.poll():
            msg = eval_conn.recv()
            if isinstance(msg, dict) and msg.get("command") == "solved":
                avg_reward = msg.get("avg_reward", 0)

                print(f"[TrainWorker] Eval worker reported 'solved' with avg reward={avg_reward:.2f}")
                
                new_weights = extract_agent_weights(agent)
                save_path = os.path.join(train_log_dir, f"ddpg_agent_weights_solved_{iteration}.pth")
                torch.save(new_weights, save_path)
                
                print(f"[TrainWorker] Saved agent weights to {save_path}")
                
                stop_flag.set()
                time.sleep(1)
        
        # Log update rate and env step rate every LOG_FREQ iterations
        if iteration % LOG_FREQ == 0:
            now = time.time()
            elapsed = now - start_time
            iterations_per_second = LOG_FREQ / elapsed  # Training updates per second
            
            current_env_steps_per_sec = None
            while env_conn.poll():
                msg = env_conn.recv()
                if isinstance(msg, dict) and msg.get("command") == "step_rate":
                    current_env_steps_per_sec = msg.get("steps_per_sec")
            
            if current_env_steps_per_sec is not None:
                ratio = iterations_per_second / current_env_steps_per_sec
                steps_per_update = current_env_steps_per_sec / iterations_per_second
                train_writer.add_scalar("Rates/UpdatesPerSec", iterations_per_second, iteration)
                train_writer.add_scalar("Rates/EnvStepsPerSec", current_env_steps_per_sec, iteration)
                train_writer.add_scalar("Rates/Ratio", ratio, iteration)
                
                if DEBUG:
                    print(f"[TrainWorker] Updates/sec: {iterations_per_second:.2f}, Env steps/sec: {current_env_steps_per_sec:.2f}, Ratio: {ratio:.4f} (or {steps_per_update:.2f} env steps/update)")
            else:
                if DEBUG:
                    print(f"[TrainWorker] Updates/sec: {iterations_per_second:.2f} (no env step rate received)")
            
            start_time = now

    print("[TrainWorker] Training loop ended, signaling shutdown...")

def extract_agent_weights(agent):
    return {
        "actor": {k: v.cpu() for k, v in agent.actor.state_dict().items()},
        "actor_target": {k: v.cpu() for k, v in agent.actor_target.state_dict().items()},
        "critic1": {k: v.cpu() for k, v in agent.critic1.state_dict().items()},
        "critic1_target": {k: v.cpu() for k, v in agent.critic1_target.state_dict().items()},
        "critic2": {k: v.cpu() for k, v in agent.critic2.state_dict().items()},
        "critic2_target": {k: v.cpu() for k, v in agent.critic2_target.state_dict().items()},
    }