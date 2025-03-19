import time
import torch
import numpy as np
import os
import signal
import threading
import queue  # Use standard library queue
from torch.utils.tensorboard import SummaryWriter
from codebase.ddpg.agent import DDPGAgent  # Updated import for DDPG
from codebase.utils.normalizer import RunningNormalizer  # your RunningNormalizer class

DEBUG = False
LOG_FREQ = 100  # Log every 100 iterations to TensorBoard

def env_message_dispatcher(env_conn, eval_conn, ack_queue, step_rate_queue, solved_queue, stop_flag):
    """
    Continuously reads from env_conn and dispatches messages into separate queues:
      - ack_queue for ack messages,
      - step_rate_queue for step rate messages,
      - solved_queue for solved messages.
    """
    while not stop_flag.is_set():
        try:
            if env_conn.poll(0.001):
                msg = env_conn.recv()
                if isinstance(msg, dict):
                    command = msg.get("command")
                    
                    if command == "ack_update":
                        ack_queue.put(msg)
                    elif command == "step_rate":
                        step_rate_queue.put(msg)
                    else:
                        # Handle unknown messages
                        print(f"[TrainWorker: EnvMessageDispatcher] Unknown message received: {msg}")
            
            if eval_conn.poll(0.001):
                msg = eval_conn.recv()
                if isinstance(msg, dict):
                    command = msg.get("command")
                    
                    if command == "ack_update":
                        ack_queue.put(msg)                    
                    elif command == "solved":
                        solved_queue.put(msg)
                    else:
                        # Handle unknown messages
                        print(f"[TrainWorker: EnvMessageDispatcher] Unknown message received: {msg}")
        except EOFError:
            break

        time.sleep(0.001)

def train_worker(
        replay_process, 
        stop_flag, 
        env_conn, 
        eval_conn, 
        gamma=0.99, 
        lr_actor=5e-3,
        actor_input_size=400,      # Actor input layer size
        actor_hidden_size=300,     # Actor hidden layer size 
        lr_critic=5e-3, 
        critic_input_size=400,      # Actor input layer size
        critic_hidden_size=300,     # Actor hidden layer size
        tau=1e-3,
        critic_clip=1.0,
        critic_weight_decay=0.0, 
        upd_w_frequency=10,  
        use_reward_norm=False,
        use_state_norm=False,
        use_reward_scaling=False,
        reward_scaling_factor=1.0,
        # Noise
        use_ou_noise=False,
        ou_noise_theta=0.15,       # Noise theta parameter
        ou_noise_sigma=0.2,        # Noise sigma parameter                  
        throttle_by=0.0,
        log_dir=None):
    """
    A training process that:
      - Instantiates a DDPGAgent to learn from replay buffer samples.
      - Periodically updates the environment and evaluation workers with new policy weights synchronously.
      - Optionally updates a state normalizer from batches and sends these updates.
      - Uses separate queues for handling different message types from the env_worker.
      - Stops if the eval worker signals that the environment is solved, or if stop_flag is set externally.
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
    agent = DDPGAgent(
        state_size=33, 
        action_size=4,         
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        tau=tau,
        critic_clip=critic_clip,
        critic_weight_decay=critic_weight_decay,
        use_ou_noise=use_ou_noise,
        ou_noise_theta=ou_noise_theta,
        ou_noise_sigma=ou_noise_sigma,
        actor_input_size=actor_input_size,
        actor_hidden_size=actor_hidden_size,
        critic_input_size=critic_input_size,
        critic_hidden_size=critic_hidden_size,
        label="TrainWorker")
    
    # Allow some time for the environment worker to start and populate the replay buffer
    if DEBUG:
        print("Waiting for replay buffer to fill...")
    
    time.sleep(5)
    
    iteration = 0
    start_time = time.time()    
    sample_times = []  # List to accumulate sampling times

    # Instantiate RunningNormalizer if using state normalization.
    if use_state_norm:
        state_normalizer = RunningNormalizer(shape=(33,), momentum=0.001)

    # Create separate queues for messages coming from env_worker
    ack_queue = queue.Queue()
    step_rate_queue = queue.Queue()
    solved_queue = queue.Queue()

    # Start dispatcher thread for env_conn messages.
    dispatcher_thread = threading.Thread(target=env_message_dispatcher,
                                         args=(env_conn, eval_conn, ack_queue, step_rate_queue, solved_queue, stop_flag))
    dispatcher_thread.daemon = True
    dispatcher_thread.start()

    # Variables for reporting iterations per second every 30 seconds.
    last_report_time = time.time()
    last_report_iterations = 0
    
    # Variables for reward normalization.
    reward_stats = {"mean": 0.0, "var": 1.0, "count": 0}

    # Main training loop.
    while not stop_flag.is_set():
        iteration += 1        
        
        # 2) Sample a batch of transitions from the replay buffer.
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
        
        # Update the state normalizer with the new batch of states if used.
        if use_state_norm:
            state_normalizer.update(transitions.state)
            transitions = transitions._replace(
                state=state_normalizer.normalize(transitions.state),
                next_state=state_normalizer.normalize(transitions.next_state)
            )

        reward = transitions.reward

        # 3) Reward scaling using PyTorch operations
        if use_reward_scaling:
            # Multiply rewards by the scaling factor and clamp between -1 and 1
            reward = torch.clamp(reward * reward_scaling_factor, -1.0, 1.0)
            transitions = transitions._replace(reward=reward)
        
        # --- Reward Normalization (Welford's algorithm style) using PyTorch ---
        if use_reward_norm:
            # Assume reward is a 1D tensor of shape [batch_size]
            n = reward.size(0)  # number of reward elements
            old_count = reward_stats["count"]
            new_count = old_count + n
        
            # Compute the sum and squared sum of the rewards using torch operations.
            reward_sum = torch.sum(reward)
            reward_sq_sum = torch.sum(reward ** 2)
        
            # Update the running mean and variance (stored as Python floats)
            new_mean = (old_count * reward_stats["mean"] + reward_sum.item()) / new_count
            new_var = ((old_count * (reward_stats["var"] + reward_stats["mean"]**2) + reward_sq_sum.item()) / new_count) - new_mean**2
        
            reward_stats["mean"] = new_mean
            reward_stats["var"] = new_var if new_var > 0 else 1.0
            reward_stats["count"] = new_count
        
            # Normalize the reward tensor
            std = torch.sqrt(torch.tensor(reward_stats["var"], dtype=reward.dtype, device=reward.device))
            normalized_reward = (reward - new_mean) / (std + 1e-8)
        
            transitions = transitions._replace(reward=normalized_reward)

        # 4) Perform a learning step with the DDPG agent.
        metrics = agent.learn(transitions)
        
        if DEBUG:
            print(f"[TrainWorker] Completed training iteration {iteration}")
        
        # Periodically send updated normalizer parameters to other workers.
        if use_state_norm and iteration % 50 == 0:
            norm_params = {
                "mean": state_normalizer.mean.tolist(),
                "var": state_normalizer.var.tolist(),
                "count": state_normalizer.count
            }

            env_conn.send({"command": "update_normalizer", "normalizer": norm_params})
            eval_conn.send({"command": "update_normalizer", "normalizer": norm_params})
            
            if DEBUG:
                print("[TrainWorker] Sent updated normalizer parameters to eval_worker.")
                
        # Log training metrics every LOG_FREQ iterations.
        if iteration % LOG_FREQ == 0:
            train_writer.add_scalar("Loss/Critic", metrics["critic_loss"], iteration)
            
            if metrics["actor_loss"] is not None:
                train_writer.add_scalar("Loss/Actor", metrics["actor_loss"], iteration)
            
            train_writer.add_scalar("Q-values/Mean_Q", metrics["current_Q_mean"], iteration)
            train_writer.add_scalar("Target_Q_Mean", metrics["target_Q_mean"], iteration)
        
        # 5) Synchronous weight update: Broadcast weights and wait for acks.
        if iteration % upd_w_frequency == 0:
            new_weights = extract_agent_weights(agent)
            
            if DEBUG:
                for name, sd in new_weights.items():
                    for key, tensor in sd.items():                                    
                        # Only compute norm for floating-point tensors
                        if torch.is_floating_point(tensor):
                            norm = torch.norm(tensor).item()
                            print(f"[TrainWorker] {name} - {key} norm: {norm:.4f}")
                        else:
                            print(f"[TrainWorker] Skipping {name} - {key} (dtype: {tensor.dtype})")
            
                print(f"[TrainWorker] Broadcasting weights at iteration {iteration}")
            
            # Send weight update command to both workers.
            eval_conn.send({"command": "update_weights", "weights": new_weights})
            env_conn.send({"command": "update_weights", "weights": new_weights})
            
            # Wait for an ack from the env_worker.
            env_ack_received = False
            try:
                ack_msg = ack_queue.get(timeout=1.0)
                if ack_msg.get("worker") == "env":
                    env_ack_received = True
                    if DEBUG:
                        print("[TrainWorker] Received ack from env_worker.")
            except queue.Empty:
                print("[TrainWorker] Warning: Env worker did not ack in time.")
            
            # Wait for an ack from the eval_worker.
            eval_ack_received = False
            try:
                ack_msg = ack_queue.get(timeout=1.0)
                if ack_msg.get("worker") == "eval":
                    eval_ack_received = True
                    if DEBUG:
                        print("[TrainWorker] Received ack from eval_worker.")
            except queue.Empty:
                print("[TrainWorker] Warning: Eval worker did not ack in time, proceeding without its ack.")
        
        # Check for "solved" message from eval_worker (from the solved_queue).
        try:
            solved_msg = solved_queue.get_nowait()
            if isinstance(solved_msg, dict) and solved_msg.get("command") == "solved":
                avg_reward = solved_msg.get("avg_reward", 0)
                print(f"[TrainWorker] Eval worker reported 'solved' with avg reward={avg_reward:.2f}")
                
                new_weights = extract_agent_weights(agent)
                save_path = os.path.join(train_log_dir, f"ddpg_agent_weights_solved_{iteration}.pth")
                torch.save(new_weights, save_path)
                
                print(f"[TrainWorker] Saved agent weights to {save_path}")
                stop_flag.set()
                time.sleep(1)
        except queue.Empty:
            pass

        # Log update rate and env step rate every n iterations.
        if iteration % LOG_FREQ == 0:
            now = time.time()
            elapsed = now - start_time
            iterations_per_second = LOG_FREQ / elapsed  # Training updates per second
            
            current_env_steps_per_sec = None
            
            # Drain all step_rate messages from the step_rate_queue; use the latest if available.
            while not step_rate_queue.empty():
                msg = step_rate_queue.get()
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
        
        # Report iterations per second every 30 seconds.
        current_time = time.time()
        if current_time - last_report_time >= 30:
            iterations_in_interval = iteration - last_report_iterations
            iterations_per_sec = iterations_in_interval / (current_time - last_report_time)
            
            print(f"[TrainWorker] Iterations per second: {iterations_per_sec:.2f}")
            
            last_report_time = current_time
            last_report_iterations = iteration
        
        if throttle_by > 0:
            time.sleep(throttle_by)

    print("[TrainWorker] Training loop ended, signaling shutdown...")

def extract_agent_weights(agent):
    return {
        "actor": {k: v.cpu() for k, v in agent.actor.state_dict().items()},
        "actor_target": {k: v.cpu() for k, v in agent.actor_target.state_dict().items()},
        "critic": {k: v.cpu() for k, v in agent.critic.state_dict().items()},
        "critic_target": {k: v.cpu() for k, v in agent.critic_target.state_dict().items()},
    }

