# training.py
from datetime import datetime
import os
import time
import multiprocessing as mp
import numpy as np

# For older numpy versions
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64

print("Current working directory:", os.getcwd())

# Create a unique log directory based on current datetime
log_dir = os.path.join("runs", "run_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
print("Logging directory:", log_dir)

# Code for replay buffer
from codebase.replay.replay_buffer import ReplayWrapper, UniformReplay
from codebase.replay.replay_proxy import ReplayProxy

# Our worker imports
from codebase.ddpg.env import env_worker
from codebase.ddpg.train import train_worker
from codebase.ddpg.eval import eval_worker

def shutdown_properly(env_process, eval_process, train_process, replay_process, stop_flag, env_parent_conn, eval_parent_conn):
    """Gracefully shut down all processes and clean up resources."""
    print("\nShutting down environment and evaluation processes...")
    stop_flag.set()

    # Give processes some time to detect the stop flag
    time.sleep(5)

    # Explicitly stop each process if it's still alive
    if env_process.is_alive():
        print("Stopping environment worker...")
        env_parent_conn.send("stop")
        env_process.join()

    if eval_process.is_alive():
        print("Stopping evaluation worker...")
        eval_parent_conn.send({"command": "stop"})
        eval_process.join()

    if train_process.is_alive():
        print("Stopping training worker...")
        train_process.terminate()  # Training might not have a stop signal, so we force terminate
        train_process.join()

    # Close the replay buffer process
    if replay_process.is_alive():
        replay_process.close()

    print("All processes terminated.")
    print("Training ended.")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    stop_flag = mp.Event()
    
    # Hyperparameters
    gamma = 0.99
    batch_size = 256
    lr_actor = 1e-4
    lr_critic = 1e-4
    upd_w_frequency = 2 # number of iterations / batches before updating the weights for the workers
    use_ou_noise = True
    exploration_start_noise = 0.2
    exploration_noise_decay = 0.9999
    reward_scaling_factor = 1.0
    use_reward_normalization = False
    use_state_norm = False

    # Throttling: Crucial for training stability (especially in multi-worker setups)
    throttle_steps_by = 0.03  # 0.0 means no throttling (increase throttle to lower steps per second)
    throttle_trainings_by = 0.08  # 0.0 means no throttling (increase throttle to lower training iterations per second)

    # Communication between env_worker and train_worker
    env_train_parent_conn, env_train_child_conn = mp.Pipe()

    # Communication between eval_worker and train_worker
    eval_train_parent_conn, eval_train_child_conn = mp.Pipe()

    # Communication between main process and env_worker (for stop signal)
    env_main_parent_conn, env_main_child_conn = mp.Pipe()

    # Communication between main process and eval_worker (for stop signal)
    eval_main_parent_conn, eval_main_child_conn = mp.Pipe()

    # Communication between env_worker and eval_worker (direct)
    env_eval_parent_conn, env_eval_child_conn = mp.Pipe()

    # Start the ReplayWrapper in a separate process
    replay_kwargs = {
        'memory_size': 100000,
        'batch_size': batch_size,
        'discount': gamma,
        'n_step': 1,
        'history_length': 1
    }

    replay_process = ReplayWrapper(
        replay_cls=UniformReplay,
        replay_kwargs=replay_kwargs, # contains batch size
        asynchronous=True
    )
    replay_proxy = ReplayProxy(replay_process.pipe)

    # Create environment process
    env_process = mp.Process(
        target=env_worker,
        args=(
            env_train_child_conn, # communication with train_worker (receives new weights)
            env_eval_child_conn,  # communication with eval_worker
            env_main_child_conn,  # NEW: Stop signal from main
            replay_proxy, 
            stop_flag, 
            "Reacher_Linux/Reacher.x86_64", 
            gamma, 
            lr_actor, 
            lr_critic,
            exploration_start_noise,            
            exploration_noise_decay,
            reward_scaling_factor,
            throttle_steps_by,
            use_ou_noise,
            use_state_norm,
            log_dir
        )
    )

    # Create evaluation process
    eval_process = mp.Process(
        target=eval_worker,
        args=(
            eval_train_child_conn, # communication with train_worker
            env_eval_parent_conn, # communication with env_worker
            eval_main_child_conn, # NEW: Stop signal from main
            stop_flag, 
            "Reacher_Linux/Reacher.x86_64", 
            30.0, 
            gamma, 
            lr_actor, 
            lr_critic,
            reward_scaling_factor,
            use_state_norm,
            log_dir, 100)
    )

    # Start environment and evaluation processes (to be started before training)
    env_process.start()
    eval_process.start()

    print("Waiting for ready signals from environment and evaluation workers...")

    # Wait for both workers to signal readiness (blocking call)
    ready_env = env_main_parent_conn.recv()
    print(f"[Main] Received ready signal from env_worker: {ready_env}")
    
    ready_eval = eval_main_parent_conn.recv()
    print(f"[Main] Received ready signal from eval_worker: {ready_eval}")

    # Create training process
    train_process = mp.Process(
        target=train_worker,
        args=(
            replay_proxy, 
            stop_flag, 
            env_train_parent_conn, # train acts as a coordinator so it needs to communicate with env and eval
            eval_train_parent_conn, 
            gamma, 
            lr_actor, 
            lr_critic,
            upd_w_frequency, 
            use_reward_normalization,
            use_ou_noise,  
            use_state_norm,   
            throttle_trainings_by,  
            log_dir
        )  # Pass the parent conns so training can send messages
    )

    # Start processes
    train_process.start()

    try:
        # Wait until the training process finishes
        train_process.join()

    except KeyboardInterrupt:
        # If Ctrl+C is detected, gracefully shut down
        print("\n[Training] Ctrl+C detected! Stopping training...")
        shutdown_properly(
            env_process, eval_process, train_process, replay_process, stop_flag, env_train_parent_conn, eval_train_parent_conn)

    finally:
        # Ensure cleanup even if an exception occurs
        shutdown_properly(
            env_process, eval_process, train_process, replay_process, stop_flag, env_train_parent_conn, eval_train_parent_conn)