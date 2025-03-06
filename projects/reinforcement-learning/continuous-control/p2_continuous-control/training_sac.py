# training_sac.py
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

# Our SAC worker imports
from codebase.sac.env_worker import env_worker
from codebase.sac.train_worker import train_worker
from codebase.sac.eval_worker import eval_worker  # Evaluation worker for SAC

def shutdown_properly(env_process, eval_process, train_process, replay_process, stop_flag, env_parent_conn, eval_parent_conn):
    """Gracefully shut down all processes and clean up resources."""
    print("\nShutting down environment and evaluation processes...")
    stop_flag.set()
    time.sleep(5)
    
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
        train_process.terminate()
        train_process.join()
    
    if replay_process.is_alive():
        replay_process.close()
    
    print("All processes terminated.")
    print("Training ended.")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    stop_flag = mp.Event()
    gamma = 0.99
    batch_size = 256
    lr_actor = 1e-4
    lr_critic = 1e-4
    lr_alpha = 1e-4  # Learning rate for temperature parameter
    upd_w_frequency = 2
    reward_scaling_factor = 1.0

    # Create pipes for inter-process communication
    env_parent_conn, env_child_conn = mp.Pipe()
    eval_parent_conn, eval_child_conn = mp.Pipe()

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
        replay_kwargs=replay_kwargs,
        asynchronous=True
    )
    replay_proxy = ReplayProxy(replay_process.pipe)

    # Create environment process for SAC
    env_process = mp.Process(
        target=env_worker,
        args=(
            env_child_conn, 
            replay_proxy, 
            stop_flag, 
            "Reacher_Linux/Reacher.x86_64", 
            gamma, 
            lr_actor, 
            lr_critic,
            reward_scaling_factor,
            log_dir
        )
    )

    # Create evaluation process for SAC
    eval_process = mp.Process(
        target=eval_worker,
        args=(
            eval_child_conn, 
            stop_flag, 
            "Reacher_Linux/Reacher.x86_64", 
            30.0, 
            gamma, 
            lr_actor, 
            lr_critic,
            reward_scaling_factor,
            log_dir, 
            100
        )
    )

    # Start environment and evaluation processes before training begins.
    env_process.start()
    eval_process.start()

    print("Waiting for ready signals from environment and evaluation workers...")
    ready_env = env_parent_conn.recv()
    print(f"[Main] Received ready signal from env_worker: {ready_env}")
    ready_eval = eval_parent_conn.recv()
    print(f"[Main] Received ready signal from eval_worker: {ready_eval}")

    # Create training process for SAC
    train_process = mp.Process(
        target=train_worker,
        args=(
            replay_proxy, 
            stop_flag, 
            env_parent_conn, 
            eval_parent_conn, 
            gamma, 
            lr_actor, 
            lr_critic,
            lr_alpha,
            upd_w_frequency,
            log_dir
        )
    )
    train_process.start()

    try:
        train_process.join()
    except KeyboardInterrupt:
        print("\n[Training] Ctrl+C detected! Stopping training...")
        shutdown_properly(
            env_process, eval_process, train_process, replay_process, stop_flag, env_parent_conn, eval_parent_conn)
    finally:
        shutdown_properly(
            env_process, eval_process, train_process, replay_process, stop_flag, env_parent_conn, eval_parent_conn)
