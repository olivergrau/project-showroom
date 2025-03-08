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

# ToDos:
# 1. Normalize state space before states get into the replay buffer (action space is already normalized)
# 2. Examine if Ornstein/Uhlenbeck noise is necessary for exploration
# 3. Update actor (every iteration - maybe update every x iterations?)
#     Here I should try a "slow-down" of the env worker (by collecting transitiosn for 20 steps and then feed them) 
#       to see if the agent can learn better

# 4. Synchrounous architecture


print("Current working directory:", os.getcwd())

# Create a unique log directory based on current datetime
log_dir = os.path.join("runs", "run_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
print("Logging directory:", log_dir)

# Code for replay buffer
from codebase.replay.replay_buffer import ReplayWrapper, UniformReplay
from codebase.replay.replay_proxy import ReplayProxy

# Our worker imports
from codebase.ddpg.env_worker import env_worker
from codebase.ddpg.train_worker import train_worker
from codebase.ddpg.eval_worker import eval_worker  # New evaluation worker

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
    gamma = 0.99
    batch_size = 256
    lr_actor = 1e-4
    lr_critic = 1e-4
    upd_w_frequency = 5 
    use_ou_noise = False
    exploration_start_noise = 0.2
    exploration_noise_decay = 0.9999
    reward_scaling_factor = 1.0
    use_reward_normalization = False
    throttle_env_by = 0.0  # 0.0 means no throttling (increase throttle to lower update rate)

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
        replay_kwargs=replay_kwargs, # contains batch size
        asynchronous=True
    )
    replay_proxy = ReplayProxy(replay_process.pipe)

    # Create environment process
    env_process = mp.Process(
        target=env_worker,
        args=(
            env_child_conn, 
            eval_child_conn,
            replay_proxy, 
            stop_flag, 
            "Reacher_Linux/Reacher.x86_64", 
            gamma, 
            lr_actor, 
            lr_critic,
            exploration_start_noise,            
            exploration_noise_decay,
            reward_scaling_factor,
            throttle_env_by,
            use_ou_noise,
            log_dir
        )
    )

    # Create evaluation process
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
            log_dir, 100)
    )

    # Start environment and evaluation processes (to be started before training)
    env_process.start()
    eval_process.start()

    print("Waiting for ready signals from environment and evaluation workers...")

    # Wait for both workers to signal readiness (blocking call)
    ready_env = env_parent_conn.recv()
    print(f"[Main] Received ready signal from env_worker: {ready_env}")
    
    ready_eval = eval_parent_conn.recv()
    print(f"[Main] Received ready signal from eval_worker: {ready_eval}")

    # Create training process
    train_process = mp.Process(
        target=train_worker,
        args=(
            replay_proxy, 
            stop_flag, 
            env_parent_conn, # train acts as a coordinator so it needs to communicate with env and eval
            eval_parent_conn, 
            gamma, 
            lr_actor, 
            lr_critic,
            upd_w_frequency, 
            use_reward_normalization,
            use_ou_noise,       
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
            env_process, eval_process, train_process, replay_process, stop_flag, env_parent_conn, eval_parent_conn)

    finally:
        # Ensure cleanup even if an exception occurs
        shutdown_properly(
            env_process, eval_process, train_process, replay_process, stop_flag, env_parent_conn, eval_parent_conn)
