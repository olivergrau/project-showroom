# sac_train_worker.py
import time
import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import signal
from codebase.sac.agent import SACAgent  # Import the SAC agent

DEBUG = False

def train_worker(
        replay_process, 
        stop_flag, 
        env_conn, 
        eval_conn, 
        gamma=0.99, 
        lr_actor=1e-3,
        lr_critic=1e-3, 
        lr_alpha=1e-3,
        upd_w_frequency=10,  
        log_dir=None):
    """
    A training process that:
      1) Instantiates a SACAgent to learn from replay buffer samples.
      2) Periodically updates the environment and evaluation workers with new policy weights.
      3) Monitors for a "solved" signal from the evaluation worker.
      4) Stops if the evaluation worker signals that the environment is solved, or if stop_flag is set externally.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    print("Training process started")
    
    # Create a log directory for training metrics.
    if log_dir is None:
        train_log_dir = os.path.join("runs", "train")
    else:
        train_log_dir = os.path.join(log_dir, "train")
    
    train_writer = SummaryWriter(log_dir=train_log_dir)

    # Instantiate the SACAgent.
    agent = SACAgent(
        state_size=33, 
        action_size=4, 
        gamma=gamma, 
        lr_actor=lr_actor, 
        lr_critic=lr_critic, 
        lr_alpha=lr_alpha,
        label="TrainWorker")
    
    # Allow time for the environment worker to start and populate the replay buffer.
    print("Waiting for replay buffer to fill...")
    time.sleep(10)
    
    iteration = 0    
    
    while not stop_flag.is_set():
        iteration += 1
        print(f"\nTraining iteration {iteration}")
        
        # Sample a batch of transitions from the replay buffer.
        transitions = None
        while True:
            transitions = replay_process.sample()  # Returns a namedtuple (e.g., Transition(*arrays))
            if transitions is None:
                print("Insufficient data in sample, waiting for more data...")
                time.sleep(1)
                continue

            # Check if the batch is nonzero.
            state_sum = transitions.state.abs().sum().item()
            action_sum = transitions.action.abs().sum().item()
            reward_sum = transitions.reward.abs().sum().item()
            next_state_sum = transitions.next_state.abs().sum().item()

            if state_sum == 0 and action_sum == 0 and reward_sum == 0 and next_state_sum == 0:
                print("Batch data is all zero, waiting for more data...")
                time.sleep(1)
            else:
                break
        
        # Perform a learning step with the SAC agent.
        metrics = agent.learn(transitions)
        print(f"Completed training iteration {iteration}")
        
        # Log training metrics via Tensorboard.
        train_writer.add_scalar("Loss/Critic", metrics["critic_loss"], iteration)
        train_writer.add_scalar("Loss/Actor", metrics["actor_loss"], iteration)
        train_writer.add_scalar("Loss/Alpha", metrics["alpha_loss"], iteration)
        train_writer.add_scalar("Alpha", metrics["alpha"], iteration)
        train_writer.add_scalar("Q-values/Mean_Q1", metrics["current_Q1_mean"], iteration)
        train_writer.add_scalar("Q-values/Mean_Q2", metrics["current_Q2_mean"], iteration)
        train_writer.add_scalar("Target_Q_Mean", metrics["target_Q_mean"], iteration)
        train_writer.add_scalar("LogProb_mean", metrics["log_prob_mean"], iteration)

        # Periodically update the eval and env workers with the latest policy weights.
        if iteration % upd_w_frequency == 0:
            new_weights = extract_agent_weights(agent)
            if DEBUG:
                for name, sd in new_weights.items():
                    for key, tensor in sd.items():
                        norm = torch.norm(tensor).item()
                        print(f"[TrainWorker] {name} - {key} norm: {norm:.4f}")

            print(f"Sending weights to eval_worker: {iteration}")
            eval_conn.send({"command": "update_weights", "weights": new_weights})

            print(f"Sending weights to env_worker: {iteration}")
            env_conn.send({"command": "update_weights", "weights": new_weights})

        # Non-blocking check for a "solved" message from eval_worker.
        while eval_conn.poll():
            msg = eval_conn.recv()

            if isinstance(msg, dict) and msg.get("command") == "solved":
                avg_reward = msg.get("avg_reward", 0)
                print(f"[TrainWorker] Eval worker reported 'solved' with avg reward={avg_reward:.2f}")
                new_weights = extract_agent_weights(agent)
                save_path = os.path.join(train_log_dir, f"sac_agent_weights_solved_{iteration}.pth")
                torch.save(new_weights, save_path)
                print(f"[TrainWorker] Saved agent weights to {save_path}")
                stop_flag.set()
                time.sleep(1)

        # Tiny sleep to reduce tight loop.
        time.sleep(1)
    
    print("[TrainWorker] Training loop ended, signaling shutdown...")

def extract_agent_weights(agent):
    return {
        "actor": {k: v.cpu().detach() for k, v in agent.actor.state_dict().items()},
        "critic": {k: v.cpu().detach() for k, v in agent.critic.state_dict().items()},
        "critic_target": {k: v.cpu().detach() for k, v in agent.critic_target.state_dict().items()},
        "log_alpha": agent.log_alpha.cpu().detach()
    }
