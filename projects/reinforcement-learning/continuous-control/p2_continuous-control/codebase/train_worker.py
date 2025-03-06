# train_worker.py
import time
import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import signal
from codebase.agent import TD3Agent  # Adjust import path to match your project

DEBUG = False

def train_worker(
        replay_process, 
        stop_flag, 
        env_conn, 
        eval_conn, 
        gamma=0.99, 
        lr_actor=5e-3,
        lr_critic=5e-3, 
        upd_w_frequency=10,  
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,     
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

    # 1) Instantiate the TD3Agent
    agent = TD3Agent(
        state_size=33, 
        action_size=4, 
        gamma=gamma, 
        lr_critic=lr_critic, 
        lr_actor=lr_actor, 
        policy_noise=policy_noise,
        noise_clip=noise_clip, 
        policy_delay=policy_delay,       
        label="TrainWorker")
    
    # Allow some time for the environment worker to start and populate the replay buffer
    print("Waiting for replay buffer to fill...")
    time.sleep(10)
    
    # Some basic loop config
    iteration = 0    
    
    # Main training loop
    while not stop_flag.is_set():
        iteration += 1
        print(f"\nTraining iteration {iteration}")
        
        # 2) Sample a batch of transitions from the replay buffer
        transitions = None
        while True:
            transitions = replay_process.sample()  # Returns a namedtuple, e.g. Transition(*arrays)
            if transitions is None:
                print("Insufficient data in sample, waiting for more data...")
                time.sleep(1)
                continue

            # Check if the batch is nonzero.
            # We check that the sum of the absolute values in key fields is nonzero.
            state_sum = transitions.state.abs().sum().item()
            action_sum = transitions.action.abs().sum().item()
            reward_sum = transitions.reward.abs().sum().item()
            next_state_sum = transitions.next_state.abs().sum().item()

            if state_sum == 0 and action_sum == 0 and reward_sum == 0 and next_state_sum == 0:
                print("Batch data is all zero, waiting for more data...")
                time.sleep(1)
            else:
                break
        
        # 3) Perform a learning step with the TD3 agent
        metrics = agent.learn(transitions)
        print(f"Completed training iteration {iteration}")
        
        # Log training metrics via SummaryWriter
        train_writer.add_scalar("Loss/Critic", metrics["critic_loss"], iteration)
        train_writer.add_scalar("Loss/Critic1", metrics["critic1_loss"], iteration)
        train_writer.add_scalar("Loss/Critic2", metrics["critic2_loss"], iteration)
        
        # Also update the eval worker on the specified evaluation frequency
        if iteration % upd_w_frequency == 0:
            # Update the environment worker with the latest policy weights            
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

        if metrics["actor_loss"] is not None:
            train_writer.add_scalar("Loss/Actor", metrics["actor_loss"], iteration)
        
        train_writer.add_scalar("Q-values/Mean_Q1", metrics["current_Q1_mean"], iteration)
        train_writer.add_scalar("Q-values/Mean_Q2", metrics["current_Q2_mean"], iteration)
        train_writer.add_scalar("Target_Q_Mean", metrics["target_Q_mean"], iteration)             

        # **Non-blocking check for "solved" message from eval_worker**
        while eval_conn.poll():  # If eval_worker sent something, read it
            msg = eval_conn.recv()
            if isinstance(msg, dict) and msg.get("command") == "solved":
                avg_reward = msg.get("avg_reward", 0)
                print(f"[TrainWorker] Eval worker reported 'solved' with avg reward={avg_reward:.2f}")
                
                # Save all weights of the TD3Agent when solved
                new_weights = extract_agent_weights(agent)
                save_path = os.path.join(train_log_dir, f"td3_agent_weights_solved_{iteration}.pth")
                torch.save(new_weights, save_path)
                
                print(f"[TrainWorker] Saved agent weights to {save_path}")

                stop_flag.set()
                time.sleep(1)
        
        # Tiny sleep to reduce tight loop
        time.sleep(1)
    
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