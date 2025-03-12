"""
training.py

A complete synchronous training loop for DDPG on the Unity Reacher environment.
This script ties together the environment wrapper, DDPGAgent, replay buffer, and
a running state normalizer into one main loop. It includes robust error handling to
ensure the Unity environment is properly shut down if any error occurs.
"""

import os
import time
import numpy as np
import torch
import traceback
from collections import namedtuple

from codebase.ddpg.agent import DDPGAgent
from codebase.utils.normalizer import RunningNormalizer
from codebase.ddpg.env import EnvWrapper
from codebase.ddpg.eval import evaluate
from codebase.replay.replay_buffer import UniformReplay
from codebase.utils.early_stopping import EarlyStopping

from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

# For older numpy versions
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64


def convert_batch_to_tensor(batch, device):
    """
    Converts each field in the sampled batch from numpy arrays to torch tensors.
    
    Args:
        batch: A Transition namedtuple containing numpy arrays.
        device: The torch device to move the tensors to.
    
    Returns:
        A Transition namedtuple where each field is a torch tensor.
    """
    # Use torch.as_tensor to avoid copying when possible.
    state = torch.as_tensor(batch.state, dtype=torch.float32, device=device)
    action = torch.as_tensor(batch.action, dtype=torch.float32, device=device)
    reward = torch.as_tensor(batch.reward, dtype=torch.float32, device=device)
    next_state = torch.as_tensor(batch.next_state, dtype=torch.float32, device=device)
    mask = torch.as_tensor(batch.mask, dtype=torch.float32, device=device)
    
    # Return a new Transition with all fields as tensors.
    Transition = type(batch)
    
    return Transition(state, action, reward, next_state, mask)

def train(
    state_size=33,
    action_size=4,
    episodes=1000,             # Total training episodes
    max_steps=1000,            # Maximum steps per episode
    batch_size=256,            # Batch size for learning
    gamma=0.99,                # Discount factor
    lr_actor=2e-4,             # Actor learning rate
    lr_critic=2e-4,            # Critic learning rate
    critic_clip=None,          # Gradient clipping for critic
    critic_weight_decay=1e-4,     # L2 weight decay for critic
    tau = 0.005,                 # Soft update parameter for target networks
    exploration_noise=0.1,     # Initial exploration noise factor
    replay_capacity=100000,    # Replay buffer capacity
    eval_frequency=10,         # Evaluate every 10 episodes
    eval_episodes=5,           # Run 5 episodes per evaluation
    eval_threshold=30.0,       # Target average reward to consider environment solved
    unity_worker_id=1
):
    LOG_FREQ = 100  # Log metrics every LOG_FREQ steps

    # ToDo: Reward normalization, Reward scaling

    # print out all hyperparameters in a nice way
    print(f"Training with hyperparameters:")
    print(f"  Episodes: {episodes}")
    print(f"  Max Steps: {max_steps}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Gamma: {gamma}")
    print(f"  Actor LR: {lr_actor}")
    print(f"  Critic LR: {lr_critic}")
    print(f"  Critic Clip: {critic_clip}")
    print(f"  Critic Weight Decay: {critic_weight_decay}")
    print(f"  Tau: {tau}")
    print(f"  Exploration Noise: {exploration_noise}")
    print(f"  Replay Capacity: {replay_capacity}")
    print(f"  Evaluation Frequency: {eval_frequency}")
    print(f"  Evaluation Episodes: {eval_episodes}")
    print(f"  Evaluation Threshold: {eval_threshold}")
    
    evaluation_rewards_window = 0.0  # Sliding window for evaluation rewards

    # Early stopping parameters: if no improvement for 'patience' evaluations, stop.
    early_stopping = EarlyStopping(patience=3, min_delta=0.4, verbose=True)

    # TensorBoard logging directory
    log_dir = os.path.join("runs", "train_ddpg", time.strftime("%Y-%m-%d_%H-%M-%S"))
    writer = SummaryWriter(log_dir=log_dir)
    
    # -------------------------------
    # Initialize modules
    # -------------------------------
    exe_path = "Reacher_Linux/Reacher.x86_64"
    
    # Use the EnvWrapper as a context manager so that it will be closed on exit.
    try:
        with EnvWrapper(exe_path, worker_id=unity_worker_id, use_graphics=False) as env:
            agent = DDPGAgent(
                state_size=state_size,
                action_size=action_size,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                tau=tau,
                critic_clip=critic_clip,
                critic_weight_decay=critic_weight_decay,
                use_ou_noise=True  # Enable OU noise for exploration.
            )

            replay_kwargs = {
                'memory_size': replay_capacity,
                'batch_size': batch_size,
                'discount': gamma,
                'n_step': 1,
                'history_length': 1
            }
            buffer = UniformReplay(**replay_kwargs)
            normalizer = RunningNormalizer(shape=(state_size,), momentum=0.001)
            
            total_steps = 0
            train_iter = 0  # Counter for learning iterations
            
            avg_reward_return = 0.0

            # Initialize actor loss tracking variables.
            actor_loss_window = []       # List to store recent actor losses.
            actor_loss_window_size = 10  # Number of recent updates to average.
            actor_loss_counter = 0       # Counter for consecutive evaluations with insufficient actor loss.
            actor_loss_patience = 5      # Number of evaluations to tolerate before stopping.
            min_actor_threshold = -0.2   # The actor loss should be below this (more negative) for learning to be considered sufficient.

            for episode in range(1, episodes + 1):
                try:
                    # Reset the environment at the start of each episode.
                    # state will have shape (num_agents, state_size)
                    state = env.reset(train_mode=True)
                except Exception as e:
                    print(f"[Training] Failed to reset environment on episode {episode}: {e}")
                    raise

                episode_reward = 0.0

                for step in range(max_steps):
                    total_steps += 1
                    try:
                        # Normalize state batch (applied to all agents)
                        norm_state = normalizer.normalize(state) if normalizer is not None else state
                        
                        # Agent selects actions for all agents (expects batch input)
                        action = agent.act(norm_state, noise=exploration_noise)
                        next_state, reward, done_flags = env.step(action)
                    except Exception as e:
                        print(f"[Training] Error during step {step} in episode {episode}: {e}")
                        raise

                    # Update normalizer with the current states (all agents)
                    if normalizer is not None:
                        normalizer.update(state)

                    # Accumulate reward as the average over agents.
                    episode_reward += np.mean(reward)

                    # Create a mask: 0 if done, 1 if not done – for each agent.
                    mask = [0 if d else 1 for d in done_flags]
                    
                    # Feed transitions for all agents in one dictionary.
                    buffer.feed({
                        'state': state,           # shape: (num_agents, state_size)
                        'action': action,         # shape: (num_agents, action_size)
                        'reward': reward,         # list or array with length=num_agents
                        'next_state': next_state, # shape: (num_agents, state_size)
                        'mask': mask              # list of 0/1 values per agent
                    })

                    # Update state for the next step.
                    state = next_state

                    # Check if all agents are done.
                    if all(done_flags):
                        break

                    # Perform learning if enough samples are available.
                    if buffer.size() >= batch_size:
                        batch = buffer.sample()
                        batch = convert_batch_to_tensor(batch, device=agent.device)
                        metrics = agent.learn(batch)
                        train_iter += 1
                        
                        # Log training metrics to TensorBoard.
                        if step % LOG_FREQ == 0:
                            writer.add_scalar("Loss/Critic", metrics["critic_loss"], train_iter)
                            writer.add_scalar("Loss/Actor", metrics["actor_loss"], train_iter)
                            writer.add_scalar("Q-values/Current", metrics["current_Q_mean"], train_iter)
                            writer.add_scalar("Q-values/Target", metrics["target_Q_mean"], train_iter)
                        
                        # Update actor loss sliding window.
                        actor_loss_window.append(metrics["actor_loss"])
                        if len(actor_loss_window) > actor_loss_window_size:
                            actor_loss_window = actor_loss_window[-actor_loss_window_size:]

                # Log episode reward (average cumulative reward per agent)
                writer.add_scalar("Episode/Reward", episode_reward, episode)
                print(f"Episode {episode:4d} | Average Reward: {episode_reward:7.2f} | Total Steps: {total_steps}")

                # Periodically evaluate the agent.
                if episode % eval_frequency == 0:
                    try:
                        avg_reward, solved = evaluate(agent, env, normalizer=normalizer, episodes=eval_episodes, threshold=eval_threshold)
                        writer.add_scalar("Eval/AverageReward", avg_reward, episode)
                        
                        print(f"Evaluation at training episode {episode}: Sliding window avg reward = {avg_reward:.2f}")
                        
                        # Compute average actor loss from the sliding window.
                        if len(actor_loss_window) > 0:
                            avg_actor_loss = np.mean(actor_loss_window)
                        else:
                            avg_actor_loss = float('inf')
                        
                        print(f"Average actor loss over last {actor_loss_window_size} updates: {avg_actor_loss:.4f}")

                        # Check actor loss criteria.
                        if avg_actor_loss > 0 or avg_actor_loss > min_actor_threshold:
                            actor_loss_counter += 1
                            print(f"Actor loss insufficient. Counter: {actor_loss_counter} / {actor_loss_patience}")
                        else:
                            actor_loss_counter = 0

                        if solved:
                            print("Environment solved! Stopping training early.")

                            # Save the agent's weights and exit training loop.
                            new_weights = extract_agent_weights(agent)
                            save_path = f"ddpg_agent_weights_solved.pth"
                            torch.save(new_weights, save_path)
                            
                            print(f"Saved agent weights to {save_path}")                            
                            
                            avg_reward_return = avg_reward
                            break

                        if actor_loss_counter >= actor_loss_patience:
                            print("Actor loss early stopping triggered. Stopping training.")
                            avg_reward_return = avg_reward
                            break
                            
                        # Early stopping check: stop training if evaluation performance hasn't improved.
                        if early_stopping.step(avg_reward):
                            print("Early stopping triggered. Stopping training.")

                            avg_reward_return = avg_reward
                            break                           
                        
                    except Exception as eval_e:
                        print(f"[Training] Evaluation failed at episode {episode}: {eval_e}")
                        # Optionally, decide whether to continue or break.
    
    except Exception as e:
        print(f"[Training] An error occurred during training: {e}")
        traceback.print_exc()
    
    finally:
        # Ensure that the environment is closed.
        try:
            env.close()
        except Exception:
            pass
        
        writer.close()
        print("[Training] Training process terminated. Unity environment shut down.")
    
    return avg_reward_return

def extract_agent_weights(agent):
    return {
        "actor": {k: v.cpu() for k, v in agent.actor.state_dict().items()},
        "actor_target": {k: v.cpu() for k, v in agent.actor_target.state_dict().items()},
        "critic": {k: v.cpu() for k, v in agent.critic.state_dict().items()},
        "critic_target": {k: v.cpu() for k, v in agent.critic_target.state_dict().items()},
    }

if __name__ == "__main__":
    train()
