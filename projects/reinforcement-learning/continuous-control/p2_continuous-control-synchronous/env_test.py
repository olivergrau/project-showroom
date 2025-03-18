"""
env_test.py

This script is designed to test the Unity Reacher environment
by sampling random actions with different strategies and printing
the reward statistics. Graphics are enabled so you can visually
inspect the behavior. The goal is to see which random action strategy
yields the highest average score (averaged over n episodes).
"""

import time
import numpy as np
from codebase.sac.env import BootstrappedEnvironment  # Adjust import as needed

# For older numpy versions
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64

def test_random_actions():
    # Configuration
    exe_path = "Reacher_Linux/Reacher.x86_64"  # Path to your Unity Reacher executable
    num_agents = 20
    action_size = 4
    max_steps = 1000       # Maximum steps per episode
    reward_threshold = 2   # Log if more than 2 agents get a reward in one step
    n_episodes = 5         # Number of episodes to run per strategy

    # Define different random action sampling strategies.
    strategies = {
        # Using a normal distribution (mean=0, std=0.5) then clipping to [-1,1].
        "normal_standard": lambda: np.clip(np.random.randn(num_agents, action_size), -1, 1),
        "early_exploration": lambda: np.clip(np.random.normal(0, 0.9, size=(num_agents, action_size)), -1, 1),
        "uniform": lambda: np.random.uniform(-1, 1, size=(num_agents, action_size)),
        # This strategy forces the magnitude of each action component to be at least 0.5.
        "stronger_05": lambda: np.sign(np.random.uniform(-1, 1, size=(num_agents, action_size))) *
                            np.random.uniform(0.5, 1, size=(num_agents, action_size)),
        # This strategy forces the magnitude of each action component to be at least 0.5.
        "stronger_075": lambda: np.sign(np.random.uniform(-1, 1, size=(num_agents, action_size))) *
                            np.random.uniform(0.5, 1, size=(num_agents, action_size)),
        # Using a normal distribution (mean=0, std=0.8) then clipping to [-1,1].
        "normal_0_dot8": lambda: np.clip(np.random.normal(0, 0.8, size=(num_agents, action_size)), -1, 1),
    }
    
    strategy_scores = {}  # To store average score for each strategy

    with BootstrappedEnvironment(exe_path, worker_id=1, use_graphics=False) as env:
        # Iterate over the different strategies.
        for strategy_name, action_fn in strategies.items():
            print(f"\n=== Testing strategy: {strategy_name} ===")
            total_strategy_score = 0.0

            for ep in range(1, n_episodes + 1):
                scores = np.zeros(num_agents)
                                
                state = env.reset(train_mode=True)
                step = 0
                while step < max_steps:
                    # Sample actions using the current strategy.
                    action = action_fn()
                    next_state, reward, done_flags = env.step(action)
                    
                    # Convert reward to a NumPy array.
                    reward_array = np.array(reward, dtype=np.float64)
                    
                    # Log if more than reward_threshold agents got a reward.
                    num_agents_rewarded = np.count_nonzero(np.abs(reward_array) > 1e-6)
                    if num_agents_rewarded > reward_threshold:
                        print(f"Episode {ep} / Step {step}: {num_agents_rewarded}/{num_agents} agents rewarded!")
                        print(f"Reward Vector: {reward_array}")
                    
                    scores += reward_array
                    state = next_state
                    step += 1
                    
                    # End episode early if any agent's episode is done.
                    if np.any(done_flags):
                        print(f"Episode {ep} ended early at step {step} due to done flags.")
                        break
                
                avg_episode_score = np.mean(scores)
                total_strategy_score += avg_episode_score
                print(f"Episode {ep} score (averaged over agents): {avg_episode_score:.4f}")
                time.sleep(1)  # Pause between episodes

            avg_strategy_score = total_strategy_score / n_episodes
            strategy_scores[strategy_name] = avg_strategy_score
            print(f"Finished testing strategy: {strategy_name}")
            print(f"Average score over {n_episodes} episodes: {avg_strategy_score:.4f}\n")
            time.sleep(2)  # Pause between strategies

        print("=== Summary of Strategies ===")
        for strat, score in strategy_scores.items():
            print(f"{strat}: {score:.4f}")
        
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        print(f"\nBest strategy: {best_strategy} with average score {strategy_scores[best_strategy]:.4f}")

if __name__ == "__main__":
    test_random_actions()
