"""
optuna_search.py

This script performs a hyperparameter search using Optuna. It uses the train function from
train_ddpg.py, which runs the synchronous DDPG training loop and returns the last average
evaluation reward. The objective for Optuna is defined as the negative average reward, so that
a higher average reward corresponds to a lower objective value.
"""

import optuna
from train_ddpg import train
import optuna

def objective(trial):
    # Suggest hyperparameters to search over.
    lr_actor = trial.suggest_float("lr_actor", 1e-5, 1e-3, log=True)
    lr_critic = trial.suggest_float("lr_critic", 1e-5, 1e-3, log=True)
    tau = trial.suggest_float("tau", 0.001, 0.01, log=True)
    exploration_noise = trial.suggest_float("exploration_noise", 0.05, 0.2)
    critic_weight_decay = trial.suggest_float("critic_weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    critic_clip = trial.suggest_float("critic_clip", 0.1, 10.0)

    # You can fix other hyperparameters here (e.g., gamma, replay_capacity, etc.)
    gamma = 0.99
    replay_capacity = 100000
    eval_frequency = 10
    eval_episodes = 5 # Number of episodes to evaluate the agent
    eval_threshold = 30.0

    # Run the training process with the current hyperparameter configuration.
    # For hyperparameter search, you may want to run fewer episodes (e.g., 500 instead of 1000)
    # to reduce computation time.
    avg_reward = train(
        state_size=33,
        action_size=4,
        episodes=50,      # fewer episodes for hyperparameter search
        max_steps=1000,
        batch_size=batch_size,
        gamma=gamma,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        critic_clip=critic_clip,
        tau=tau,
        exploration_noise=exploration_noise,
        replay_capacity=replay_capacity,
        critic_weight_decay=critic_weight_decay,
        eval_frequency=eval_frequency,
        eval_episodes=eval_episodes,
        eval_threshold=eval_threshold,
        unity_worker_id=0
    )

    # Since we want to maximize reward, return its negative for Optuna minimization.
    return -avg_reward

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    
    print("Best trial:")
    trial = study.best_trial

    print("  Value (negative average reward):", trial.value)
    print("  Parameters:")
    
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
