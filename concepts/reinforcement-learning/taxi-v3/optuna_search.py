from agent import DoubleQLearningAgent, ExpectedSarsaAgent, QLearningAgent
from monitor import interact
import gym
import optuna

def objective(trial):
    """Objective function for Optuna. We sample hyperparams, run training, and return the best reward."""
    # Sample hyperparameters
    epsilon = trial.suggest_float("epsilon", 0.01, 0.5, log=True)
    alpha = trial.suggest_float("alpha", 0.01, 0.7, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 1.0)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.95, 0.9999, log=True)
    alpha_decay = trial.suggest_float("alpha_decay", 0.95, 0.9999, log=True)
    min_epsilon = trial.suggest_float("min_epsilon", 0.001, 0.05, log=True)
    min_alpha = trial.suggest_float("min_alpha", 0.0001, 0.01, log=True)

    # Create environment and agent
    env = gym.make("Taxi-v3", render_mode=None)
    agent = DoubleQLearningAgent(
        nA=env.action_space.n,
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma
    )

    # Train the agent
    _, best_avg_reward = interact(
        env=env,
        agent=agent,
        num_episodes=50000,        # you can increase if you want a more thorough search
        window=100,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        alpha_decay=alpha_decay,
        min_alpha=min_alpha,
        patience=2500
    )

    # We want to maximize best_avg_reward, so return it directly
    return best_avg_reward

if __name__ == "__main__":
    # 'maximize' since we want to find max best_avg_reward
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=500)  # Adjust n_trials as needed

    print("\n\nBest trial:")
    best_trial = study.best_trial
    print(f"  Value (Best Avg Reward): {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")