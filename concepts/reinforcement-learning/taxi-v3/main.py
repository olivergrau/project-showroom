from agent import ExpectedSarsaAgent, QLearningAgent
from monitor import interact
import gym

# Best trial:
#  Value (Best Avg Reward): 8.46
#  Params:
#    epsilon: 0.027946563766637098
#    alpha: 0.6238732908332515
#    gamma: 0.9742249127511525
#    epsilon_decay: 0.9820606002535831
#    alpha_decay: 0.999369669995233
#    min_epsilon: 0.001264509062443988
#    min_alpha: 0.00034017921680262335

env = gym.make('Taxi-v3')
agent = QLearningAgent(
    nA=env.action_space.n,
    epsilon=0.027946563766637098,
    alpha=0.6238732908332515,
    gamma=0.9742249127511525
)
avg_rewards, best_avg_reward = interact(
    env, agent, 
    num_episodes=20000, 
    window=100, 
    epsilon_decay=0.9820606002535831, 
    min_epsilon=0.001264509062443988,
    alpha_decay=0.999369669995233,
    min_alpha=0.00034017921680262335,
    patience=5000)