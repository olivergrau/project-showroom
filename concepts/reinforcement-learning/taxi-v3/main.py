from agent import Agent
from monitor import interact
import gym

env = gym.make('Taxi-v3')
agent = Agent(
    nA=env.action_space.n,
    epsilon=0.005,
    alpha=0.1,
    gamma=1.0
)
avg_rewards, best_avg_reward = interact(env, agent)