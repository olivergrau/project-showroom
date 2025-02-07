from collections import deque
import sys
import math
import numpy as np

def interact(
        env, agent, 
        num_episodes=20000, 
        window=100, 
        epsilon_decay=0.999, 
        min_epsilon=0.01, 
        alpha_decay=0.999, 
        min_alpha=0.01, 
        patience=2000):
    """ Monitor agent's performance.
    
    Params
    ======
    - env: instance of OpenAI Gym's Taxi-v3 environment
    - agent: instance of class Agent (see Agent.py for details)
    - num_episodes: number of episodes of agent-environment interaction
    - window: number of episodes to consider when calculating average rewards

    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    """
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    
    # initialize best average reward
    best_avg_reward = -math.inf
    
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)

    # Track how many episodes since last improvement
    episodes_since_improvement = 0

    # for each episode
    for i_episode in range(1, num_episodes+1):
        # begin the episode
        state, _ = env.reset()
        
        # initialize the sampled reward
        samp_reward = 0

        while True:
            # agent selects an action
            action = agent.select_action(state)
            
            # agent performs the selected action
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # agent performs internal updates based on sampled experience
            agent.step(state, action, reward, next_state, terminated or truncated)
            
            # update the sampled reward
            samp_reward += reward
            
            # update the state (s <- s') to next time step
            state = next_state
            
            if terminated or truncated:
                # save final sampled reward
                samp_rewards.append(samp_reward)
                break

        if (i_episode >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            
            # append to deque
            avg_rewards.append(avg_reward)
            
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                episodes_since_improvement = 0
            else:
                episodes_since_improvement += 1

        # monitor progress
        print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, best_avg_reward), end="")
        sys.stdout.flush()
        
        # check if task is solved (according to OpenAI Gym)
        if best_avg_reward >= 9.7:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            break
            
        # check if we should stop early
        if episodes_since_improvement >= patience:
            print('\nStopping early due to lack of improvement.')
            break

        if i_episode == num_episodes: print('\n')

        # implement epsilon decay
        if epsilon_decay is not None:
            agent.epsilon = max(agent.epsilon * epsilon_decay, min_epsilon)

        # implement alpha decay
        if alpha_decay is not None:
            agent.alpha = max(agent.alpha * alpha_decay, min_alpha)

    return avg_rewards, best_avg_reward