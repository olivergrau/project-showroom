import numpy as np
from collections import defaultdict

class DoubleQLearningAgent:
    def __init__(self, nA=6, epsilon=0.005, alpha=0.1, gamma=1.0):
        """
        Initialize the Double Q-Learning agent.
        
        Params
        ======
        - nA: number of actions available to the agent.
        - epsilon: exploration rate for epsilon-greedy action selection.
        - alpha: learning rate.
        - gamma: discount factor.
        """
        self.nA = nA
        # Two separate Q-tables
        self.Q1 = defaultdict(lambda: np.zeros(self.nA))
        self.Q2 = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        assert 0 <= self.epsilon <= 1, "Epsilon must be between 0 and 1"
        assert 0 <= self.alpha <= 1, "Alpha must be between 0 and 1"
        assert 0 <= self.gamma <= 1, "Gamma must be between 0 and 1"

    def select_action(self, state):
        """
        Given the state, select an action using an epsilon-greedy policy.
        The policy is based on the sum of Q1 and Q2 values.
        
        Params
        ======
        - state: the current state of the environment.
        
        Returns
        =======
        - action: an integer, compatible with the task's action space.
        """
        # Combine Q-values from both tables for behavior
        q_values = self.Q1[state] + self.Q2[state]
        policy_s = self._epsilon_greedy_probs(q_values)
        action = np.random.choice(np.arange(self.nA), p=policy_s)
        return action

    def step(self, state, action, reward, next_state, done):
        """
        Update the agent's knowledge, using the most recently sampled tuple.
        
        In Double Q-learning, one of the two Q-tables is chosen at random for updating.
        When updating one table, the greedy action is determined using that same table, but
        the Q-value from the other table is used to compute the target.
        
        Params
        ======
        - state: the previous state of the environment.
        - action: the agent's previous choice of action.
        - reward: last reward received.
        - next_state: the current state of the environment.
        - done: whether the episode is complete (True or False).
        """
        # If we reached a terminal state, the target is simply the reward.
        if done:
            target = reward
            # Choose randomly which table to update.
            if np.random.rand() < 0.5:
                self.Q1[state][action] += self.alpha * (target - self.Q1[state][action])
            else:
                self.Q2[state][action] += self.alpha * (target - self.Q2[state][action])
        else:
            # Randomly decide which Q-table to update
            if np.random.rand() < 0.5:
                # Update Q1:
                # 1. Choose the greedy action from Q1 in the next state.
                a_max = np.argmax(self.Q1[next_state])
                # 2. Use Q2 to evaluate that action.
                target = reward + self.gamma * self.Q2[next_state][a_max]
                self.Q1[state][action] += self.alpha * (target - self.Q1[state][action])
            else:
                # Update Q2:
                # 1. Choose the greedy action from Q2 in the next state.
                a_max = np.argmax(self.Q2[next_state])
                # 2. Use Q1 to evaluate that action.
                target = reward + self.gamma * self.Q1[next_state][a_max]
                self.Q2[state][action] += self.alpha * (target - self.Q2[state][action])

    def _epsilon_greedy_probs(self, Q_s):
        """
        Obtains the action probabilities corresponding to an epsilon-greedy policy.
        
        Params
        ======
        - Q_s: a vector of Q-values for the available actions.
        
        Returns
        =======
        - policy_s: a vector of probabilities for each action.
        """
        epsilon = self.epsilon
        policy_s = np.ones(self.nA) * epsilon / self.nA
        # Ensure that the greedy action gets the majority of the probability
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s


class QLearningAgent:
    def __init__(self, nA=6, epsilon=0.005, alpha=0.1, gamma=1.0):
        """Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        - epsilon: exploration rate for epsilon-greedy action selection
        - alpha: learning rate
        - gamma: discount factor
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        assert 0 <= self.epsilon <= 1, "Epsilon must be between 0 and 1"
        assert 0 <= self.alpha <= 1, "Alpha must be between 0 and 1"
        assert 0 <= self.gamma <= 1, "Gamma must be between 0 and 1"

    def select_action(self, state):
        """Given the state, select an action using an epsilon-greedy policy.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy_s = self._epsilon_greedy_probs(self.Q[state])
        action = np.random.choice(np.arange(self.nA), p=policy_s)
        return action

    def step(self, state, action, reward, next_state, done):
        """Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if done:
            # If next_state is terminal, the target is just the reward.
            target = reward
        else:
            # Q-learning: use the max over next state's Q-values
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        # Update Q-value using the Q-learning update rule
        self.Q[state][action] = self._update_Q(self.Q[state][action], target)

    def _update_Q(self, Qsa, target):
        """Updates the action-value function estimate using the Q-learning update rule."""
        return Qsa + self.alpha * (target - Qsa)

    def _epsilon_greedy_probs(self, Q_s):
        """Obtains the action probabilities corresponding to an epsilon-greedy policy."""
        epsilon = self.epsilon
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        
        return policy_s

class ExpectedSarsaAgent:

    def __init__(self, nA=6, epsilon=0.005, alpha=0.1, gamma=1.0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        assert 0 <= self.epsilon <= 1, "Epsilon must be between 0 and 1"
        assert 0 <= self.alpha <= 1, "Alpha must be between 0 and 1"
        assert 0 <= self.gamma <= 1, "Gamma must be between 0 and 1"

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy_s = self._epsilon_greedy_probs(self.Q[state])
        
        # pick action A
        action = np.random.choice(np.arange(self.nA), p=policy_s)

        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # If terminal, set expected Q-value to zero.
        if done:
            expected_value = 0
        else:
            # Compute epsilon-greedy probabilities for the next state
            policy_next = self._epsilon_greedy_probs(self.Q[next_state])
            expected_value = np.dot(self.Q[next_state], policy_next)
        
        # Update using expected SARSA update rule
        self.Q[state][action] = self._update_Q(
            self.Q[state][action], expected_value, reward)

    def _update_Q(self, Qsa, Qsa_next, reward):
        """ updates the action-value function estimate using the most recent time step """
        return Qsa + (self.alpha * (reward + (self.gamma * Qsa_next) - Qsa))

    def _epsilon_greedy_probs(self, Q_s):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        epsilon = self.epsilon
        
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s