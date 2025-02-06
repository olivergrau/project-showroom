import numpy as np
from collections import defaultdict

class Agent:

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
        policy_s = self._epsilon_greedy_probs(self.Q[state])

        # update Q (using the expected value of Q, so all actions are considered)
        self.Q[state][action] = self._update_Q(
            self.Q[state][action], np.dot(self.Q[next_state], policy_s), reward)      

    def _update_Q(self, Qsa, Qsa_next, reward):
        """ updates the action-value function estimate using the most recent time step """
        return Qsa + (self.alpha * (reward + (self.gamma * Qsa_next) - Qsa))

    def _epsilon_greedy_probs(self, Q_s):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        epsilon = self.epsilon
        
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s