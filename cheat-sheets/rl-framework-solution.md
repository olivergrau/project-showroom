# Reinforcement Learning Cheat Sheet: RL Framework - The Solution

## Policies
- A **policy** \( \pi \) defines the agent's behavior at any state.
- It is a mapping from states to probabilities of selecting each possible action:
  \[
  \pi(a | s) = Pr(A_t = a | S_t = s)
  \]
- Two types of policies:
  - **Deterministic policy**: Always selects the same action for a given state.
  - **Stochastic policy**: Assigns probabilities to different actions.

## State-Value Functions
- The **value function** \( V(s) \) estimates the expected cumulative reward starting from state \( s \) and following policy \( \pi \):
  \[
  V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} | S_0 = s \right]
  \]
- Measures the desirability of being in a particular state under a given policy.

## Bellman Equations
- The Bellman equation expresses the relationship between the value of a state and the values of its successor states:
  \[
  V^\pi(s) = \sum_{a \in A} \pi(a | s) \sum_{s'} P(s' | s, a) \left[ R(s, a) + \gamma V^\pi(s') \right]
  \]
- It provides a recursive decomposition of the value function.

## Optimality
- The **optimal policy** \( \pi^* \) is the policy that maximizes the expected return from any state.
- The corresponding **optimal value function**:
  \[
  V^*(s) = \max_\pi V^\pi(s)
  \]
- The agent's goal is to find \( \pi^* \).

## Action-Value Functions
- The **Q-value function** \( Q(s, a) \) estimates the expected return for taking action \( a \) in state \( s \) and following policy \( \pi \):
  \[
  Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} | S_0 = s, A_0 = a \right]
  \]
- The optimal action-value function:
  \[
  Q^*(s, a) = \max_\pi Q^\pi(s, a)
  \]
- The **Bellman optimality equation** for Q-values:
  \[
  Q^*(s, a) = \mathbb{E} \left[ R + \gamma \max_{a'} Q^*(s', a') | S = s, A = a \right]
  \]

## Optimal Policies
- An **optimal policy** \( \pi^* \) chooses actions that maximize the Q-value:
  \[
  \pi^*(s) = \arg\max_a Q^*(s, a)
  \]
- Finding the optimal policy is the main goal of reinforcement learning.

## Pseudo Algorithm: Policy Iteration
```python
initialize V(s) arbitrarily for all s
repeat:
    policy evaluation:
        compute V(s) for current policy π
    policy improvement:
        update π to be greedy w.r.t. V(s)
until policy converges
```
