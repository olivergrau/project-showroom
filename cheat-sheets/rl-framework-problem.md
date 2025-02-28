# Reinforcement Learning Cheat Sheet: RL Framework - The Problem

## Episodic vs Continuing Tasks
- **Episodic Tasks**: Tasks that have a definite beginning and end, such as a game of chess or a robotic arm picking up an object.
- **Continuing Tasks**: Tasks that do not have a natural termination, such as a thermostat controlling room temperature.

## Reward Hypothesis
- The reward hypothesis states that **all goals can be described as the maximization of cumulative rewards**.

## Goals and Rewards
- A reinforcement learning agent seeks to maximize the total expected reward over time.
- The choice of reward function is crucial as it guides the learning process.

## Cumulative Reward
- The total reward an agent collects over time is known as **cumulative reward**.
- The objective of an RL agent is to maximize this reward.

## Discounted Return
- **Return (G_t)**: The sum of future rewards from time step *t*:
  \[
  G_t = R_{t+1} + R_{t+2} + R_{t+3} + ... + R_T
  \]
- **Discounted Return**: Future rewards are often discounted using a discount factor \( \gamma \) where \( 0 \leq \gamma \leq 1 \):
  \[
  G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... + \gamma^{T-t} R_T
  \]

## Markov Decision Process (MDP)
- An MDP is defined as a **tuple** \( (S, A, P, R, \gamma) \) where:
  - **S**: Set of states
  - **A**: Set of actions
  - **P**: State transition probability function \( P(s' | s, a) \)
  - **R**: Reward function \( R(s, a) \)
  - **γ**: Discount factor (0 ≤ γ ≤ 1)

## One-Step Dynamics of MDPs
- The probability of transitioning from state \( s \) to state \( s' \) given action \( a \):
  \[
  P(s' | s, a) = Pr(S_{t+1} = s' | S_t = s, A_t = a)
  \]
- The expected reward for taking action \( a \) in state \( s \):
  \[
  R(s, a) = \mathbb{E}[R_{t+1} | S_t = s, A_t = a]
  \]

## Finite vs Infinite MDPs
- **Finite MDP**: The state and action spaces contain a finite number of elements.
- **Infinite MDP**: The state or action space is continuous or infinitely large.
