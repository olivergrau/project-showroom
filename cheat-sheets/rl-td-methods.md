# Reinforcement Learning Cheat Sheet: Temporal-Difference (TD) Methods

## Introduction
- Temporal-Difference (TD) learning is a mix of **Monte Carlo methods** and **Dynamic Programming**.
- It updates value estimates based on observed **rewards and estimates of future values** rather than waiting for the entire episode to finish.
- **TD(0) update rule**:
  \[
  V(s_t) \leftarrow V(s_t) + \alpha (R_{t+1} + \gamma V(s_{t+1}) - V(s_t))
  \]
- **Key advantage**: Can learn **online**, without needing to store full episodes.

## TD Control: Theory
- TD control methods estimate **action-value functions** and improve policies iteratively.
- **General TD(0) update for Q-values**:
  \[
  Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha (R_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t))
  \]

## Greedy in the Limit of Infinite Exploration (GLIE)
- A policy satisfies **GLIE** if:
  1. Every state-action pair is visited infinitely often.
  2. The policy converges to a **greedy policy** over time.
- **Achieved by**:
  - Using an \( \epsilon \)-greedy strategy where \( \epsilon \) decreases over time.
  - Ensuring continued exploration in early learning.

## TD Control: Sarsa
- **On-policy TD control method**.
- Updates Q-values using the action **actually taken** by the policy.
- **Update rule**:
  \[
  Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha (R_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t))
  \]
- **Pseudo Algorithm**:
```python
initialize Q(s, a) arbitrarily for all s, a
for each episode:
    initialize s
    choose a using ε-greedy policy
    for each step in episode:
        take action a, observe reward r and next state s'
        choose next action a' using ε-greedy policy
        update Q(s, a) using Sarsa update rule
        s, a = s', a'
```

## TD Control: Q-Learning (Sarsamax)
- **Off-policy TD control method**.
- Uses **maximum** Q-value of the next state rather than the action taken by the policy.
- **Update rule**:
  \[
  Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha (R_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t))
  \]
- **Pseudo Algorithm**:
```python
initialize Q(s, a) arbitrarily for all s, a
for each episode:
    initialize s
    for each step in episode:
        choose a using ε-greedy policy
        take action a, observe reward r and next state s'
        update Q(s, a) using Q-learning update rule
        s = s'
```

## TD Control: Expected Sarsa
- Like Sarsa, but instead of using the next action’s Q-value directly, it uses the **expected Q-value** under the current policy.
- **Update rule**:
  \[
  Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left( R_{t+1} + \gamma \sum_{a'} \pi(a' | s_{t+1}) Q(s_{t+1}, a') - Q(s_t, a_t) \right)
  \]
- **Reduces variance** compared to standard Sarsa.

## Summary
| Method        | Policy Type | Update Target |
|--------------|------------|--------------|
| **Sarsa** | On-policy | Q-value of actual next action |
| **Q-Learning** | Off-policy | Max Q-value of next state |
| **Expected Sarsa** | On-policy | Expected Q-value under policy |

## Comparing DP and MC Methods

### Key Differences
- **Dynamic Programming (DP)**:
  - Requires a *model*: $p(s',r\mid s,a)$.
  - Bellman “sweeps” over states using full transition probabilities.
  - Examples: Iterative Policy Evaluation, Policy Iteration, Value Iteration.

- **Monte Carlo (MC)**:
  - No model needed; learns from sampled *episodes*.
  - Uses *actual returns* from episodes to update value estimates (first‐visit or every‐visit).
  - Examples: First‐Visit MC Prediction, GLIE MC Control, Constant‐alpha MC Control.

---

## 8. DP, MC, and TD (Full Summary Table)

| **Aspect**            | **Dynamic Programming (DP)**                                                                                   | **Monte Carlo (MC)**                                                                                    | **Temporal-Difference (TD)**                                                                                                                                    |
|:----------------------|:----------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Model Required?**   | **Yes** (needs $p(s',r\mid s,a)$ to do Bellman sums)                                                           | **No** (learns directly from returns of sampled episodes)                                                | **No** (learns from step-by-step experience, does not require model)                                                                                             |
| **Update Mechanism**  | Full Bellman backup (summing over all next states, rewards)                                                     | Averages *actual returns* after entire episodes                                                          | **Bootstrapping** from current estimates, e.g. one-step TD: <br>$V(s)\leftarrow V(s) + \alpha[R + \gamma\,V(s') - V(s)]$                                       |
| **Timing of Updates** | Typically “sweeps” (synchronous) over the state space                                                           | At episode end (for first-visit or every-visit MC)                                                       | After **every step** (online, incremental); can also do n-step or λ-returns                                                                                     |
| **Pros**              | - Guaranteed stable solutions <br>- Can use full knowledge of transitions                                       | - Simpler if model is unknown <br>- Straightforward to implement if you can generate episodes            | - Often more data-efficient <br>- Works well in continuing tasks <br>- Updates before knowing the full return                                                                 |
| **Cons**              | - Needs complete transition model <br>- Potentially expensive for large or continuous state/action spaces        | - Must wait for the end of each episode <br>- Can have high variance of returns                          | - Introduces bias from bootstrapping <br>- More complex algorithm design (on-policy vs off-policy, etc.)                                                         |
| **Examples**          | Iterative Policy Evaluation, Policy Iteration, Value Iteration                                                  | First-Visit / Every-Visit MC Prediction, GLIE MC Control, Constant-alpha MC                              | TD(0), Sarsa, Q-Learning, n-step TD, TD(λ), Dyna-Q, etc.                                                                                                        |

---
