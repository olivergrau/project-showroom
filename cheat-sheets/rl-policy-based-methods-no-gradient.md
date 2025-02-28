# Reinforcement Learning Cheat Sheet: Policy-Based Methods (Part 1)

## Introduction
- Policy-based methods **directly optimize policies** instead of estimating value functions.
- Useful for **high-dimensional** or **continuous action spaces** where value-based methods struggle.

## Why Policy-Based Methods?
- **Action-value methods** (Q-learning, DQN) struggle with large or continuous action spaces.
- Policy-based methods:
  - Can **learn stochastic policies**.
  - Avoid **value function approximation challenges**.
  - Can be used in **on-policy learning**.

## Differentiation from Action-Value / State-Value Functions
| Method | Learns? | Suitable for? |
|--------|--------|--------------|
| **Value-Based (Q-learning, DQN)** | Q-values \( Q(s, a) \) | Discrete actions |
| **Policy-Based (Policy Gradient, PPO)** | Policy \( \pi(a | s) \) | Continuous & high-dimensional actions |

## Policy Function Approximation
- Policies can be represented as:
  - **Tabular representations** (for small state spaces).
  - **Neural networks** (for complex tasks).
  - **Parameter vectors** (for simple continuous problems).

## Hill Climbing
- A simple **black-box** optimization method.
- Iteratively updates parameters based on performance.

### Hill Climbing Algorithm:
```python
initialize policy parameters θ randomly
for each iteration:
    generate episodes using policy π_θ
    compute total reward G for each episode
    update θ in the direction of better G
```

## Cross Entropy Method (CEM)
- A **sampling-based** optimization method.
- Generates multiple policy samples and selects the best ones to update parameters.

### Cross Entropy Method Algorithm:
```python
initialize mean μ and standard deviation σ
for each iteration:
    sample N candidate policies θ_i ~ N(μ, σ)
    evaluate each policy θ_i on episodes
    select top k performing policies
    update μ and σ based on selected policies
```

## More Black-Box Optimization (Without Gradient Methods)
- **Evolution Strategies (ES)**: Uses evolutionary algorithms for optimization.
- **Genetic Algorithms (GA)**: Simulates natural selection to evolve better policies.
- **Simulated Annealing**: Explores solutions probabilistically, reducing randomness over time.

