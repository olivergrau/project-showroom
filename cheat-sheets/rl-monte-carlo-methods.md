# Reinforcement Learning Cheat Sheet: Monte Carlo Methods

## Concept of Monte Carlo Methods
- Monte Carlo (MC) methods use **sampled experience** to estimate value functions and optimize policies.
- They require **episodic tasks** (each episode must terminate).
- Estimation is based on averaging observed returns.

## MC Prediction
- Used to estimate the **value function** from sampled episodes.
- The **Monte Carlo return estimate** for state \( s \) is:
  \[
  V(s) = \mathbb{E}[G_t | S_t = s]
  \]
- Two types:
  - **First-visit MC**: Averages returns only from the first occurrence of state \( s \) in each episode.
  - **Every-visit MC**: Averages returns from all occurrences of state \( s \).

## Greedy Policies
- A greedy policy always selects the action with the highest estimated value:
  \[
  \pi(s) = \arg\max_a Q(s, a)
  \]
- **Problem**: Greedy policies may get stuck in **local optima** due to lack of exploration.

## Epsilon-Greedy Policies
- Introduces **exploration** by selecting a random action with probability \( \epsilon \) and the greedy action otherwise.
- Defined as:
  \[
  \pi(a|s) =
  \begin{cases} 
  1 - \epsilon + \frac{\epsilon}{|A|}, & \text{if } a = \arg\max Q(s, a) \\
  \frac{\epsilon}{|A|}, & \text{otherwise}
  \end{cases}
  \]
- Helps prevent getting stuck in suboptimal solutions.

## MC-Control
- Uses **Monte Carlo estimation** to optimize policies.
- **Policy iteration** with MC:
  1. **Policy evaluation** using MC estimates.
  2. **Policy improvement** using an \( \epsilon \)-greedy update.

## Exploration vs. Exploitation
- **Exploration**: Trying new actions to discover better strategies.
- **Exploitation**: Using known information to maximize immediate reward.
- **Balancing exploration and exploitation** is crucial for learning optimal policies.

## Incremental Mean
- Used for **efficiently computing sample averages**.
- The update formula:
  \[
  Q_{n+1} = Q_n + \frac{1}{n} (x_n - Q_n)
  \]
- Avoids storing all previous values.

## Constant-Alpha
- Instead of using sample averages, a **constant step-size** \( \alpha \) is used:
  \[
  Q(s, a) \leftarrow Q(s, a) + \alpha (G - Q(s, a))
  \]
- Allows the value function to adapt **more quickly** to new data.

## Pseudo Algorithm: First-Visit MC Control with Epsilon-Greedy
```python
initialize Q(s, a) arbitrarily for all s, a
initialize π to be arbitrary
for each episode:
    generate an episode using π
    compute returns G_t for each state-action pair
    update Q(s, a) using first-visit MC estimate
    update π to be ε-greedy w.r.t. Q
```
