# Reinforcement Learning Cheat Sheet: RL in Continuous Spaces

## Introduction
- Many real-world problems have **continuous state and action spaces** (e.g., robotics, finance, control systems).
- Standard tabular RL methods **do not scale** well to continuous domains.
- Solutions include:
  - **Discretization**
  - **Function approximation**

## Discrete vs Continuous Spaces
| Type | Example |
|------|---------|
| **Discrete Spaces** | Chess board, Grid-world |
| **Continuous Spaces** | Robot arm positions, Stock prices |

- **Challenges in Continuous Spaces**:
  - Infinite number of states/actions.
  - Harder to explore effectively.
  - Requires function approximation.

## Discretization
- **Converting continuous spaces into discrete representations**.
- **Uniform discretization**: Divide space into equally spaced bins.
- **Adaptive discretization**: Dynamically adjust bins based on experience.

## Tile Coding
- **Breaks the state space into overlapping tilings**.
- Allows for **better generalization** across states.
- **Example**: A robot’s position might be covered by multiple overlapping tiles.

## Coarse Coding
- Similar to tile coding but **each feature covers a large portion** of the state space.
- Allows for **smooth generalization** over large areas.

## Function Approximation
- Instead of storing values for each state-action pair, **approximate value functions**.
- Used when **state space is too large**.

## Linear Function Approximation
- Approximates Q-values using a linear combination of features:
  \[
  Q(s, a) = \theta_0 + \theta_1 f_1(s, a) + \theta_2 f_2(s, a) + ... + \theta_n f_n(s, a)
  \]
- **Update rule**:
  \[
  \theta \leftarrow \theta + \alpha (R + \gamma Q(s', a') - Q(s, a)) \nabla_\theta Q(s, a)
  \]
- Computationally efficient but may not capture complex relationships.

## Kernel Functions
- Kernel-based methods use **weighted similarity** between states.
- Example: **Gaussian RBF kernel**:
  \[
  K(s, s') = e^{-\frac{||s - s'||^2}{2 \sigma^2}}
  \]
- Useful for **non-parametric function approximation**.

## Non-Linear Function Approximation
- Uses **deep learning** or **neural networks** to approximate value functions.
- **Deep Q-Networks (DQN)**:
  - Replace Q-table with a neural network.
  - Update using **gradient descent**.

## Pseudo Algorithm: Function Approximation with TD Learning
```python
initialize weights θ randomly
for each episode:
    initialize state s
    for each step in episode:
        choose action a using ε-greedy policy
        take action a, observe reward r and new state s'
        compute TD target: target = r + γ * Q(s', a'; θ)
        compute loss: L = (target - Q(s, a; θ))^2
        update weights θ using gradient descent
        s = s'
```
## Coarse Coding vs Tile Coding

**Both** are ways to discretize continuous states via **overlapping features** that produce sparse binary feature vectors.  
- **Coarse Coding** is the *general concept* of having broad, possibly irregular “receptive fields.”  
- **Tile Coding** is a *specific type* of coarse coding using multiple offset grids (“tilings”).

| **Aspect**               | **Coarse Coding (General)**                                                          | **Tile Coding (Specific Form)**                                                                                                       |
|:-------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------|
| **Layout of Regions**    | Arbitrary shapes/placements of large overlapping “receptive fields”                  | Multiple offset **grids** partitioning the space into “tiles”                                                                         |
| **Overlap Structure**    | Overlaps depend on how you place circles, ellipses, or custom shapes                 | Overlap arises by offsetting each tiling. A state activates exactly one tile *per* tiling                                             |
| **Activation Pattern**   | A state typically “turns on” all receptive fields that contain it                     | A state “turns on” exactly **one** tile in each tiling (so a small, fixed number of active features)                                   |
| **Design Complexity**    | More flexible but can be more ad hoc (shapes, domain knowledge, random coverage)     | Straightforward to parameterize (just choose number of tilings, tile size, and offsets)                                               |
| **Common Usage**         | When you want custom or domain‐informed partitions                                   | Often the go‐to method in RL for continuous state spaces (e.g. the classic “Mountain Car” environment) because of ease of implementation |

---