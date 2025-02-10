# Reinforcement Learning Cheat Sheet

This cheat sheet summarizes the key equations, examples, and comparisons we discussed.

---

## 1. Expectation Formulas

1. **Unconditional Expectation** (discrete):
   $$
   E[X] \;=\; \sum_{x_i} x_i \,\Pr\{X = x_i\}.
   $$

2. **Conditional Expectation**:
   $$
   E[X \mid Y = y_j] \;=\; \sum_{x_i} x_i \,\Pr\{X = x_i \mid Y = y_j\}.
   $$

3. **Conditional Expectation with Law of Total Expectation**:
   $$
   E[X \mid Y = y_j]
   \;=\;
   \sum_{z_k} \Pr\{Z = z_k \mid Y = y_j\}\,\bigl[\,E[X \mid Y=y_j,\, Z=z_k]\bigr].
   $$

---

## 2. Example of These Formulas (Dice Roll)

- Let $X$ be the outcome of rolling a fair 6‐sided die $\{1,2,3,4,5,6\}$.

1. **$E[X]$**:
   $$
   E[X] 
   \;=\;
   \sum_{x=1}^{6} x \cdot \frac{1}{6}
   \;=\;
   3.5.
   $$

2. **$E[X \mid Y=\text{“Even”}]$** (where $Y=$ "Even"/"Odd"):
   - Even faces: $\{2,4,6\}$. Conditional probability of each even face is $\tfrac{1}{3}$.
   $$
   E[X \mid Y=\text{“Even”}]
   \;=\; 2\cdot \tfrac{1}{3} + 4\cdot \tfrac{1}{3} + 6\cdot \tfrac{1}{3}
   \;=\; 4.
   $$

3. **$E[X \mid Y=\text{“Even”}]$ via another variable $Z$**:
   - For instance $Z$=“Less than 4” or “At least 4.”  
   - Then
     $$
     E[X \mid Y=\text{“Even”}]
     \;=\;
     \sum_{z} \Pr\{Z=z \mid Y=\text{“Even”}\}
        \, E[X \mid Y=\text{“Even”}, Z=z].
     $$
   - You would sum over the ways $Z$ can occur within the even faces, etc.

---

## 3. Quick Example: E[X | Y = "Prime"] Without Using Z

- For a fair die, the prime faces are $\{2,3,5\}$.
- Given $Y=\text{“Prime”}$, each prime face has probability $\tfrac{1}{3}$.
$$
E[X \mid Y=\text{“Prime”}]
=\;
\frac{2+3+5}{3}
=\;
\frac{10}{3}
=\;
3.\overline{3}.
$$

---

## 4. Four Equations in the MDP

1. **$p(s', r \mid s,a)$**:
   $$
   p(s', r \mid s,a)
   \;=\;
   \Pr\{S_t = s',\,R_t = r \;\mid\; S_{t-1} = s,\,A_{t-1}=a\}.
   $$
   Probability of next state‐reward pair $(s',r)$ given current state‐action $(s,a)$.

2. **$p(s' \mid s,a)$**:
   $$
   p(s' \mid s,a)
   \;=\;
   \sum_{r} p(s', r \mid s,a).
   $$
   Probability of transitioning to state $s'$ from $(s,a)$ (marginalizing out reward).

3. **$r(s,a) = E[R_t \mid S_{t-1}=s,\,A_{t-1}=a]$**:
   $$
   r(s,a)
   \;=\;
   \sum_{r}\sum_{s'} p(s', r \mid s,a)\,\bigl[r\bigr].
   $$
   Expected one‐step reward from taking action $a$ in state $s$.

4. **$r(s', s, a) = E[R_t \mid S_{t-1}=s,\; A_{t-1}=a,\; S_t=s']$**:
   $$
   r(s', s, a)
   \;=\;
   \sum_{r}
   r \,\frac{p(s', r \mid s,a)}{p(s'\mid s,a)}.
   $$
   Expected reward given that you definitely landed in $s'$.

---

## 5. Derivation of Equation (4)

- We want
  $$
  r(s', s, a)
  =
  E[R_t \mid S_{t-1}=s,\,A_{t-1}=a,\,S_t = s'].
  $$
- By definition of conditional probability:
  $$
  \Pr\{R_t = r \mid s, a, s'\}
  =
  \frac{\Pr\{S_t=s', R_t = r \mid s,a\}}
       {\Pr\{S_t=s' \mid s,a\}}.
  $$
- Hence,
  $$
  r(s', s, a)
  =
  \sum_{r} r \,\frac{p(s',r \mid s,a)}{p(s'\mid s,a)}.
  $$

---

## 6. First Bellman Equation for $v_\pi(s)$

$$
v_{\pi}(s)
\;=\;
\sum_{a \in \mathcal{A}} \pi(a \mid s)\,
\sum_{s'\in\mathcal{S}} \sum_{r\in\mathcal{R}}
p(s', r \mid s,a)\,\bigl[r + \gamma\,v_{\pi}(s')\bigr].
$$

Or in **expectation** form:
$$
v_{\pi}(s)
=
\mathbb{E}_{\pi}\bigl[R_{t+1} + \gamma\,v_{\pi}(S_{t+1})
\;\big|\;S_t = s\bigr].
$$

**Interpretation**: The value of a state under $\pi$ is the expected immediate reward plus the discounted value of the next state, *averaged* over all actions chosen by $\pi$ and all state transitions $p$.

---

## 7. Comparing DP and MC Methods

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

## 9. Coarse Coding vs Tile Coding

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

