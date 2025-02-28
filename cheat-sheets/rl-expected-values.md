# Expected Values Cheat Sheet

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

