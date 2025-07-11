The reward normalization code is performing **online (incremental) normalization** using **running mean and variance**. Let's break it down step by step.

---

## **Mathematical Explanation**
### **1. Running Mean and Variance Update**
We maintain running statistics for the **mean** and **variance** of the rewards. Instead of storing all past rewards (which would require infinite memory), we update these statistics **incrementally** using the following formulas:

#### **Mean Update:**
\[
\mu_{\text{new}} = \frac{N_{\text{old}} \cdot \mu_{\text{old}} + \sum r}{N_{\text{new}}}
\]
where:
- \( \mu_{\text{new}} \) is the updated mean
- \( N_{\text{old}} \) is the previous count of rewards seen
- \( N_{\text{new}} = N_{\text{old}} + n \) is the new total count
- \( \sum r \) is the sum of the new batch of rewards
- \( \mu_{\text{old}} \) is the previous mean

#### **Variance Update (Welford's Method Approximation)**
The variance is updated using a formula derived from Welford's method:
\[
\sigma^2_{\text{new}} = \frac{N_{\text{old}} \cdot (\sigma^2_{\text{old}} + \mu_{\text{old}}^2) + \sum r^2}{N_{\text{new}}} - \mu_{\text{new}}^2
\]
where:
- \( \sigma^2_{\text{new}} \) is the new variance
- \( \sigma^2_{\text{old}} \) is the old variance
- \( \sum r^2 \) is the sum of squares of the new rewards

---

## **2. Implementation Breakdown**
Let's go line by line.

```python
reward = np.array(reward)
n = len(reward)
old_count = reward_stats["count"]
new_count = old_count + n
```
- `reward` is converted into a NumPy array to allow vectorized operations.
- `n` stores how many new reward values were observed in this batch.
- `old_count` keeps track of how many rewards have been seen before.
- `new_count` updates the total count of rewards.

#### **Mean Update:**
```python
new_mean = (old_count * reward_stats["mean"] + np.sum(reward)) / new_count
```
- We compute the new mean by taking a weighted average of the **previous mean** and the **current batch sum**.
- This avoids having to store all past rewards.

#### **Variance Update:**
```python
new_var = ((old_count * (reward_stats["var"] + reward_stats["mean"]**2) + np.sum(reward**2)) / new_count) - new_mean**2
```
- The first term inside the parentheses computes the new **sum of squared differences** using the running variance update formula.
- The second term subtracts the new mean squared, ensuring the variance is correctly centered.

#### **Store Updated Values:**
```python
reward_stats["mean"] = new_mean
reward_stats["var"] = new_var if new_var > 0 else 1.0
reward_stats["count"] = new_count
```
- We update the running **mean**, **variance**, and **count**.
- If variance becomes zero (which happens in early steps), we set it to **1.0** to avoid division by zero.

---

## **3. Normalizing the Reward**
```python
reward = (reward - reward_stats["mean"]) / (np.sqrt(reward_stats["var"]) + 1e-8)
```
- We **subtract the mean** to center the rewards around **zero**.
- We **divide by the standard deviation** \( (\sqrt{\sigma^2}) \) to ensure unit variance.
- The `1e-8` term avoids division by zero.

---

## **Example Calculation**
### **Step 1: Assume Running Statistics**
Let's say before this batch, we had:
- `reward_stats["mean"] = 0.2`
- `reward_stats["var"] = 0.05`
- `reward_stats["count"] = 100`

And the current batch of rewards is:
\[
r = [0.1, 0.15, 0.2]
\]
(So `n = 3`)

### **Step 2: Compute New Mean**
\[
\mu_{\text{new}} = \frac{(100 \times 0.2) + (0.1 + 0.15 + 0.2)}{100 + 3} = \frac{20 + 0.45}{103} \approx 0.198
\]

### **Step 3: Compute New Variance**
First, compute the sum of squares:
\[
\sum r^2 = (0.1^2 + 0.15^2 + 0.2^2) = (0.01 + 0.0225 + 0.04) = 0.0725
\]

\[
\sigma^2_{\text{new}} = \frac{100 \times (0.05 + 0.2^2) + 0.0725}{103} - 0.198^2
\]

Approximating, we get:
\[
\sigma^2_{\text{new}} \approx 0.048
\]

### **Step 4: Normalize Rewards**
Each reward is normalized as:
\[
r_{\text{normalized}} = \frac{r - 0.198}{\sqrt{0.048} + 1e-8}
\]

For example, for **r = 0.1**:
\[
\frac{0.1 - 0.198}{\sqrt{0.048}} \approx \frac{-0.098}{0.219} \approx -0.447
\]

---

## **Final Takeaways**
✅ **Why is this useful?**
- It helps stabilize training by keeping rewards in a consistent range.
- It prevents extreme reward values from dominating updates.

✅ **Why might rewards go negative?**
- If many rewards are small (like 0.1), the running mean may be larger than some rewards, leading to **negative normalized rewards**.

✅ **When should you use it?**
- When raw rewards have very different magnitudes, making updates unstable.
- When training agents in environments with sparse rewards.