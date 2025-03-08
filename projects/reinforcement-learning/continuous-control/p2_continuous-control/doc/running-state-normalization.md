Exactly—the challenge is that you only see mini-batches (or the 20-agent data per step) and not the full state distribution. In practice, most implementations use a running (online) normalization scheme. This works as follows:

1. **Maintain Running Statistics:**  
   You create a "normalizer" object (or module) that maintains a running estimate of the mean and variance of the states. As new batches of states come in (from the env_worker), you update these estimates using an exponential moving average or an algorithm like Welford’s method.

2. **Normalize On the Fly:**  
   Each time you get a new state (or batch of states), you normalize them using:  
   \[
   \text{normalized\_state} = \frac{\text{state} - \text{running\_mean}}{\sqrt{\text{running\_var}} + \epsilon}
   \]
   Here, \(\epsilon\) is a small constant to avoid division by zero.

3. **Consistent Normalization Across Components:**  
   - **Env_worker:** When collecting data, normalize states before storing them in the replay buffer.  
   - **Train_worker & Eval_worker:** Use the same normalization parameters when processing the states.  
   Since you update the running estimates continuously, your training and evaluation components always operate on similarly normalized data.

4. **Handling Non-Stationarity:**  
   The state distribution might change over time (non-stationary environment), so the running statistics need to adapt. Using an exponential moving average (with a small update rate) allows the estimates to slowly adjust.

### **Example Implementation**

Here’s a very simplified illustration:

```python
class RunningNormalizer:
    def __init__(self, shape, momentum=0.001, epsilon=1e-8):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.momentum = momentum
        self.epsilon = epsilon
        self.count = 0

    def update(self, batch):
        # Compute batch statistics:
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]
        
        # Update running mean and variance with exponential moving average:
        self.mean = (1 - self.momentum) * self.mean + self.momentum * batch_mean
        self.var = (1 - self.momentum) * self.var + self.momentum * batch_var
        self.count += batch_count

    def normalize(self, batch):
        return (batch - self.mean) / (np.sqrt(self.var) + self.epsilon)
```

You could integrate this object in your env_worker. For example:

```python
# At initialization in env_worker:
state_normalizer = RunningNormalizer(shape=(33,), momentum=0.001)

# When you get raw states from the environment:
raw_states = env_info.vector_observations  # shape: (num_agents, state_size)
state_normalizer.update(raw_states)
normalized_states = state_normalizer.normalize(raw_states)

# Then, store normalized_states in the replay buffer.
```

Make sure that the same normalizer (or a copy of its parameters) is used in the train_worker and eval_worker when processing states. You may need to share these parameters (e.g., via a shared memory object, or by periodically sending the current normalization parameters to the training and evaluation processes).

### **Summary**

- **Yes, you cannot know the true mean and standard deviation from each mini-batch alone.**
- **The common solution is to maintain running estimates** (using an online update rule) of the mean and variance.
- **All components should then normalize states consistently** using these running estimates.

This approach is widely used in deep RL implementations and helps to stabilize training even when only a subset of the entire state distribution is observed at each time step.