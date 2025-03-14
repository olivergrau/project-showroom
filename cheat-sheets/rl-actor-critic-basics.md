
# Actor-Critic Algorithm Explanation

## Basic Actor-Critic Algorithm

A basic Actor-Critic algorithm consists of two neural networks:  
- The **Actor**, which decides actions based on a policy \(\pi_	\theta(a \mid s)\).  
- The **Critic**, which evaluates states with a value function \(V_\phi(s)\).  

At each time step, we:  
1. Take an action \(a_t\) from the policy \(\pi_	\theta(a_t \mid s_t)\).  
2. Receive a reward \(r_t\) and the next state \(s_{t+1}\).  
3. Compute the **TD error**:  
   \[
   \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
   \]
4. **Critic update** using Mean Squared Error:  
   \[
   \phi \leftarrow \phi + \alpha_v \delta_t 
\nabla_\phi V_\phi(s_t)
   \]
5. **Actor update** using the policy gradient:  
   \[
   	\theta \leftarrow 	\theta + \alpha_	\theta \delta_t 
\nabla_	\theta \log \pi_	\theta(a_t \mid s_t)
   \]

### **Pseudocode**

```
Initialize θ (actor parameters), φ (critic parameters)

loop over episodes:
    Initialize state s
    repeat until episode ends:
        # 1) Choose action from the policy
        a ~ πθ(a | s)
        
        # 2) Step in the environment
        s_next, r = EnvironmentStep(s, a)
        
        # 3) Compute TD error for the critic
        δ = r + γ * Vφ(s_next) - Vφ(s)
        
        # 4) Critic update: reduce TD error
        φ ← φ + α_v * δ * ∇φ Vφ(s)
        
        # 5) Actor update: policy gradient
        θ ← θ + α_θ * δ * ∇θ log πθ(a | s)
        
        # 6) Advance
        s ← s_next
    end repeat
end loop
```

## **Implementation in a Neural Network**

In practical implementations, gradient updates are realized through **loss functions**:

### **Critic Loss: TD Learning (MSE Loss)**
\(
\mathcal{L}_{	\text{critic}} = (r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t))^2
\)

```python
# Compute TD target
with torch.no_grad():
    td_target = reward + gamma * value_net(next_state) * (1 - done)

# Compute critic loss
value_estimate = value_net(state)
critic_loss = torch.nn.functional.mse_loss(value_estimate, td_target)

# Perform gradient descent step
critic_optimizer.zero_grad()
critic_loss.backward()
critic_optimizer.step()
```

### **Actor Loss: Policy Gradient**
\(
\mathcal{L}_{	\text{actor}} = - \delta_t \log \pi_	heta(a_t \mid s_t)
\)

```python
# Compute policy loss (negative policy gradient)
log_prob = torch.log(policy_net(state).gather(1, action.unsqueeze(1)))
actor_loss = - (td_error.detach() * log_prob).mean()

# Perform gradient ascent step
actor_optimizer.zero_grad()
actor_loss.backward()
actor_optimizer.step()
```

### **Why Use `.detach()` for TD Error?**
- The critic is trained using \(\delta_t\), but the actor should **not** backpropagate through the critic.
- Using `.detach()` ensures that **gradients only affect the actor**.

## **Summary**
- **The critic** minimizes MSE loss to learn value estimation.
- **The actor** uses policy gradients to improve action selection.
- **Both updates are interdependent**, where the actor improves based on critic feedback.

This is the foundation for advanced methods like A2C, PPO, and TD3! 🚀
