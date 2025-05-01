# DDPGAgent encapsulates the DDPG algorithm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from codebase.ddpg.net.actor import Actor  # same actor network
from codebase.ddpg.net.critic import Critic  # same critic network

DEBUG = False  # Set to True for detailed debugging
PRINT_EVERY = 1  # Print debug information every PRINT_EVERY steps
MAX_PRINT_STEPS = 50  # Maximum number of steps to print debug information

class OrnsteinUhlenbeckNoise:
    """
    Implements Ornstein-Uhlenbeck process for temporally correlated noise.
    """
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = np.ones(self.size) * mu
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.state = self.mu.copy()
    
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(*self.state.shape)
        self.state = self.state + dx
        return self.state
    
class DDPGAgent:
    def __init__(
        self,
        state_size=33,
        action_size=4,
        actor_input_size=400,      # Actor input layer size
        actor_hidden_size=300,     # Actor hidden layer size
        critic_input_size=400,     # Critic input layer size
        critic_hidden_size=300,    # Critic hidden layer size
        lr_actor=1e-3,
        lr_critic=1e-3,
        critic_clip_norm=None,
        critic_weight_decay=0.0,
        gamma=0.99,
        tau=0.005,
        device=None,
        label="DDPGAgent",
        use_ou_noise=True,
        ou_noise_theta=0.15,
        ou_noise_sigma=0.2,
        use_batch_norm=False,
        use_prioritized_replay=False,
    ):
        self.ou_noise_theta = ou_noise_theta
        self.ou_noise_sigma = ou_noise_sigma
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.total_it = 0
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent_id = "DDPGAgent (" + label + ")"
        self.use_ou_noise = use_ou_noise
        self.critic_clip = critic_clip_norm
        self.critic_weight_decay = critic_weight_decay
        self.use_prioritized_replay = use_prioritized_replay

        print()
        print(f"{self.agent_id}: Using critic clip: {self.critic_clip}")
        print(f"{self.agent_id}: Using critic weight decay: {self.critic_weight_decay}")
        print(f"{self.agent_id}: Using device: {self.device}")
        print(f"{self.agent_id}: Using gamma: {gamma}")
        print(f"{self.agent_id}: Using tau: {tau}")
        print(f"{self.agent_id}: Using actor input size: {actor_input_size}")
        print(f"{self.agent_id}: Using actor hidden size: {actor_hidden_size}")
        print(f"{self.agent_id}: Using critic input size: {critic_input_size}")
        print(f"{self.agent_id}: Using critic hidden size: {critic_hidden_size}")
        print(f"{self.agent_id}: Using actor learning rate: {lr_actor}")
        print(f"{self.agent_id}: Using critic learning rate: {lr_critic}")
        print(f"{self.agent_id}: Using Ornstein-Uhlenbeck noise: {use_ou_noise}")
        print(f"{self.agent_id}: Using OU noise theta: {ou_noise_theta}")
        print(f"{self.agent_id}: Using OU noise sigma: {ou_noise_sigma}")
        print(f"{self.agent_id}: Using batch normalization: {use_batch_norm}")
        print(f"{self.agent_id}: Using prioritized replay: {use_prioritized_replay}")
        print()        
        
        if self.use_ou_noise:
            self.ou_noise = OrnsteinUhlenbeckNoise(action_size, mu=0.0, theta=ou_noise_theta, sigma=ou_noise_sigma)
            print(f"{self.agent_id}: Ornstein-Uhlenbeck noise enabled with theta={ou_noise_theta}, sigma={ou_noise_sigma}")

        # Initialize actor and its target
        self.actor = Actor(
            state_size, 
            action_size, 
            hidden1=actor_input_size, 
            hidden2=actor_hidden_size, 
            init_type="kaiming",
            use_batch_norm=use_batch_norm).to(self.device)
        
        self.actor_target = Actor(
            state_size, 
            action_size, 
            hidden1=actor_input_size, 
            hidden2=actor_hidden_size,
            use_batch_norm=use_batch_norm).to(self.device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Initialize a single critic and its target
        self.critic = Critic(
            state_size, 
            action_size, 
            hidden1=critic_input_size, 
            hidden2=critic_hidden_size, 
            init_type="kaiming",
            use_batch_norm=use_batch_norm,
            dropout_prob=None).to(self.device)
        
        self.critic_target = Critic(
            state_size, 
            action_size, 
            hidden1=critic_input_size, 
            hidden2=critic_hidden_size,
            use_batch_norm=use_batch_norm).to(self.device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=self.critic_weight_decay or 0.0)

    def act(self, state, noise=0.0):
        """
        Given a state, select an action. Optionally, add exploration noise.
        State can be a single state or a batch of states.
        """
        state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        self.actor.eval()  # Switch to evaluation mode
        
        with torch.no_grad():
            action = self.actor(state)
        
        self.actor.train()  # Optionally, revert back to training mode if needed

        if noise != 0.0 and self.use_ou_noise:
            # Use Ornstein-Uhlenbeck noise for temporally correlated exploration
            ou_noise = self.ou_noise.sample() * noise
            
            # Convert OU noise to a tensor
            ou_noise_tensor = torch.from_numpy(ou_noise).to(self.device).float()
            
            # If batch size > 1, replicate noise across the batch
            if action.shape[0] > 1:
                ou_noise_tensor = ou_noise_tensor.unsqueeze(0).expand(action.shape[0], -1)

            action = action + ou_noise_tensor
        elif noise != 0.0:
            # Fallback to Gaussian noise if OU noise is disabled
            action = action + torch.randn_like(action) * noise
        
        # Clamp the actions to the valid range [-1, 1]
        return action.clamp(-1, 1).detach().cpu().numpy()
    
    def reset_noise(self):
        """
        Reset the Ornstein-Uhlenbeck noise process. This should be called at the start of each new episode.
        """
        if self.use_ou_noise:
            self.ou_noise.reset()

    def learn(self, batch, is_weights=None):
        """
        Update the DDPG agent networks based on a batch of transitions.
        Batch is expected to be a namedtuple or similar structure containing:
        - state: shape [batch_size, state_size]
        - action: shape [batch_size, action_size]
        - reward: shape [batch_size]
        - next_state: shape [batch_size, state_size]
        - mask: shape [batch_size] (1 if not done, 0 if done)
    
        Returns a dictionary of metrics for logging, including a per-sample TD error
        so you can update priorities in your replay buffer.
        """
        self.total_it += 1
        
        # Unpack batch and move to the appropriate device
        state = batch.state.to(self.device)
        action = batch.action.to(self.device)
        reward = batch.reward.unsqueeze(1).to(self.device)
        next_state = batch.next_state.to(self.device)
        mask = batch.mask.unsqueeze(1).to(self.device)

        if is_weights is None:
            is_weights = torch.ones_like(reward)

        # For logging reward stats
        reward_log = (
            reward.mean().item(),
            reward.std().item(),
            reward.min().item(),
            reward.max().item(),
        )
    
        # --- Critic Update ---
        with torch.no_grad():
            # noise = (torch.randn_like(action) * 0.1).clamp(-0.2, 0.2)
            # noisy_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)
            
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target = reward + mask * self.gamma * target_Q
            # clamp:     target = reward + mask * self.gamma * torch.clamp(target_Q, min=-10.0, max=10.0)

        # 2. Compute current Q
        current_Q = self.critic(state, action)
    
        # 3. Critic loss
        if self.use_prioritized_replay:
            td_error = current_Q - target
            #weights = batch.is_weight.unsqueeze(1)  # shape: [batch_size, 1]
            critic_loss = (is_weights * td_error ** 2).mean()
        else:
            critic_loss = nn.MSELoss()(current_Q, target)
            #critic_loss = nn.SmoothL1Loss()(current_Q, target)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
    
        # Optionally clip gradients to avoid exploding gradients
        if self.critic_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_clip)
    
        self.critic_optimizer.step()
    
        # --- Actor Update ---
        # Actor aims to maximize Q(state, actor(state)) => we do gradient ascent (negative sign for descent)
        actor_loss = -self.critic(state, self.actor(state)).mean()
    
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
        # Soft update target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
    
        # --- Compute TD Errors for Prioritized Replay ---
        # We do this after the backward pass so it doesn't affect training computations
        with torch.no_grad():
            # Absolute difference between current_Q and target for each sample
            # Shape: [batch_size, 1]
            td_error = torch.abs(current_Q - target).squeeze(1)  # => [batch_size]
    
        metrics = {
            "critic_loss": critic_loss.item(),
            "current_Q_mean": current_Q.mean().item(),
            "target_Q_mean": target_Q.mean().item(),
            "actor_loss": actor_loss.item(),
            "batch_reward_mean": reward_log[0],
            "batch_reward_std": reward_log[1],
            "batch_reward_min": reward_log[2],
            "batch_reward_max": reward_log[3],
            "td_error": td_error.detach().cpu().numpy(),
            "td_error_mean": td_error.mean().item(),
            "td_error_std": td_error.std().item(),

            # ✅ Diagnostic metrics:
            "state_mean": state.mean().item(),
            "state_std": state.std().item(),
            "action_mean": action.mean().item(),
            "action_std": action.std().item(),
            "target_Q_std": target_Q.std().item(),
            "current_Q_min": current_Q.min().item(),
            "current_Q_max": current_Q.max().item(),
        }
    
        return metrics


    def soft_update(self, net, target_net):
        """
        Perform Polyak averaging to update the target network:
        target_param = tau * param + (1 - tau) * target_param
        """
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)