# DDPGAgent encapsulates the DDPG algorithm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from codebase.ddpg.net.actor import Actor  # same actor network
from codebase.ddpg.net.critic import Critic  # same critic network

DEBUG = False

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
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        use_reward_normalization=False,
        device=None,
        label="DDPGAgent",
        use_ou_noise=True
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.total_it = 0
        self.use_reward_normalization = use_reward_normalization
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent_id = "DDPGAgent (" + label + ")"
        self.use_ou_noise = use_ou_noise

        print()
        print(f"{self.agent_id}: Using device: {self.device}")
        print(f"{self.agent_id}: use_reward_norm={self.use_reward_normalization}, state_size={state_size}, action_size={action_size}, actor_lr={lr_actor}, critic_lr={lr_critic}, gamma={gamma}, tau={tau}")
        
        if self.use_ou_noise:
            self.ou_noise = OrnsteinUhlenbeckNoise(action_size, mu=0.0, theta=0.15, sigma=0.2)
            print(f"{self.agent_id}: Ornstein-Uhlenbeck noise enabled with theta=0.15, sigma=0.2")

        # Initialize actor and its target
        self.actor = Actor(state_size, action_size).to(self.device)
        self.actor_target = Actor(state_size, action_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Initialize a single critic and its target
        self.critic = Critic(state_size, action_size).to(self.device)
        self.critic_target = Critic(state_size, action_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1e-5)

    def act(self, state, noise=0.0):
        """
        Given a state, select an action. Optionally, add exploration noise.
        State can be a single state or a batch of states.
        """
        state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        action = self.actor(state)
        
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
    
    def learn(self, batch):
        """
        Update the DDPG agent networks based on a batch of transitions.
        Batch is expected to be a namedtuple or similar structure containing:
        - state: shape [batch_size, state_size]
        - action: shape [batch_size, action_size]
        - reward: shape [batch_size]
        - next_state: shape [batch_size, state_size]
        - mask: shape [batch_size] (1 if not done, 0 if done)
        Returns a dictionary of metrics for logging.
        """
        self.total_it += 1

        # Unpack batch and move to the appropriate device
        state = batch.state.to(self.device)
        action = batch.action.to(self.device)
        reward = batch.reward.unsqueeze(1).to(self.device)
        next_state = batch.next_state.to(self.device)
        mask = batch.mask.unsqueeze(1).to(self.device)
        
        # Debug: Print batch statistics for plausibility check
        if DEBUG:
            print(f"[{self.agent_id}]: === Batch Statistics ===")
            print(f"[{self.agent_id}]: State: shape={state.shape}, min={state.min().item():.4f}, max={state.max().item():.4f}, mean={state.mean().item():.4f}")
            print(f"[{self.agent_id}]: Action: shape={action.shape}, min={action.min().item():.4f}, max={action.max().item():.4f}, mean={action.mean().item():.4f}")
            print(f"[{self.agent_id}]: Reward: shape={reward.shape}, min={reward.min().item():.4f}, max={reward.max().item():.4f}, mean={reward.mean().item():.4f}")
            print(f"[{self.agent_id}]: Next_state: shape={next_state.shape}, min={next_state.min().item():.4f}, max={next_state.max().item():.4f}, mean={next_state.mean().item():.4f}")
            print(f"[{self.agent_id}]: Mask: shape={mask.shape}, unique values: {mask.unique()}")  

        epsilon = 1e-6  # small value to prevent division by zero

        # Optionally apply z-normalization to the scaled rewards
        if self.use_reward_normalization:
            r_mean = reward.mean()
            r_std  = reward.std() + epsilon
            normalized_reward = (reward - r_mean) / r_std
        else:
            normalized_reward = reward
            
        # Compute target Q-values without target noise (DDPG)
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target = normalized_reward + mask * self.gamma * target_Q

        # Get current Q estimates from the critic
        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Update actor (gradient ascent)
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        if DEBUG:
            for name, param in self.actor.named_parameters():
                if param.grad is not None:
                    print(f"[{self.agent_id}] Actor gradient {name}: {param.grad.abs().mean().item()}")

        self.actor_optimizer.step()
            
        # Soft update target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
            
        metrics = {
            "critic_loss": critic_loss.item(),
            "current_Q_mean": current_Q.mean().item(),
            "target_Q_mean": target_Q.mean().item(),
            "scaled_reward_mean": reward.mean().item(),
            "normalized_reward_mean": normalized_reward.mean().item(),
            "actor_loss": actor_loss.item(),
        }
        
        return metrics

    def soft_update(self, net, target_net):
        """
        Perform Polyak averaging to update the target network:
        target_param = tau * param + (1 - tau) * target_param
        """
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)