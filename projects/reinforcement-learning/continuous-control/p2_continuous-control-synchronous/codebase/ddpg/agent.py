# DDPGAgent encapsulates the DDPG algorithm
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from codebase.ddpg.net.actor import Actor  # same actor network
from codebase.ddpg.net.critic import Critic  # same critic network

class OrnsteinUhlenbeckNoise:
    """
    Implements Ornstein-Uhlenbeck process for temporally correlated noise.
    """

    def __init__(self, size, seed=0, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process.

        :param size: Integer. Dimension of each state
        :param seed: Integer. Random seed
        :param mu: Float. Mean of the distribution
        :param theta: Float. Rate of the mean reversion of the distribution
        :param sigma: Float. Volatility of the distribution
        """
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self._rng  = np.random.RandomState(seed)
        self.reset()

        print(f"Ornstein-Uhlenbeck noise enabled with theta={theta}, sigma={sigma}")

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * self._rng.standard_normal(self.size)
        self.state = x + dx
        return self.state
    
class DDPGAgent:
    def __init__(
        self,
        num_agents=20,
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
        seed=0
    ):
        self.num_agents = num_agents
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
                
        if self.use_ou_noise:
            self.ou_noise = OrnsteinUhlenbeckNoise((num_agents, action_size), theta=ou_noise_theta, sigma=ou_noise_sigma, seed=seed)

        print()
        print(f"{self.agent_id}: Using num_agents: {num_agents}")
        print(f"{self.agent_id}: Using state size: {state_size}")
        print(f"{self.agent_id}: Using action size: {action_size}")
        print(f"{self.agent_id}: Using SEED: {seed}")
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
        print()        
        
        # Initialize actor and its target
        self.actor = Actor(
            state_size, 
            action_size, 
            hidden1=actor_input_size, 
            hidden2=actor_hidden_size,
            seed=seed
            ).to(self.device)
        
        self.actor_target = Actor(
            state_size, 
            action_size, 
            hidden1=actor_input_size, 
            hidden2=actor_hidden_size,
            seed=seed
            ).to(self.device)
        
        #self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Initialize a single critic and its target
        self.critic = Critic(
            state_size, 
            action_size, 
            hidden1=critic_input_size, 
            hidden2=critic_hidden_size, 
            seed=seed
            ).to(self.device)
        
        self.critic_target = Critic(
            state_size, 
            action_size, 
            hidden1=critic_input_size, 
            hidden2=critic_hidden_size,
            seed=seed
            ).to(self.device)
        
        #self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=self.critic_weight_decay or 0.0)

        self.random_seed = random.seed(seed)

    def act(self, state, eval=False, noise_scale=1.0):
        """
        Given a state, select an action. Optionally, add exploration noise.
        State can be a single state or a batch of states.
        """
        state = torch.FloatTensor(state).to(self.device)
        
        self.actor.eval()  # Switch to evaluation mode
        
        with torch.no_grad():
            action = self.actor(state)
        
        self.actor.train()  # Optionally, revert back to training mode if needed

        noise = None

        if not eval and self.use_ou_noise:
            # Use Ornstein-Uhlenbeck noise for temporally correlated exploration
            noise = self.ou_noise.sample() # * noise_scale

            # Convert OU noise to a tensor
            noise_tensor = torch.from_numpy(noise).to(self.device).float()
            
            action = action + noise_tensor
        elif not eval and noise_scale != 0.0 and noise_scale is not None:
            # Fallback to Gaussian noise if OU noise is disabled
            noise = torch.randn_like(action) * noise_scale
            action = action + torch.randn_like(action) * noise_scale
        
        # Clamp the actions to the valid range [-1, 1]
        return action.clamp(-1, 1).detach().cpu().numpy(), noise
    
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
    
        Returns a dictionary of metrics for logging, including a per-sample TD error
        so you can update priorities in your replay buffer.
        """
        self.total_it += 1
        
        # Unpack batch and move to the appropriate device
        # state = batch.state.to(self.device)
        # action = batch.action.to(self.device)
        # reward = batch.reward.unsqueeze(1).to(self.device)
        # next_state = batch.next_state.to(self.device)
        # mask = batch.mask.unsqueeze(1).to(self.device)

        # Unpack batch
        state, action, reward, next_state, mask = batch

        # For logging reward stats
        reward_log = (
            reward.mean().item(),
            reward.std().item(),
            reward.min().item(),
            reward.max().item(),
        )
    
        # --- Critic Update ---
        with torch.no_grad():
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target = reward + mask * self.gamma * target_Q

        # 2. Compute current Q
        current_Q = self.critic(state, action)
    
        # 3. Critic loss
        critic_loss = nn.MSELoss()(current_Q, target)

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
            "state_min": state.min().item(),
            "state_max": state.max().item(),
            "action_mean": action.mean().item(),
            "action_std": action.std().item(),
            "action_min": action.min().item(),
            "action_max": action.max().item(),
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