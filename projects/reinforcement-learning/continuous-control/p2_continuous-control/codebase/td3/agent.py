# TD3Agent encapsulates the TD3 algorithm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from codebase.td3.net.actor import Actor
from codebase.td3.net.critic import Critic

DEBUG = False

class TD3Agent:
    def __init__(
        self,
        state_size=33,
        action_size=4,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        use_reward_normalization=False,
        device=None,
        label="TD3Agent", # to identify the agent
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent_id = "TD3Agent (" + label + ")"
        self.use_reward_normalization = use_reward_normalization

        print()
        print(f"{self.agent_id}: Using device: {self.device}")

        # print out all set parameters
        print(f"{self.agent_id}: use_reward_norm={use_reward_normalization}, state_size={state_size}, action_size={action_size}, actor_lr={lr_actor}, critic_lr={lr_critic}, gamma={gamma}, tau={tau}, policy_noise={policy_noise}, noise_clip={noise_clip}, policy_delay={policy_delay}")

        # Initialize actor and its target
        self.actor = Actor(state_size, action_size).to(self.device)
        self.actor_target = Actor(state_size, action_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Initialize two critics and their targets
        self.critic1 = Critic(state_size, action_size).to(self.device)
        self.critic2 = Critic(state_size, action_size).to(self.device)
        self.critic1_target = Critic(state_size, action_size).to(self.device)
        self.critic2_target = Critic(state_size, action_size).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        #self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic, weight_decay=1e-4)
        #self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic, weight_decay=1e-4)

    
    def act(self, state, noise=0.0):
        """
        Given a state, select an action. Optionally, add exploration noise.
        State can be a single state or a batch of states.
        """
        state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        action = self.actor(state)

        if noise != 0.0:
            action = action + torch.randn_like(action) * noise
        
        # Clamp the actions to the valid range [-1, 1]
        return action.clamp(-1, 1).detach().cpu().numpy()
    
    def learn(self, batch):
        """
        Update the TD3 agent networks based on a batch of transitions.
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
        state      = batch.state.to(self.device)
        action     = batch.action.to(self.device)
        scaled_reward = batch.reward.unsqueeze(1).to(self.device)
        next_state = batch.next_state.to(self.device)
        mask       = batch.mask.unsqueeze(1).to(self.device)
        
        # Debug: Print batch statistics for plausibility
        if DEBUG:
            print("[{self.agent_id}]: === Batch Statistics ===")
            print(f"[{self.agent_id}]: State: shape={state.shape}, min={state.min().item():.4f}, max={state.max().item():.4f}, mean={state.mean().item():.4f}")
            print(f"[{self.agent_id}]: Action: shape={action.shape}, min={action.min().item():.4f}, max={action.max().item():.4f}, mean={action.mean().item():.4f}")
            print(f"[{self.agent_id}]: Reward: shape={scaled_reward.shape}, min={scaled_reward.min().item():.4f}, max={scaled_reward.max().item():.4f}, mean={scaled_reward.mean().item():.4f}")
            print(f"[{self.agent_id}]: Next_state: shape={next_state.shape}, min={next_state.min().item():.4f}, max={next_state.max().item():.4f}, mean={next_state.mean().item():.4f}")
            print(f"[{self.agent_id}]: Mask: shape={mask.shape}, unique values: {mask.unique()}")  

        epsilon = 1e-6  # small value to prevent division by zero

        # Optionally apply z-normalization to the scaled rewards
        if self.use_reward_normalization:
            r_mean = scaled_reward.mean()
            r_std  = scaled_reward.std() + epsilon  # prevent division by zero
            normalized_reward = (scaled_reward - r_mean) / r_std
        else:
            normalized_reward = scaled_reward
            
        # Compute target actions with noise for target policy smoothing
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
            
            # Compute target Q-values from target critic networks
            target_Q1 = self.critic1_target(next_state, next_action)
            target_Q2 = self.critic2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            if DEBUG:
                print(f"[{self.agent_id}]: Target_Q1 mean: {target_Q1.mean().item()}, min: {target_Q1.min().item()}, max: {target_Q1.max().item()}")
                print(f"[{self.agent_id}]: Target_Q2 mean: {target_Q2.mean().item()}, min: {target_Q2.min().item()}, max: {target_Q2.max().item()}")
                print(f"[{self.agent_id}]: Target_Q mean: {target_Q.mean().item()}, min: {target_Q.min().item()}, max: {target_Q.max().item()}")

            target = normalized_reward + mask * self.gamma * target_Q
            #target = reward + mask * self.gamma * torch.clamp(target_Q, min=-10, max=100)

        # Get current Q estimates from the critics
        current_Q1 = self.critic1(state, action)
        current_Q2 = self.critic2(state, action)
        critic1_loss = F.mse_loss(current_Q1, target)
        critic2_loss = F.mse_loss(current_Q2, target)
        critic_loss = critic1_loss + critic2_loss
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        
        # Prepare a dictionary of metrics
        metrics = {
            "critic_loss": critic_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "current_Q1_mean": current_Q1.mean().item(),
            "current_Q2_mean": current_Q2.mean().item(),
            "target_Q_mean": target_Q.mean().item(),
            "scaled_reward_mean": scaled_reward.mean().item(),
            "normalized_reward_mean": normalized_reward.mean().item(),
        }
        
        # Delayed actor update: only update actor and targets every `policy_delay` iterations
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            # Debugging: Check gradient magnitudes
            if DEBUG:
                for name, param in self.actor.named_parameters():
                    if param.grad is not None:
                        print(f"[{self.agent_id}] Actor gradient {name}: {param.grad.abs().mean().item()}")

            self.actor_optimizer.step()
            
            # Soft update target networks
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)
            
            metrics["actor_loss"] = actor_loss.item()
        else:
            metrics["actor_loss"] = None  # Actor not updated this iteration

        return metrics

    
    def soft_update(self, net, target_net):
        """
        Perform Polyak averaging to update the target network:
        target_param = tau * param + (1 - tau) * target_param
        """
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)