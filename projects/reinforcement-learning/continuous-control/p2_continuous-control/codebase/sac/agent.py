import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from codebase.sac.net.actor import Actor  # our SAC actor network
from codebase.sac.net.critic import Critic  # our SAC critic network

DEBUG = True

class SACAgent:
    def __init__(
        self,
        state_size=33,
        action_size=4,
        lr_actor=1e-3,
        lr_critic=1e-3,
        lr_alpha=1e-3,
        gamma=0.99,
        tau=0.005,
        target_entropy=None,
        device=None,
        label="SACAgent",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent_id = "SACAgent (" + label + ")"
        self.total_it = 0

        print(f"\n{self.agent_id}: Using device: {self.device}")
        print(f"{self.agent_id}: state_size={state_size}, action_size={action_size}, "
              f"lr_actor={lr_actor}, lr_critic={lr_critic}, gamma={gamma}, tau={tau}")

        # Initialize SAC Actor network and its optimizer
        self.actor = Actor(state_size, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Initialize SAC Critic network and its target network
        self.critic = Critic(state_size, action_size).to(self.device)
        self.critic_target = Critic(state_size, action_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Temperature parameter for entropy regularization.
        # We learn log_alpha to ensure alpha is always positive.
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

        # Set target entropy (default: -action_size)
        self.target_entropy = target_entropy if target_entropy is not None else -action_size

    def act(self, state, evaluate=False):
        """
        Given a state, select an action.
        When evaluate=True, returns a deterministic action (tanh(mean)).
        Otherwise, samples stochastically from the policy.
        """
        state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor(state)
                action = torch.tanh(mean)

            return action.clamp(-1, 1).detach().cpu().numpy()
        else:
            action, _, _ = self.actor.sample(state)

            return action.clamp(-1, 1).detach().cpu().numpy()

    def learn(self, batch):
        """
        Update the SAC agent based on a batch of transitions.
        The batch is expected to be a namedtuple or similar structure containing:
          - state: shape [batch_size, state_size]
          - action: shape [batch_size, action_size]
          - reward: shape [batch_size]
          - next_state: shape [batch_size, state_size]
          - mask: shape [batch_size] (1 if not done, 0 if done)
        Returns a dictionary of metrics for logging.
        """
        self.total_it += 1

        # Unpack the batch and move to the appropriate device.
        state = batch.state.to(self.device)
        action = batch.action.to(self.device)
        reward = batch.reward.unsqueeze(1).to(self.device)
        next_state = batch.next_state.to(self.device)
        mask = batch.mask.unsqueeze(1).to(self.device)

        # ----------------------
        # Critic Update
        # ----------------------
        with torch.no_grad():
            # Sample action for next state and compute its log probability.
            next_action, next_log_prob, _ = self.actor.sample(next_state)

            # Evaluate target Q-values using the target critic network.
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            
            # Compute the entropy-adjusted target value.
            alpha = self.log_alpha.exp()
            target_value = target_Q - alpha * next_log_prob
            
            # Compute the final target.
            target = reward + mask * self.gamma * target_value

        # Current Q estimates from the critic.
        current_Q1, current_Q2 = self.critic(state, action)
        critic1_loss = F.mse_loss(current_Q1, target)
        critic2_loss = F.mse_loss(current_Q2, target)
        critic_loss = critic1_loss + critic2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------------
        # Actor Update
        # ----------------------
        # Sample actions from the current policy.
        sampled_action, log_prob, _ = self.actor.sample(state)

        # Evaluate the Q-value for these actions.
        Q1, Q2 = self.critic(state, sampled_action)
        Q = torch.min(Q1, Q2)
        
        # Actor loss: minimize (alpha * log_prob - Q)
        actor_loss = (alpha * log_prob - Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------
        # Temperature (alpha) Update
        # ----------------------
        # The loss to adjust the temperature parameter alpha.
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ----------------------
        # Soft Update of Critic Target Networks
        # ----------------------
        self.soft_update(self.critic, self.critic_target)

        metrics = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": alpha.item(),
            "current_Q1_mean": current_Q1.mean().item(),
            "current_Q2_mean": current_Q2.mean().item(),
            "target_Q_mean": target.mean().item(),
            "log_prob_mean": log_prob.mean().item(),
        }
        return metrics

    def soft_update(self, net, target_net):
        """
        Perform Polyak averaging to update the target network:
        target_param = tau * param + (1 - tau) * target_param
        """
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
