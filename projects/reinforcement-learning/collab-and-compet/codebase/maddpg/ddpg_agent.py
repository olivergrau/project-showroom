import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from codebase.maddpg.net.actor_critic import Actor, Critic

class DDPGAgent:
    """
    A per-agent DDPG wrapper used inside a MADDPG setting,
    with optional debug/telemetry instrumentation.
    """
    def __init__(self,
                 agent_id: int,
                 obs_size: int,
                 action_size: int,
                 full_obs_size: int,
                 full_action_size: int,
                 actor_hidden=[128, 128],
                 critic_hidden=[256, 256],
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 tau=1e-3,
                 gamma=0.99,
                 actor_use_layer_norm=False,
                 critic_use_batch_norm=False,
                 ou_noise_sigma=0.2,
                 ou_noise_theta=0.15,
                 seed=0,
                 device='cpu',
                 debug: bool = False):
        
        self.agent_id               = agent_id
        self.obs_size               = obs_size
        self.action_size            = action_size
        self.full_obs_size          = full_obs_size
        self.full_action_size       = full_action_size
        self.tau                    = tau
        self.gamma                  = gamma
        self.device                 = torch.device(device)
        self.debug                  = debug

        # print out hyperparameters nicely
        print(f"[Agent {self.agent_id}] Hyperparameters:")
        print(f"  - actor_hidden: {actor_hidden}")
        print(f"  - critic_hidden: {critic_hidden}")
        print(f"  - actor_lr: {actor_lr}")
        print(f"  - critic_lr: {critic_lr}")
        print(f"  - tau: {tau}")
        print(f"  - gamma: {gamma}")
        print(f"  - actor_use_layer_norm: {actor_use_layer_norm}")
        print(f"  - critic_use_batch_norm: {critic_use_batch_norm}")
        print(f"  - ou_noise_sigma: {ou_noise_sigma}")
        print(f"  - ou_noise_theta: {ou_noise_theta}")
        print(f"  - seed: {seed}")
        print(f"  - device: {self.device}")
        print(f"  - debug: {debug}")

        # ——— Actor and optimizer ———
        self.actor = Actor(obs_size, action_size, actor_hidden, seed,
                           use_layer_norm=actor_use_layer_norm).to(self.device)
        
        self.actor_target = Actor(obs_size, action_size, actor_hidden, seed,
                                  use_layer_norm=actor_use_layer_norm).to(self.device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # ——— Critic and optimizer ———
        self.critic = Critic(full_obs_size, full_action_size, critic_hidden, seed,
                             use_batch_norm=critic_use_batch_norm, dropout_p=0.0).to(self.device)
        
        self.critic_target = Critic(full_obs_size, full_action_size, critic_hidden, seed,
                                    use_batch_norm=critic_use_batch_norm, dropout_p=0.0).to(self.device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=0.0)

        self.critic_target.eval()
        self.actor_target.eval()
        
        # ——— Exploration noise ———
        self.noise = OUNoise(action_size, seed, sigma=ou_noise_sigma, theta=ou_noise_theta)


    def act(self, obs, eval: bool = False):
        """Get action + optional OU noise."""
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval()
        
        with torch.no_grad():
            base_action = self.actor(x).squeeze(0).cpu().numpy()
        
        self.actor.train()

        if eval:
            return np.clip(base_action, -1, 1), None

        noise = self.noise.sample()
        
        return np.clip(base_action + noise, -1, 1), noise

    def reset(self):
        self.noise.reset(randomize=True)

    def soft_update(self, local_model, target_model):
        """Polyak‐averaging update of target network."""
        for tp, lp in zip(target_model.parameters(), local_model.parameters()):
            tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)

    def learn(self,
              obs_batch, rewards, dones,
              obs_all, actions_all,
              next_obs_all, next_actions_all):
        """
        Perform a single DDPG update:
          1) critic step
          2) actor step
          3) soft‐update targets
        Returns a dict of scalar metrics.
        """

        # ——— 1. Critic update ———
        with torch.no_grad():
            q_next = self.critic_target(next_obs_all, next_actions_all) # global values for critic
            q_target = rewards + self.gamma * q_next * (1 - dones)

        q_expected = self.critic(obs_all, actions_all)
        critic_loss = nn.MSELoss()(q_expected, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # optional gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

        self.critic_optimizer.step()

        # ——— 2. Actor update ———
        pred_action = self.actor(obs_batch) # only this agent’s obs
        actions_all_pred = self._integrate_actor_pred_actions(actions_all, pred_action)

        actor_loss = -self.critic(obs_all, actions_all_pred).mean()
        
        # optional telemetry before backward
        #if self.debug:
        #    self._telemetry_before_actor(obs_batch, actor_loss)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)

        grad_norm = self._compute_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        # ——— 3. Soft‐update target nets ———
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor,  self.actor_target)

        # ——— 4. Telemetry after update ———
        dqda_mean = None
        if self.debug:
            dqda_mean = self._compute_dqda(obs_all, actions_all, pred_action)

        metrics = {
            'critic_loss': critic_loss.item(),
            'actor_loss':  actor_loss.item(),
            'q_expected':  q_expected.mean().item(),
            'q_target':    q_target.mean().item(),
            'grad_norm':   grad_norm,
            'actions':     pred_action.detach().cpu().numpy(), # is the detach ok here?
        }

        if dqda_mean is not None:
            metrics['dqda_mean'] = dqda_mean

        return metrics

    # ——— Helpers ———

    def _integrate_actor_pred_actions(self, actions_all, my_pred):
        """Re‐insert this agent’s newly predicted actions into the joint action vector."""

        start, end = self.agent_id*self.action_size, (self.agent_id+1)*self.action_size
        
        return torch.cat([
            actions_all[:, :start],
            my_pred,
            actions_all[:, end:]
        ], dim=1)

    def _compute_grad_norm(self, parameters):
        """L2‐norm of all actor gradients."""
        total = 0.0
        for p in parameters:
            if p.grad is not None:
                total += p.grad.norm(2).item()**2
        
        return total**0.5    

    def _telemetry_before_actor(self, obs_batch, actor_loss):
        """Optionally log or print actor‐output stats before backprop."""
        with torch.no_grad():
            out = self.actor(obs_batch)
            mi, ma, st = out.min().item(), out.max().item(), out.std().item()
        
        print(f"[Agent {self.agent_id}] BEFORE actor backward → "
              f"out=[{mi:.4f},{ma:.4f}], std={st:.4f}, loss={actor_loss.item():.6f}")

    def _compute_dqda(self, obs_all, actions_all, pred_action):
        """Recompute ∂Q/∂a for this agent’s action dims only."""
        watched = pred_action.clone().detach().requires_grad_(True)
        full = actions_all.clone().detach()
        start, end = self.agent_id*self.action_size, (self.agent_id+1)*self.action_size
        full[:, start:end] = watched

        q = self.critic(obs_all, full).mean()
        dqda = torch.autograd.grad(q, watched, retain_graph=False)[0]
        
        return dqda.abs().mean().item()


class OUNoise:
    """Ornstein–Uhlenbeck noise for continuous actions."""
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        self.mu    = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.rng   = np.random.RandomState(seed)
        self.size  = size
        self.state = None
        self.reset()

        # print out hyperparameters nicely
        print(f"[OUNoise] Hyperparameters:")
        print(f"  - mu: {mu}")
        print(f"  - theta: {theta}")
        print(f"  - sigma: {sigma}")
        print(f"  - seed: {seed}")
        print(f"  - size: {size}")
        print(f"[OUNoise] Initial state: {self.state}")

    def reset(self, randomize=False):
        self.state = self.mu.copy()

        if randomize:
            self.state = np.random.uniform(-1.0, 1.0, size=self.size)


    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * self.rng.randn(len(self.state))
        self.state += dx
        
        return self.state
