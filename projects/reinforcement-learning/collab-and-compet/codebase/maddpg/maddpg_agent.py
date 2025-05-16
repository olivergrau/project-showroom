import torch
from codebase.maddpg.ddpg_agent import DDPGAgent

class MADDPGAgent:
    """
    Multi-Agent DDPG Coordinator for Unity Tennis (2 agents).
    """

    def __init__(self,
                 num_agents,
                 obs_size,
                 action_size,
                 actor_hidden=[128, 128],
                 critic_hidden=[128, 128],
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 gamma=0.99,
                 tau=1e-3,
                 device='cpu',
                 seed=0,
                 use_action_noise=False,
                 noise_type: str = "gaussian", # ou or gaussian
                 noise_params: dict = None,    # 
                ):
        
        self.num_agents = num_agents
        self.obs_size = obs_size
        self.action_size = action_size
        self.device = torch.device(device)
        self.gamma = gamma
        self.use_action_noise = use_action_noise
        self.training_logs = [[] for _ in range(num_agents)]
        self.noise_type = noise_type
        self.noise_params = noise_params

        # print out hyperparameters nicely
        print(f"[MADDPG] Hyperparameters:")
        print(f"  - num_agents: {num_agents}")
        print(f"  - obs_size: {obs_size}")
        print(f"  - action_size: {action_size}")
        print(f"  - actor_hidden: {actor_hidden}")
        print(f"  - critic_hidden: {critic_hidden}")
        print(f"  - actor_lr: {actor_lr}")
        print(f"  - critic_lr: {critic_lr}")
        print(f"  - tau: {tau}")
        print(f"  - gamma: {gamma}")
        print(f"  - device: {self.device}")
        print(f"  - use_action_noise: {use_action_noise}")
        print(f"  - noise_type: {noise_type}")
        print(f"  - noise_params: {noise_params}")
        print(f"  - seed: {seed}")

        # Each agent sees its own obs & acts, but uses global info in critic
        self.agents = [
            DDPGAgent(
                agent_id=i,
                obs_size=obs_size,
                action_size=action_size,
                full_obs_size=obs_size * num_agents,
                full_action_size=action_size * num_agents,
                actor_hidden=actor_hidden,
                critic_hidden=critic_hidden,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                tau=tau,
                gamma=gamma,
                device=device,
                seed=seed + i,  
                critic_use_layer_norm=True, # without batch norm, training is unstable actions are immediately saturated
                actor_use_layer_norm=True,                 
                noise_type=noise_type,
                noise_params=noise_params,      
                
                debug=True                            
            )
            for i in range(num_agents)
        ]

    def act(self, states, eval=False):
        """Get actions from all agents for a given list of local observations."""
        actions = []
        noises = []
        for i, agent in enumerate(self.agents): # for each agent i select action a_i
            action, noise = agent.act(states[i], eval=eval)  # (1, action_size)
            actions.append(action)
            noises.append(noise)  # (1, action_size)
        
        return actions, noises  # list of np arrays

    def reset(self):
        for agent in self.agents:
            agent.reset()
    
    def step(self, samples):
        """
        Perform a learning step given a batch of experience from the buffer.
        Args:
            samples: tuple of tensors from replay buffer
                - states:      (B, num_agents, obs_size)
                - actions:     (B, num_agents, action_size)
                - rewards:     (B, num_agents)
                - next_states: (B, num_agents, obs_size)
                - dones:       (B, num_agents)
        """
        states, actions, rewards, next_states, dones = samples
        B = states.shape[0]

        # 1) Flatten for critic
        obs_all        = states.view(B, -1) # remove agent dim
        actions_all    = actions.view(B, -1) # remove agent dim
        next_obs_all   = next_states.view(B, -1) # remove agent dim

        # 2) Compute next_actions_all with small target noise
        with torch.no_grad():
            next_actions_all = []

            for i, agent in enumerate(self.agents):
                a_next = agent.actor_target(next_states[:, i, :])  # (B, A)

                if self.use_action_noise:
                    # TD3-style target policy smoothing
                    noise_std = 0.1
                    noise_clip = 0.3
                    noise = torch.clamp(
                        torch.randn_like(a_next) * noise_std,
                        -noise_clip, noise_clip
                    )
                    a_next = torch.clamp(a_next + noise, -1.0, 1.0)

                next_actions_all.append(a_next)

            next_actions_all = torch.cat(next_actions_all, dim=-1)  # (B, num_agents × A)

        # 3) Let each agent learn
        for i, agent in enumerate(self.agents):
            metrics = agent.learn(
                obs_batch=    states[:, i, :],           # (B, obs_size)
                rewards=      rewards[:, i].unsqueeze(-1),
                dones=        dones[:, i].unsqueeze(-1),
                obs_all=      obs_all,                   # (B, full_obs_size)
                actions_all=  actions_all,
                next_obs_all= next_obs_all,
                next_actions_all= next_actions_all
            )
            self.training_logs[i].append(metrics)
