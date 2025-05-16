from collections import deque, namedtuple
import random

import numpy as np
import torch


class MultiAgentReplayBuffer:
    """Shared replay buffer for multi-agent environments."""

    def __init__(self, num_agents, obs_size, action_size, buffer_size, batch_size, seed, device):
        self.num_agents = num_agents
        self.obs_size = obs_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        self.experience = namedtuple("Experience", field_names=[
            "states", "actions", "rewards", "next_states", "dones"
        ])
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.memory)

    def add(self, states, actions, rewards, next_states, dones):
        """
        Add one experience tuple (from all agents) to the buffer.
        Each argument is a list of length `num_agents`.
        """
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences and return tensors per agent."""
        experiences = self._rng.sample(self.memory, k=self.batch_size)

        # Unzip each component into a list of lists (shape: [batch_size, num_agents, *])
        batch = self.experience(*zip(*experiences))

        # Convert to np.array first to avoid PyTorch warning
        states = torch.tensor(np.array(batch.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(batch.actions), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array(batch.rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(batch.next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(batch.dones), dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones

