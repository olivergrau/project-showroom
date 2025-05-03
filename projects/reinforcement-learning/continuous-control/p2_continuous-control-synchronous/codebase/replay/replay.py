import random
import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Args:
            action_size: Integer. Dimension of each action
            buffer_size: Integer. Maximum size of buffer
            batch_size: Integer. Size of each training batch
            seed: Integer. Random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self._rng = random.Random(seed)
        self.device = device

    def __str__(self):
        return 'ReplayBuffer_class'

    def __repr__(self):
        return 'ReplayBuffer_class'

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def size(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.

        Args:
            state: The previous state of the environment
            action: Integer. Previous action selected by the agent
            reward: Float. Reward value
            next_state: The current state of the environment
            done: Boolean. Whether the episode is complete
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = self._rng.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(self.device)

        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).float().to(self.device)

        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(self.device)

        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(self.device)

        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(self.device)

        return states, actions, rewards, next_states, dones

