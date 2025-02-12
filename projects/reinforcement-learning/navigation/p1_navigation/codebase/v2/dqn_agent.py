import numpy as np
import random
from collections import namedtuple, deque

from codebase.v2.model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-5               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
ALPHA = 0.6             # prioritization factor
BETA = 0.4              # importance-sampling factor
BETA_FRAMES = 100000    # number of frames over which beta will be annealed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.last_loss = None
        self.last_q_values = None
        self.last_beta = None

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, alpha=ALPHA, beta_start=BETA)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # print device
        print(f"Agent's Device: {device}")
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                loss_value = self.learn(experiences, GAMMA)
                

                self.last_beta = self.memory.current_beta
                self.last_loss = loss_value                

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        
        self.qnetwork_local.train()

        self.last_q_values = action_values.cpu().data.numpy()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(self.last_q_values)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Update value parameters using a batch of experience tuples, incorporating
        importance-sampling (IS) weights for prioritized experience replay.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (states, actions, rewards, 
                next_states, dones, is_weights, indices) tensors from prioritized sampling.
            gamma (float): discount factor
        """
        # Unpack the prioritized batch (now with IS weights and indices)
        states, actions, rewards, next_states, dones, is_weights, indices = experiences
        
        # Compute Q-values expected from the local model for the selected actions.
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute Q-targets for next states using the target network.
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute the element-wise MSE loss (without reduction).
        loss = F.mse_loss(Q_expected, Q_targets, reduction='none')
        
        # Multiply the loss for each sample by its corresponding IS weight.
        weighted_loss = (loss * is_weights).mean()

        # Optimize the network by performing a gradient descent step.
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()
        
        # Soft-update the target network parameters.
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        
        # Compute the new priorities (absolute TD error + a small constant)
        with torch.no_grad():
            td_errors = torch.abs(Q_expected - Q_targets).cpu().numpy().squeeze()
        
        new_priorities = td_errors + 1e-6 # Add small constant to avoid zero priorities
        
        # Update the priorities for these experiences in the replay buffer
        self.memory.update_priorities(indices, new_priorities)

        return weighted_loss.item()
                 
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples with prioritized experience replay and β annealing."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha, beta_start=0.4, beta_frames=100000):
        """Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            beta_start (float): initial value of beta (for IS correction)
            beta_frames (int): number of sample calls over which beta will anneal to 1.0
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)
        
        # Beta annealing parameters
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.alpha = alpha
        self.current_beta = beta_start

        self.sample_calls = 0  # Counter for how many times sample() has been called

        # print out Buffer properties
        print(f"Buffer size: {buffer_size}, Batch size: {batch_size}, Beta start: {beta_start}, Beta frames: {beta_frames}")

    def max_priority(self):
        """Return the maximum priority in the current memory.
        
        Returns:
            float: the maximum priority found; if memory is empty, returns 1.0.
        """
        if len(self.memory) == 0:
            return 1.0
        
        return max(e.priority for e in self.memory if e.priority is not None)
    
    def update_priorities(self, indices, new_priorities):
        """
        Update the priorities of sampled experiences.

        Params:
            indices (list or array): indices of the experiences to update.
            new_priorities (list or array): new priority values for these experiences.
        """
        for idx, priority in zip(indices, new_priorities):
            # Retrieve the current experience.
            e = self.memory[idx]
            
            # Create a new experience with the updated priority.
            updated_experience = self.experience(e.state, e.action, e.reward, e.next_state, e.done, priority)
            
            # Replace the old experience with the updated one.
            self.memory[idx] = updated_experience

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.
        
        If no priority is provided, assign the maximum priority currently in memory.
        """
        priority = self.max_priority()
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory using priorities and annealed β.
        
        Params:
            alpha (float): controls how much prioritization is used (0 means uniform sampling)
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, is_weights, indices)
        """
        # Increment the sample counter and anneal beta accordingly.
        self.sample_calls += 1
        self.current_beta = min(1.0, self.beta_start + self.beta_increment * self.sample_calls)

        # Extract priorities from all experiences.
        priorities = np.array([e.priority for e in self.memory])
        
        # Compute the sampling probabilities.
        prob_alpha = priorities ** self.alpha
        probs = prob_alpha / prob_alpha.sum()
        
        # Sample indices according to the probabilities.
        indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False, p=probs)
        experiences = [self.memory[i] for i in indices]
        
        # Compute importance-sampling (IS) weights:
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.current_beta)
        weights /= weights.max()  # Normalize for stability
        
        # Convert experiences to PyTorch tensors.
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        # Convert weights to a tensor (unsqueeze to match the batch dimension for later loss scaling).
        is_weights = torch.from_numpy(weights).float().unsqueeze(1).to(device)
    
        return (states, actions, rewards, next_states, dones, is_weights, indices)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
