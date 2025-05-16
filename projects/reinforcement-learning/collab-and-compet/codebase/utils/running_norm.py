import numpy as np
import torch

class RunningNormalizer:
    """
    Running z-score normalizer that dynamically handles both numpy arrays and torch tensors.
    Tracks one mean+var per feature (excluding batch dimensions).
    """
    def __init__(self, shape, momentum=0.01, epsilon=1e-8):
        self.momentum = momentum
        self.epsilon  = epsilon
        self.shape    = shape       # e.g. (obs_size,)
        self.count    = 0
        self.mean     = None
        self.var      = None

    def _init_numpy(self):
        self.mean = np.zeros(self.shape, dtype=np.float32)
        self.var  = np.ones(self.shape, dtype=np.float32)

    def _init_torch(self, device):
        self.mean = torch.zeros(self.shape, dtype=torch.float32, device=device)
        self.var  = torch.ones(self.shape, dtype=torch.float32, device=device)

    def update(self, batch):
        # batch can be numpy array or torch tensor of shape (B, 2, obs_size)
        # we want to treat every agent-step as one sample: flatten first two dims
        if isinstance(batch, np.ndarray):
            if self.mean is None or not isinstance(self.mean, np.ndarray):
                self._init_numpy()
            # flatten B x num_agents into N = B*num_agents samples
            flat = batch.reshape(-1, self.shape[0])
            batch_mean = flat.mean(axis=0)
            batch_var  = flat.var( axis=0)
            # EMA update
            self.mean = (1 - self.momentum) * self.mean + self.momentum * batch_mean
            self.var  = (1 - self.momentum) * self.var  + self.momentum * batch_var
            self.count += flat.shape[0]

        elif isinstance(batch, torch.Tensor):
            if self.mean is None or not isinstance(self.mean, torch.Tensor):
                self._init_torch(batch.device)
            elif self.mean.device != batch.device:
                self.mean = self.mean.to(batch.device)
                self.var  = self.var.to(batch.device)
            # flatten
            flat = batch.view(-1, self.shape[0])
            batch_mean = flat.mean(dim=0)
            batch_var  = flat.var( dim=0, unbiased=False)
            self.mean = (1 - self.momentum) * self.mean + self.momentum * batch_mean
            self.var  = (1 - self.momentum) * self.var  + self.momentum * batch_var
            self.count += flat.shape[0]

        else:
            raise TypeError("RunningNormalizer.update expects numpy.ndarray or torch.Tensor")

    def normalize(self, batch):
        if isinstance(batch, np.ndarray):
            if self.mean is None or not isinstance(self.mean, np.ndarray):
                self._init_numpy()
            # broadcast subtract/divide over shape (obs_size,)
            return (batch - self.mean) / (np.sqrt(self.var) + self.epsilon)

        elif isinstance(batch, torch.Tensor):
            if self.mean is None or not isinstance(self.mean, torch.Tensor):
                self._init_torch(batch.device)
            elif self.mean.device != batch.device:
                # bring mean/var onto the same device as batch
                self.mean = self.mean.to(batch.device)
                self.var  = self.var.to(batch.device)
            return (batch - self.mean) / (torch.sqrt(self.var) + self.epsilon)

        else:
            raise TypeError("RunningNormalizer.normalize expects numpy.ndarray or torch.Tensor")
