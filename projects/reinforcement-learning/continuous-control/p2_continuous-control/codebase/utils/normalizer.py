import numpy as np


class RunningNormalizer:
    def __init__(self, shape, momentum=0.001, epsilon=1e-8):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.momentum = momentum
        self.epsilon = epsilon
        self.count = 0

    def update(self, batch):
        # Compute batch statistics:
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]
        
        # Update running mean and variance with exponential moving average:
        self.mean = (1 - self.momentum) * self.mean + self.momentum * batch_mean
        self.var = (1 - self.momentum) * self.var + self.momentum * batch_var
        self.count += batch_count

    def normalize(self, batch):
        return (batch - self.mean) / (np.sqrt(self.var) + self.epsilon)
