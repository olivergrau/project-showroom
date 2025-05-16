# noise.py
import abc
import numpy as np

class Noise(abc.ABC):
    """Abstract noise interface."""
    @abc.abstractmethod
    def reset(self, randomize: bool = False):
        """Reset any internal state (called at episode start)."""
        pass

    @abc.abstractmethod
    def sample(self) -> np.ndarray:
        """Draw one noise sample (shape = action dim)."""
        pass


class OUNoise(Noise):
    """Ornstein–Uhlenbeck noise process."""
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        self.mu    = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size  = size
        self.rng   = np.random.RandomState(seed)
        self.state = None
        self.reset()

        print(f"[OUNoise] mu={mu}, theta={theta}, sigma={sigma}, seed={seed}, size={size}")

    def reset(self, randomize: bool = False):
        self.state = self.mu.copy()
        if randomize:
            self.state = np.random.uniform(-1.0, 1.0, size=self.size)

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) \
             + self.sigma * self.rng.randn(self.size)
        self.state += dx
        return self.state


class GaussianNoise(Noise):
    """Zero-mean Gaussian (i.i.d.) noise."""
    def __init__(self, size, seed=None, mu=0.0, sigma=0.2):
        self.mu    = mu
        self.sigma = sigma
        self.size  = size
        self.rng   = np.random.RandomState(seed) if seed is not None else np.random
        print(f"[GaussianNoise] mu={mu}, sigma={sigma}, seed={seed}, size={size}")

    def reset(self, randomize: bool = False):
        # no persistent state
        pass

    def sample(self) -> np.ndarray:
        return self.rng.normal(self.mu, self.sigma, size=self.size)
