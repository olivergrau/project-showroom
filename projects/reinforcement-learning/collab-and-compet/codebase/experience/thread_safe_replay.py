import threading

from codebase.experience.replay import MultiAgentReplayBuffer

class ThreadSafeReplayBuffer(MultiAgentReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()

    def add(self, *args, **kwargs):
        with self._lock:
            super().add(*args, **kwargs)

    def sample(self):
        with self._lock:
            return super().sample()
