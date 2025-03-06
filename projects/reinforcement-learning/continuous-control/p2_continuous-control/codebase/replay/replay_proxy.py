from codebase.replay.replay_buffer import ReplayWrapper, Transition  # Replay buffer implementation

class ReplayProxy:
    def __init__(self, conn):
        self.conn = conn
        self.cache = None

    def feed(self, exp):
        self.conn.send([ReplayWrapper.FEED, exp])

    def sample(self):
        self.conn.send([ReplayWrapper.SAMPLE, None])
        cache_id, data = self.conn.recv()

        # If data is None, then use the previously cached value. (No new data available)
        if data is None:
            if self.cache is None:
                raise RuntimeError("No cached data available for sampling.")
            data = self.cache
        else:
            # Update cache with the new data
            self.cache = data
        
        # Return the expected Transition namedtuple using the data from the cache.
        return Transition(*data[cache_id])

# (Add other methods if needed, e.g., update_priorities)
