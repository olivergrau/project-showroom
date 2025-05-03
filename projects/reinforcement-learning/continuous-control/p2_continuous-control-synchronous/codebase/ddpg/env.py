import time
import subprocess
import traceback
import numpy as np
from unityagents import UnityEnvironment
import gc

class BootstrappedEnvironment:
    def __init__(
            self, 
            exe_path, 
            worker_id=0, 
            use_graphics=False, 
            preprocess_fn=None, 
            max_retries=5, 
            retry_delay=2, 
            reward_shaping_fn=None):
        """
        Initialize the Unity environment wrapper.
        """
        self.exe_path = exe_path
        self.worker_id = worker_id
        self.use_graphics = use_graphics
        self.preprocess_fn = preprocess_fn
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._closed = False
        self.reward_shaping_fn = reward_shaping_fn
        self.total_steps = 0

        print(f"[BootstrappedEnvironment] Initializing Unity environment with exe_path: {self.exe_path}, worker_id: {self.worker_id}, use_graphics: {self.use_graphics}")
        print(f"[BootstrappedEnvironment] Preprocess function: {self.preprocess_fn}, Max retries: {self.max_retries}, Retry delay: {self.retry_delay}")
        print(f"[BootstrappedEnvironment] Reward shaping function: {self.reward_shaping_fn}")

        try:
            self.env = UnityEnvironment(file_name=self.exe_path, no_graphics=not self.use_graphics, worker_id=self.worker_id)
        except Exception as e:
            print(f"[BootstrappedEnvironment] Failed to initialize UnityEnvironment (worker_id: {self.worker_id}): {e}")
            self.env = None
            raise

        self.brain_name = self.env.brain_names[0]

    def reset(self, train_mode=True):
        attempt = 0
        while attempt < self.max_retries:
            try:
                env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
                raw_state = env_info.vector_observations
                
                if self.preprocess_fn:
                    return self.preprocess_fn(raw_state)
                
                return raw_state
            except Exception as e:
                print(f"[BootstrappedEnvironment] Error during reset: {e}. Attempt {attempt + 1}/{self.max_retries}. Retrying in {self.retry_delay} seconds...")
                attempt += 1
                time.sleep(self.retry_delay)
        
        self.close()
        raise RuntimeError("Failed to reset Unity environment after maximum retries.")

    def step(self, actions):
        try:
            env_info = self.env.step(actions)[self.brain_name]
            raw_next_state = env_info.vector_observations
            env_reward = np.array(env_info.rewards, dtype=np.float32)
            done = env_info.local_done
            
            # Optional preprocessing of the next state
            if self.preprocess_fn:
                next_state = self.preprocess_fn(raw_next_state)
            else:
                next_state = raw_next_state
            
            g_norm = None

            # Apply reward shaping if provided
            if self.reward_shaping_fn is not None:
                reward, shaped, g_norm = self.reward_shaping_fn(next_state, env_reward, self.total_steps)                
            else:
                reward = env_reward

            self.total_steps += 1

            return next_state, (reward, env_reward, shaped if "shaped" in locals() else None, g_norm) , done
        except Exception as e:
            print(f"[BootstrappedEnvironment] Error during step: {e}. Attempting to close environment.")
            traceback.print_exc()
            self.close()
            raise

    def close(self):
        """
        Close the Unity environment if not already closed, and then attempt to wipe lingering processes.
        """
        if self.env is not None and not self._closed:
            try:
                self.env.close()
                self._closed = True
                self.env = None
                                
                gc.collect()

                print("[BootstrappedEnvironment] Waiting 5 seconds for closing the environment...")
                time.sleep(5)
                
                self._wipe_unity_processes()
                print("[BootstrappedEnvironment] Unity environment closed and processes wiped successfully.")
            except Exception as e:
                print(f"[BootstrappedEnvironment] Error while closing Unity environment: {e}")

    def _wipe_unity_processes(self):
        """
        Attempts to kill lingering Unity/Reacher processes from the OS.
        This method uses the 'pkill' command which is Linux-specific.
        """
        try:
            # Using pkill with the exe_path should kill any process that was started with that executable.
            subprocess.call(["pkill", "-f", self.exe_path])
            time.sleep(5)
            print("[BootstrappedEnvironment] Successfully wiped Unity processes from OS.")
        except Exception as e:
            print(f"[BootstrappedEnvironment] Error while wiping Unity processes: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()
