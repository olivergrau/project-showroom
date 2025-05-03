import os
import time
import numpy as np
import torch
import traceback
from dataclasses import dataclass, field
from typing import Optional, Protocol, Tuple
from torch.utils.tensorboard import SummaryWriter
from codebase.ddpg.agent import DDPGAgent
from codebase.ddpg.env import BootstrappedEnvironment
from codebase.ddpg.eval import evaluate
from codebase.replay.replay import ReplayBuffer
from codebase.utils.normalizer import RunningNormalizer
from codebase.utils.early_stopping import EarlyStopping
from optuna.exceptions import TrialPruned

class VelocityRewardShaper:
    def __init__(self, 
                 alpha: float = 1.0, 
                 momentum: float = 0.99, 
                 initial_baseline: float = 0.05,
                 decay_alpha_from: Optional[float] = None,
                 decay_alpha_to:   Optional[float] = None,
                 decay_alpha_until: Optional[int]  = None,
                 env_positive_boost: float = 0.3,
                 slowdown_weight:    float = 0.1,
                 slowdown_dist_thr:  float = 0.3,
                 proximity_weight:   float = 0.2):
        """
        alpha               – initial blending weight for shaping
        decay_alpha_from    – start value for alpha decay
        decay_alpha_to      – final value for alpha decay
        decay_alpha_until   – number of steps over which to decay alpha
        env_positive_boost  – bonus shaping when env_reward > 0
        slowdown_weight     – penalty for moving fast when near the goal
        slowdown_dist_thr   – distance threshold under which to start slowing
        proximity_weight    – bonus proportional to (1 - normalized distance)
        """
        self.alpha = alpha
        self.alpha_init = decay_alpha_from or alpha
        self.alpha_target = decay_alpha_to or alpha
        self.alpha_decay_until = decay_alpha_until

        self.momentum = momentum
        self.baseline = initial_baseline

        self.env_positive_boost = env_positive_boost
        self.slowdown_weight    = slowdown_weight
        self.slowdown_dist_thr  = slowdown_dist_thr
        self.proximity_weight   = proximity_weight

        # for logging
        self.last_goal_distance = None

    def _update_alpha(self, total_steps: int):
        if self.alpha_decay_until and total_steps < self.alpha_decay_until:
            frac = total_steps / self.alpha_decay_until
            self.alpha = (
                self.alpha_target +
                (self.alpha_init - self.alpha_target) * (1.0 - frac)
            )
        elif self.alpha_decay_until:
            self.alpha = self.alpha_target

    def __call__(self,
                 next_state: np.ndarray,
                 reward:     np.ndarray,
                 total_steps: Optional[int] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
        # 0) decay alpha if configured
        if total_steps is not None:
            self._update_alpha(total_steps)

        # 1) extract velocity & goal-vector
        v      = next_state[:, 23:26]
        goal   = next_state[:, 26:29]
        v_norm = np.linalg.norm(v, axis=1) + 1e-8
        g_norm = np.linalg.norm(goal, axis=1) + 1e-8
        cos_sim = np.sum(v * goal, axis=1) / (v_norm * g_norm)

        # log for distance diagnostics
        self.last_goal_distance = g_norm

        # 2) base shaping: encourage movement toward goal
        shaped = cos_sim * v_norm

        # 3) boost when env reward > 0 (partial or full goal contact)
        shaped += (reward > 0).astype(np.float64) * self.env_positive_boost

        # 4) slowdown penalty: discourage high speed when near the goal
        near_goal = (g_norm < self.slowdown_dist_thr).astype(np.float64)
        slowdown_penalty = v_norm * near_goal * self.slowdown_weight
        shaped -= slowdown_penalty

        # 5) proximity bonus: reward being closer to the goal
        #    (we normalize distance by a rough max of 1.0)
        proximity_term = (1.0 - np.clip(g_norm, 0.0, 1.0))
        shaped += proximity_term * self.proximity_weight

        # 6) update moving baseline from env reward
        reward_max = np.max(reward) + 1e-6
        self.baseline = (
            self.momentum * self.baseline +
            (1 - self.momentum) * reward_max
        )

        # 7) scale shaped to match baseline & current alpha
        shaped_peak = np.max(np.abs(shaped)) + 1e-6
        scaled_shaped = self.alpha * shaped / shaped_peak * self.baseline

        # return combined (env + shaped) and the shaping term
        return reward + scaled_shaped, scaled_shaped, g_norm
    
# Example usage:
# shaper = VelocityRewardShaper(momentum=0.995, initial_baseline=0.05, 
#                           alpha=3.5, decay_alpha_from=3.5, 
#                           decay_alpha_to=0.5, decay_alpha_until=50000,
#                           env_positive_boost=1.0,
#                           slowdown_weight = 0.1,
#                           slowdown_dist_thr = 0.3,
#                           proximity_weight = 0.2)