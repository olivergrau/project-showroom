# train_maddpg.py
import numpy as np

import os
import time
import random
import json
import torch

from argparse import ArgumentParser
from codebase.maddpg.maddpg_agent import MADDPGAgent
from codebase.experience.replay import MultiAgentReplayBuffer
from codebase.maddpg.trainer import Trainer

# Monkey patch missing attributes for newer numpy versions
if not hasattr(np, "float_"):
    np.float_ = np.float64
    
if not hasattr(np, "int_"):
    np.int_ = np.int64

def setup_args():
    parser = ArgumentParser("MADDPG Training with JSON config overrides")
    parser.add_argument('--config',          type=str,   default=None,
                        help="Path to JSON config file")
    
    parser.add_argument('--env_path',        type=str,   default="Tennis_Linux/Tennis.x86_64",
                        help="Path to Unity env executable")
    parser.add_argument('--log_dir',         type=str,   default="runs/train_maddpg",
                        help="Base dir for TensorBoard logs")
    
    parser.add_argument('--num_agents',      type=int,   default=2)
    parser.add_argument('--obs_size',        type=int,   default=24)
    
    parser.add_argument('--action_size',     type=int,   default=2)
    parser.add_argument('--buffer_size',     type=int,   default=int(1e5))
    parser.add_argument('--batch_size',      type=int,   default=128)
    parser.add_argument('--actor_hidden',    nargs='+', type=int,   default=[128, 128],
                        help="Actor hidden layer sizes")
    parser.add_argument('--critic_hidden',   nargs='+', type=int,   default=[128, 128],
                        help="Critic hidden layer sizes")
    parser.add_argument('--lr_actor',        type=float, default=1e-3)
    parser.add_argument('--lr_critic',       type=float, default=1e-4)
    parser.add_argument('--gamma',           type=float, default=0.99)
    parser.add_argument('--tau',             type=float, default=1e-3)
    parser.add_argument('--use_action_noise',action='store_true', default=False)
    parser.add_argument('--ou_noise_sigma',  type=float, default=0.05)
    parser.add_argument('--ou_noise_theta',  type=float, default=0.15)
    parser.add_argument('--train_every',     type=int,   default=1)
    parser.add_argument('--warmup_steps',    type=int,   default=0)
    parser.add_argument('--n_episodes',      type=int,   default=2000)
    parser.add_argument('--max_steps',       type=int,   default=1000)
    parser.add_argument('--seed',            type=int,   default=0)
    args = parser.parse_args()

    # Override defaults with JSON config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)

    return args

def make_agent_and_buffer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # replay buffer
    buffer = MultiAgentReplayBuffer(
        num_agents  = args.num_agents,
        obs_size    = args.obs_size,
        action_size = args.action_size,
        buffer_size = args.buffer_size,
        batch_size  = args.batch_size,
        seed        = args.seed,
        device      = device
    )

    # MADDPG agent
    agent = MADDPGAgent(
        num_agents       = args.num_agents,
        obs_size         = args.obs_size,
        action_size      = args.action_size,
        actor_hidden     = args.actor_hidden,
        critic_hidden    = args.critic_hidden,
        actor_lr         = args.lr_actor,
        critic_lr        = args.lr_critic,
        tau              = args.tau,
        gamma            = args.gamma,
        device           = device,
        seed             = args.seed,
        use_action_noise = args.use_action_noise,
        ou_noise_sigma   = args.ou_noise_sigma,
        ou_noise_theta   = args.ou_noise_theta,
    )

    return agent, buffer

def main():
    args = setup_args()
    agent, buffer = make_agent_and_buffer(args)

    # timestamped log dir for reproducibility
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, f"{timestamp}_seed{args.seed}")
    os.makedirs(log_dir, exist_ok=True)

    trainer = Trainer(
        env_path      = args.env_path,
        maddpg_agent  = agent,
        replay_buffer = buffer,
        num_agents    = args.num_agents,
        obs_size      = args.obs_size,
        action_size   = args.action_size,
        max_steps     = args.max_steps,
        batch_size    = args.batch_size,
        train_every   = args.train_every,
        warmup_steps  = args.warmup_steps,
        log_dir       = log_dir,
        use_state_norm= False
    )

    ep_scores = trainer.train(n_episodes=args.n_episodes)

    print("Training completed.")
    #print(f"Scores: {ep_scores}")

    # Print the mean of the last 10 episode rewards as a single float
    mean_ep_score = float(np.mean(ep_scores[-10:]))
    print(mean_ep_score)

if __name__ == "__main__":
    main()
