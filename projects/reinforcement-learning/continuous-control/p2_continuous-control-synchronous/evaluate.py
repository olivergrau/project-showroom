import numpy as np
import torch
from codebase.ddpg.agent import DDPGAgent
from unityagents import UnityEnvironment

def evaluate_ddpg(env_path: str,
                  weights_path: str,
                  n_episodes: int = 100,
                  no_graphics: bool = True) -> list:
    """
    Run a trained DDPGAgent for n_episodes in the Unity Reacher env and
    return a list of per-episode average scores (over all agents).
    """
    # 1) Launch the Unity environment
    env = UnityEnvironment(file_name=env_path, no_graphics=no_graphics)
    env.reset()
    brain_name = env.brain_names[0]

    # 2) Instantiate the agent with the same hyperparameters you used for training
    agent = DDPGAgent(
        num_agents=20,
        state_size=33,
        action_size=4,
        actor_input_size=256,     # match your training config
        actor_hidden_size=128,
        critic_input_size=256,
        critic_hidden_size=128,
        lr_actor=1e-4,
        lr_critic=1e-4,
        gamma=0.99,
        tau=1e-3,
        use_ou_noise=False,       # no exploration during evaluation
        seed=0
    )
    agent.actor.eval()           # put networks in eval mode
    agent.critic.eval()

    # 3) Load the saved weights
    checkpoint = torch.load(weights_path, map_location=agent.device, weights_only=True)
    for net in ['actor', 'actor_target', 'critic', 'critic_target']:
        getattr(agent, net).load_state_dict(checkpoint[net])

    # 4) Run n_episodes and collect scores
    scores = []
    for ep in range(1, n_episodes + 1):
        # reset env in inference mode
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations       # shape: (20, 33)
        episode_scores = np.zeros(agent.num_agents)

        done = False
        while not done:
            # get actions from deterministic policy (eval=True ⇒ no OU noise) :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
            actions, _ = agent.act(states, eval=True)
            env_info = env.step(actions)[brain_name]
            states = env_info.vector_observations
            rewards = np.array(env_info.rewards)
            dones   = np.array(env_info.local_done)

            episode_scores += rewards
            done = dones.any()   # stop when any agent signals done

        # average over the 20 agents
        avg_score = episode_scores.mean()
        scores.append(avg_score)
        print(f"Episode {ep:3d}  Avg Score: {avg_score:.2f}")

    env.close()
    return scores

if __name__ == "__main__":
    ENV_PATH     = "Reacher_Linux/Reacher.x86_64"
    WEIGHTS_PATH = "saved_weights/train_ddpg/ddpg_weights_ep_40_2025-05-03_21-00-17.pth"   # path to your saved weights dict

    # This is insane! Outrageous! 100 episodes in a row? The folks at Udacity must be all masochist! 20 episodes is enough to get a good idea of the performance of my agent.
    N_EPISODES   = 100 

    scores = evaluate_ddpg(ENV_PATH, WEIGHTS_PATH, n_episodes=N_EPISODES, no_graphics=True)