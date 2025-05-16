import numpy as np # important for multiprocessing -> import as first

# Monkey patch missing attributes for newer numpy versions
if not hasattr(np, "float_"):
    np.float_ = np.float64
    
if not hasattr(np, "int_"):
    np.int_ = np.int64

import json
import subprocess
import optuna

def objective_noise_search(trial):
    # 1) Sample only the noise hyperparameters
    ou_theta = trial.suggest_float('ou_noise_theta', 0.05, 0.3)
    ou_sigma = trial.suggest_float('ou_noise_sigma', 0.05, 0.3)
    seed     = 1000 + trial.number

    # 2) Fixed hyperparameters
    config = {
        "lr_actor":       1e-4,
        "lr_critic":      1e-3,
        "gamma":          0.99,
        "tau":            0.01,
        "actor_hidden":   [128, 128],
        "critic_hidden":  [128, 128],
        "ou_noise_theta": ou_theta,
        "ou_noise_sigma": ou_sigma,
        "seed":           seed,
        "n_episodes":     200,
        "max_steps":      500
    }

    # 3) Write trial config to disk
    cfg_path = f"./trial_configs/noise_trial_{trial.number}.json"
    with open(cfg_path, 'w') as f:
        json.dump(config, f)

    # 4) Launch training as subprocess
    cmd = ["python", "train_maddpg.py", "--config", cfg_path]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=600,
            env=None  # or pass customised env if needed
        )
    except subprocess.CalledProcessError as e:
        print(f"[Trial {trial.number}] Error:\n{e.stderr}")
        return float("nan")
    except subprocess.TimeoutExpired:
        print(f"[Trial {trial.number}] Timeout")
        return float("nan")

    # 5) Parse last line of stdout as the reward
    out_lines = proc.stdout.strip().splitlines()
    try:
        return float(out_lines[-1])
    except ValueError:
        print(f"[Trial {trial.number}] Couldn't parse reward from: '{out_lines[-1]}'")
        return float("nan")
    
def objective_default(trial):
    # 1) Sample hyperparameters
    lr_actor = trial.suggest_float('lr_actor',  1e-5,   1e-3, log=True)
    lr_critic= trial.suggest_float('lr_critic', 1e-5,   1e-3, log=True)
    gamma    = trial.suggest_float('gamma',     0.90,   0.995)
    tau      = trial.suggest_float('tau',       1e-4,   5e-3)
    h        = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    ou_theta = trial.suggest_float('ou_noise_theta', 0.05, 0.3)
    ou_sigma = trial.suggest_float('ou_noise_sigma', 0.05, 0.3)
    seed     = 1000 + trial.number

    # 2) Write trial-specific config
    config = {
        "lr_actor":       lr_actor,
        "lr_critic":      lr_critic,
        "gamma":          gamma,
        "tau":            tau,
        "actor_hidden":   [h, h],
        "critic_hidden":  [h, h],
        "ou_noise_sigma": ou_sigma,
        "ou_noise_theta": ou_theta,
        "seed":           seed,
        "n_episodes":     250,
        "max_steps":      500
    }
    cfg_path = f"./trial_configs/trial_config_{trial.number}.json"
    with open(cfg_path, 'w') as f:
        json.dump(config, f)

    # 3) Launch train_maddpg.py as a separate process
    cmd = ["python", "train_maddpg.py", "--config", cfg_path]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=600
        )
    except subprocess.CalledProcessError as e:
        print(f"[Trial {trial.number}] Error:\n{e.stderr}")
        return float("nan")
    except subprocess.TimeoutExpired:
        print(f"[Trial {trial.number}] Timeout")
        return float("nan")

    # 4) Parse the last line of stdout as the objective
    out_lines = proc.stdout.strip().splitlines()
    last = out_lines[-1]
    
    try:
        return float(last)
    except ValueError:
        print(f"[Trial {trial.number}] Couldn't parse reward from: '{last}'")
        return float("nan")

if __name__ == "__main__":
    storage = "sqlite:///optuna_maddpg.db"
    study = optuna.create_study(
        study_name="maddpg_optuna",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective_noise_search, n_trials=50) # timeout=3600)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    for k, v in trial.params.items():
        print(f"  {k}: {v}")
