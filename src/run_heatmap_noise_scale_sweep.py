"""Sweep over n = 3, 4 AND noise_scale = 0.00, 0.05, 0.10, ..., 2.00 for a
single HeatmapGroup(g=4, noise_scale=noise_scale, n=n, value=1.0) - i.e.
every (n, noise_scale) combination on a 2 x 41 grid, one PPO training run
each. weight_decay is held fixed at 0.0 for every run (unlike
run_k_weight_decay_sweep.py, which sweeps weight_decay too) - this sweep
is only about how HeatmapGroup's own difficulty knobs (n, noise_scale)
affect learnability and weight magnitude, with regularization strength
held out of the picture entirely. Exactly the same run/log/cleanup shape
as run_k_weight_decay_sweep.py otherwise - same PPO hyperparameters, same
weight_norm split (actor/critic/total), same exponential-learning-curve
fit reused (not duplicated) from run_exponential_fit_sweep.py, same
per-run temp-file cleanup so nothing accumulates across the sweep.

That's 2 * 41 = 82 separate training runs. At TOTAL_TIMESTEPS=200_000
each, this is a long sweep (a few hours - budget accordingly, and feel
free to kill it early: SUMMARY_CSV_PATH is written one row at a time as
each run finishes, so nothing already-completed is lost).

noise_scale=0.0 is a genuinely valid, meaningful point here (not just a
boundary filler): per HeatmapGroup's own docstring, noise_scale=0 makes
the additive-noise term vanish entirely and the shift a no-op, so f_out
is just gradient*heatmap_weights raised to each column's power - a
noise-free (but still power-scaled, still only-f_out-is-observed) version
of the task. HeatmapGroup used to reject noise_scale<=0 outright; that
constructor check was loosened to allow exactly 0.0 (still rejects
negative values) specifically so this sweep's requested range could
include it.

Each run ALSO gets an exponential learning-curve fit (L - A*exp(-x/tau))
computed from its own in-training data (the rolling-average per-episode
stochastic training signal, needs LOG_TRAINING_DATA on) - reusing
run_exponential_fit_sweep.py's exact fitting function (imported below,
not duplicated), same as run_k_weight_decay_sweep.py does for MarginGroup.

NOTE: only the standard, always-on end-of-training greedy eval train()
itself runs is used for hit_rate below - see run_k_weight_decay_sweep.py's
own docstring for why the old periodic-during-training eval mechanism
isn't used here either.

Output: eval_logs/sweep_heatmap_noise_scale_summary.csv with columns:
    n, noise_scale, weight_decay, hit_rate, correct, episodes, mean_reward,
    weight_norm_actor, weight_norm_critic, weight_norm_total,
    fit_status, fit_L, fit_A, fit_tau,
    fit_L_err, fit_A_err, fit_tau_err, r_squared, rmse, fit_points_used

`weight_decay` is included (always 0.0) purely so this file has the same
shape as sweep_k_weight_decay_summary.csv and can be joined/compared
against it directly. fit_status is "ok" or "failed" (curve_fit didn't
converge, or there wasn't enough data left after skipping the noisy early
portion) - the fit_*/r_squared/rmse columns are left blank on failure
rather than something misleading like 0.

LOG_TRAINING_DATA is on (needed for the fit - see above), but each run's
train_log/eval CSVs and model checkpoint are still deleted right after
that run's row is written - same disk-blowup avoidance as
run_k_weight_decay_sweep.py.

Run:  python3 run_heatmap_noise_scale_sweep.py
"""
import csv
import os

import numpy as np
import torch

from Utilities.bandit_env import HeatmapGroup
from run_exponential_fit_sweep import _fit_train_curve
from trainPPO import train

# --- sweep-specific config ---
G = 4
VALUE = 1.0
N_VALUES = [3, 4]

# noise_scale = 0.00, 0.05, 0.10, ..., 2.00 (41 values) - see module
# docstring for why 0.0 is included and valid here.
NOISE_SCALE_VALUES = [round(0.05 * i, 10) for i in range(41)]

# Held fixed at 0 for every run in this sweep - see module docstring.
WEIGHT_DECAY = 0.0

# Output path - overwritten fresh at the start of every sweep run, then
# appended to one row at a time as each (n, noise_scale) run finishes.
SUMMARY_CSV_PATH = "eval_logs/sweep_heatmap_noise_scale_summary.csv"

# Reused every iteration so per-run temp files get overwritten (not
# accumulated) and then explicitly deleted right after each run's row is
# written - see module docstring.
TEMP_LABEL = "heatmap_noise_scale_sweep_temp"

# --- environment config (groups is NOT here - built by the loop) ---
INCORRECT_REWARD = 0.0
N_ENVS = 8

# --- training loop config ---
TOTAL_TIMESTEPS = 200_000
PROGRESS_BAR = False
LOG_TRAINING_DATA = True  # on - needed for the TRAIN fit, see module docstring
LOG_INTERVAL = None
PRINT_FINAL_SUMMARY = False

# --- PPO hyperparameters (same as run_k_weight_decay_sweep.py) ---
DEVICE = "cpu"
VERBOSE = 0
SEED = None
LEARNING_RATE = 3e-4
N_STEPS = 512
BATCH_SIZE = 512
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
NET_ARCH_PI = (64, 32)
NET_ARCH_VF = (64, 32)
ACTOR_WEIGHT_DECAY = None
CRITIC_WEIGHT_DECAY = None
PPO_KWARGS = None

# --- evaluation config ---
EVAL_EPISODES = 500

# --- output config ---
WEIGHTS_DIR = "weights"
EVAL_LOGS_DIR = "eval_logs"

FIELDNAMES = [
    "n", "noise_scale", "weight_decay", "hit_rate", "correct", "episodes", "mean_reward",
    "weight_norm_actor", "weight_norm_critic", "weight_norm_total",
    "fit_status", "fit_L", "fit_A", "fit_tau",
    "fit_L_err", "fit_A_err", "fit_tau_err",
    "r_squared", "rmse", "fit_points_used",
]


def weight_norm(parameters):
    """L2 norm across every element of every tensor in `parameters` -
    same helper as run_k_weight_decay_sweep.py's/run_trainPPO_sweep.py's."""
    total = 0.0
    for p in parameters:
        total += float(torch.sum(p.detach() ** 2))
    return total ** 0.5


def run_sweep():
    os.makedirs(EVAL_LOGS_DIR, exist_ok=True)

    with open(SUMMARY_CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    train_log_path = f"{EVAL_LOGS_DIR}/ppo_{TEMP_LABEL}_train_log.csv"
    eval_log_path = f"{EVAL_LOGS_DIR}/ppo_{TEMP_LABEL}_eval.csv"
    checkpoint_path = f"{WEIGHTS_DIR}/ppo_{TEMP_LABEL}.zip"

    total_runs = len(N_VALUES) * len(NOISE_SCALE_VALUES)
    run_num = 0

    for n in N_VALUES:
        for noise_scale in NOISE_SCALE_VALUES:
            run_num += 1

            groups = [HeatmapGroup(g=G, noise_scale=noise_scale, n=n, value=VALUE)]

            result = train(
                groups,
                incorrect_reward=INCORRECT_REWARD,
                n_envs=N_ENVS,
                total_timesteps=TOTAL_TIMESTEPS,
                label=TEMP_LABEL,
                progress_bar=PROGRESS_BAR,
                log_training_data=LOG_TRAINING_DATA,
                log_interval=LOG_INTERVAL,
                print_final_summary=PRINT_FINAL_SUMMARY,
                device=DEVICE,
                verbose=VERBOSE,
                seed=SEED,
                learning_rate=LEARNING_RATE,
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
                clip_range=CLIP_RANGE,
                ent_coef=ENT_COEF,
                vf_coef=VF_COEF,
                max_grad_norm=MAX_GRAD_NORM,
                net_arch_pi=NET_ARCH_PI,
                net_arch_vf=NET_ARCH_VF,
                weight_decay=WEIGHT_DECAY,
                actor_weight_decay=ACTOR_WEIGHT_DECAY,
                critic_weight_decay=CRITIC_WEIGHT_DECAY,
                ppo_kwargs=PPO_KWARGS,
                eval_episodes=EVAL_EPISODES,
                weights_dir=WEIGHTS_DIR,
                eval_logs_dir=EVAL_LOGS_DIR,
            )

            fit = _fit_train_curve(train_log_path)

            policy = result.model.policy
            actor_params = list(policy.mlp_extractor.policy_net.parameters()) + list(
                policy.action_net.parameters()
            )
            critic_params = list(policy.mlp_extractor.value_net.parameters()) + list(
                policy.value_net.parameters()
            )
            norm_actor = weight_norm(actor_params)
            norm_critic = weight_norm(critic_params)
            norm_total = weight_norm(policy.parameters())

            row = {
                "n": n,
                "noise_scale": noise_scale,
                "weight_decay": WEIGHT_DECAY,
                "hit_rate": result.hit_rate,
                "correct": result.correct,
                "episodes": result.episodes,
                "mean_reward": result.mean_reward,
                "weight_norm_actor": norm_actor,
                "weight_norm_critic": norm_critic,
                "weight_norm_total": norm_total,
            }
            row.update(fit)
            with open(SUMMARY_CSV_PATH, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

            # The fit has already served its only purpose - delete every
            # per-run file now so nothing accumulates across 82 runs.
            for path in (train_log_path, eval_log_path, checkpoint_path):
                if os.path.exists(path):
                    os.remove(path)

            if fit["fit_status"] == "ok":
                fit_note = f"L={fit['fit_L']:.3f} R^2={fit['r_squared']:.3f}"
            else:
                fit_note = "failed"
            print(
                f"[{run_num}/{total_runs}] n={n} noise_scale={noise_scale:.2f}: "
                f"hit_rate={result.hit_rate:.1%}  weight_norm={norm_total:.2f}  "
                f"fit[{fit_note}]"
            )

    print(f"Summary -> {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    run_sweep()
