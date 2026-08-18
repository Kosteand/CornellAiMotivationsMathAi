"""2D sweep over k = 1, 2, ..., 25 AND weight_decay = 0, N/20, 2*N/20, ...,
N for a single MarginGroup(g=4, value=1.0, delta=1/k) - i.e. every
(k, weight_decay) combination on a 25 x 21 grid, one PPO training run
each. Records hit_rate and weight norms (actor/critic/total) for every
cell, so you can see how learnability (hit_rate) and weight magnitude
trade off against each other as BOTH complexity and regularization
strength vary together - not just weight_decay's effect at one fixed k,
or k's effect at one fixed weight_decay.

That's 25 * 21 = 525 separate training runs. At TOTAL_TIMESTEPS=200_000
each, this is a LONG sweep (many hours - budget accordingly, and feel
free to kill it early: SUMMARY_CSV_PATH is written one row at a time as
each run finishes, so nothing already-completed is lost).

Picking N (the weight_decay ceiling): tested empirically, not guessed,
specifically at k=25 - the HARDEST k in this sweep's range, since that's
where weight_decay's regularization pressure is most likely to fight
against the model actually learning the task, and the whole grid's top
row (k=25, weight_decay=N) needs to be a run that still actually
learns. At TOTAL_TIMESTEPS=100_000 (half of this script's actual
TOTAL_TIMESTEPS below, i.e. LESS forgiving), weight_decay=2.0 at k=25
still reached 100% hit_rate (weight_decay=3.0 still reached 99%) - a
shorter, 40_000-timestep version of the same test had weight_decay=2.0
only reaching 78%, which shows this is a "needs enough training time to
overcome the extra regularization pressure" effect, not a hard ceiling -
and 200_000 timesteps (double the 100_000-timestep test that already
passed) gives real headroom beyond that. N=2.0 below, giving
weight_decay values 0.0, 0.1, 0.2, ..., 2.0.

Each run ALSO gets an exponential learning-curve fit (L - A*exp(-x/tau))
computed from its own in-training data (the rolling-average per-episode
stochastic training signal, needs LOG_TRAINING_DATA on) - reusing
run_exponential_fit_sweep.py's exact fitting function (imported below,
not duplicated), so results here are directly comparable to that
script's sweep_summary.csv, cell-for-cell where k and weight_decay=0
overlap.

NOTE: this used to ALSO fit a second curve to a periodic DURING-training
GREEDY-eval history (PeriodicEvalCallback). That mechanism turned out to
be broken and has been removed here entirely, same as in
run_exponential_fit_sweep.py - the only evaluation left per run is the
standard, always-on one train() runs ONCE at the very end (which is what
hit_rate below comes from). It'll get fixed and reintroduced later.

Output: eval_logs/sweep_k_weight_decay_summary.csv (deliberately not
sweep_summary.csv or sweep_summary_weight_decay.csv - both already used
by other scripts) with columns:
    k, weight_decay, hit_rate, correct, episodes, mean_reward,
    weight_norm_actor, weight_norm_critic, weight_norm_total,
    fit_status, fit_L, fit_A, fit_tau,
    fit_L_err, fit_A_err, fit_tau_err, r_squared, rmse, fit_points_used

fit_status is "ok" or "failed" (curve_fit didn't converge, or there
wasn't enough data left after skipping the noisy early portion) - the
fit_*/r_squared/rmse columns are left blank on failure rather than
something misleading like 0.

LOG_TRAINING_DATA is on (needed for the fit - see above), but each run's
train_log/eval CSVs and model checkpoint are still deleted right after
that run's row is written - at 525 runs, letting any of that accumulate
would be the same kind of disk blowup earlier sweeps already hit once.

Run:  python3 run_k_weight_decay_sweep.py
"""
import csv
import os

import numpy as np
import torch

from Utilities.bandit_env import MarginGroup
from run_exponential_fit_sweep import _fit_train_curve
from trainPPO import train

# --- sweep-specific config ---
G = 4
VALUE = 1.0
K_VALUES = list(range(1, 26))  # k = 1, 2, ..., 25

# See module docstring for how N was chosen.
N = 2.0
WEIGHT_DECAY_VALUES = [round(N * i / 20.0, 10) for i in range(21)]  # 0, N/20, ..., N

# Output path - overwritten fresh at the start of every sweep run, then
# appended to one row at a time as each (k, weight_decay) run finishes.
SUMMARY_CSV_PATH = "eval_logs/sweep_k_weight_decay_summary.csv"

# Reused every iteration so per-run temp files get overwritten (not
# accumulated) and then explicitly deleted right after each run's row is
# written - see module docstring.
TEMP_LABEL = "k_wd_sweep_temp"

# --- environment config (groups is NOT here - built by the loop) ---
INCORRECT_REWARD = 0.0
N_ENVS = 8

# --- training loop config ---
TOTAL_TIMESTEPS = 200_000
PROGRESS_BAR = False
LOG_TRAINING_DATA = True  # on - needed for the TRAIN fit, see module docstring
LOG_INTERVAL = None
PRINT_FINAL_SUMMARY = False

# --- PPO hyperparameters (same as run_trainPPO_sweep.py) ---
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
    "k", "weight_decay", "hit_rate", "correct", "episodes", "mean_reward",
    "weight_norm_actor", "weight_norm_critic", "weight_norm_total",
    "fit_status", "fit_L", "fit_A", "fit_tau",
    "fit_L_err", "fit_A_err", "fit_tau_err",
    "r_squared", "rmse", "fit_points_used",
]


def weight_norm(parameters):
    """L2 norm across every element of every tensor in `parameters` -
    same helper as run_trainPPO_sweep.py's."""
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

    total_runs = len(K_VALUES) * len(WEIGHT_DECAY_VALUES)
    run_num = 0

    for k in K_VALUES:
        groups = [MarginGroup(g=G, delta=1.0 / k, value=VALUE)]

        for weight_decay in WEIGHT_DECAY_VALUES:
            run_num += 1

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
                weight_decay=weight_decay,
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
                "k": k,
                "weight_decay": weight_decay,
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
            # per-run file now so nothing accumulates across 525 runs
            # (see module docstring).
            for path in (train_log_path, eval_log_path, checkpoint_path):
                if os.path.exists(path):
                    os.remove(path)

            if fit["fit_status"] == "ok":
                fit_note = f"L={fit['fit_L']:.3f} R^2={fit['r_squared']:.3f}"
            else:
                fit_note = "failed"
            print(
                f"[{run_num}/{total_runs}] k={k} weight_decay={weight_decay:.3f}: "
                f"hit_rate={result.hit_rate:.1%}  weight_norm={norm_total:.2f}  "
                f"fit[{fit_note}]"
            )

    print(f"Summary -> {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    run_sweep()
