"""One-off sweep: train PPO on MarginGroup(g=4, delta=1/k, value=1.0) for
k = 1, 2, ..., 50, and for EACH k fit an exponential learning curve
(L - A*exp(-x/tau)) to that run's rolling-average per-episode training
"correct" signal - the same approach the original version of this script
(and plot_sweep_results.fit_learning_curves) used.

NOTE: this used to also fit a second curve to a periodic DURING-training
GREEDY-eval history (trainPPO.train()'s periodic_eval_freq/
periodic_eval_episodes via PeriodicEvalCallback), specifically to work
around the training-signal fit's asymptote sitting below hit_rate. That
periodic-eval mechanism turned out to be broken and has been removed
here entirely - the only evaluation left is the standard, always-on
one train() runs ONCE at the very end of each run (which is what
hit_rate below comes from). It'll get fixed and reintroduced later.

Output: eval_logs/sweep_summary.csv with columns
    k, hit_rate,
    fit_status, fit_L, fit_A, fit_tau,
    fit_L_err, fit_A_err, fit_tau_err, r_squared, rmse, fit_points_used

fit_status is "ok" or "failed" (curve_fit didn't converge, or there
wasn't enough data left after skipping the noisy early portion) - the
fit_*/r_squared/rmse columns are left blank on failure rather than
something misleading like 0.

LOG_TRAINING_DATA is on (needed for the fit), but each run's train_log/
eval CSVs and model checkpoint are still deleted right after that run's
fit is computed - nothing accumulates across the sweep.

Run:  python3 run_exponential_fit_sweep.py
"""
import csv
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from Utilities.bandit_env import MarginGroup
from trainPPO import train

# --- sweep-specific config ---
G = 4
VALUE = 1.0
K_VALUES = list(range(1, 51))  # k = 1, 2, ..., 50

# Output path - overwritten fresh at the start of every sweep run.
SUMMARY_CSV_PATH = "eval_logs/sweep_summary.csv"

# Reused every iteration so per-run temp files get overwritten (not
# accumulated) and then explicitly deleted right after each run's fit is
# computed - see module docstring.
TEMP_LABEL = "exp_fit_sweep_temp"

# --- environment config (groups is NOT here - built by the loop) ---
INCORRECT_REWARD = 0.0
N_ENVS = 8

# --- training loop config ---
TOTAL_TIMESTEPS = 200_000
PROGRESS_BAR = False
# On: needed for the fit (see module docstring) - deleted right after
# each run's fit is computed, so it never accumulates.
LOG_TRAINING_DATA = True
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
WEIGHT_DECAY = 0.0
ACTOR_WEIGHT_DECAY = None
CRITIC_WEIGHT_DECAY = None
PPO_KWARGS = None

# --- evaluation config ---
EVAL_EPISODES = 500

# Fit config - smooth the raw per-episode "correct" signal with a
# ROLLING_WINDOW-episode rolling mean before fitting. SKIP_FIRST_TRAIN_EPISODES
# used to drop the first 1000 of those (rationale: too noisy early on,
# before the policy settles into a trend) - now 0, so the fit uses the
# WHOLE rolling-average series, including that early stretch.
ROLLING_WINDOW = 200
SKIP_FIRST_TRAIN_EPISODES = 0

# --- output config ---
WEIGHTS_DIR = "weights"
EVAL_LOGS_DIR = "eval_logs"

FIELDNAMES = [
    "k", "hit_rate",
    "fit_status", "fit_L", "fit_A", "fit_tau",
    "fit_L_err", "fit_A_err", "fit_tau_err",
    "r_squared", "rmse", "fit_points_used",
]

_EMPTY_FIT = {
    "fit_status": "failed",
    "fit_L": None, "fit_A": None, "fit_tau": None,
    "fit_L_err": None, "fit_A_err": None, "fit_tau_err": None,
    "r_squared": None, "rmse": None, "fit_points_used": 0,
}


def _exp_func(x, L, A, tau):
    # Exponential approach to an asymptote L, starting `A` below it, with
    # time constant tau - same functional form as
    # plot_sweep_results._build_fit_specs' "exponential" candidate.
    return L - A * np.exp(-x / tau)


def _fit_exp_curve(x, y, sigma=None):
    """Core exponential curve_fit wrapper - fits L - A*exp(-x/tau) to
    (x, y), optionally with per-point sigma (inverse-variance weighting;
    absolute_sigma=True), and returns a dict with fit_status/fit_L/
    fit_A/fit_tau/*_err/r_squared/rmse/fit_points_used. fit_status=
    "failed" (everything else None/0) if there's nothing to fit or
    curve_fit doesn't converge."""
    if len(x) < 3:
        return dict(_EMPTY_FIT)

    p0 = [float(y[-1]), float(y[-1] - y[0]), max((x[-1] - x[0]) / 3.0, 1.0)]
    try:
        if sigma is None:
            popt, pcov = curve_fit(_exp_func, x, y, p0=p0, maxfev=20000)
        else:
            popt, pcov = curve_fit(
                _exp_func, x, y, p0=p0, sigma=sigma, absolute_sigma=True, maxfev=20000,
            )
    except RuntimeError:
        return dict(_EMPTY_FIT)

    residuals = y - _exp_func(x, *popt)
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    param_errs = np.sqrt(np.diag(pcov))

    L, A, tau = (float(v) for v in popt)
    L_err, A_err, tau_err = (float(v) for v in param_errs)

    return {
        "fit_status": "ok",
        "fit_L": L, "fit_A": A, "fit_tau": tau,
        "fit_L_err": L_err, "fit_A_err": A_err, "fit_tau_err": tau_err,
        "r_squared": r_squared, "rmse": rmse, "fit_points_used": len(x),
    }


def _as_bool_int(series):
    """Coerce a 'correct' column to 0/1 ints, whether it was read back as
    real bools or as "True"/"False" strings."""
    if series.dtype == object:
        series = series.map(lambda v: str(v).strip().lower() == "true")
    return series.astype(int)


def _fit_train_curve(train_log_path):
    """Fit an exponential curve to a single run's raw per-episode
    train_log CSV (the BanditDataLoggerCallback output), after smoothing
    with a ROLLING_WINDOW-episode rolling mean and skipping the first
    SKIP_FIRST_TRAIN_EPISODES of those (0 by default now - see that
    constant's definition above; set it back above 0 if the fit needs to
    ignore an early noisy stretch again). Unweighted: adjacent
    rolling-mean points share most of their underlying episodes (heavily
    autocorrelated), so a binomial inverse-variance weighting wouldn't be
    a meaningful description of each point's true independent
    information content anyway."""
    if not os.path.exists(train_log_path):
        return dict(_EMPTY_FIT)

    df = pd.read_csv(train_log_path, usecols=["correct"])
    df["correct"] = _as_bool_int(df["correct"])
    rolling = df["correct"].rolling(window=ROLLING_WINDOW, min_periods=1).mean()

    x_full = rolling.index.to_numpy(dtype=float)
    y_full = rolling.to_numpy(dtype=float)

    if SKIP_FIRST_TRAIN_EPISODES >= len(x_full):
        return dict(_EMPTY_FIT)
    x, y = x_full[SKIP_FIRST_TRAIN_EPISODES:], y_full[SKIP_FIRST_TRAIN_EPISODES:]

    return _fit_exp_curve(x, y, sigma=None)


def run_sweep():
    os.makedirs(EVAL_LOGS_DIR, exist_ok=True)

    with open(SUMMARY_CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    train_log_path = f"{EVAL_LOGS_DIR}/ppo_{TEMP_LABEL}_train_log.csv"
    eval_log_path = f"{EVAL_LOGS_DIR}/ppo_{TEMP_LABEL}_eval.csv"
    checkpoint_path = f"{WEIGHTS_DIR}/ppo_{TEMP_LABEL}.zip"

    for k in K_VALUES:
        groups = [MarginGroup(g=G, delta=1.0 / k, value=VALUE)]

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

        hit_rate = result.hit_rate
        fit = _fit_train_curve(train_log_path)

        row = {"k": k, "hit_rate": hit_rate}
        row.update(fit)
        with open(SUMMARY_CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

        # The fit has already served its only purpose - delete every
        # per-run file now so nothing accumulates across 50 runs.
        # TEMP_LABEL being fixed means these paths get overwritten next
        # iteration anyway, but deleting them explicitly means nothing is
        # left sitting on disk even if the sweep is killed partway
        # through.
        for path in (train_log_path, eval_log_path, checkpoint_path):
            if os.path.exists(path):
                os.remove(path)

        if fit["fit_status"] == "ok":
            note = f"L={fit['fit_L']:.3f} R^2={fit['r_squared']:.3f}"
        else:
            note = "failed"
        print(f"k={k}: hit_rate={hit_rate:.1%}  fit[{note}]")

    print(f"Summary -> {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    run_sweep()
