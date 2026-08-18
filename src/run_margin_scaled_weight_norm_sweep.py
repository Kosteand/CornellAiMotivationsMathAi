"""Sweep over k = 1..25 x s = 10, 100, 1000 for a single scaled
MarginGroup(g=4, delta=s/k, value=1.0, s=s) - i.e. the SAME relative
margin as a plain MarginGroup(delta=1/k) at each k, just with everything
(the random backdrop AND the margin) scaled up by s. weight_decay is held
fixed at 0.0, same as every other per-layer weight-norm sweep in this
project. See run_indifference_batch.py's mixed_group_factory docstring
for the ("margin_scaled", k, s) spec this mirrors - this script exists to
build up the same kind of per-group weight_norm/tau lookup table for
scaled MarginGroup that run_per_layer_weight_norm_rerun.py already built
for plain MarginGroup and HeatmapGroup(n=3), so scaled-margin pairs can be
plugged into the same M-vs-weight_norm-ratio analysis.

75 runs total (25 k's x 3 s's) at TOTAL_TIMESTEPS=200_000 each - a few
hours, budgeted the same way as the other sweeps in this project (each
run ~10 seconds per the per_layer_weight_norm_rerun.py timing note, so
this should run considerably faster in practice than that estimate).
SUMMARY_CSV_PATH is written one row at a time as each run finishes, so
nothing already-completed is lost if this gets killed early.

STANDING CONSTRAINT (as of 2026-08-12, per direct instruction): do not
run MarginGroup (scaled or not) for k > 25, and do not run HeatmapGroup
for noise_scale > 2.0 or n != 3, unless told otherwise in the future -
hit_rate sometimes drops below 1.0 outside those ranges (confirmed against
per_layer_weight_norm_rerun.csv: margin k=30 -> hit_rate=0.998; every
heatmap noise_scale>2.0 point checked has hit_rate 0.992-0.998), which
means the policy hasn't fully solved the task there and its weight_norm
is no longer a clean stand-in for "the weight norm of a converged
solution" - it's contaminated by whatever the policy was still doing
wrong. MARGIN_SCALED_KS below is already capped at 25 for exactly this
reason (k=1..25, not up to 30) - do not extend this sweep past k=25
without checking hit_rate at the new k first.

Per-layer breakdown, output columns, weight_norm() helper, and the
exponential learning-curve fit are all copied unchanged from
run_per_layer_weight_norm_rerun.py so the two files' rows are directly
comparable/joinable - see that file's docstring for the full explanation
of what wn_policy_net_0/2, wn_action_net, and the critic equivalents mean.

Output: eval_logs/margin_scaled_weight_norm_sweep.csv with columns:
    k, s, delta, weight_decay,
    hit_rate, correct, episodes, mean_reward,
    wn_policy_net_0, wn_policy_net_2, wn_action_net, weight_norm_actor,
    wn_value_net_0, wn_value_net_2, wn_value_net_out, weight_norm_critic,
    weight_norm_total,
    fit_status, fit_L, fit_A, fit_tau,
    fit_L_err, fit_A_err, fit_tau_err, r_squared, rmse, fit_points_used

`delta` (= s/k) is included alongside k/s so a row's actual margin is
visible without recomputing it.

Run:  python3 run_margin_scaled_weight_norm_sweep.py
"""
import csv
import os

import torch

from Utilities.bandit_env import MarginGroup
from run_exponential_fit_sweep import _fit_train_curve
from trainPPO import train

# --- sweep-specific config ---
G = 4
VALUE = 1.0

MARGIN_SCALED_KS = list(range(1, 26))  # 1, 2, ..., 25 - see module docstring
MARGIN_SCALED_S_VALUES = [10, 100, 1000]

WEIGHT_DECAY = 0.0  # fixed at 0 for every run, matching the other sweeps

SUMMARY_CSV_PATH = "eval_logs/margin_scaled_weight_norm_sweep.csv"
TEMP_LABEL = "margin_scaled_weight_norm_sweep_temp"

INCORRECT_REWARD = 0.0
N_ENVS = 8

TOTAL_TIMESTEPS = 200_000
PROGRESS_BAR = False
LOG_TRAINING_DATA = True
LOG_INTERVAL = None
PRINT_FINAL_SUMMARY = False

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

EVAL_EPISODES = 500

WEIGHTS_DIR = "weights"
EVAL_LOGS_DIR = "eval_logs"

FIELDNAMES = [
    "k", "s", "delta", "weight_decay",
    "hit_rate", "correct", "episodes", "mean_reward",
    "wn_policy_net_0", "wn_policy_net_2", "wn_action_net", "weight_norm_actor",
    "wn_value_net_0", "wn_value_net_2", "wn_value_net_out", "weight_norm_critic",
    "weight_norm_total",
    "fit_status", "fit_L", "fit_A", "fit_tau",
    "fit_L_err", "fit_A_err", "fit_tau_err",
    "r_squared", "rmse", "fit_points_used",
]


def layer_norm(*tensors):
    """L2 norm across every element of every tensor passed in - identical
    to the helper of the same name in run_per_layer_weight_norm_rerun.py."""
    total = 0.0
    for t in tensors:
        total += float(torch.sum(t.detach() ** 2))
    return total ** 0.5


def run_one(k, s):
    delta = s / k
    groups = [MarginGroup(g=G, delta=delta, value=VALUE, s=s)]

    train_log_path = f"{EVAL_LOGS_DIR}/ppo_{TEMP_LABEL}_train_log.csv"
    eval_log_path = f"{EVAL_LOGS_DIR}/ppo_{TEMP_LABEL}_eval.csv"
    checkpoint_path = f"{WEIGHTS_DIR}/ppo_{TEMP_LABEL}.zip"

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
    pn = dict(policy.mlp_extractor.policy_net.named_parameters())
    vn = dict(policy.mlp_extractor.value_net.named_parameters())
    an = dict(policy.action_net.named_parameters())
    vout = dict(policy.value_net.named_parameters())

    wn_policy_net_0 = layer_norm(pn["0.weight"], pn["0.bias"])
    wn_policy_net_2 = layer_norm(pn["2.weight"], pn["2.bias"])
    wn_action_net = layer_norm(an["weight"], an["bias"])
    weight_norm_actor = (
        wn_policy_net_0 ** 2 + wn_policy_net_2 ** 2 + wn_action_net ** 2
    ) ** 0.5

    wn_value_net_0 = layer_norm(vn["0.weight"], vn["0.bias"])
    wn_value_net_2 = layer_norm(vn["2.weight"], vn["2.bias"])
    wn_value_net_out = layer_norm(vout["weight"], vout["bias"])
    weight_norm_critic = (
        wn_value_net_0 ** 2 + wn_value_net_2 ** 2 + wn_value_net_out ** 2
    ) ** 0.5

    weight_norm_total = (weight_norm_actor ** 2 + weight_norm_critic ** 2) ** 0.5

    row = {
        "k": k,
        "s": s,
        "delta": delta,
        "weight_decay": WEIGHT_DECAY,
        "hit_rate": result.hit_rate,
        "correct": result.correct,
        "episodes": result.episodes,
        "mean_reward": result.mean_reward,
        "wn_policy_net_0": wn_policy_net_0,
        "wn_policy_net_2": wn_policy_net_2,
        "wn_action_net": wn_action_net,
        "weight_norm_actor": weight_norm_actor,
        "wn_value_net_0": wn_value_net_0,
        "wn_value_net_2": wn_value_net_2,
        "wn_value_net_out": wn_value_net_out,
        "weight_norm_critic": weight_norm_critic,
        "weight_norm_total": weight_norm_total,
    }
    row.update(fit)
    with open(SUMMARY_CSV_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    for path in (train_log_path, eval_log_path, checkpoint_path):
        if os.path.exists(path):
            os.remove(path)

    return result, fit, weight_norm_actor


def run_sweep():
    os.makedirs(EVAL_LOGS_DIR, exist_ok=True)

    with open(SUMMARY_CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    total_runs = len(MARGIN_SCALED_KS) * len(MARGIN_SCALED_S_VALUES)
    run_num = 0

    for s in MARGIN_SCALED_S_VALUES:
        for k in MARGIN_SCALED_KS:
            run_num += 1
            result, fit, wn_actor = run_one(k, s)
            fit_note = (
                f"L={fit['fit_L']:.3f} R^2={fit['r_squared']:.3f}"
                if fit["fit_status"] == "ok" else "failed"
            )
            hit_flag = "" if result.hit_rate >= 1.0 else "  <-- hit_rate < 1.0!"
            print(
                f"[{run_num}/{total_runs}] k={k} s={s} (delta={s/k:.4g}): "
                f"hit_rate={result.hit_rate:.1%}  weight_norm_actor={wn_actor:.2f}  "
                f"fit[{fit_note}]{hit_flag}"
            )

    print(f"Summary -> {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    run_sweep()
