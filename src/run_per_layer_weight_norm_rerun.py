"""Rerun the exact single-group training used by run_k_weight_decay_sweep.py
(MarginGroup) and run_heatmap_noise_scale_sweep.py (HeatmapGroup, n=3), but
this time save the weight norm of EVERY layer separately, not just one
combined actor/critic/total number.

Why: the existing weight_norm_actor figure sums squared elements across the
WHOLE actor sub-network (both hidden layers plus the output layer) into one
L2 norm. Only the first hidden layer's parameter COUNT actually depends on
observation_size (its weight matrix is shape (64, observation_size)) - the
second hidden layer (64 -> 32) and the output layer (32 -> g) don't change
shape at all between MarginGroup and HeatmapGroup runs (g is fixed at 4
everywhere in this project's batches so far). Collapsing all of that into
one norm may be exactly what's making the MarginGroup <-> HeatmapGroup
weight-norm-ratio comparison unreliable - this script exists to test that
hypothesis directly, by keeping every layer's norm separate so they can be
analyzed (e.g. first-layer-only ratios) instead of re-fit blind.

Sweeps a WIDER range than the batches measured so far, not just the exact
values that happened to appear in batch_summary_2/3/6.csv - see MARGIN_KS /
HEATMAP_NOISE_SCALES_HIGH / HEATMAP_NOISE_SCALES_LOW below:
    MARGIN_KS: every integer k from 1 to 30 inclusive (30 values).
    HEATMAP_NOISE_SCALES_HIGH: 1.00, 1.05, 1.10, ..., 4.00 (61 values, n=3
        fixed - n itself wasn't part of what this rerun is checking).
    HEATMAP_NOISE_SCALES_LOW: 0.00, 0.05, 0.10, ..., 0.95 (20 values, n=3)
        - added afterwards to cover the noise_scale<1 pairs from
        batch_summary_6.csv (0.0, 0.25, 0.5, 0.75) that the first version
        of this script didn't reach, since it only started at 1.0.

APPEND, DON'T OVERWRITE: MARGIN_KS and HEATMAP_NOISE_SCALES_HIGH were
already run once and are already sitting in SUMMARY_CSV_PATH from that
run. run_sweep() only re-executes HEATMAP_NOISE_SCALES_LOW (the genuinely
new values) and opens SUMMARY_CSV_PATH in append mode - it writes the
header only if the file doesn't exist yet, and never truncates an
existing file. That's 20 new runs at TOTAL_TIMESTEPS=200_000 each (a few
hours - budget accordingly; still written one row at a time, so nothing
already-done, from this run or the earlier one, is lost if this gets
killed early). If you ever do want to regenerate MARGIN_KS /
HEATMAP_NOISE_SCALES_HIGH from scratch, delete SUMMARY_CSV_PATH first and
add them back into run_sweep()'s loop - they're left defined below (not
deleted) so that full-range definition doesn't have to be re-derived.

All other config (PPO hyperparameters, net_arch, weight_decay=0,
total_timesteps, eval_episodes, the exponential learning-curve fit) is
copied unchanged from run_k_weight_decay_sweep.py /
run_heatmap_noise_scale_sweep.py so results are directly comparable to
(and, for the ranges those covered, a superset of) what's already been
measured - this is a rerun for MORE detail and MORE range, not a
different experiment.

Per-layer breakdown captured (matching the actual SB3 ActorCriticPolicy
module names for net_arch_pi=net_arch_vf=(64, 32), confirmed by inspection):
    actor:  mlp_extractor.policy_net.0  (observation_size -> 64)
            mlp_extractor.policy_net.2  (64 -> 32)
            action_net                 (32 -> g)
    critic: mlp_extractor.value_net.0   (observation_size -> 64)
            mlp_extractor.value_net.2   (64 -> 32)
            value_net                  (32 -> 1)
Each layer's norm is the L2 norm over BOTH that layer's weight matrix and
its bias vector together (same "sum of squares of every element" approach
weight_norm() already uses elsewhere in this project, just scoped to one
layer instead of a whole sub-network). The old combined actor/critic/total
numbers are also written on every row (computed as a sanity-check cross-sum
of the per-layer numbers, not independently) so this file's rows can be
checked against - and used interchangeably with - the existing
sweep_k_weight_decay_summary.csv / sweep_heatmap_noise_scale_summary.csv
rows for the same (k) / (noise_scale, n=3) values.

Output: eval_logs/per_layer_weight_norm_rerun.csv with columns:
    group_type, k, noise_scale, n, weight_decay,
    hit_rate, correct, episodes, mean_reward,
    wn_policy_net_0, wn_policy_net_2, wn_action_net, weight_norm_actor,
    wn_value_net_0, wn_value_net_2, wn_value_net_out, weight_norm_critic,
    weight_norm_total,
    fit_status, fit_L, fit_A, fit_tau,
    fit_L_err, fit_A_err, fit_tau_err, r_squared, rmse, fit_points_used

`group_type` is "margin" or "heatmap"; the irrelevant spec column for each
row (noise_scale/n for margin rows, k for heatmap rows) is left blank
rather than 0 - 0 would misleadingly look like a real spec value.

Same per-run temp-file cleanup as the other sweep scripts (LOG_TRAINING_DATA
is on for the fit, but train_log/eval CSVs and the checkpoint are deleted
right after each run's row is written) and the same one-row-at-a-time CSV
append so nothing already-completed is lost if this is killed early.

Run:  python3 run_per_layer_weight_norm_rerun.py
"""
import csv
import os

import numpy as np
import torch

from Utilities.bandit_env import MarginGroup, HeatmapGroup
from run_exponential_fit_sweep import _fit_train_curve
from trainPPO import train

# --- wide sweep ranges (wider than what the batches happened to use) ---
G = 4
VALUE = 1.0

MARGIN_KS = list(range(1, 31))  # 1, 2, ..., 30 - already run, kept for reference only

# 1.00, 1.05, ..., 4.00 - already run, kept for reference only (see module
# docstring's APPEND, DON'T OVERWRITE section - run_sweep() below does NOT
# loop over MARGIN_KS or HEATMAP_NOISE_SCALES_HIGH anymore).
HEATMAP_NOISE_SCALES_HIGH = [round(1.0 + 0.05 * i, 10) for i in range(61)]

# 0.00, 0.05, ..., 0.95 - the NEW values this edit adds, to cover the
# noise_scale<1 heatmap pairs from batch_summary_6.csv that
# HEATMAP_NOISE_SCALES_HIGH doesn't reach. Built the same
# round(0.05*i, 10) way as every other noise_scale grid in this project,
# so it lines up exactly on the same 0.05 grid (0.25, 0.5, 0.75, etc. are
# all included).
HEATMAP_NOISE_SCALES_LOW = [round(0.05 * i, 10) for i in range(20)]

HEATMAP_N = 3  # every heatmap pair used in the batches was n=3

WEIGHT_DECAY = 0.0  # fixed at 0 for every run, matching both source sweeps

SUMMARY_CSV_PATH = "eval_logs/per_layer_weight_norm_rerun.csv"
TEMP_LABEL = "per_layer_weight_norm_rerun_temp"

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
    "group_type", "k", "noise_scale", "n", "weight_decay",
    "hit_rate", "correct", "episodes", "mean_reward",
    "wn_policy_net_0", "wn_policy_net_2", "wn_action_net", "weight_norm_actor",
    "wn_value_net_0", "wn_value_net_2", "wn_value_net_out", "weight_norm_critic",
    "weight_norm_total",
    "fit_status", "fit_L", "fit_A", "fit_tau",
    "fit_L_err", "fit_A_err", "fit_tau_err",
    "r_squared", "rmse", "fit_points_used",
]


def layer_norm(*tensors):
    """L2 norm across every element of every tensor passed in - same
    "sum of squares" approach as weight_norm() in the other sweep scripts,
    just scoped to whichever tensors (e.g. one layer's weight+bias) are
    passed rather than a whole sub-network."""
    total = 0.0
    for t in tensors:
        total += float(torch.sum(t.detach() ** 2))
    return total ** 0.5


def run_one(group_type, groups, k=None, noise_scale=None, n=None):
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
        "group_type": group_type,
        "k": k if k is not None else "",
        "noise_scale": noise_scale if noise_scale is not None else "",
        "n": n if n is not None else "",
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
    """Only runs HEATMAP_NOISE_SCALES_LOW (the newly-added noise_scale<1
    values) and APPENDS them to SUMMARY_CSV_PATH - MARGIN_KS and
    HEATMAP_NOISE_SCALES_HIGH were already run in an earlier pass and are
    already sitting in that file, so this does not touch or repeat them.
    The header is written only if the file doesn't already exist (a fresh
    run from scratch); an existing file is opened in append mode and left
    otherwise untouched, so the earlier 91 rows stay exactly as they are."""
    os.makedirs(EVAL_LOGS_DIR, exist_ok=True)

    file_is_new = not os.path.exists(SUMMARY_CSV_PATH)
    if file_is_new:
        with open(SUMMARY_CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    total_runs = len(HEATMAP_NOISE_SCALES_LOW)
    run_num = 0

    for noise_scale in HEATMAP_NOISE_SCALES_LOW:
        run_num += 1
        groups = [HeatmapGroup(g=G, noise_scale=noise_scale, n=HEATMAP_N, value=VALUE)]
        result, fit, wn_actor = run_one("heatmap", groups, noise_scale=noise_scale, n=HEATMAP_N)
        fit_note = f"L={fit['fit_L']:.3f} R^2={fit['r_squared']:.3f}" if fit["fit_status"] == "ok" else "failed"
        print(f"[{run_num}/{total_runs}] heatmap noise_scale={noise_scale:.2f} n={HEATMAP_N}: "
              f"hit_rate={result.hit_rate:.1%}  weight_norm_actor={wn_actor:.2f}  fit[{fit_note}]")

    print(f"Appended {total_runs} new rows -> {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    run_sweep()
