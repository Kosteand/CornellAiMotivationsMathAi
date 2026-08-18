"""Sweep over k = 1..25 x s = 0.1, 0.01, 0.001 for a single scaled
MarginGroup(g=4, delta=s/k, value=1.0, s=s) - the SAME relative margin as
a plain MarginGroup(delta=1/k) at each k, just with everything (the random
backdrop AND the margin) scaled DOWN below the default s=1, rather than up
as in run_margin_scaled_weight_norm_sweep.py (which covered s=10, 100,
1000). This is the complementary direction: that script tested whether
inflating MarginGroup's absolute magnitude toward HeatmapGroup's natural
scale closes the cross-type weight-norm-ratio-to-M gap (it does, see
plot_M_fits); this one asks what happens as absolute magnitude shrinks
well below 1 instead - does weight_norm keep climbing (extrapolating the
monotonic decrease-in-s trend backwards), or does something else happen
(e.g. numerical/optimization difficulty at very small margins)?

Output/columns/per-layer breakdown/exponential fit are all copied
unchanged from run_margin_scaled_weight_norm_sweep.py (which itself
mirrors run_per_layer_weight_norm_rerun.py) so all three files' rows are
directly comparable/joinable on (k, s, ...).

25 runs total (25 k's x 3 s's) at TOTAL_TIMESTEPS=200_000 each.
SUMMARY_CSV_PATH is written one row at a time as each run finishes, so
nothing already-completed is lost if this gets killed early.

STANDING CONSTRAINT (as of 2026-08-12, per direct instruction): do not run
MarginGroup (scaled or not) for k > 25 unless told otherwise - see
run_margin_scaled_weight_norm_sweep.py's docstring for the hit_rate
evidence behind this. MARGIN_SCALED_KS below is already capped at 25.

NOTE: this is new territory - every other margin_scaled sweep so far only
tested s >= 10 (i.e. inflating magnitude). s=0.1/0.01/0.001 shrinks the
raw margin/backdrop instead, which was NOT covered by the hit_rate checks
that motivated the standing constraint above (those only checked s=1 and
s=10/100/1000). run_sweep() below prints a "<-- hit_rate < 1.0!" flag
inline for any point that doesn't fully converge, same as the other
sweeps, so any such point can be excluded from downstream M-fitting
exactly like the >25/>2.0-noise_scale exclusions were - just don't assume
convergence holds here the way it does for s=1..1000 without checking.

Output: eval_logs/margin_scaled_weight_norm_sweep_small_s.csv with columns:
    k, s, delta, weight_decay,
    hit_rate, correct, episodes, mean_reward,
    wn_policy_net_0, wn_policy_net_2, wn_action_net, weight_norm_actor,
    wn_value_net_0, wn_value_net_2, wn_value_net_out, weight_norm_critic,
    weight_norm_total,
    fit_status, fit_L, fit_A, fit_tau,
    fit_L_err, fit_A_err, fit_tau_err, r_squared, rmse, fit_points_used

`delta` (= s/k) is included alongside k/s so a row's actual margin is
visible without recomputing it - at these s values delta is tiny (e.g.
k=25, s=0.001 -> delta=0.00004).

Run:  python3 run_margin_scaled_weight_norm_sweep_small_s.py
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
MARGIN_SCALED_S_VALUES = [0.1, 0.01, 0.001]

WEIGHT_DECAY = 0.0  # fixed at 0 for every run, matching the other sweeps

SUMMARY_CSV_PATH = "eval_logs/margin_scaled_weight_norm_sweep_small_s.csv"
TEMP_LABEL = "margin_scaled_weight_norm_sweep_small_s_temp"

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
    to the helper of the same name in run_per_layer_weight_norm_rerun.py
    and run_margin_scaled_weight_norm_sweep.py."""
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
                f"[{run_num}/{total_runs}] k={k} s={s} (delta={s/k:.6g}): "
                f"hit_rate={result.hit_rate:.1%}  weight_norm_actor={wn_actor:.2f}  "
                f"fit[{fit_note}]{hit_flag}"
            )

    print(f"Summary -> {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    run_sweep()
