"""Sweep MarginGroup's ``err`` parameter (added 2026-08-13 - see
Utilities/bandit_env.py's MarginGroup docstring) from 0.01 to 0.75 at a
single fixed (k, s), measuring weight_norm/hit_rate/proxy_hit_rate/tau at
each point, to see how much label noise the network can absorb before
training quality degrades.

``err`` is the per-episode probability that the option actually REWARDED
as correct is NOT the one the proxy (argmax of the observation) points
to - i.e. it injects label noise on top of an otherwise perfectly clean
margin task, WITHOUT changing the observation's distribution at all (the
same closed-form expected_magnitude() in Utilities/weight_norm_data.py
still applies unchanged - err has no effect on the observation, only on
which label the environment rewards).

TWO hit-rate metrics are measured per point (added 2026-08-14):
  - hit_rate: how often the trained policy's greedy action matches the
    option actually REWARDED that episode - the noisy, possibly-mislabeled
    target when err > 0. This is trainPPO.train()'s own built-in eval,
    unchanged from before.
  - proxy_hit_rate: how often the policy's action matches the option the
    PROXY points to (argmax of the observation) - the "intended" target,
    regardless of whether that episode's reward went to it. Measured by a
    SEPARATE fresh-episode eval pass (evaluate_proxy() below), since
    trainPPO.evaluate() only ever compares against the (possibly noisy)
    rewarded label. Note MarginGroup doesn't need to change for this - the
    proxy's pick is always exactly argmax(observation) by construction,
    independent of err (see MarginGroup.sample's docstring), so this is
    computable directly from each episode's observation with no extra
    plumbing through the env.
Comparing the two tells you whether the network is still tracking the
TRUE underlying margin signal even as err corrupts what it's rewarded
for (proxy_hit_rate staying high while hit_rate falls with err), or
whether it's instead learning to chase the noisy reward itself (both
degrading together).

ERR_VALUES below is finer near 0 (increments of 0.01 up to 0.20, where the
network is expected to still mostly cope) and coarser above that
(increments of 0.05 up to 0.75, where degradation - if any - should
already be visible), per direct instruction:
    0.01, 0.02, ..., 0.20   (20 points, step 0.01)
    0.25, 0.30, ..., 0.75   (11 points, step 0.05)
31 points total. err=0.0 is deliberately EXCLUDED (per direct instruction,
2026-08-14) - it isn't run by this sweep at all. This is not the same as
"err=0.0 has no data": weight_norm_data.csv already has an UNRELATED
(margin, k=9, s=1.0, err=0.0) row from a much older sweep
(per_layer_weight_norm_rerun_7.csv) that 4 existing indifference_data.csv
pairs depend on for their weight_norm lookup - that row is left alone on
purpose (removing it would silently break those 4 pairs); this sweep just
never touches err=0.0 itself, before OR after this file's err=0 point was
dropped.

S is FIXED at 1.0 (per direct instruction). K_VALUES below (added
2026-08-14, per direct instruction) is now the FULL k=1..25 range - this
is a genuine (k, err) GRID (25 k's x 31 err's = 775 runs total), not the
single-k=9 sweep this file started as. K=9 alone is still worth reading
off the results as the original reference point, but every other k in
1..25 now gets its own full err sweep too.

STANDING CONSTRAINT (per direct instruction, 2026-08-12, re-affirmed for
every subsequent MarginGroup sweep in this project): do not run MarginGroup
for k > 25 unless told otherwise - K_VALUES is capped at 25 for exactly
this reason.

Output: writes directly to the unified weight_norm_data.csv via
Utilities.weight_norm_data.upsert_record (the established "future sweeps"
convention - see that module's docstring - extended to overwrite rather
than just append). Every row's group_type is "margin", with err actually
populated (every OTHER margin row in weight_norm_data.csv predates err
and is retroactively err=0.0 - this sweep is what actually exercises
nonzero err for the first time). RE-RUNNING THIS SCRIPT OVERWRITES each
err value's previous row in place (upsert_record removes the old row for
that exact spec, if present, before writing the new one) rather than
skipping already-present points or duplicating them - per direct
instruction (2026-08-14), since this sweep's own measurements (e.g. after
changing what it measures, as just happened by adding proxy_hit_rate) are
meant to be replaced by a re-run, not accumulated alongside stale
versions of themselves. Written one row at a time as each run finishes,
so nothing already-completed is lost if this gets killed early.

Run:  python3 run_err_weight_norm_sweep.py
"""
import os

import numpy as np
import torch

from Utilities import weight_norm_data as wnd
from Utilities.bandit_env import BanditEnv, MarginGroup
from run_exponential_fit_sweep import _fit_train_curve
from trainPPO import train

# --- sweep-specific config ---
G = 4
VALUE = 1.0

# K_VALUES: the full k=1..25 range (per direct instruction, 2026-08-14) -
# every one of these gets its own full ERR_VALUES sweep below, i.e. a
# (k, err) grid, not just the single k=9 point this file started as. k=9
# specifically remains this project's go-to "middling difficulty"
# reference point if you want one k to look at first.
K_VALUES = list(range(1, 26))
S = 1.0  # fixed per direct instruction

ERR_VALUES = (
    [round(0.01 * i, 2) for i in range(1, 21)]   # 0.01, 0.02, ..., 0.20
    + [round(0.25 + 0.05 * i, 2) for i in range(0, 11)]  # 0.25, ..., 0.75
)  # err=0.0 deliberately excluded - see module docstring

WEIGHT_DECAY = 0.0  # fixed at 0, matching every other weight-norm sweep

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
# Episodes for the SEPARATE proxy_hit_rate eval pass (evaluate_proxy
# below) - kept equal to EVAL_EPISODES by default so both hit-rate metrics
# have the same statistical precision; these are FRESH episodes, not the
# same rollouts hit_rate was measured on (trainPPO.evaluate() doesn't
# expose per-episode observations, so re-measuring on a second pass is
# simpler than threading that through).
PROXY_EVAL_EPISODES = EVAL_EPISODES

WEIGHTS_DIR = "weights"
EVAL_LOGS_DIR = "eval_logs"

TEMP_LABEL = "err_weight_norm_sweep_temp"


def layer_norm(*tensors):
    """L2 norm across every element of every tensor passed in - identical
    to the helper of the same name in the other weight-norm sweep
    scripts."""
    total = 0.0
    for t in tensors:
        total += float(torch.sum(t.detach() ** 2))
    return total ** 0.5


def evaluate_proxy(model, groups, incorrect_reward, episodes):
    """Greedy accuracy against the PROXY's pick (argmax of the
    observation) rather than the (possibly err-corrupted) rewarded label -
    mirrors trainPPO.evaluate()'s structure exactly, but compares the
    action to argmax(obs) instead of info["correct"]. Only meaningful for
    a single-group MarginGroup env (the assumption this whole script
    makes) - argmax of the FULL observation is only guaranteed to equal
    "the proxy's pick for group i" when there's exactly one group, since
    with multiple groups concatenated together the global argmax could
    fall in a different group's block entirely.

    Returns (proxy_correct, episodes, mean_reward) - same shape as
    trainPPO.evaluate()'s return, for a drop-in-comparable pair of
    metrics."""
    if len(groups) != 1:
        raise ValueError(
            "evaluate_proxy assumes exactly one group (argmax(obs) only "
            f"equals the proxy's pick in that case); got {len(groups)}"
        )

    env = BanditEnv(groups=groups, incorrect_reward=incorrect_reward)

    proxy_correct = 0
    total_reward = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        proxy_label = int(np.argmax(obs))
        obs, reward, terminated, truncated, info = env.step(int(action))
        proxy_correct += int(action == proxy_label)
        total_reward += reward

    return proxy_correct, episodes, total_reward / episodes


def run_one(k, err):
    delta = S / k
    groups = [MarginGroup(g=G, delta=delta, value=VALUE, s=S, err=err)]

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

    # Separate eval pass for proxy_hit_rate (see module docstring / this
    # function's docstring above the training call for why it's a second
    # pass rather than reused from `result`).
    proxy_correct, proxy_episodes, _proxy_mean_reward = evaluate_proxy(
        result.model, groups, INCORRECT_REWARD, PROXY_EVAL_EPISODES,
    )
    proxy_hit_rate = proxy_correct / proxy_episodes

    row = {
        "group_type": "margin",
        "k": k,
        "s": S,
        "err": err,
        "delta": delta,
        "weight_decay": WEIGHT_DECAY,
        "hit_rate": result.hit_rate,
        "proxy_hit_rate": proxy_hit_rate,
        "correct": result.correct,
        "proxy_correct": proxy_correct,
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
        "source_file": "run_err_weight_norm_sweep.py",
    }
    row.update(fit)

    # upsert (not append): re-running this sweep is meant to OVERWRITE its
    # own previous rows in place - see module docstring.
    wnd.upsert_record(row)

    for path in (train_log_path, eval_log_path, checkpoint_path):
        if os.path.exists(path):
            os.remove(path)

    return result, fit, weight_norm_actor, proxy_hit_rate


def run_sweep():
    os.makedirs(EVAL_LOGS_DIR, exist_ok=True)

    total_runs = len(K_VALUES) * len(ERR_VALUES)
    run_num = 0

    for k in K_VALUES:
        for err in ERR_VALUES:
            run_num += 1
            result, fit, wn_actor, proxy_hit_rate = run_one(k, err)
            fit_note = (
                f"L={fit['fit_L']:.3f} tau={fit['fit_tau']:.3g} R^2={fit['r_squared']:.3f}"
                if fit["fit_status"] == "ok" else "failed"
            )
            hit_flag = "" if result.hit_rate >= 1.0 else "  <-- hit_rate < 1.0!"
            print(
                f"[{run_num}/{total_runs}] k={k} s={S} err={err}: "
                f"hit_rate={result.hit_rate:.1%}  proxy_hit_rate={proxy_hit_rate:.1%}  "
                f"weight_norm_actor={wn_actor:.2f}  fit[{fit_note}]{hit_flag}"
            )

    print("Done - see weight_norm_data.csv for every row written "
          f"(group_type='margin', k in {K_VALUES[0]}..{K_VALUES[-1]}, s={S}).")


if __name__ == "__main__":
    run_sweep()
