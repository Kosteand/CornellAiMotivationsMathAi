"""Hardcoded (no-training) baseline built from a REAL PPO ActorCriticPolicy
instance - net_arch_pi=(64, 32), Tanh activations, exactly matching
run_trainPPO_sweep.py's actual architecture - with hand-set actor weights.
The critic/value net is left at its random PyTorch initialization and never
touched: deterministic action selection only ever consults the actor, so
the value net's weights are structurally present (SB3 always builds one)
but completely irrelevant here.

Why not just re-embed the "3*x_i - sum(others)" linear rule directly, the
way run_hardcoded_baseline.py's standalone HardcodedMarginModel does?
Because that rule needs the raw x values to survive each Linear layer
UNCHANGED, and PPO's actual hidden activation is Tanh (SB3's MlpPolicy
default - trainPPO.py never overrides it). Tanh only behaves like the
identity function in a shrinking neighborhood around 0; away from that it
saturates, and no finite width undoes that distortion exactly - only
approximately, by keeping every intermediate value tiny.

The fix: don't try to preserve VALUES through the network - preserve
RANK. Tanh is strictly increasing (and odd: tanh(-t) = -tanh(t)), which
means it preserves order globally, at every magnitude, saturated or not:
tanh(a) > tanh(b) whenever a > b, full stop, no small-signal caveat
needed. So instead of passing x itself through, layer 1 computes every
PAIRWISE DIFFERENCE tanh(x_i - x_j) for each ordered pair of actions
(i, j), i != j (a "round robin tournament" - one hidden unit per
comparison). Layer 2 sums up, for each action i, its g-1 comparisons
against every other action, giving score_i = sum_{j != i} tanh(x_i - x_j),
then Tanh is applied again (SB3 activates every hidden layer). The final
action_net layer (plain Linear, no activation - SB3 never activates the
output layer) just passes those g values through unchanged into the
action logits.

Proof this reproduces argmax(x) exactly: for any two actions i, k with
x_i > x_k,

    score_i - score_k = 2*tanh(x_i - x_k) + sum_{j != i,k} [tanh(x_i - x_j) - tanh(x_k - x_j)]

using tanh's oddness to fold the two "self" terms together. The first term
is positive because tanh is odd and strictly increasing and x_i > x_k. Every
term in the remaining sum is also positive, because x_i - x_j > x_k - x_j
for every shared j, and tanh is strictly increasing - so a bigger argument
gives a strictly bigger tanh. Every term in score_i - score_k is therefore
positive, so score_i > score_k whenever x_i > x_k: score ranks actions
IDENTICALLY to x, exactly, for any inputs, saturated or not. Applying the
outer Tanh in layer 2 doesn't disturb this either, since it's just another
strictly increasing function applied to each score_i - it can't reorder
them. So argmax(final logits) = argmax(x) = MarginGroup's true label,
always - same oracle guarantee as run_hardcoded_baseline.py's version, just
realized as literal weights inside the real trained architecture instead
of a separate hand-written function.

Run:  python3 run_hardcoded_ppo_baseline.py
"""
import csv
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from Utilities.bandit_env import MarginGroup
from trainPPO import evaluate, make_env

# --- config ---
G = 4
VALUE = 1.0

# Must match (or exceed) the capacity this construction needs: layer 1
# needs >= g*(g-1) units (one per ordered pair), layer 2 needs >= g units
# (one per action's summed score). (64, 32) is run_trainPPO_sweep.py's
# actual architecture, and comfortably clears both (12 and 4 needed).
NET_ARCH_PI = (64, 32)

K_VALUES = None
SWEEP_SUMMARY_CSV_PATH = "eval_logs/sweep_summary.csv"

EVAL_EPISODES = 500
INCORRECT_REWARD = 0.0

OUTPUT_CSV_PATH = "eval_logs/hardcoded_ppo_baseline_summary.csv"


def build_hardcoded_ppo_model(g, net_arch_pi):
    """
    Build a real PPO model (MlpPolicy, Tanh hidden activations,
    net_arch_pi hidden layers) whose ACTOR weights are hand-set (never
    trained) to exactly reproduce argmax(x) via the pairwise-tanh-
    difference construction described in the module docstring. The
    critic/value net is left at its random init and never used.
    """
    if len(net_arch_pi) != 2:
        raise ValueError(
            "this construction is written for exactly 2 hidden layers "
            f"(got net_arch_pi={net_arch_pi}) - it maps directly onto "
            "policy_net[0] and policy_net[2]."
        )

    pairs = [(i, j) for i in range(g) for j in range(g) if i != j]
    if net_arch_pi[0] < len(pairs):
        raise ValueError(
            f"net_arch_pi[0]={net_arch_pi[0]} is too small - need at "
            f"least g*(g-1)={len(pairs)} units, one per ordered pair."
        )
    if net_arch_pi[1] < g:
        raise ValueError(
            f"net_arch_pi[1]={net_arch_pi[1]} is too small - need at "
            f"least g={g} units, one per action's summed score."
        )

    dummy_env = DummyVecEnv(
        [make_env([MarginGroup(g=g, delta=0.5, value=VALUE)], INCORRECT_REWARD)]
    )
    model = PPO(
        "MlpPolicy",
        dummy_env,
        policy_kwargs=dict(net_arch=dict(pi=list(net_arch_pi), vf=list(net_arch_pi))),
        device="cpu",
    )

    policy_net = model.policy.mlp_extractor.policy_net
    layer1 = policy_net[0]  # Linear(g, net_arch_pi[0])
    layer2 = policy_net[2]  # Linear(net_arch_pi[0], net_arch_pi[1])
    action_net = model.policy.action_net  # Linear(net_arch_pi[1], g), no activation

    with torch.no_grad():
        # Layer 1: one hidden unit per ordered pair (i, j) computes
        # (before its Tanh) x_i - x_j. Unused units (beyond len(pairs))
        # stay all-zero, i.e. constant 0 - harmless.
        layer1.weight.zero_()
        layer1.bias.zero_()
        for unit, (i, j) in enumerate(pairs):
            layer1.weight[unit, i] = 1.0
            layer1.weight[unit, j] = -1.0

        # Layer 2: hidden unit `action` (before its Tanh) sums every
        # pairwise-comparison unit whose first index is `action` -
        # i.e. score_action = sum_{j != action} tanh(x_action - x_j).
        # Unused units (beyond g) stay all-zero.
        layer2.weight.zero_()
        layer2.bias.zero_()
        for action in range(g):
            for unit, (i, j) in enumerate(pairs):
                if i == action:
                    layer2.weight[action, unit] = 1.0

        # action_net: plain linear pass-through of the g score units into
        # the g action logits - no activation here (SB3 never activates
        # the output layer), so this step introduces no distortion at all.
        action_net.weight.zero_()
        action_net.bias.zero_()
        for action in range(g):
            action_net.weight[action, action] = 1.0

    return model


def _load_k_values():
    if K_VALUES is not None:
        return list(K_VALUES)
    if not os.path.exists(SWEEP_SUMMARY_CSV_PATH):
        raise FileNotFoundError(
            f"{SWEEP_SUMMARY_CSV_PATH} not found and K_VALUES is None - "
            "either run run_trainPPO_sweep.py first, or set K_VALUES "
            "explicitly above."
        )
    with open(SWEEP_SUMMARY_CSV_PATH, newline="") as f:
        return [float(row["k"]) for row in csv.DictReader(f)]


def run_baseline():
    k_values = _load_k_values()
    model = build_hardcoded_ppo_model(G, NET_ARCH_PI)

    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH) or ".", exist_ok=True)
    with open(OUTPUT_CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["k", "hit_rate"])

    for k in k_values:
        groups = [MarginGroup(g=G, delta=1.0 / k, value=VALUE)]
        correct, episodes, _mean_reward = evaluate(
            model, groups, INCORRECT_REWARD, EVAL_EPISODES
        )
        hit_rate = correct / episodes

        with open(OUTPUT_CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([k, hit_rate])

        print(f"k={k}: hit_rate={hit_rate:.1%}")

    print(f"Baseline summary -> {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    run_baseline()
