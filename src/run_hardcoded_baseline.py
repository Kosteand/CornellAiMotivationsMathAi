"""No-training baseline: evaluate a hand-set (not learned) decision rule on
the same MarginGroup(g=4, delta=1/k) sweep run_trainPPO_sweep.py runs, to
see how far a simple hardcoded rule gets without any PPO training at all.

Decision rule (HardcodedMarginModel): for each option i,

    score_i = 3 * x_i - (sum of the other 3 options' values)

then ReLU, then argmax over the g scores - same "shape" as the trained
policy (a linear map into per-action scores, a nonlinearity, then greedy
action selection), just with hand-set weights instead of learned ones.

Algebraically, score_i = 3*x_i - (total - x_i) = 4*x_i - total, where
`total` is the sum of ALL g values. Since every option's score has the
exact same constant (`total`) subtracted off, the scores are ranked in
exactly the same order as the raw x_i values - and MarginGroup's correct
answer is always argmax(x) by construction (see Utilities/bandit_env.py).
So this rule reproduces the correct answer regardless of the margin
(delta) size: it's a "perfect if you already know the trick" oracle, not
a learning curve. Comparing its flat ~100% line to the actual PPO sweep's
hit_rate curve shows exactly how much of that gap is genuine learning
difficulty for PPO versus how much is just PPO not having found this
closed-form solution.

By default this reads the k values to test from SWEEP_SUMMARY_CSV_PATH -
i.e. the exact k's your PPO sweep actually ran - so the two curves line up
point-for-point when plotted together (e.g. via plot_sweep_results.
plot_sweep_summary(csv_path="eval_logs/hardcoded_baseline_summary.csv")).
Set K_VALUES explicitly to test a different range instead.

Run:  python3 run_hardcoded_baseline.py
"""
import csv
import os

import numpy as np

from Utilities.bandit_env import MarginGroup
from trainPPO import evaluate

# --- config ---
G = 4
VALUE = 1.0

# k values to test. None => read from SWEEP_SUMMARY_CSV_PATH's "k" column
# (the same k's your PPO sweep ran). Set to an explicit list, e.g.
# [1.0, 2.0, 5.0, 10.0, 50.0], to test a different range instead.
K_VALUES = None
SWEEP_SUMMARY_CSV_PATH = "eval_logs/sweep_summary.csv"

EVAL_EPISODES = 500
INCORRECT_REWARD = 0.0

OUTPUT_CSV_PATH = "eval_logs/hardcoded_baseline_summary.csv"


class HardcodedMarginModel:
    """
    Hand-set (zero training) baseline for a BanditEnv built from a single
    MarginGroup. Implements the SB3-style `predict(obs, deterministic)`
    interface so it drops straight into trainPPO.evaluate() in place of a
    trained PPO model.

    g is inferred from the observation's length at predict time, so this
    works for any g, not just g=4 - the "multiply by 3, subtract the
    other 3" rule is the g=4 special case of the general
    score_i = (g-1)*x_i - sum_{j!=i} x_j = g*x_i - sum(x) rule.
    """

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        x = np.asarray(observation, dtype=np.float64).reshape(-1)
        total = float(x.sum())
        scores = len(x) * x - total  # score_i = (g-1)*x_i - sum_{j!=i} x_j
        scores = np.maximum(scores, 0.0)  # ReLU
        action = int(np.argmax(scores))
        return action, None


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
    model = HardcodedMarginModel()

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
