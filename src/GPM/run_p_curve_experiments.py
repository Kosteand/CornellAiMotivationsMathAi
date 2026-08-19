"""
Estimate p(x) = P(a freshly-trained PPO policy "goes for" the VARIABLE
target | variable target's reward = x), as a smooth function of x, for
several two-group BanditEnv setups - complementary to
find_indifference_reward.py's single-point M estimate (which only asks
"where does p cross 0.5", assuming a parametric sigmoid shape). Here we
instead train at a handful of x-values, and let a Gaussian Process (fit
in plot_p_curve_results.py) interpolate/extrapolate the WHOLE curve with
an honest, non-parametric 95% confidence band - useful for seeing things
a single-point M estimate can't, e.g. how SHARP the transition is (steep
= the two targets are cleanly separated in "how much reward they need";
shallow = the trade-off is fuzzy/noisy), or whether the curve is even
sigmoid-shaped at all.

Definition of p, matching find_indifference_reward.py's own convention
(see that module's docstring): every action falls in EXACTLY ONE group's
action block (BanditEnv gives each group its own disjoint block), so
"went for variable" is unambiguous and always defined - it's
BanditEnv.info["chosen_group"] == 1 (the variable group is always index 1
here), regardless of whether the policy actually landed the correct
option within that block. This is a PREFERENCE statistic, not an
accuracy statistic - same thing find_indifference_reward.py's Y_hat
measures (just the complement: that module tracks P(went for FIXED), this
one tracks P(went for VARIABLE) directly, since that's what's strictly
increasing in x - "going for the bigger-reward target more often as its
reward grows" is the natural monotone quantity here).

Method
------
For each test config (see TESTS below):

  1. BRACKET SEARCH (cheap, few seeds per point): starting from
     x = value_fixed, walk outward in BOTH directions by doubling steps
     (x * 2, x * 4, ... and x / 2, x / 4, ...) until a low x is found
     where the pooled preference rate is confidently near 0, and a high x
     is found where it's confidently near 1. This exploits monotonicity
     exactly the way the user asked: p only needs to be traced from ~0 to
     ~100, never beyond, so once both extremes are bracketed the search
     stops - no dense sampling wasted on regions where the answer is
     already obviously 0 or 1.

  2. DENSE SAMPLING (most of the run budget): once [x_lo, x_hi] is
     bracketed, place N_DENSE_POINTS log-spaced x-values across it
     (log-spaced, not linear, since we plot/fit against log(x) - see
     module docstring rationale in plot_p_curve_results.py) and train
     N_SEEDS_DENSE fresh seeds at EACH of those points. Multiple seeds
     per point matter a lot here: find_indifference_reward.py's own
     docstring flags that trained policies show real run-to-run
     preference variance (a seed often fully commits to one target
     rather than landing near 50/50) - a single seed per x would let one
     idiosyncratic run dominate that point's estimate. Every seed's own
     y_hat (fraction of its hits_per_run eval episodes that went for
     variable) is kept as a SEPARATE observation (not just pooled into
     one point estimate) - that's what plot_p_curve_results.py's GP fits
     on directly, so the between-seed spread becomes part of the fitted
     uncertainty automatically, rather than being thrown away by
     averaging first.

Output: one CSV per test in OUTPUT_DIR, one row per (x, seed) - columns
"test", "x", "seed", "phase" ("bracket" or "dense"), "hits_per_run",
"chose_fixed", "chose_variable", "y_hat" (== chose_variable / hits_per_run,
i.e. this seed's own estimate of p at this x). plot_p_curve_results.py
reads these CSVs, fits the GP, plots, and prints an interpretation.

This file only TRAINS and EVALUATES - it deliberately does no plotting or
GP fitting itself, so a long training run (likely well over an hour for
all TESTS at the defaults below) can be kicked off standalone (e.g. in a
background/nohup process) and the analysis re-run separately/repeatedly
afterward without re-training anything.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# 2026-08-19: this file moved from src/ into src/GPM/ (alongside
# p_curve_data/p_curve_plots and the rest of the p-curve pipeline) as
# part of a repo cleanup. It still needs Utilities.bandit_env/trainPPO/
# find_indifference_reward, which now live in the self-contained
# src/M_comparison_background/ subfolder - inserting that folder onto
# sys.path (computed from THIS FILE's own location, not the current
# working directory) keeps those imports working regardless of where
# this script is invoked from.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "M_comparison_background",
))

from Utilities.bandit_env import MarginGroup, HeatmapGroup
from trainPPO import train
from find_indifference_reward import _evaluate_two_group_hits

# Resolved relative to this file's own location (not the current working
# directory) so p_curve_data/ (this script's sibling directory, now under
# src/GPM/) is found the same way whether this runs as
# `python3 run_p_curve_experiments.py` from within GPM/, or as
# `python3 GPM/run_p_curve_experiments.py` from src/, or from anywhere else.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p_curve_data")

# ----------------------------------------------------------------------
# TESTS - 7 two-group configs spanning: a symmetric MarginGroup baseline,
# an asymmetric-difficulty MarginGroup pair, a MarginGroup-vs-HeatmapGroup
# cross-family comparison, an asymmetric-difficulty HeatmapGroup pair, a
# MarginGroup pair isolating label noise (err), a MarginGroup pair
# isolating background-scale (s) mismatch, and a MarginGroup pair varying
# the number of options (g) - i.e. "two normal MarginGroup groups" first,
# then progressively "mixing it in with Heatmap groups, changing error
# rates and scales" as requested.
#
# Every test is a dict: {"name", "value_fixed", "incorrect_reward",
# "fixed_group": (value) -> Group, "variable_group": (value) -> Group}.
# `fixed_group`/`variable_group` are called with the group's OWN reward
# value only - value_fixed for the fixed side (always), and the swept x
# for the variable side - so a test just needs to say how each side's
# Group is built as a function of its own reward value.
# ----------------------------------------------------------------------
TESTS = [
    {
        "name": "margin_vs_margin_baseline",
        "value_fixed": 1.0,
        "incorrect_reward": 0.0,
        "fixed_group": lambda value: MarginGroup(g=4, k=1.0, value=value),
        "variable_group": lambda value: MarginGroup(g=4, k=1.0, value=value),
        # Same delta on both sides -> by the project's own symmetry
        # argument, p should cross 0.5 right at x = value_fixed = 1.0.
        # This is the sanity-check test: if the GP's fitted crossing
        # isn't close to 1.0 here, something upstream is wrong.
    },
    {
        "name": "margin_vs_margin_harder_variable",
        "value_fixed": 1.0,
        "incorrect_reward": 0.0,
        "fixed_group": lambda value: MarginGroup(g=4, k=1.0, value=value),
        "variable_group": lambda value: MarginGroup(g=4, k=4.0, value=value),
        # Variable side has a 4x smaller margin (harder to learn) - expect
        # the crossing to sit ABOVE x=1.0: the harder target needs extra
        # reward to compensate for its weaker learning signal.
    },
    {
        "name": "margin_vs_heatmap",
        "value_fixed": 1.0,
        "incorrect_reward": 0.0,
        "fixed_group": lambda value: MarginGroup(g=4, k=1.0, value=value),
        "variable_group": lambda value: HeatmapGroup(g=4, noise_scale=1.0, n=3, value=value),
        # Cross-family comparison - this project's central question in
        # miniature: how much reward does a HeatmapGroup target (a
        # fundamentally different kind of complexity - implicit
        # inversion, not margin/signal-detection) need to compete with a
        # MarginGroup target of "baseline" difficulty.
    },
    {
        "name": "heatmap_vs_heatmap_harder_variable",
        "value_fixed": 1.0,
        "incorrect_reward": 0.0,
        "fixed_group": lambda value: HeatmapGroup(g=4, noise_scale=1.0, n=2, value=value),
        "variable_group": lambda value: HeatmapGroup(g=4, noise_scale=1.0, n=6, value=value),
        # Both sides HeatmapGroup, variable has 3x more columns/powers to
        # invert (harder) - expect the crossing above x=1.0, and lets you
        # compare "how much reward per unit of extra n" against test 2's
        # "how much reward per unit of extra margin difficulty" on the
        # same p(x)-vs-log(x) axes.
    },
    {
        "name": "margin_vs_margin_label_noise",
        "value_fixed": 1.0,
        "incorrect_reward": 0.0,
        "fixed_group": lambda value: MarginGroup(g=4, k=1.0, value=value, err=0.0),
        "variable_group": lambda value: MarginGroup(g=4, k=1.0, value=value, err=0.2),
        # SAME margin as the fixed side - only difference is a 20%
        # per-episode chance the variable side's rewarded option isn't
        # the one its own observation actually points to (see
        # MarginGroup's `err`). Isolates label noise's own effect on
        # p(x), separate from margin/complexity per se.
    },
    {
        "name": "margin_vs_margin_scale_mismatch",
        "value_fixed": 1.0,
        "incorrect_reward": 0.0,
        "fixed_group": lambda value: MarginGroup(g=4, k=1.0, value=value, s=1.0),
        "variable_group": lambda value: MarginGroup(g=4, k=5.0, value=value, s=5.0),
        # SAME absolute margin (delta = s/k = 1.0) but a 5x larger random-score
        # backdrop (s=5.0) on the variable side - i.e. a 5x smaller
        # RELATIVE margin (delta/s). Isolates scale's own effect,
        # separate from changing delta directly (test 2).
    },
    {
        "name": "margin_vs_margin_more_options",
        "value_fixed": 1.0,
        "incorrect_reward": 0.0,
        "fixed_group": lambda value: MarginGroup(g=4, k=1.0, value=value),
        "variable_group": lambda value: MarginGroup(g=8, k=1.0, value=value),
        # Same margin, but the variable side has twice as many options
        # (g=8 vs 4) - a different "complexity" axis (bigger action
        # block / observation dimension for that side) than margin size,
        # heatmap depth, err, or scale.
    },
]

# ----------------------------------------------------------------------
# Budget / training knobs - tune these down for a quick smoke test, or up
# for the full run. At ~70 runs/test x 7 tests = ~490 runs total, and
# ~59s/run at the project's normal total_timesteps=200_000 (measured
# directly on this machine), that's several hours - see this file's
# docstring re: running it standalone/in the background.
# ----------------------------------------------------------------------
N_SEEDS_BRACKET = 2       # seeds per point during the cheap bracket search
N_SEEDS_DENSE = 8         # seeds per point during the dense sampling stage
RUN_BUDGET_PER_TEST = 70  # total training runs to spend per test (approx)
HITS_PER_RUN = 400        # eval episodes per trained seed (cheap; not counted in RUN_BUDGET)
BASE_SEED = 0
BRACKET_LOW_THRESHOLD = 0.10   # pooled y_hat below this = "confidently near 0"
BRACKET_HIGH_THRESHOLD = 0.90  # pooled y_hat above this = "confidently near 1"
MAX_BRACKET_STEPS = 8          # safety cap per direction (x *= 2 each step)
MIN_X_FLOOR = 1e-3             # never search x below this (keeps log(x) finite)
TRAIN_KWARGS = dict(
    total_timesteps=200_000,   # matches the rest of this project's calibration data
    verbose=0,
    progress_bar=False,
    log_training_data=False,
    save_model=False,
    eval_episodes=1,           # train()'s own built-in eval is unused here - we do our own
)


def _run_one_seed(test, x, seed, hits_per_run=HITS_PER_RUN, train_kwargs=None):
    """Train one fresh PPO policy at variable-group value=x and return
    (y_hat, chose_fixed, chose_variable) - y_hat = chose_variable /
    hits_per_run = this seed's own estimate of P(went for variable)."""
    train_kwargs = dict(train_kwargs or TRAIN_KWARGS)
    x = max(float(x), 0.0)
    groups = [test["fixed_group"](test["value_fixed"]), test["variable_group"](x)]
    result = train(groups, incorrect_reward=test["incorrect_reward"], seed=seed, **train_kwargs)
    hf, hv, misses, cf, cv = _evaluate_two_group_hits(
        result.model, groups, test["incorrect_reward"], hits_per_run,
    )
    y_hat = cv / hits_per_run
    return y_hat, cf, cv


def _pooled_mean(rows):
    """Mean y_hat across a list of (y_hat, cf, cv) tuples."""
    return float(np.mean([r[0] for r in rows])) if rows else float("nan")


def _find_bracket(test, seed_counter, n_seeds=N_SEEDS_BRACKET, hits_per_run=HITS_PER_RUN,
                   train_kwargs=None, rows_out=None):
    """Walk outward from x=value_fixed in both directions (doubling the
    step each time) until a low-x point with pooled y_hat <
    BRACKET_LOW_THRESHOLD and a high-x point with pooled y_hat >
    BRACKET_HIGH_THRESHOLD are both found. Every point evaluated along
    the way is appended to rows_out (so this cheap stage's data isn't
    wasted - it still feeds the GP later). Returns
    (x_lo, x_hi, seed_counter)."""
    x0 = test["value_fixed"] if test["value_fixed"] > 0 else 1.0

    def evaluate(x, phase):
        nonlocal seed_counter
        rows = []
        for _ in range(n_seeds):
            y_hat, cf, cv = _run_one_seed(test, x, seed_counter, hits_per_run, train_kwargs)
            rows_out.append({
                "test": test["name"], "x": x, "seed": seed_counter, "phase": phase,
                "hits_per_run": hits_per_run, "chose_fixed": cf, "chose_variable": cv,
                "y_hat": y_hat,
            })
            rows.append((y_hat, cf, cv))
            seed_counter += 1
        mean_y = _pooled_mean(rows)
        print(f"  [bracket:{test['name']}] x={x:.4f}  mean_y_hat_variable={mean_y:.3f}")
        return mean_y

    # Always evaluate the natural starting point once - informative
    # regardless of which direction the search ends up walking.
    mean_at_x0 = evaluate(x0, "bracket")

    # Search upward for x_hi (pooled y_hat confidently near 1).
    x_hi = x0
    mean_hi = mean_at_x0
    step = x0
    tries = 0
    while mean_hi < BRACKET_HIGH_THRESHOLD and tries < MAX_BRACKET_STEPS:
        x_hi = x_hi + step
        step *= 2.0
        mean_hi = evaluate(x_hi, "bracket")
        tries += 1

    # Search downward for x_lo (pooled y_hat confidently near 0).
    x_lo = x0
    mean_lo = mean_at_x0
    step = x0 / 2.0
    tries = 0
    while mean_lo > BRACKET_LOW_THRESHOLD and tries < MAX_BRACKET_STEPS and x_lo > MIN_X_FLOOR:
        x_lo = max(x_lo - step, MIN_X_FLOOR)
        step *= 2.0
        mean_lo = evaluate(x_lo, "bracket")
        tries += 1
        if x_lo <= MIN_X_FLOOR:
            break

    if mean_hi < BRACKET_HIGH_THRESHOLD:
        print(f"  WARNING [{test['name']}]: never confidently reached p~1 "
              f"(stopped at x={x_hi:.4f}, mean_y={mean_hi:.3f}) within "
              f"{MAX_BRACKET_STEPS} doubling steps - using the highest x "
              f"tried as x_hi anyway.")
    if mean_lo > BRACKET_LOW_THRESHOLD:
        print(f"  WARNING [{test['name']}]: never confidently reached p~0 "
              f"(stopped at x={x_lo:.4f}, mean_y={mean_lo:.3f}) - using the "
              f"lowest x tried (or the {MIN_X_FLOOR} floor) as x_lo anyway.")

    return x_lo, x_hi, seed_counter


def run_test(test, run_budget=RUN_BUDGET_PER_TEST, n_seeds_bracket=N_SEEDS_BRACKET,
             n_seeds_dense=N_SEEDS_DENSE, hits_per_run=HITS_PER_RUN,
             train_kwargs=None, base_seed=BASE_SEED):
    """Run the full bracket-then-dense-sampling procedure for one test
    config and return a DataFrame with one row per (x, seed) - see this
    module's docstring for the column list."""
    print(f"\n=== {test['name']} ===")
    rows = []
    seed_counter = base_seed

    t0 = time.time()
    x_lo, x_hi, seed_counter = _find_bracket(
        test, seed_counter, n_seeds=n_seeds_bracket, hits_per_run=hits_per_run,
        train_kwargs=train_kwargs, rows_out=rows,
    )
    bracket_runs = len(rows)
    print(f"  bracket found: x_lo={x_lo:.4f}  x_hi={x_hi:.4f}  "
          f"({bracket_runs} runs spent)")

    remaining_budget = max(run_budget - bracket_runs, n_seeds_dense)
    n_dense_points = max(remaining_budget // n_seeds_dense, 5)
    x_grid = np.geomspace(x_lo, x_hi, num=n_dense_points)
    print(f"  dense grid: {n_dense_points} points x {n_seeds_dense} seeds "
          f"= {n_dense_points * n_seeds_dense} more runs")

    for x in x_grid:
        x = float(x)
        for _ in range(n_seeds_dense):
            y_hat, cf, cv = _run_one_seed(test, x, seed_counter, hits_per_run, train_kwargs)
            rows.append({
                "test": test["name"], "x": x, "seed": seed_counter, "phase": "dense",
                "hits_per_run": hits_per_run, "chose_fixed": cf, "chose_variable": cv,
                "y_hat": y_hat,
            })
            seed_counter += 1
        this_x_rows = [r for r in rows if r["x"] == x and r["phase"] == "dense"]
        print(f"  [dense:{test['name']}] x={x:.4f}  "
              f"mean_y_hat_variable={np.mean([r['y_hat'] for r in this_x_rows]):.3f}")

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    total_runs = len(df)
    print(f"  done: {total_runs} total runs, {elapsed / 60.0:.1f} min "
          f"({elapsed / max(total_runs, 1):.1f} s/run)")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", nargs="*", default=None,
                         help="Subset of test names to run (default: all in TESTS).")
    parser.add_argument("--smoke-test", action="store_true",
                         help="Drastically reduced scale (short training, few seeds/"
                              "points) to sanity-check the pipeline runs end-to-end "
                              "without errors. NOT meant to produce meaningful curves.")
    args = parser.parse_args()

    tests_to_run = TESTS
    if args.tests:
        wanted = set(args.tests)
        tests_to_run = [t for t in TESTS if t["name"] in wanted]
        missing = wanted - {t["name"] for t in tests_to_run}
        if missing:
            raise SystemExit(f"Unknown test name(s): {missing} "
                              f"(available: {[t['name'] for t in TESTS]})")

    train_kwargs = dict(TRAIN_KWARGS)
    n_seeds_bracket, n_seeds_dense, run_budget, hits_per_run = (
        N_SEEDS_BRACKET, N_SEEDS_DENSE, RUN_BUDGET_PER_TEST, HITS_PER_RUN,
    )
    if args.smoke_test:
        train_kwargs["total_timesteps"] = 3_000
        n_seeds_bracket, n_seeds_dense, run_budget, hits_per_run = 1, 2, 10, 50
        print("*** SMOKE TEST MODE: tiny scale, results are meaningless, "
              "only checking the pipeline runs without errors. ***")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for test in tests_to_run:
        df = run_test(
            test, run_budget=run_budget, n_seeds_bracket=n_seeds_bracket,
            n_seeds_dense=n_seeds_dense, hits_per_run=hits_per_run,
            train_kwargs=train_kwargs,
        )
        out_path = os.path.join(OUTPUT_DIR, f"{test['name']}.csv")
        df.to_csv(out_path, index=False)
        print(f"  wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
