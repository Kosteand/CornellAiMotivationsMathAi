"""
Adaptive replacement for run_p_curve_experiments.py's fixed bracket-then-
dense-grid procedure. Instead of a fixed run budget, this CONTINUOUSLY
trains more seeds - adding new x-values where needed, and adding more
seeds at existing x-values where needed - until, for each test:

  1. Both tails are CONFIRMED FLAT: the lowest sampled x has a CI entirely
     below FLAT_EPS (default 0.02), and the highest sampled x has a CI
     entirely above 1 - FLAT_EPS. (Not just "confidently on one side of
     0.5" the way run_p_curve_experiments.py's bracket search checked -
     genuinely flat, so the plotted curve visibly levels off at both
     ends, per your feedback that some plots didn't reach a flat p=0/1
     line.)
  2. Every gap between two adjacent sampled x-values has a CI width no
     wider than TARGET_CI_WIDTH (default 0.05) ANYWHERE inside that gap -
     see "Monotonicity-based confidence band" below for what "CI width
     inside a gap" means and why it's computable without sampling every
     point directly.

Once both conditions hold, it keeps going anyway (adding seeds to
whichever point has the loosest CI) until MAX_RUNS_PER_TEST (200) NEW
runs have happened for that test THIS INVOCATION - so you get the
tightest curve your 200-run budget can buy, not just the minimum that
clears the bar. Runs already on disk from a previous invocation (see
"Resumable" below) do NOT count against this budget - only runs added
during the current call to run_test_adaptive()/main() do. If 200 new
runs are reached before the conditions are ever satisfied, it stops
anyway (a safety cap, not a target) and says so.

Monotonicity-based confidence band (the "avoid over-sampling where we
already know p=0" part of your request)
-------------------------------------------------------------------------
p(x) is strictly increasing. So if x_i < x_j are two ALREADY-SAMPLED
points with their own (correctly-computed) confidence intervals
[lo_i, hi_i] and [lo_j, hi_j], then for ANY x with x_i <= x <= x_j:

    p(x_i) <= p(x) <= p(x_j)   (monotonicity)
    => lo_i <= p(x_i) <= p(x) <= p(x_j) <= hi_j   (chaining through the CIs)
    => p(x) is guaranteed to fall in [lo_i, hi_j] with the SAME confidence
       level, PROVIDED [lo_i, hi_i] and [lo_j, hi_j] themselves hold.

So the width of a VALID confidence band anywhere strictly between two
sampled points x_i and x_j is exactly (hi_j - lo_i) - constant across the
whole gap, not shrinking as you approach either point (there's no way to
do better without another sample in between). This is what "requirement
met" (#2 above) actually checks: max over every adjacent pair of
(hi_{i+1} - lo_i) <= TARGET_CI_WIDTH. It's also exactly how the flat
tails get extended for free: once x_lo's CI is entirely below FLAT_EPS,
EVERY x < x_lo also has p(x) <= hi(x_lo) < FLAT_EPS by the same argument -
i.e. once you've nailed down "p is ~0 here," you already know it's ~0
everywhere further down too, with NO extra sampling. This is a simple
heuristic envelope, not a rigorously calibrated SIMULTANEOUS confidence
band (each point's own CI is computed independently, at its own nominal
95%) - consistent with the general rigor level of the confidence-sequence
math already in this project's find_indifference_reward.py, but worth
knowing if you want to push this further later.

Why this replaces a GP's predictive std as the plotted band: away from
data, a GP's predictive variance is governed by its fitted kernel
hyperparameters, not the actual data density - it can come out either far
too wide OR (misleadingly) tight depending on what the optimizer happened
to fit, which is exactly the "weirdly tight in places with no data" you
flagged. The monotonicity envelope above only ever reflects what's
actually been measured (plus the one monotonicity assumption baked into
this whole project - p is strictly increasing in the target's own
reward), so it can't misbehave that way. plot_p_curve_results.py still
fits a GP for the MEAN curve (to visualize the shape - logistic vs
linear vs something else), but plots THIS envelope as the band.

Per-point confidence interval: each point can be tightened by adding
MORE SEEDS. Its own CI takes the WIDER of two independent estimates - so
whichever source of uncertainty is currently bigger dominates, rather
than either one being silently ignored:

  - Pooled Wilson interval on ALL episodes across ALL seeds at that x
    (total trials = n_seeds * hits_per_run) - captures ordinary sampling
    noise, and shrinks as more seeds/episodes pile up.
  - The seed-to-seed spread itself: mean +/- 1.96 * (sample std of each
    seed's own y_hat across seeds) / sqrt(n_seeds) - this is what
    actually captures run-to-run POLICY variance (find_indifference_
    reward.py's own docs flag this as a real, large effect - trained
    policies often fully commit to one target rather than landing near
    50/50), which a pooled-episode count alone would understate if
    different seeds behave very differently. Requires >= 2 seeds to even
    define - with only 1 seed this contributes [0, 1] (maximally
    uninformative), which is why every new x-value always gets at least
    2 starter seeds before its CI is trusted at all.

Resumable: if data for a test already exists (e.g. from a previous run of
this script OR run_p_curve_experiments.py) at OUTPUT_DIR/{test}.csv, it's
loaded and built on top of - existing points get MORE seeds added where
their CI needs tightening, rather than starting over. Pre-existing runs
do NOT count against this invocation's 200-new-run budget; each call to
run_test_adaptive() adds up to 200 MORE runs on top of whatever was
already there.

Two mitigations for a real observed "overconfidence" problem
-------------------------------------------------------------------------
Real runs showed individual points' CIs sometimes far too tight relative
to how much nearby (even near-duplicate-x) points disagreed - e.g. one
x=1.273 batch reporting p in [0.997, 1.0] right next to another x=1.273
batch (from a later insert landing almost on top of the first) reporting
p in [0.54, 0.85]. Root cause: near the indifference point, individual
SEEDS behave bimodally (each training run fully commits to one target
rather than landing near 50/50 - see seed_sem_ci's docstring), so a
SMALL batch of seeds has a real chance of landing unanimously (or
near-unanimously) on one side purely by luck, which makes that batch's
own sample variance - and therefore its CI - look artificially tight
even though the true variance at that x is large. Two changes address
this without touching the Wilson/pseudo-replication side of the CI:

  1. STEEP-REGION STARTING SEED COUNT (see steep_insert_seed_count()): a
     freshly-inserted point starts with MORE than the usual
     INITIAL_SEEDS_NEW_POINT seeds when the gap it's splitting looks like
     a real, steep transition (i.e. its width is mostly "signal", not
     already explained by its endpoints' own noise) rather than a flat
     region. More starting seeds directly lowers the chance of an
     unlucky unanimous small batch (P(unanimous) ~ 2*0.5^n falls from
     12.5% at n=4 to well under 1% at n=8+). Tail-extension and bootstrap
     points, which have no gap to gauge steepness from yet, still use the
     plain INITIAL_SEEDS_NEW_POINT.

  2. VARIANCE SHRINKAGE TOWARD NEIGHBORING POINTS (see
     seed_sem_ci_shrunk()): a point's seed-to-seed variance estimate is
     blended with a "regional" variance estimate pooled from its
     immediate left/right neighbors' own seeds, weighted by
     n_own / (n_own + VARIANCE_SHRINKAGE_PRIOR_N) in favor of the point's
     own data. With only a handful of seeds, the point's own sample
     variance is unreliable and the (typically larger, more honest)
     neighboring variance pulls the CI wider; once a point has plenty of
     its own seeds (e.g. 100), the weight on the borrowed regional
     variance shrinks toward zero and the point's own data dominates
     completely, matching that a point that's genuinely been measured a
     lot needs no help from its neighbors.

A separate, related problem: STOP SAMPLING WHERE THE SHAPE IS ALREADY
RESOLVED (see is_locally_resolved() / RESOLVED_GAP_FRAC_OF_RANGE)
-------------------------------------------------------------------------
Near an indifference point, a point's own CI HEIGHT can stay wide almost
indefinitely (the irreducible seed-to-seed variance discussed above and
in the MAX_SEEDS_PER_POINT_FRAC comment) - but that's not the same as the
curve's SHAPE being under-resolved there. If a region already has many
closely-spaced sampled x-values (fine x-RESOLUTION), the shape is already
strongly pinned down regardless of how tall each point's own CI still is
- further tightening or inserting there teaches nothing new about the
shape, even though the old scoring (purely "would this shrink a CI
height") kept treating it as high-value. This was a real observed bug:
one test kept grinding a visually-already-resolved transition region
(dozens of x-values spaced ~1e-4 apart in log(x)) while genuinely
under-resolved gaps elsewhere in the same run went untouched. The fix:
once a gap between two sampled points is smaller than
RESOLVED_GAP_FRAC_OF_RANGE of the test's whole bracketed log(x) range,
that gap is no longer eligible for INSERT, and neither endpoint is
eligible for TIGHTEN if every gap touching it is that small too - see
is_locally_resolved()'s docstring for the exact rule. A point with a
still-wide neighboring gap on even one side is untouched by this and
stays fully eligible, since there's genuine unresolved territory nearby.

Run: python3 sample_p_curve_adaptive.py [--tests NAME ...] [--smoke-test]
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

# 2026-08-19: see run_p_curve_experiments.py's matching comment - this
# file also moved into src/GPM/. `from run_p_curve_experiments import ...`
# below already triggers that file's own sys.path fix as a side effect of
# importing it, but this is added directly too for robustness (e.g. if
# this file is ever imported before run_p_curve_experiments.py for any
# reason).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_p_curve_experiments import TESTS, _run_one_seed, TRAIN_KWARGS, HITS_PER_RUN

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p_curve_data")
Z_95 = 1.959963985

TARGET_CI_WIDTH = 0.05     # max allowed width anywhere inside a gap between sampled points
FLAT_EPS = 0.02            # a tail counts as "confirmed flat" once its CI is entirely within this of 0/1
MAX_RUNS_PER_TEST = 200    # hard cap AND the target once requirements are met (see module docstring) -
                           # counts only NEW runs added THIS invocation; pre-existing runs loaded via
                           # resume do NOT count against this budget (see run_test_adaptive's new_runs()).
SEED_BATCH = 4             # seeds added per "tighten an existing point" step
INITIAL_SEEDS_NEW_POINT = 4  # starter seeds for a brand-new x-value (bootstrap, or tail extension, or insert)
MIN_X_FLOOR = 1e-4          # never search x below this (keeps log(x) finite; also a safety floor
                            # for the tail-extension loop, which would otherwise halve forever)
MIN_SEEDS_FOR_INSERT_ELIGIBILITY_MULT = 2  # a gap's two endpoints must each have at least
                            # this many times initial_seeds_new_point seeds before that gap
                            # is eligible for INSERT scoring at all - see the big comment
                            # above the insert-candidate loop in run_test_adaptive() for why
                            # (avoids a real observed bug: fresh, few-seed points can look
                            # artificially tight by chance, triggering a cascade of inserts
                            # into an already-fine-enough region instead of tightening first).

VARIANCE_SHRINKAGE_PRIOR_N = INITIAL_SEEDS_NEW_POINT  # equivalent pseudo-seed-count given
                            # to the REGIONAL (neighbor-borrowed) variance estimate when
                            # shrinking a point's own seed-to-seed variance toward it - see
                            # seed_sem_ci_shrunk()'s docstring. Tied to INITIAL_SEEDS_NEW_POINT
                            # so the prior's "weight" is comparable to one fresh starter batch;
                            # a point's own data dominates once it has much more than this.
MAX_INITIAL_SEEDS_STEEP_MULT = 3  # cap on how many times INITIAL_SEEDS_NEW_POINT a freshly
                            # inserted point may start with when the gap it's splitting
                            # looks maximally steep - see steep_insert_seed_count().

RESOLVED_GAP_FRAC_OF_RANGE = 0.0025  # once a gap between two sampled x-values is already
                            # this small a fraction of the test's whole bracketed log(x)
                            # range, treat it (and the points touching it) as "shape-
                            # resolved" and NEVER spend more budget there - not on
                            # inserting further into it, and not on tightening either
                            # endpoint - regardless of how wide that endpoint's own CI
                            # HEIGHT still is. See is_locally_resolved()'s docstring for
                            # the full reasoning (this was a real observed bug: the
                            # sampler kept grinding on an already visually-resolved
                            # transition because its CI height never met TARGET_CI_WIDTH,
                            # even though its x-RESOLUTION had long since been more than
                            # enough to pin down the curve's shape). 0.0025 (~1/400) is
                            # chosen to roughly match plot_p_curve_results.py's own
                            # N_GRID=400 plotting resolution - there's no point resolving
                            # x finer than what the plot itself can even render.

# --- per-point "give up" cap (fixes the runaway-tightening bug) ---
# See the big comment in run_test_adaptive() for WHY this exists: near
# p(x)~0.5, seed-to-seed policy variance (different training seeds fully
# committing to different targets) can genuinely require thousands of
# seeds to shrink to TARGET_CI_WIDTH - an unattainable amount inside any
# realistic budget. Without a cap, the loop always re-picks whichever
# point/gap is "worst," which is always this same point, so it burns the
# ENTIRE budget on one x-value and never improves anything else. This
# caps how many seeds any single x-value may receive from TIGHTENING
# actions (tail-tighten, gap-tighten, refine-extra) before it's marked
# "accepted" and the algorithm moves on to whatever it actually CAN fix
# with the budget remaining. Expressed as a fraction of max_runs so it
# scales with --smoke-test's tiny budget too.
MAX_SEEDS_PER_POINT_FRAC = 0.15
MIN_MAX_SEEDS_PER_POINT = 20

# NOTE: an earlier version of this file had a separate "per-gap
# affordability gate" here (skip a gap outright if a cost-estimate said
# fixing it would take more than some fraction of the budget). That's
# gone now - it's superseded by run_test_adaptive()'s BEST-EXPECTED-
# IMPROVEMENT-PER-RUN scoring (see the big comment in that function),
# which achieves the same "don't chase hopeless gaps" effect continuously
# or via a hard per-attempt gate: a gap whose width is mostly explained by
# its endpoints' own (possibly irreducible) noise scores near zero for
# insertion automatically, without needing a separate cutoff.


def wilson_ci(successes, n, z=Z_95):
    """Wilson score interval for a binomial proportion - well-behaved
    (unlike the naive normal approximation) even when successes is 0 or
    n, which happens constantly here (many seeds land exactly 0% or 100%
    at the extremes)."""
    if n <= 0:
        return 0.0, 1.0
    lo, hi = proportion_confint(successes, n, alpha=0.05, method="wilson")
    return float(max(lo, 0.0)), float(min(hi, 1.0))


def seed_sem_ci(y_hats, z=Z_95):
    """95% CI on the MEAN of per-seed y_hats, from their own sample
    variance across seeds - captures run-to-run policy variance (see
    module docstring). [0, 1] (maximally wide - "we don't know yet") if
    fewer than 2 seeds, since sample variance is undefined with n=1."""
    n = len(y_hats)
    if n < 2:
        return 0.0, 1.0
    mean = float(np.mean(y_hats))
    sem = float(np.std(y_hats, ddof=1) / np.sqrt(n))
    return max(mean - z * sem, 0.0), min(mean + z * sem, 1.0)


def seed_sem_ci_shrunk(y_hats, neighbor_y_hats, prior_n=VARIANCE_SHRINKAGE_PRIOR_N, z=Z_95):
    """Like seed_sem_ci, but the sample VARIANCE used to build the CI is
    shrunk toward a regional estimate borrowed from this point's
    immediate neighbors (see module docstring's "overconfidence" section
    for the full motivation). own_var and regional_var are blended as

        var_used = w * own_var + (1 - w) * regional_var,
        w = n / (n + prior_n)

    so the regional estimate matters most when this point has few of its
    own seeds (n small -> w small) and its influence VANISHES as this
    point's own seed count grows (n >> prior_n -> w -> 1, var_used ->
    own_var) - a point with plenty of its own data needs no help from
    its neighbors, by construction, regardless of what its neighbors
    look like. The SEM itself still divides by this point's own n (not
    n + prior_n): the neighbors only inform WHAT VARIANCE TO ASSUME, they
    don't pretend to be extra observations of this point's own mean.
    Falls back to the plain own-variance-only estimate if fewer than 2
    neighbor y_hats are available (nothing usable to borrow yet)."""
    n = len(y_hats)
    if n < 2:
        return 0.0, 1.0
    mean = float(np.mean(y_hats))
    own_var = float(np.var(y_hats, ddof=1))
    if len(neighbor_y_hats) >= 2:
        regional_var = float(np.var(neighbor_y_hats, ddof=1))
        w = n / (n + prior_n)
        var_used = w * own_var + (1.0 - w) * regional_var
    else:
        var_used = own_var
    sem = float(np.sqrt(var_used / n))
    return max(mean - z * sem, 0.0), min(mean + z * sem, 1.0)


def point_ci(rows_at_x, neighbor_y_hats=None):
    """This x-value's own confidence interval: the WIDER of the pooled
    Wilson interval and the seed-to-seed interval (see module docstring
    for why both are computed and why the wider one is used). When
    `neighbor_y_hats` (y_hats pooled from this point's immediate sampled
    neighbors) is given, the seed-to-seed interval uses the
    variance-shrinkage version (seed_sem_ci_shrunk) instead of the plain
    one - see its docstring. Returns (lo, hi, n_seeds, n_episodes_total).
    """
    y_hats = [r["y_hat"] for r in rows_at_x]
    total_success = sum(r["chose_variable"] for r in rows_at_x)
    total_n = sum(r["hits_per_run"] for r in rows_at_x)
    w_lo, w_hi = wilson_ci(total_success, total_n)
    if neighbor_y_hats:
        s_lo, s_hi = seed_sem_ci_shrunk(y_hats, neighbor_y_hats)
    else:
        s_lo, s_hi = seed_sem_ci(y_hats)
    return min(w_lo, s_lo), max(w_hi, s_hi), len(y_hats), total_n


def compute_all_point_cis(points):
    """CI for every sampled x in `points` (a dict x -> list of row
    dicts), using the neighbor-informed variance-shrinkage version of
    point_ci - each point's regional prior is borrowed from its
    immediate left/right neighbors among these SAME points. Returns
    {x: (lo, hi, n_seeds, n_episodes_total)}."""
    xs = sorted(points.keys())
    cis = {}
    for i, x in enumerate(xs):
        neighbor_y_hats = []
        if i > 0:
            neighbor_y_hats.extend(r["y_hat"] for r in points[xs[i - 1]])
        if i < len(xs) - 1:
            neighbor_y_hats.extend(r["y_hat"] for r in points[xs[i + 1]])
        cis[x] = point_ci(points[x], neighbor_y_hats)
    return cis


def steep_insert_seed_count(signal, base=INITIAL_SEEDS_NEW_POINT,
                             max_mult=MAX_INITIAL_SEEDS_STEEP_MULT):
    """How many seeds to start a freshly-inserted point with, scaled up
    when the gap being split looks like a steep/uncertain transition
    rather than a flat, already-well-explained one (see module
    docstring's "overconfidence" section). `signal` is the gap's width
    NOT already explained by its endpoints' own noise (the same quantity
    used to score insert candidates) - a gap dominated by signal rather
    than by own_i/own_j noise is exactly the kind of real, sharp
    transition where a lucky/unlucky small starting batch would do the
    most damage. Scales linearly from `base` (signal=0) up to
    `base * max_mult` as signal approaches its plausible max of 1.0."""
    scale = 1.0 + (max_mult - 1.0) * min(max(signal, 0.0), 1.0)
    return int(round(base * scale))


def resolved_gap_log_x(xs, frac=RESOLVED_GAP_FRAC_OF_RANGE):
    """The log(x) gap width, below which a gap is considered
    "shape-resolved" (see RESOLVED_GAP_FRAC_OF_RANGE's docstring) - a
    fixed fraction of the test's current full bracketed log(x) range
    (xs[0] to xs[-1]). Recomputed fresh each call since the bracket can
    still be growing during tail extension; by phase 2 (where this is
    actually used) it's settled."""
    if len(xs) < 2:
        return 0.0
    return frac * (np.log(xs[-1]) - np.log(xs[0]))


def is_locally_resolved(xs, k, min_gap_log_x):
    """True if xs[k] is already boxed in on EVERY side it has a neighbor
    by a gap <= min_gap_log_x - i.e. this point's local x-RESOLUTION is
    already fine enough to strongly pin down the curve's shape there, so
    no further sampling (tightening this point's own CI further, or
    inserting yet another point even closer) would add meaningful new
    shape information - regardless of how wide this point's own CI
    HEIGHT still is. This is the fix for a real observed bug: near an
    indifference point, p's CI height can stay wide almost indefinitely
    (irreducible seed-to-seed variance - see the big comment on
    MAX_SEEDS_PER_POINT_FRAC), so scoring purely on "will this shrink the
    reported CI height" kept the sampler grinding on a region that was
    ALREADY visually resolved by its sheer sampling density, while other,
    genuinely under-resolved regions sat untouched. A point with a still-
    wide neighboring gap on at least one side is NOT considered resolved
    (there's real unresolved territory on that side), so it stays
    eligible for both tighten and insert scoring as before."""
    gaps = []
    if k > 0:
        gaps.append(np.log(xs[k]) - np.log(xs[k - 1]))
    if k < len(xs) - 1:
        gaps.append(np.log(xs[k + 1]) - np.log(xs[k]))
    if not gaps:
        return False  # no neighbors at all yet - can't be "resolved"
    return all(g <= min_gap_log_x for g in gaps)


def pooled_mean(rows_at_x):
    """Plain pooled point estimate (chose_variable / hits_per_run,
    summed across every seed at this x) - used ONLY to decide whether a
    boundary point's own mean already looks flat (in which case it just
    needs a TIGHTER CI, via more seeds at the SAME x) versus genuinely
    not flat yet (in which case we need to extend to a new x further
    out). Using the CI itself for this decision would never let a
    boundary point "graduate" to flat by adding more seeds - a Wilson
    upper bound on 0 successes out of n stays a few percent above 0
    until n is fairly large (e.g. ~190 for the bound to clear 2%), so
    checking hi<=flat_eps alone would keep pushing to new, redundant
    lower/higher x's instead of just tightening the one we already have."""
    total_success = sum(r["chose_variable"] for r in rows_at_x)
    total_n = sum(r["hits_per_run"] for r in rows_at_x)
    return total_success / total_n if total_n > 0 else float("nan")


def load_existing(test_name):
    path = os.path.join(OUTPUT_DIR, f"{test_name}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df.to_dict("records")
    return []


def run_test_adaptive(test, hits_per_run=HITS_PER_RUN, train_kwargs=None,
                       target_width=TARGET_CI_WIDTH, flat_eps=FLAT_EPS,
                       max_runs=MAX_RUNS_PER_TEST, seed_batch=SEED_BATCH,
                       initial_seeds_new_point=INITIAL_SEEDS_NEW_POINT,
                       csv_path=None):
    print(f"\n=== {test['name']} ===")
    rows = load_existing(test["name"])
    existing_count = len(rows)  # runs already on disk BEFORE this invocation - excluded from the budget below
    if rows:
        print(f"  resuming from {existing_count} existing runs (NOT counted against this "
              f"invocation's {max_runs}-new-run budget - this run will add up to {max_runs} more)")
    seed_counter = (max((r["seed"] for r in rows), default=-1) + 1)
    points = defaultdict(list)
    for r in rows:
        points[r["x"]].append(r)

    # --- per-point give-up cap (see MAX_SEEDS_PER_POINT_FRAC's docstring
    # above). WHY THIS IS NEEDED: right at/near the true indifference point,
    # p(x) ~ 0.5. Individual trained policies tend to fully COMMIT to one
    # target rather than landing near 50/50 (this is a real, previously-
    # documented effect - see find_indifference_reward.py's own docstring
    # and this module's docstring on seed_sem_ci). So at that one x-value,
    # roughly half the SEEDS land near y_hat=0 and half near y_hat=1 - the
    # across-seed standard deviation approaches its theoretical max (~0.5),
    # not the near-zero std you'd see away from the indifference point. The
    # seed-mean's 95% half-width there is ~1.96*0.5/sqrt(n_seeds), so
    # shrinking it to the target (0.025) needs n_seeds ~ (1.96*0.5/0.025)^2
    # ~= 1537 SEEDS - not episodes, SEEDS (more episodes per seed doesn't
    # help at all, since a single trained policy's greedy choice is highly
    # consistent across its own episodes; only training MORE independent
    # policies reduces this). That's ~1537 * hits_per_run runs just for one
    # x-value, far past any realistic budget. Without a cap, the old loop
    # always re-selects "whichever point/gap is worst," which is always
    # this same point (it never stops being the worst), so 100% of the
    # budget got poured into it while every other gap sat untouched - this
    # IS the "kept sampling at the same point over and over" bug. It's also
    # why the "estimated runs left" kept climbing rather than shrinking:
    # each re-estimate recomputes the sample std from the (now larger, more
    # representative) pool of seeds, and small early samples of a ~0.5-std
    # bimodal population systematically under-estimate the true std by
    # chance - so the projection kept revising itself upward as reality
    # caught up, rather than the code doing anything internally
    # inconsistent. The fix isn't a smarter estimate (the estimate was
    # revealing a true, large number) - it's giving the algorithm
    # permission to accept "this point is as tight as this budget affords"
    # and spend the rest of the budget where it actually helps.
    max_seeds_per_point = max(MIN_MAX_SEEDS_PER_POINT,
                               int(max_runs * MAX_SEEDS_PER_POINT_FRAC))
    given_up = set()  # x-values that hit the cap; excluded from future tighten targets

    t0 = time.time()

    def new_runs():
        """Runs added DURING this invocation only - excludes existing_count
        (the runs already on disk before this call started), per the
        "200 NEW runs, don't count already-existing ones" requirement."""
        return len(rows) - existing_count

    def remaining_budget():
        return max(max_runs - new_runs(), 0)

    def seeds_at(x):
        return len(points[x])

    def add_seeds(x, n, phase):
        """Adds up to n seeds at x, but NEVER more than remaining_budget()
        - this is what actually enforces "~200 NEW runs, never over" (the old
        code could overshoot max_runs by up to a full batch). Returns the
        number of seeds actually added (0 if the budget was already
        exhausted) so callers can tell when nothing happened."""
        nonlocal seed_counter
        n = min(n, remaining_budget())
        for _ in range(n):
            y_hat, cf, cv = _run_one_seed(test, x, seed_counter, hits_per_run, train_kwargs)
            row = {
                "test": test["name"], "x": x, "seed": seed_counter, "phase": phase,
                "hits_per_run": hits_per_run, "chose_fixed": cf, "chose_variable": cv,
                "y_hat": y_hat,
            }
            rows.append(row)
            points[x].append(row)
            seed_counter += 1
        if n > 0 and csv_path is not None:
            pd.DataFrame(rows).to_csv(csv_path, index=False)  # checkpoint after every batch
        if seeds_at(x) >= max_seeds_per_point and x not in given_up:
            given_up.add(x)
            print(f"  [{test['name']}] log(x)={np.log(x):.5f} hit the per-point cap of "
                  f"{max_seeds_per_point} seeds without reaching the target CI width - "
                  f"accepting its current CI and moving budget elsewhere.")
        return n

    if not points:
        x0 = test["value_fixed"] if test["value_fixed"] > 0 else 1.0
        add_seeds(x0, initial_seeds_new_point, "bootstrap")

    while new_runs() < max_runs:
        xs = sorted(points.keys())
        cis = compute_all_point_cis(points)

        # --- 1. tail extension: keep pushing outward until both ends are
        # confirmed flat (see module docstring). At each boundary, first
        # check whether its own POINT ESTIMATE already looks flat - if
        # so, the fix is to TIGHTEN it (more seeds at the same x), not
        # push further out (see pooled_mean's docstring for why checking
        # the CI alone would never let a boundary point "graduate": a
        # Wilson upper bound on 0 successes stays a few percent above 0
        # until n is fairly large). Only extend to a genuinely new x when
        # the boundary's own mean hasn't reached the flat threshold yet.
        # If a boundary has hit its per-point cap, accept it as flat
        # (best effort) rather than grinding on it forever. ---
        x_lo, x_hi = xs[0], xs[-1]
        lo_flat = cis[x_lo][1] <= flat_eps or x_lo in given_up
        hi_flat = cis[x_hi][0] >= 1.0 - flat_eps or x_hi in given_up

        if not lo_flat:
            if pooled_mean(points[x_lo]) <= flat_eps:
                if add_seeds(x_lo, seed_batch, "tighten_tail_low") == 0:
                    break
                new_ci = compute_all_point_cis(points)[x_lo]
                if x_lo in given_up and new_ci[1] > flat_eps:
                    # Same escape hatch as the MIN_X_FLOOR case below, just
                    # triggered from the OTHER direction: the per-point cap
                    # kicked in before this tail's CI actually cleared
                    # flat_eps. lo_flat will read True next iteration (via
                    # "or x_lo in given_up") and the algorithm will move on,
                    # but - unlike a normal interior point hitting the cap -
                    # this means the LOW TAIL IS NOT ACTUALLY CONFIRMED FLAT,
                    # only accepted as a best-effort stand-in. Said explicitly
                    # here (not just the generic "hit the per-point cap"
                    # message every add_seeds call already prints) so this
                    # doesn't silently masquerade as a genuine confirmation.
                    print(f"  [{test['name']}] WARNING: log(x_lo)={np.log(x_lo):.5f} hit its "
                          f"per-point cap while still not confirmed flat (CI hi="
                          f"{new_ci[1]:.4f} > flat_eps={flat_eps}) - accepting it as the "
                          f"low tail anyway; this test's low end may not truly reach 0.")
                print(f"  [{test['name']}] {100.0 * new_runs() / max_runs:.1f}% - "
                      f"log(x_lo)={np.log(x_lo):.5f} already looks flat but CI still too "
                      f"wide ({cis[x_lo][0]:.4f}, {cis[x_lo][1]:.4f}) - tightening it.")
                continue
            new_x = x_lo / 2.0
            if new_x < MIN_X_FLOOR:
                print(f"  [{test['name']}] WARNING: x_lo hit the {MIN_X_FLOOR} floor "
                      f"without confirming p~0 (current CI={cis[x_lo][:2]}) - treating "
                      f"the low tail as flat anyway; this test's low end may not truly "
                      f"reach 0.")
                lo_flat = True
            else:
                if add_seeds(new_x, initial_seeds_new_point, "extend_low") == 0:
                    break
                print(f"  [{test['name']}] {100.0 * new_runs() / max_runs:.1f}% - "
                      f"extending low tail to log(x)={np.log(new_x):.5f}")
                continue

        if not hi_flat:
            if pooled_mean(points[x_hi]) >= 1.0 - flat_eps:
                if add_seeds(x_hi, seed_batch, "tighten_tail_high") == 0:
                    break
                new_ci = compute_all_point_cis(points)[x_hi]
                if x_hi in given_up and new_ci[0] < 1.0 - flat_eps:
                    print(f"  [{test['name']}] WARNING: log(x_hi)={np.log(x_hi):.5f} hit its "
                          f"per-point cap while still not confirmed flat (CI lo="
                          f"{new_ci[0]:.4f} < 1-flat_eps={1.0 - flat_eps}) - accepting it "
                          f"as the high tail anyway; this test's high end may not truly "
                          f"reach 1.")
                print(f"  [{test['name']}] {100.0 * new_runs() / max_runs:.1f}% - "
                      f"log(x_hi)={np.log(x_hi):.5f} already looks flat but CI still too "
                      f"wide ({cis[x_hi][0]:.4f}, {cis[x_hi][1]:.4f}) - tightening it.")
                continue
            new_x = x_hi * 2.0
            if add_seeds(new_x, initial_seeds_new_point, "extend_high") == 0:
                break
            print(f"  [{test['name']}] {100.0 * new_runs() / max_runs:.1f}% - "
                  f"extending high tail to log(x)={np.log(new_x):.5f}")
            continue

        # --- 2. BEST-EXPECTED-IMPROVEMENT-PER-RUN: both tails are flat
        # (or accepted). Instead of ranking gaps worst-width-first (what
        # this used to do, replacing the earlier "affordability gate"
        # patch entirely - see below for why that's no longer needed),
        # every possible action - tightening any live point, OR inserting
        # into any gap - is scored by its ESTIMATED WIDTH REDUCTION PER
        # RUN SPENT, and the single best-scoring action is taken. This is
        # re-computed fresh every iteration, so priority continuously
        # shifts to wherever the next batch of seeds would help most.
        #
        # MONOTONICITY BONUS (this is the "getting more data near the
        # midpoint helps tighten OTHER gaps too" part): the monotonicity
        # envelope (see module docstring) makes each sampled point's own
        # CI do double duty - its HI bound is the right edge of the gap
        # to its LEFT, and its LO bound is the left edge of the gap to its
        # RIGHT. Tightening a point's own CI narrows both bounds roughly
        # symmetrically (~half the width reduction on each side), so an
        # INTERIOR point (with a gap on both sides) delivers its full
        # width-reduction as combined benefit to two neighboring gaps at
        # once, while a point at either end of the sampled range only
        # benefits its single neighboring gap. A point flanked by two
        # still-wide gaps is therefore worth roughly TWICE as much per run
        # as an equivalent edge point - exactly the effect you described.
        #
        # TIGHTEN score: estimate the own-CI shrink from adding seed_batch
        # more seeds via the same ~1/sqrt(n) scaling used throughout this
        # pipeline (own_after = own * sqrt(n/(n+batch))), split half to
        # each live neighbor, divided by the run cost (seed_batch).
        #
        # INSERT score: of a gap's current width, only the portion NOT
        # already explained by its two endpoints' own noise (own_i, own_j)
        # is something insertion could ever address - the rest is
        # irreducible per-point variance that no amount of x-resolution
        # fixes (this is the "signal vs noise" distinction from the
        # THRESHOLD FIX below, now applied continuously instead of as a
        # hard gate). That residual ("signal"), divided by the cost of
        # seeding a new point (initial_seeds_new_point), is the score.
        # This ALONE reproduces the old per-gap affordability gate's
        # effect without needing a separate hard cutoff: a gap dominated
        # by noise near a capped, high-variance point (like x=1.0 at an
        # indifference point) naturally scores near zero for insertion,
        # since own_i/own_j already explain nearly all of its width - so
        # it never gets picked ahead of gaps with real, addressable signal.
        #
        # MIN-SEEDS-FOR-INSERT-ELIGIBILITY GATE (separate from the above,
        # added after real runs showed a cascade of near-duplicate x's
        # right at an indifference point - see plot_p_curve_results.py's
        # module docstring for how that showed up: dozens of x-values only
        # 1e-4 or less apart, several with inconsistent-looking intervals).
        # WHY the signal-vs-noise math above didn't already prevent this: a
        # BRAND NEW point only has initial_seeds_new_point (4, by default)
        # seeds, and with that few, own_i/own_j can look ARTIFICIALLY TIGHT
        # by pure chance (a small sample of a bimodal, high-variance
        # population can easily land all on one side) - this is the exact
        # "small early samples systematically under-estimate the true
        # std" effect already documented above on MAX_SEEDS_PER_POINT_FRAC.
        # A falsely-tight own_i/own_j makes "signal" look positive even
        # when the gap is actually noise-dominated, so INSERT keeps
        # winning against a freshly-inserted point's own neighbor gaps
        # before TIGHTEN ever gets a chance to reveal the point's true
        # (wider) width - bisecting the same small region over and over
        # instead of spending seeds where they'd add real information. The
        # fix: a gap's endpoints must each have at least
        # MIN_SEEDS_FOR_INSERT_ELIGIBILITY_MULT * initial_seeds_new_point
        # seeds before that gap is even ELIGIBLE for insertion - below
        # that, only TIGHTEN scores for those points, so seeds go toward
        # revealing their TRUE width first. This doesn't touch the
        # legitimate "hopeless gap near a capped point" case at all - a
        # capped point already has plenty of seeds (it hit the cap), so it
        # clears this bar easily and its insert score correctly stays ~0
        # via the signal-vs-noise math above, same as before.
        min_seeds_for_insert = MIN_SEEDS_FOR_INSERT_ELIGIBILITY_MULT * initial_seeds_new_point
        xs = sorted(points.keys())
        cis = compute_all_point_cis(points)

        candidates = []  # (score, kind, a, b, signal) - b/signal are None for "tighten"
        min_gap_log_x = resolved_gap_log_x(xs)

        for k, x in enumerate(xs):
            if x in given_up:
                continue
            if is_locally_resolved(xs, k, min_gap_log_x):
                continue  # already densely sampled enough here - shape is resolved,
                          # not worth tightening further regardless of CI height (see
                          # is_locally_resolved()'s docstring)
            n_neighbors = (1 if k > 0 else 0) + (1 if k < len(xs) - 1 else 0)
            if n_neighbors == 0:
                continue  # only one sampled point total - nothing to compare against yet
            own = cis[x][1] - cis[x][0]
            n_seeds = cis[x][2]
            own_after = own * np.sqrt(n_seeds / (n_seeds + seed_batch))
            delta_w = own - own_after
            benefit = delta_w * n_neighbors / 2.0
            candidates.append((benefit / seed_batch, "tighten", x, None, None))

        for i in range(len(xs) - 1):
            x_i, x_j = xs[i], xs[i + 1]
            if (np.log(x_j) - np.log(x_i)) <= min_gap_log_x:
                continue  # already fine enough x-resolution to pin down the shape here -
                          # subdividing further wouldn't teach us anything new (see
                          # is_locally_resolved()'s docstring)
            new_x = float(np.exp((np.log(x_i) + np.log(x_j)) / 2.0))
            if not (x_i < new_x < x_j):
                continue  # no floating-point room left to bisect this gap further
            if cis[x_i][2] < min_seeds_for_insert or cis[x_j][2] < min_seeds_for_insert:
                continue  # an endpoint is too young for its own CI to be trusted yet (see above)
            width = cis[x_j][1] - cis[x_i][0]
            own_i = cis[x_i][1] - cis[x_i][0]
            own_j = cis[x_j][1] - cis[x_j][0]
            signal = max(width - own_i - own_j, 0.0)
            candidates.append((signal / initial_seeds_new_point, "insert", x_i, x_j, signal))

        if not candidates:
            print(f"  [{test['name']}] {100.0 * new_runs() / max_runs:.1f}% - no remaining "
                  f"action has any measurable expected benefit (every remaining point/gap "
                  f"is either per-point-capped or already shape-resolved by its x-density) "
                  f"- stopping early with the best fit this data supports.")
            break

        score, kind, a, b, signal = max(candidates, key=lambda c: c[0])
        max_gap_width = max((cis[xs[i + 1]][1] - cis[xs[i]][0] for i in range(len(xs) - 1)), default=0.0)
        met_note = " (target width already met everywhere - polishing further)" if max_gap_width <= target_width else ""

        if kind == "tighten":
            if add_seeds(a, seed_batch, "tighten") == 0:
                break
            print(f"  [{test['name']}] {100.0 * new_runs() / max_runs:.1f}%{met_note} - best "
                  f"expected improvement/run: tightening log(x)={np.log(a):.5f} "
                  f"(own CI width {cis[a][1] - cis[a][0]:.4f}, score {score:.5f}).")
        else:
            new_x = float(np.exp((np.log(a) + np.log(b)) / 2.0))
            n_seeds_new = steep_insert_seed_count(signal)
            if add_seeds(new_x, n_seeds_new, "insert") == 0:
                break
            print(f"  [{test['name']}] {100.0 * new_runs() / max_runs:.1f}%{met_note} - best "
                  f"expected improvement/run: inserting new log(x)={np.log(new_x):.5f} "
                  f"between log(x)={np.log(a):.5f} and log(x)={np.log(b):.5f} "
                  f"(gap width {cis[b][1] - cis[a][0]:.4f}, score {score:.5f}, "
                  f"starting with {n_seeds_new} seeds - signal {signal:.4f}).")

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    n_given_up = len(given_up)
    added_this_run = new_runs()
    print(f"  done: {added_this_run} new runs added this invocation "
          f"({existing_count} pre-existing + {added_this_run} new = {len(df)} total runs on disk), "
          f"{df['x'].nunique()} distinct x-values, {n_given_up} point(s) hit the per-point cap, "
          f"{elapsed / 60.0:.1f} min this session ({elapsed / max(added_this_run, 1):.1f} s/run)")
    return df


def run_global_budget(tests, hits_per_run=HITS_PER_RUN, train_kwargs=None,
                       target_width=TARGET_CI_WIDTH, flat_eps=FLAT_EPS,
                       total_budget=1000, seed_batch=SEED_BATCH,
                       initial_seeds_new_point=INITIAL_SEEDS_NEW_POINT,
                       output_dir=OUTPUT_DIR):
    """Like run_test_adaptive(), but with ONE shared budget of
    `total_budget` NEW runs spent across ALL `tests` at once, instead of a
    separate budget per test. Every batch of seeds goes wherever the
    BEST-EXPECTED-IMPROVEMENT-PER-RUN score is highest ACROSS THE WHOLE
    SET of comparisons - so a test whose curve is already tight gets left
    alone and a test that's still noisy/under-sampled gets more of the
    1000 runs, rather than every test getting a fixed, possibly
    mismatched, equal share.

    Two-pass structure (same reasoning as run_test_adaptive's phases, just
    applied across tests rather than within one):

      1. TAIL EXTENSION, one test at a time, in the order given. A test's
         curve can't be scored/compared meaningfully via the phase-2
         benefit-per-run formula until it has at least a low and a high
         boundary point that are genuinely part of a bracketed range - so
         this is done as a priority pass, not folded into the same
         cross-test scoring competition as phase 2. It still draws from
         the SAME shared budget (this isn't a separate allowance), so a
         test needing an unusually long walk to bracket its tails will
         still show up as spending more of the 1000 than a test that
         brackets quickly - that's the expected, correct behavior, not a
         bug.
      2. GLOBAL BEST-EXPECTED-IMPROVEMENT-PER-RUN: once every test's tails
         are flat (or accepted - same per-point give-up cap logic as
         run_test_adaptive, see its big comment for why that's needed),
         every test's own tighten/insert candidates (identical scoring
         formula to run_test_adaptive's phase 2) are pooled into ONE list
         and the single best-scoring action ANYWHERE is taken each
         iteration - so, e.g., if test A's worst gap is already down to
         0.03 width but test B's worst gap is still 0.15, B's candidates
         will keep winning until they're comparably tight, then priority
         shifts to whichever test would benefit most next. This is
         literally the same per-test scoring math, just run over a
         combined candidate pool instead of one test's own.

    Returns {test_name: DataFrame} for every test in `tests`."""
    print(f"\n=== GLOBAL BUDGET MODE: {total_budget} total NEW runs shared across "
          f"{len(tests)} tests ===")

    # Per-point give-up cap, scaled to each test's FAIR SHARE of the
    # shared budget (total_budget / len(tests)) rather than the whole
    # budget - otherwise one point could be allowed to eat a number of
    # seeds calibrated as if it had the entire 1000-run budget to itself,
    # defeating the point of sharing.
    per_test_share = max(total_budget / max(len(tests), 1), 1.0)
    max_seeds_per_point = max(MIN_MAX_SEEDS_PER_POINT,
                               int(per_test_share * MAX_SEEDS_PER_POINT_FRAC))

    states = {}
    for test in tests:
        rows = load_existing(test["name"])
        points = defaultdict(list)
        for r in rows:
            points[r["x"]].append(r)
        states[test["name"]] = {
            "test": test,
            "rows": rows,
            "existing_count": len(rows),
            "seed_counter": max((r["seed"] for r in rows), default=-1) + 1,
            "points": points,
            "given_up": set(),
            "csv_path": os.path.join(output_dir, f"{test['name']}.csv"),
        }
        if rows:
            print(f"  [{test['name']}] resuming from {len(rows)} existing runs "
                  f"(not counted against the shared {total_budget}-run budget)")

    t0 = time.time()

    def spent():
        return sum(len(s["rows"]) - s["existing_count"] for s in states.values())

    def remaining():
        return max(total_budget - spent(), 0)

    def add_seeds(state, x, n, phase):
        n = min(n, remaining())
        for _ in range(n):
            y_hat, cf, cv = _run_one_seed(state["test"], x, state["seed_counter"],
                                           hits_per_run, train_kwargs)
            row = {
                "test": state["test"]["name"], "x": x, "seed": state["seed_counter"],
                "phase": phase, "hits_per_run": hits_per_run, "chose_fixed": cf,
                "chose_variable": cv, "y_hat": y_hat,
            }
            state["rows"].append(row)
            state["points"][x].append(row)
            state["seed_counter"] += 1
        if n > 0 and state["csv_path"] is not None:
            pd.DataFrame(state["rows"]).to_csv(state["csv_path"], index=False)
        if len(state["points"][x]) >= max_seeds_per_point and x not in state["given_up"]:
            state["given_up"].add(x)
            print(f"  [{state['test']['name']}] log(x)={np.log(x):.5f} hit the per-point "
                  f"cap of {max_seeds_per_point} seeds (this test's fair share of the "
                  f"shared budget) without reaching the target CI width - accepting its "
                  f"current CI and moving budget elsewhere.")
        return n

    # --- Pass 1: tail extension, one test at a time, shared budget ---
    for test in tests:
        state = states[test["name"]]
        name = test["name"]

        if not state["points"]:
            if remaining() <= 0:
                break
            x0 = test["value_fixed"] if test["value_fixed"] > 0 else 1.0
            add_seeds(state, x0, initial_seeds_new_point, "bootstrap")

        while remaining() > 0:
            xs = sorted(state["points"].keys())
            cis = compute_all_point_cis(state["points"])
            x_lo, x_hi = xs[0], xs[-1]
            lo_flat = cis[x_lo][1] <= flat_eps or x_lo in state["given_up"]
            hi_flat = cis[x_hi][0] >= 1.0 - flat_eps or x_hi in state["given_up"]

            if lo_flat and hi_flat:
                break  # this test's tails are done - move to the next test

            if not lo_flat:
                if pooled_mean(state["points"][x_lo]) <= flat_eps:
                    if add_seeds(state, x_lo, seed_batch, "tighten_tail_low") == 0:
                        break
                    new_ci = compute_all_point_cis(state["points"])[x_lo]
                    if x_lo in state["given_up"] and new_ci[1] > flat_eps:
                        print(f"  [{name}] WARNING: log(x_lo)={np.log(x_lo):.5f} hit its "
                              f"per-point cap while still not confirmed flat (CI hi="
                              f"{new_ci[1]:.4f} > flat_eps={flat_eps}) - accepting it as "
                              f"the low tail anyway; this test's low end may not truly "
                              f"reach 0.")
                    print(f"  [{name}] {100.0 * spent() / total_budget:.1f}% (global) - "
                          f"log(x_lo)={np.log(x_lo):.5f} looks flat but CI still too wide "
                          f"- tightening it.")
                    continue
                new_x = x_lo / 2.0
                if new_x < MIN_X_FLOOR:
                    print(f"  [{name}] WARNING: x_lo hit the {MIN_X_FLOOR} floor without "
                          f"confirming p~0 (current CI={cis[x_lo][:2]}) - treating the low "
                          f"tail as flat anyway; this test's low end may not truly reach 0.")
                else:
                    if add_seeds(state, new_x, initial_seeds_new_point, "extend_low") == 0:
                        break
                    print(f"  [{name}] {100.0 * spent() / total_budget:.1f}% (global) - "
                          f"extending low tail to log(x)={np.log(new_x):.5f}")
                    continue

            if not hi_flat:
                if pooled_mean(state["points"][x_hi]) >= 1.0 - flat_eps:
                    if add_seeds(state, x_hi, seed_batch, "tighten_tail_high") == 0:
                        break
                    new_ci = compute_all_point_cis(state["points"])[x_hi]
                    if x_hi in state["given_up"] and new_ci[0] < 1.0 - flat_eps:
                        print(f"  [{name}] WARNING: log(x_hi)={np.log(x_hi):.5f} hit its "
                              f"per-point cap while still not confirmed flat (CI lo="
                              f"{new_ci[0]:.4f} < 1-flat_eps={1.0 - flat_eps}) - accepting "
                              f"it as the high tail anyway; this test's high end may not "
                              f"truly reach 1.")
                    print(f"  [{name}] {100.0 * spent() / total_budget:.1f}% (global) - "
                          f"log(x_hi)={np.log(x_hi):.5f} looks flat but CI still too wide "
                          f"- tightening it.")
                    continue
                new_x = x_hi * 2.0
                if add_seeds(state, new_x, initial_seeds_new_point, "extend_high") == 0:
                    break
                print(f"  [{name}] {100.0 * spent() / total_budget:.1f}% (global) - "
                      f"extending high tail to log(x)={np.log(new_x):.5f}")
                continue

            break  # both flat now (last add_seeds call flipped one of them)

        if remaining() <= 0:
            print(f"  shared budget exhausted during tail extension - stopping "
                  f"({100.0 * spent() / total_budget:.1f}% of budget spent).")
            break

    # --- Pass 2: GLOBAL best-expected-improvement-per-run, pooled across
    # every test's own candidates (see docstring above). ---
    min_seeds_for_insert = MIN_SEEDS_FOR_INSERT_ELIGIBILITY_MULT * initial_seeds_new_point
    while remaining() > 0:
        all_candidates = []  # (score, test_name, kind, a, b, signal) - b/signal None for "tighten"

        for test in tests:
            state = states[test["name"]]
            xs = sorted(state["points"].keys())
            if len(xs) < 2:
                continue  # no gaps/neighbors yet for this test (still mid tail-extension)
            cis = compute_all_point_cis(state["points"])
            min_gap_log_x = resolved_gap_log_x(xs)

            for k, x in enumerate(xs):
                if x in state["given_up"]:
                    continue
                if is_locally_resolved(xs, k, min_gap_log_x):
                    continue  # already densely sampled enough here - see
                              # is_locally_resolved()'s docstring
                n_neighbors = (1 if k > 0 else 0) + (1 if k < len(xs) - 1 else 0)
                if n_neighbors == 0:
                    continue
                own = cis[x][1] - cis[x][0]
                n_seeds = cis[x][2]
                own_after = own * np.sqrt(n_seeds / (n_seeds + seed_batch))
                delta_w = own - own_after
                benefit = delta_w * n_neighbors / 2.0
                all_candidates.append((benefit / seed_batch, test["name"], "tighten", x, None, None))

            for i in range(len(xs) - 1):
                x_i, x_j = xs[i], xs[i + 1]
                if (np.log(x_j) - np.log(x_i)) <= min_gap_log_x:
                    continue  # already fine enough x-resolution here - see
                              # is_locally_resolved()'s docstring
                new_x = float(np.exp((np.log(x_i) + np.log(x_j)) / 2.0))
                if not (x_i < new_x < x_j):
                    continue
                if cis[x_i][2] < min_seeds_for_insert or cis[x_j][2] < min_seeds_for_insert:
                    continue  # an endpoint is too young for its own CI to be trusted yet
                width = cis[x_j][1] - cis[x_i][0]
                own_i = cis[x_i][1] - cis[x_i][0]
                own_j = cis[x_j][1] - cis[x_j][0]
                signal = max(width - own_i - own_j, 0.0)
                all_candidates.append((signal / initial_seeds_new_point, test["name"],
                                        "insert", x_i, x_j, signal))

        if not all_candidates:
            print(f"  {100.0 * spent() / total_budget:.1f}% (global) - no remaining action "
                  f"has any measurable expected benefit anywhere across all {len(tests)} "
                  f"tests (every remaining point/gap is either per-point-capped or already "
                  f"shape-resolved by its x-density) - stopping early with the best fit "
                  f"this data supports.")
            break

        score, name, kind, a, b, signal = max(all_candidates, key=lambda c: c[0])
        state = states[name]
        cis = compute_all_point_cis(state["points"])

        if kind == "tighten":
            if add_seeds(state, a, seed_batch, "tighten") == 0:
                break
            print(f"  [{name}] {100.0 * spent() / total_budget:.1f}% (global) - best "
                  f"expected improvement/run ACROSS ALL TESTS: tightening "
                  f"log(x)={np.log(a):.5f} (own CI width {cis[a][1] - cis[a][0]:.4f}, "
                  f"score {score:.5f}).")
        else:
            new_x = float(np.exp((np.log(a) + np.log(b)) / 2.0))
            n_seeds_new = steep_insert_seed_count(signal)
            if add_seeds(state, new_x, n_seeds_new, "insert") == 0:
                break
            print(f"  [{name}] {100.0 * spent() / total_budget:.1f}% (global) - best "
                  f"expected improvement/run ACROSS ALL TESTS: inserting new "
                  f"log(x)={np.log(new_x):.5f} between log(x)={np.log(a):.5f} and "
                  f"log(x)={np.log(b):.5f} (gap width {cis[b][1] - cis[a][0]:.4f}, "
                  f"score {score:.5f}, starting with {n_seeds_new} seeds - signal {signal:.4f}).")

    elapsed = time.time() - t0
    total_spent = spent()
    print(f"\n  GLOBAL BUDGET done: {total_spent}/{total_budget} new runs spent across "
          f"{len(tests)} tests, {elapsed / 60.0:.1f} min total "
          f"({elapsed / max(total_spent, 1):.1f} s/run).")
    result = {}
    for test in tests:
        state = states[test["name"]]
        df = pd.DataFrame(state["rows"])
        added = len(state["rows"]) - state["existing_count"]
        print(f"    [{test['name']}] {added} new runs "
              f"({state['existing_count']} pre-existing + {added} new = {len(df)} total), "
              f"{df['x'].nunique() if len(df) else 0} distinct x-values, "
              f"{len(state['given_up'])} point(s) hit the per-point cap")
        result[test["name"]] = df
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", nargs="*", default=None,
                         help="Subset of test names to run (default: all in TESTS).")
    parser.add_argument("--smoke-test", action="store_true",
                         help="Tiny scale (short training, small max_runs) to sanity-check "
                              "the pipeline runs end-to-end. NOT meant to produce meaningful curves.")
    parser.add_argument("--global-budget", type=int, nargs="?", const=1000, default=None,
                         help="Instead of a separate %d-run budget PER TEST, spend this many "
                              "TOTAL new runs (default 1000 if you pass --global-budget with no "
                              "number) shared across every selected test - each batch of seeds "
                              "goes wherever the best-expected-improvement-per-run score is "
                              "highest ACROSS ALL comparisons, not just within one test. See "
                              "run_global_budget()'s docstring for the full reasoning." % MAX_RUNS_PER_TEST)
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
    hits_per_run = HITS_PER_RUN
    max_runs = MAX_RUNS_PER_TEST
    target_width = TARGET_CI_WIDTH
    flat_eps = FLAT_EPS
    seed_batch = SEED_BATCH
    initial_seeds_new_point = INITIAL_SEEDS_NEW_POINT
    global_budget = args.global_budget
    if args.smoke_test:
        train_kwargs["total_timesteps"] = 3_000
        hits_per_run = 50
        max_runs = 24
        seed_batch = 2
        initial_seeds_new_point = 2
        if global_budget is not None:
            global_budget = 24 * len(tests_to_run)
        print("*** SMOKE TEST MODE: tiny scale, results are meaningless, only checking "
              "the pipeline runs without errors. ***")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if global_budget is not None:
        run_global_budget(
            tests_to_run, hits_per_run=hits_per_run, train_kwargs=train_kwargs,
            target_width=target_width, flat_eps=flat_eps, total_budget=global_budget,
            seed_batch=seed_batch, initial_seeds_new_point=initial_seeds_new_point,
            output_dir=OUTPUT_DIR,
        )
        return

    for test in tests_to_run:
        csv_path = os.path.join(OUTPUT_DIR, f"{test['name']}.csv")
        run_test_adaptive(
            test, hits_per_run=hits_per_run, train_kwargs=train_kwargs,
            target_width=target_width, flat_eps=flat_eps, max_runs=max_runs,
            seed_batch=seed_batch, initial_seeds_new_point=initial_seeds_new_point,
            csv_path=csv_path,
        )
        print(f"  wrote {csv_path}")


if __name__ == "__main__":
    main()
