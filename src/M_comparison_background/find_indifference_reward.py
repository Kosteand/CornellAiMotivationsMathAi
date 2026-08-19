"""Estimate the reward "indifference point" between two MarginGroup
targets of possibly different learning difficulty (k).

Setup
-----
A BanditEnv is built with exactly two MarginGroup targets sharing the
same g:

  - group_fixed:    k = k_fixed,    value = value_fixed (1.0 by default)
  - group_variable: k = k_variable, value = x            (searched over,
                     never negative - x=0 is allowed, meaning "no reward
                     at all for hitting the variable target")

BanditEnv now gives group_fixed and group_variable their OWN disjoint
block of actions (see BanditEnv's own docstring) - e.g. with g=4 each,
actions 0-3 belong to group_fixed and 4-7 belong to group_variable. This
means every action unambiguously "goes for" (falls in the block of)
exactly one of the two groups - there is no longer any way for a single
action to be simultaneously correct for both (that WAS possible under
the old shared-action-space design). Every episode the trained (greedy)
policy's action falls in EXACTLY ONE group's block - BanditEnv.
info["chosen_group"] - and either IS that block's correct option this
episode (a HIT, BanditEnv.info["matched_group"]) or ISN'T (a miss within
that block). The core statistic this whole module is built around is
based on which block the policy went for, NOT on whether it actually won
there - i.e. it's a measure of PREFERENCE/behavior, not of correctness.
For a single trained policy evaluated over some number of episodes,
define:

    Y_hat(x) = target_fixed / (target_fixed + target_variable)
             = target_fixed / episodes

i.e. the fraction of episodes where the policy's action fell in
group_fixed's block rather than group_variable's - since every action
falls in exactly one block, this denominator always equals the total
episode count (there's no "went for neither" case to exclude the way
there was under the old shared-action-space design). The goal is
`x* = M`: the value of x at which a freshly-trained policy is genuinely
indifferent between going for the two targets, i.e. E[Y_hat(x=M)] = 0.5 -
equivalently, P(goes for group_fixed) = 0.5. Whether the policy actually
LANDS the correct option within whichever block it chose (hit_fixed/
hit_variable/missed, from matched_group) is tracked separately after
every training run (see _train_and_get_hits) as a secondary diagnostic -
useful for sanity-checking, but not what M is fit to. It's logged to the
run CSV, not printed - the terminal only ever shows the current x and
that run's target_rate_fixed (plus the running CI during certify) -
see find_indifference_reward's docstring for the full logging story.

If k_fixed == k_variable, M should land at x = value_fixed by symmetry
(equal difficulty -> equal reward is the indifference point). If
k_variable is HARDER to learn than k_fixed, M should land ABOVE
value_fixed - the harder target needs extra reward to compensate for its
weaker/noisier learning signal, and exactly how much M exceeds
value_fixed is itself a measurement of "how much reward compensates for
how much complexity," which is this project's whole question.

Two-phase design (ported from a similar setup built on another branch,
adapted here to BanditEnv/MarginGroup/trainPPO.train() and to varying
REWARD VALUE rather than varying k):

  Phase 1 - find_candidate_reward(): cheap heuristic SEARCH. Neither the
  number of x values tried nor the number of seeds trained at each x is
  a preset count - both are driven by confidence:

    - At a given x, seeds are trained ONE AT A TIME and their per-seed
      Y_hat values accumulate into the SAME anytime-valid confidence
      sequence certify_reward uses (Robbins normal-mixture bound, fixed
      Hoeffding variance proxy sigma^2=1/4). As soon as that running CI
      sits entirely above or entirely below 0.5, this point's DIRECTION
      is settled confidently and no more seeds are trained there.
      `max_seeds_per_point` is a safety cap only (a point that's
      genuinely near 0.5 - e.g. it's close to M itself - may never
      resolve a confident direction, so something has to stop it
      eventually), not a target.
    - Which x to try next is chosen adaptively: start at x = value_fixed
      (the natural prior - "same reward as the fixed target" is the
      right guess whenever the two targets are similarly hard, or the
      midpoint of `search_range` if one is given), and walk outward,
      DOUBLING the step each time the confident direction is still wrong
      (so being further off costs more points, not more precision) until
      the crossing is bracketed, then bisect within the bracket. The
      search stops once the bracket has narrowed below `bracket_tol` AND
      the fitted M has stayed stable (within `m_stable_tol`) for
      `m_stable_window` consecutive points in a row - i.e. once it's
      fairly confident it's found the right value, not after a
      predetermined number of points. `max_points` is again a safety cap
      only.

  Every seed's (target_fixed, target_variable) counts - i.e. how many of
  its evaluation episodes went for each group's action block, regardless
  of whether it won there - at a given x are POOLED into one Binomial
  observation per point, and P(went for fixed | some target) =
  sigmoid(-4*beta*(x - M)) is fit to those pooled counts via maximum
  likelihood (not least-squares on each point's mean Y_hat weighted by an
  estimated variance - estimating variance from only a few seeds is
  itself so noisy that inverse-variance weighting could swing a point's
  influence by orders of magnitude purely from sampling luck). Multiple
  SEEDS (not just more eval episodes at one seed) are what actually let
  this see how much a trained policy's own preference varies run-to-run
  at the same x - a real, large effect here, since policies often fully
  commit to one target rather than landing near 50/50; more eval episodes
  alone only shrinks how precisely you measure one already-trained
  policy's own rate, not this run-to-run variance (see hits_per_point's
  own note in find_candidate_reward's docstring).

  Returns M_hat, beta_hat, and diagnostics - explicitly NO statistical
  guarantee on its own; this phase exists to propose a good candidate for
  phase 2.

  Phase 2 - certify_reward(): a separate, from-scratch, SEQUENTIAL test
  at ONE fixed x (normally the candidate M_hat from phase 1). After every
  fresh run's Y_hat, it recomputes the running mean and the same
  anytime-valid confidence sequence (fixed Hoeffding variance proxy
  sigma^2=1/4 - not estimated from the data, which is what keeps the
  sequence valid to check after every single new run without inflating
  the false-positive rate past alpha). This is what makes certify_reward
  an actual verification that P(goes for fixed) is in (lo, hi) - default
  (0.40, 0.60) - with 95% confidence (alpha=0.05), rather than just a
  bigger, still-unverified estimate: it stops and ACCEPTS as soon as the
  CI sits entirely inside (lo, hi); stops and REJECTS as soon as the CI
  sits entirely outside. `max_runs` is a large safety valve (default
  5000), not a realistic target - converging can genuinely take a few
  hundred runs when the true probability sits close to a boundary, so a
  cap in the tens is nowhere near enough headroom; `target_n` tunes how
  the bound's tightness is distributed across the run count (see its own
  docstring) without changing the safety cap itself. A leftover "stalled"
  exclusion mechanism (for a run whose per-episode target counts summed
  to 0) still exists for defensive robustness but should never actually
  trigger now: every action falls in exactly one group's block, so
  target_fixed + target_variable always equals the episode count.

find_indifference_reward() ties both phases together in one call.
Without verify, it runs the search ONCE and returns that estimate - a
solid, cheap number with no guarantee attached. With verify=True, it
keeps going until it actually lands a certified reward: it certifies the
search's candidate, and if certify_reward doesn't ACCEPT (it either
REJECTs or runs out of its - very generous - run budget inconclusively),
it folds that certify batch's own pooled hit counts in as one more (very
informative, since certify runs many episodes) data point, refits the
candidate from every point seen so far (original search points + every
certify attempt), and certifies again at the new candidate - repeating
until a certify batch is ACCEPTED or `max_search_iterations` attempts are
exhausted (a safety valve against looping forever if 0.5 genuinely isn't
achievable in the range being searched; exhausting it returns the best
candidate found so far with certified=False, not an exception). Each
certify attempt is still a from-scratch batch of runs - phase 1's or an
earlier phase 2 attempt's own training runs are never reused as evidence
for a LATER certify batch's accept/reject decision itself (only their
pooled counts feed the next candidate's refit) - so the final ACCEPT is
still a clean, uncontaminated test of the candidate that was actually
accepted.

IMPORTANT - alpha-splitting across attempts: each individual certify
attempt is only valid at its own alpha in isolation. If the loop above
ever needs more than one attempt (a REJECT or inconclusive batch followed
by a retry at a refit candidate), running each attempt at the full,
flat `alpha` would let the FAMILY-WISE false-accept probability across
the whole verify loop climb as high as 1-(1-alpha)**max_search_iterations
in the worst case (e.g. ~40% for alpha=0.05, max_search_iterations=10) -
because the guarantee has to cover every possible attempt the procedure
is willing to make, not just however many attempts happened to actually
run. To keep the OVERALL procedure's false-accept rate bounded by
`alpha`, find_indifference_reward() Bonferroni-splits it before calling
certify_reward: every attempt is tested at
alpha/max_search_iterations, not alpha. This makes each attempt's CI
correspondingly wider/slower to shrink (roughly 1.3x more runs needed
for the same tightness at max_search_iterations=10), which is the
correct price for the overall guarantee actually holding regardless of
how many attempts end up being used.

Run directly (`python3 find_indifference_reward.py`) for a tiny smoke
test - real use should call find_indifference_reward() from another
script with hyperparameters sized for your actual budget.

Not MarginGroup-only: every function above (the search's pooled Binomial
counts, the sigmoid MLE fit, the anytime-valid confidence sequence) only
ever operates on which of the two groups' action BLOCK the trained policy
went for - it never looks at what kind of Group produced that block. The
one place a Group actually gets constructed is `_build_groups`, via a
pluggable `group_factory(g, difficulty, value) -> Group` argument
threaded through find_candidate_reward()/certify_reward()/
find_indifference_reward() - defaulting to plain MarginGroup(delta=1/k)
when not given, i.e. every example and default above still applies
unchanged. Pass a different group_factory to run this whole pipeline
against AlternatingGroup, HeatmapGroup, or any other BanditEnv-compatible
Group instead; see _build_groups' and find_indifference_reward's own
docstrings, and run_indifference_batch.py for worked examples.
"""
import csv
import os

import numpy as np
import torch
from scipy.optimize import minimize

from Utilities.bandit_env import BanditEnv, MarginGroup
from trainPPO import train

# --- Fixed, known variance proxy for any [0, 1]-bounded random variable
# (Hoeffding's lemma) - used by every anytime-valid confidence sequence
# below (both certify_reward's and the search phase's per-point
# direction check). NOT estimated from data; see certify_reward's
# docstring for why that matters.
_HOEFFDING_SIGMA2 = 0.25

# Every column any record type below might populate. A single wide CSV
# (one row per event - a training run, a search point's pooled summary, a
# fit refit, a certify run, a verify iteration, or the final result) is
# simplest to consume in pandas/Excel later - unused columns for a given
# row are just left blank. `record_type` says what kind of row it is.
_CSV_FIELDNAMES = [
    "record_type", "iteration", "phase", "x", "seed",
    "k_fixed", "k_variable", "g", "value_fixed", "incorrect_reward",
    "hits_per_run",
    "hits_fixed", "hits_variable", "misses", "chose_fixed", "chose_variable",
    "target_rate_variable_pct", "target_rate_fixed_pct",
    "hit_rate_variable_pct", "hit_rate_fixed_pct", "missed_rate_pct",
    "model_weight_norm",
    "seeds_used", "n_fixed_total", "n_variable_total",
    "avg_target_rate_variable_pct", "seed_y_hats",
    "fitted_M", "fitted_beta", "fit_converged",
    "fitted_M_exp", "fitted_beta_exp", "fit_exp_converged",
    "certify_n", "certify_mean", "certify_ci_lo", "certify_ci_hi",
    "certify_status",
    "final_M", "final_beta", "certified",
]


class _RunCsvLogger:
    """Writes one row per event (see `_CSV_FIELDNAMES`) to `path` -
    everything that used to be printed to the terminal every run, plus
    anything else cheap to compute (e.g. `model_weight_norm`), now lands
    here instead. Opens in write mode (truncates any previous run's log
    at the same path) and flushes after every row, so the file is always
    readable mid-run rather than only after a clean exit. `None` path
    disables logging entirely (`.log()` becomes a no-op) - useful for
    callers that don't want a CSV at all."""

    def __init__(self, path):
        self.path = path
        self._file = None
        self._writer = None
        if path is not None:
            dirname = os.path.dirname(path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            self._file = open(path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=_CSV_FIELDNAMES)
            self._writer.writeheader()
            self._file.flush()

    def log(self, record_type, **fields):
        if self._writer is None:
            return
        row = {name: "" for name in _CSV_FIELDNAMES}
        row["record_type"] = record_type
        for key, value in fields.items():
            row[key] = value
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()


def _model_weight_norm(model):
    """Total L2 norm across every parameter tensor in the trained PPO
    policy (actor + critic + shared layers, whatever the net_arch has) -
    a single cheap-to-compute scalar summarizing "how far the weights
    have moved from initialization" for this run. Purely a diagnostic
    logged to the CSV; nothing in the search/certify math reads it."""
    total_sq = 0.0
    for p in model.policy.parameters():
        total_sq += float(torch.sum(p.detach() ** 2))
    return float(np.sqrt(total_sq))


def _neg_log_likelihood_exp(params, xs, n_fixed, n_variable):
    """Negative log-likelihood of the same pooled Binomial counts
    _neg_log_likelihood uses, but under an EXPONENTIAL-tailed alternative
    to the logistic sigmoid: P(goes for fixed) is the Laplace-distribution
    CDF centered at M with rate beta -
        0.5*exp(beta*(x-M))         for x <= M
        1 - 0.5*exp(-beta*(x-M))    for x >  M
    - i.e. exponential (not logistic) decay away from the crossing point
    on each side. Still bounded in [0, 1], still monotonic, still exactly
    0.5 at x=M, but with sharper/thinner tails than the sigmoid - a useful
    alternative fit to sanity-check M against, not a replacement for it
    (the sigmoid fit is still what drives the search's own decisions).
    beta is optimized in log-space for the same reason as
    _neg_log_likelihood: keeps the unconstrained optimizer from wandering
    into beta <= 0."""
    M, log_beta = params
    beta = np.exp(log_beta)
    nll = 0.0
    for x, nf, nv in zip(xs, n_fixed, n_variable):
        n = nf + nv
        if n == 0:
            continue
        d = x - M
        if d <= 0:
            p_fixed = 0.5 * np.exp(beta * d)
        else:
            p_fixed = 1.0 - 0.5 * np.exp(-beta * d)
        p_fixed = float(np.clip(p_fixed, 1e-12, 1.0 - 1e-12))
        nll -= nf * np.log(p_fixed) + nv * np.log(1.0 - p_fixed)
    return float(nll)


def _fit_exponential_mle(xs, n_fixed, n_variable, m0):
    """Fit (M, beta) for the exponential-tailed alternative curve above -
    same MLE approach as _fit_sigmoid_mle, just a different curve shape.
    Returns (M_hat, beta_hat, scipy OptimizeResult)."""
    xs = np.asarray(xs, dtype=float)
    n_fixed = np.asarray(n_fixed, dtype=float)
    n_variable = np.asarray(n_variable, dtype=float)

    x0 = [float(m0), 0.0]
    res = minimize(
        _neg_log_likelihood_exp, x0, args=(xs, n_fixed, n_variable), method="Nelder-Mead",
    )
    M_hat, log_beta_hat = res.x
    beta_hat = float(np.exp(log_beta_hat))
    return float(M_hat), beta_hat, res


def _default_group_factory(g, difficulty, value):
    """Default `group_factory` - preserves this module's original,
    MarginGroup-only behavior exactly: `difficulty` IS `k` directly now
    that MarginGroup's constructor takes `k` instead of `delta` (at the
    default s=1.0, `k = s / delta = difficulty` reproduces the old
    `delta = 1.0 / difficulty` behavior exactly). Pass a different
    `group_factory(g, difficulty, value) -> Group` to
    find_candidate_reward()/certify_reward()/find_indifference_reward()
    to use a different Group type entirely - see their docstrings'
    "group_factory" note."""
    return MarginGroup(g=g, k=difficulty, value=value)


def _build_groups(g, k_fixed, k_variable, value_fixed, value_variable, group_factory=None):
    """Build the [fixed, variable] two-group list `_evaluate_two_group_hits`
    trains/evaluates against. `group_factory` (defaulting to
    `_default_group_factory`, i.e. MarginGroup with delta=1/k) is called
    once per side as `group_factory(g, k_fixed_or_variable, value)` - it
    decides what `k_fixed`/`k_variable` actually MEAN and what Group
    subclass gets built from them. This is the one place a non-MarginGroup
    target gets constructed; every other function in this module is
    already Group-agnostic (BanditEnv, _evaluate_two_group_hits, etc. only
    ever touch the generic Group interface - g/observation_*/sample), so
    swapping `group_factory` is enough to run this whole search/certify
    pipeline against AlternatingGroup, HeatmapGroup, or any other
    BanditEnv-compatible Group without touching anything else here."""
    group_factory = group_factory or _default_group_factory
    return [
        group_factory(g, k_fixed, value_fixed),
        group_factory(g, k_variable, value_variable),
    ]


def _evaluate_two_group_hits(model, groups, incorrect_reward, episodes):
    """Run `episodes` deterministic (greedy) BanditEnv episodes against a
    two-group [fixed, variable] setup and tally both:

      - which group (if any) the policy actually HIT (BanditEnv's
        "matched_group" - the action fell in that group's action block
        AND was that block's correct option this episode), and
      - which group's action BLOCK the policy CHOSE at all (BanditEnv's
        "chosen_group" - whether or not it was the right option within
        that block).

    Since BanditEnv now gives each group its own disjoint block of the
    action space (see BanditEnv's own docstring), these two things are
    always well-defined and unambiguous: "chosen_group" is never None for
    a valid action (every action belongs to exactly one group's block),
    while "matched_group" is None on a miss (chose a block, guessed
    wrong within it). There's no longer any collision case where a
    single action could satisfy both groups at once - that could only
    happen under the old shared-action-space design.

    Returns (hits_fixed, hits_variable, misses, chose_fixed, chose_variable):
    hits_fixed + hits_variable + misses == episodes (matched_group-based -
    exactly one of these three per episode), and separately
    chose_fixed + chose_variable == episodes (chosen_group-based - every
    action falls in exactly one block, so there's no "chose neither" to
    account for)."""
    env = BanditEnv(groups=groups, incorrect_reward=incorrect_reward)
    hits_fixed = hits_variable = misses = 0
    chose_fixed = chose_variable = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        raw_action, _ = model.predict(obs, deterministic=True)
        action = int(raw_action)
        _, _, _, _, info = env.step(action)
        matched = info["matched_group"]
        chosen = info["chosen_group"]
        if matched == 0:
            hits_fixed += 1
        elif matched == 1:
            hits_variable += 1
        else:
            misses += 1
        if chosen == 0:
            chose_fixed += 1
        elif chosen == 1:
            chose_variable += 1
    return hits_fixed, hits_variable, misses, chose_fixed, chose_variable


def _train_and_get_hits(
    x, k_fixed, k_variable, g, value_fixed, incorrect_reward, hits_per_run,
    seed, train_kwargs, csv_logger=None, phase=None, iteration=None,
    group_factory=None,
):
    """One from-scratch training run at variable-group value=x, followed
    by a dedicated `hits_per_run`-episode greedy evaluation tallying
    per-group hits (NOT trainPPO.train()'s own built-in final eval, which
    only reports aggregate correct/episodes across ALL groups combined -
    we need the fixed-vs-variable breakdown, so this runs its own
    separate evaluation loop via _evaluate_two_group_hits). x is clamped
    to >= 0 defensively - the variable reward can never be negative.

    Terminal only gets a one-line "x, target_rate_fixed" print here -
    (target_rate_fixed is what the certify/search running mean and CI are
    actually computed from - see the module docstring's Y_hat definition -
    so it moves in the same direction the printed CI does, unlike its
    complement target_rate_variable) -
    every other per-run detail computed (hit_fixed/hit_variable/missed,
    chose_fixed/chose_variable, the model's post-training weight L2 norm)
    goes to `csv_logger` instead, if one was passed (see _RunCsvLogger).
    `phase` ("search" or "certify") and `iteration` (the enclosing
    verify() iteration number, if any) are just passed through to the
    CSV row for context - they don't affect the run itself.

    `group_factory` is forwarded to `_build_groups` (see its docstring) -
    defaults to plain MarginGroup(delta=1/k) when None."""
    x = max(float(x), 0.0)
    groups = _build_groups(g, k_fixed, k_variable, value_fixed, x, group_factory=group_factory)
    result = train(groups, incorrect_reward=incorrect_reward, seed=seed, **train_kwargs)
    hf, hv, misses, cf, cv = _evaluate_two_group_hits(
        result.model, groups, incorrect_reward, hits_per_run,
    )
    target_rate_variable = 100.0 * cv / hits_per_run
    target_rate_fixed = 100.0 * cf / hits_per_run
    hit_rate_variable = 100.0 * hv / hits_per_run
    hit_rate_fixed = 100.0 * hf / hits_per_run
    missed_rate = 100.0 * misses / hits_per_run
    weight_norm = _model_weight_norm(result.model)

    print(f"[{phase or 'run'}] x={x:.4f}  target_rate_fixed={target_rate_fixed:.1f}%")

    if csv_logger is not None:
        csv_logger.log(
            "training_run",
            iteration=iteration, phase=phase, x=x, seed=seed,
            k_fixed=k_fixed, k_variable=k_variable, g=g,
            value_fixed=value_fixed, incorrect_reward=incorrect_reward,
            hits_per_run=hits_per_run,
            hits_fixed=hf, hits_variable=hv, misses=misses,
            chose_fixed=cf, chose_variable=cv,
            target_rate_variable_pct=target_rate_variable,
            target_rate_fixed_pct=target_rate_fixed,
            hit_rate_variable_pct=hit_rate_variable,
            hit_rate_fixed_pct=hit_rate_fixed,
            missed_rate_pct=missed_rate,
            model_weight_norm=weight_norm,
        )
    # cf + cv always equals hits_per_run (every action chooses exactly one
    # group's action block - see BanditEnv's docstring), so these are what
    # the rest of this module now treats as "hits_fixed"/"hits_variable":
    # the core search/certify statistics are driven by which target the
    # policy went for, not by whether it actually landed the correct
    # option within that target's block. The 3rd slot (historically
    # "misses") is always 0 here - kept only so callers elsewhere in this
    # module that unpack a 3-tuple don't need to change.
    return cf, cv, 0


def _half_width(n, sigma2=_HOEFFDING_SIGMA2, alpha=0.05, n0=None):
    """Anytime-valid confidence-sequence half-width for the running mean
    of n independent, mean-stationary observations each sub-Gaussian with
    a FIXED, known variance proxy sigma2 (0.25 by default - Hoeffding's
    lemma's guarantee for any [0,1]-bounded random variable, deliberately
    NOT estimated from the data - estimating it would break the
    martingale argument this bound relies on, making it invalid to peek
    at repeatedly). This is a standard Robbins normal-mixture confidence
    sequence (Darling & Robbins 1967; see also Howard et al. 2021's
    "always-valid" sub-Gaussian mixture boundary): for a sum
    S_n = sum_{i=1}^n (Y_i - mu), |S_n| stays within
    sqrt(2*v*log(sqrt(v/sigma2)/alpha)) - where v = n*rho2 + sigma2 -
    SIMULTANEOUSLY for every n, with probability >= 1-alpha; dividing by
    n gives the half-width on the mean itself. This is exactly what makes
    certify_reward's accept/reject decision (and the search phase's
    per-point direction check) a genuine (1-alpha)-confidence statement,
    rather than just a bigger point estimate.

    rho2 (the mixture's tuning variance) is set so the bound is tightest
    around n0 observations - defaults to n0=n if not given, which is a
    reasonable choice when you don't have a specific target sample size
    in mind ahead of time; passing an actual anticipated sample size (as
    certify_reward's `target_n` and the search phase's
    `target_n_per_point` do) tunes it to be tightest there instead. Note
    that n0 only changes how the tightness is DISTRIBUTED across n - the
    bound stays valid at every n for any fixed choice of n0.
    """
    if n <= 0:
        return float("inf")
    if n0 is None:
        n0 = n
    rho2 = sigma2 / max(n0, 1)
    v = n * rho2 + sigma2
    bound_on_sum = np.sqrt(2.0 * v * np.log(np.sqrt(v / sigma2) / alpha))
    return float(bound_on_sum / n)


def _evaluate_point_until_confident(
    x, k_fixed, k_variable, g, value_fixed, incorrect_reward,
    hits_per_point, seed_offset, train_kwargs,
    alpha=0.05, target_n_per_point=30, min_seeds_per_point=2,
    max_seeds_per_point=40, stall_patience=5,
    csv_logger=None, iteration=None, group_factory=None,
):
    """Train seeds one at a time at this x - NOT a preset count - until
    the pooled evidence confidently places Y_hat on one side of 0.5 (via
    the same anytime-valid confidence sequence certify_reward uses), or
    max_seeds_per_point is hit (a safety cap, not a target - a point that
    is genuinely near 0.5, e.g. it's close to M itself, may never resolve
    a confident direction). Returns (record, seeds_used) where record is
    {"x", "n_fixed", "n_variable", "seed_y_hats"} (seed_y_hats has one
    entry per trained seed - None for a seed that stalled, i.e. hit
    neither target in its hits_per_point episodes).

    The pooled per-point summary (seeds_used, avg_target_rate_variable,
    every seed's own y_hat) that used to print to the terminal after every
    point now only goes to `csv_logger` (record_type="search_point") -
    each individual seed's run still gets its own one-line terminal print
    from _train_and_get_hits."""
    x = max(float(x), 0.0)
    n_fixed_total = 0
    n_variable_total = 0
    seed_y_hats = []
    y_hats_for_ci = []  # excludes stalled seeds
    consecutive_stalls = 0
    seeds_used = 0

    while seeds_used < max_seeds_per_point:
        seed = seed_offset + seeds_used
        seeds_used += 1
        hf, hv, misses = _train_and_get_hits(
            x, k_fixed, k_variable, g, value_fixed, incorrect_reward,
            hits_per_point, seed, train_kwargs,
            csv_logger=csv_logger, phase="search", iteration=iteration,
            group_factory=group_factory,
        )
        n_fixed_total += hf
        n_variable_total += hv
        denom = hf + hv

        if denom == 0:
            seed_y_hats.append(None)
            consecutive_stalls += 1
            if consecutive_stalls >= stall_patience:
                break
            continue

        consecutive_stalls = 0
        y = hf / denom
        seed_y_hats.append(y)
        y_hats_for_ci.append(y)

        n = len(y_hats_for_ci)
        if n >= min_seeds_per_point:
            mean = float(np.mean(y_hats_for_ci))
            hw = _half_width(n, alpha=alpha, n0=target_n_per_point)
            if mean - hw > 0.5 or mean + hw < 0.5:
                break  # confidently on one side of 0.5 - direction settled

    record = {
        "x": x,
        "n_fixed": n_fixed_total,
        "n_variable": n_variable_total,
        "seed_y_hats": seed_y_hats,
    }
    total_episodes = n_fixed_total + n_variable_total
    avg_target_rate_variable = (
        100.0 * n_variable_total / total_episodes if total_episodes > 0 else float("nan")
    )
    if csv_logger is not None:
        csv_logger.log(
            "search_point",
            iteration=iteration, phase="search", x=x,
            k_fixed=k_fixed, k_variable=k_variable, g=g,
            value_fixed=value_fixed, incorrect_reward=incorrect_reward,
            seeds_used=seeds_used,
            n_fixed_total=n_fixed_total, n_variable_total=n_variable_total,
            avg_target_rate_variable_pct=avg_target_rate_variable,
            seed_y_hats=seed_y_hats,
        )
    return record, seeds_used


def _pooled_y(record):
    """Pooled Y_hat for a search-point record, or None if every seed
    stalled (0 total hits on either target)."""
    denom = record["n_fixed"] + record["n_variable"]
    return record["n_fixed"] / denom if denom > 0 else None


def find_candidate_reward(
    k_fixed,
    k_variable,
    search_range=None,
    hits_per_point=500,
    g=4,
    value_fixed=1.0,
    incorrect_reward=0.0,
    base_seed=0,
    train_kwargs=None,
    extra_points=None,
    start_x_hint=None,
    initial_step_hint=None,
    search_alpha=0.05,
    target_n_per_point=30,
    min_seeds_per_point=2,
    max_seeds_per_point=40,
    bracket_tol=None,
    m_stable_tol=None,
    m_stable_window=3,
    max_points=200,
    csv_logger=None,
    iteration=None,
    fit_exponential=False,
    group_factory=None,
):
    """Phase 1 (heuristic search, NO statistical guarantee) - see module
    docstring. `train_kwargs` is forwarded verbatim to trainPPO.train()
    for every run (e.g. total_timesteps, learning_rate, n_steps, ...) -
    see find_indifference_reward() for a version that exposes those as
    individually named arguments instead of a raw dict.

    group_factory: optional `(g, difficulty, value) -> Group` callable
    controlling what `k_fixed`/`k_variable` actually build - see
    _build_groups' docstring. Defaults to plain MarginGroup(delta=1/k)
    when None, i.e. exactly this module's original behavior. Every other
    part of the search (the confidence-driven point/seed logic, the
    bracket/stability stopping criteria, the sigmoid MLE fit) only ever
    touches the pooled (x, n_fixed, n_variable) counts - none of it
    assumes MarginGroup specifically, so a different group_factory (e.g.
    building AlternatingGroup or HeatmapGroup instead) is a drop-in swap.

    Neither the number of x values tried nor the number of seeds trained
    at each x is fixed in advance - see the module docstring's Phase 1
    section for the full confidence-driven algorithm. In short:
      - `search_range`: optional (low, high) hint for x (the variable
        group's reward value) - if given, only used to pick the starting
        point (its midpoint) and initial step size; it does NOT fix how
        many points get evaluated. If None, the search starts at
        x = value_fixed instead. x never goes negative: 0 (no reward at
        all for the variable target) is the floor.
      - `search_alpha`, `target_n_per_point`, `min_seeds_per_point`,
        `max_seeds_per_point`: control the per-point "how many seeds
        until confident about direction" check (see
        _evaluate_point_until_confident). `max_seeds_per_point` is a
        safety cap, not a target.
      - `bracket_tol`, `m_stable_tol`, `m_stable_window`: control when
        the OVERALL search stops - as soon as EITHER the bracket around
        the crossing has narrowed below `bracket_tol` (defaults to
        max(1e-3, 0.01 * value_fixed)) OR the fitted M has stayed within
        `m_stable_tol` (same default) of itself for `m_stable_window`
        consecutive points - whichever happens first. A tight-enough
        bracket is, on its own, already enough precision; it doesn't also
        need to wait for the fit to separately stabilize (and vice
        versa - a stable fit is enough even if a clean bracket was never
        found, e.g. no search_range was given and the initial guess
        happened to land very close to M). `max_points` is a safety cap
        on the total number of x values tried, not a target.

    hits_per_point: how many deterministic (greedy) evaluation episodes
    EACH trained seed is scored on at a given x, to produce that seed's
    own (target_fixed, target_variable) counts - i.e. how many of its
    episodes went for each group's action block. It is NOT how many
    training runs happen at that point (that's decided by the confidence
    check above) - it only controls how precisely a single already-
    trained policy's own target rate is measured. Raising it shrinks the
    noise in each seed's individual Y_hat, but does nothing about how
    much trained policies differ from each other run-to-run at the same x
    (that run-to-run spread is real and often large - see the module
    docstring - and is only addressed by training more seeds, not more
    hits_per_point).

    extra_points: optional list of already-evaluated point dicts
    ({"x", "n_fixed", "n_variable"}, "seed_y_hats" optional) to fold into
    the MLE fit alongside anything evaluated by this call - used by
    find_indifference_reward's verify loop to reuse earlier certify
    batches' pooled counts as extra information when refitting, without
    re-running their training.

    start_x_hint / initial_step_hint: optional override for where the
    bisection walk STARTS (see the algorithm description above) - used
    by find_indifference_reward's verify loop on its 2nd+ iteration so a
    re-search doesn't re-walk x=value_fixed, 2*value_fixed, ... from
    scratch every time when earlier iterations already narrowed down
    roughly where M is. Only changes the STARTING point/step; every
    downstream mechanism (doubling the step while the wrong direction
    keeps being confirmed, bracketing once the sign flips, and the
    bracket_tol/m_stable_tol/m_stable_window stopping criteria) is
    completely unchanged, so a wrong or stale hint costs a few extra
    points recovering from it - via the same doubling logic that already
    handles a bad guess when no hint is given - never a wrong or
    under-verified answer. Ignored if `search_range` is given (an
    explicit range is a stronger, deliberate signal than a hint from a
    previous iteration). `start_x_hint` alone (no `initial_step_hint`)
    falls back to the same step `value_fixed` would give.

    Returns a dict: {"M", "beta", "converged", "points", "note"}. "points"
    is a list of per-x dicts with the pooled counts and each individual
    seed's own Y_hat (None for a seed that stalled), so you can inspect
    run-to-run spread directly rather than only the pooled fit. "points"
    includes extra_points, if any, appended after this call's own
    newly-evaluated points. If `fit_exponential=True`, the dict also has
    "M_exp"/"beta_exp"/"exp_converged" - an alternative fit using an
    exponential-tailed (Laplace CDF) curve instead of the logistic
    sigmoid (see _fit_exponential_mle). This is a comparison/diagnostic
    only: it's cheap to compute (just another Nelder-Mead fit over the
    SAME pooled counts, no extra training), and it never influences the
    search's own bracket/stability decisions or the returned "M"/"beta" -
    those always come from the sigmoid fit. All three (sigmoid and, if
    requested, exponential) fits are logged to `csv_logger` after every
    refit if one was passed.
    """
    train_kwargs = dict(train_kwargs or {})

    if bracket_tol is None:
        bracket_tol = max(1e-3, 0.01 * max(value_fixed, 1e-6))
    if m_stable_tol is None:
        m_stable_tol = max(1e-3, 0.01 * max(value_fixed, 1e-6))

    point_records = []
    seed_counter = 0
    m_history = []
    last_fit = (None, None, None)  # (M_hat, beta_hat, opt_result)
    last_fit_exp = (None, None, None)  # (M_hat_exp, beta_hat_exp, opt_result_exp)

    def evaluate(x):
        nonlocal seed_counter
        record, used = _evaluate_point_until_confident(
            x, k_fixed, k_variable, g, value_fixed, incorrect_reward,
            hits_per_point, base_seed + seed_counter, train_kwargs,
            alpha=search_alpha, target_n_per_point=target_n_per_point,
            min_seeds_per_point=min_seeds_per_point,
            max_seeds_per_point=max_seeds_per_point,
            csv_logger=csv_logger, iteration=iteration,
            group_factory=group_factory,
        )
        seed_counter += used
        point_records.append(record)
        return record

    def refit():
        nonlocal last_fit, last_fit_exp
        pts = point_records + list(extra_points or [])
        xs_arr = np.array([r["x"] for r in pts])
        nf_arr = np.array([r["n_fixed"] for r in pts])
        nv_arr = np.array([r["n_variable"] for r in pts])
        m0 = float(value_fixed) if value_fixed > 0 else float(np.mean(xs_arr))
        M_hat, beta_hat, opt_result = _fit_sigmoid_mle(xs_arr, nf_arr, nv_arr, m0)
        M_hat = max(M_hat, 0.0)  # the variable reward can never be negative
        last_fit = (M_hat, beta_hat, opt_result)
        m_history.append(M_hat)

        M_hat_exp = beta_hat_exp = opt_result_exp = None
        if fit_exponential:
            M_hat_exp, beta_hat_exp, opt_result_exp = _fit_exponential_mle(
                xs_arr, nf_arr, nv_arr, m0,
            )
            M_hat_exp = max(M_hat_exp, 0.0)
            last_fit_exp = (M_hat_exp, beta_hat_exp, opt_result_exp)

        if csv_logger is not None:
            csv_logger.log(
                "search_refit",
                iteration=iteration, phase="search",
                fitted_M=M_hat, fitted_beta=beta_hat,
                fit_converged=bool(opt_result.success),
                fitted_M_exp=M_hat_exp, fitted_beta_exp=beta_hat_exp,
                fit_exp_converged=(
                    bool(opt_result_exp.success) if opt_result_exp is not None else ""
                ),
            )
        return M_hat, beta_hat, opt_result

    def m_is_stable():
        if len(m_history) < m_stable_window:
            return False
        recent = m_history[-m_stable_window:]
        return (max(recent) - min(recent)) < m_stable_tol

    if search_range is not None:
        lo, hi = search_range
        start_x = max(0.5 * (float(lo) + float(hi)), 0.0)
        step = max(abs(float(hi) - float(lo)) / 2.0, 1e-6)
    elif start_x_hint is not None:
        # See start_x_hint's own docstring note above - only the
        # STARTING point/step changes; the doubling/bracket/stability
        # logic below is untouched, so a wrong hint just costs a few
        # extra points, not a wrong answer.
        start_x = max(float(start_x_hint), 0.0)
        step = (
            max(float(initial_step_hint), 1e-6) if initial_step_hint is not None
            else (value_fixed if value_fixed > 0 else 1.0)
        )
    else:
        start_x = max(value_fixed if value_fixed > 0 else 1.0, 0.0)
        step = value_fixed if value_fixed > 0 else 1.0

    x = start_x
    direction = None  # "up" or "down", once we've made one exploratory move
    x_prev, y_prev = None, None
    bracket = None  # (x_lo, y_lo, x_hi, y_hi) once found

    points_used = 0
    while points_used < max_points:
        record = evaluate(x)
        points_used += 1
        y = _pooled_y(record)
        # A stalled point (every seed hit neither target) gives no
        # direction signal - treat it as "exactly balanced" so the
        # search keeps moving rather than getting stuck, but it still
        # contributes its (zero) counts to the eventual fit.
        y_for_decision = 0.5 if y is None else y

        if bracket is not None:
            x_lo, y_lo, x_hi, y_hi = bracket
            if y_for_decision > 0.5:
                x_lo, y_lo = x, y_for_decision
            else:
                x_hi, y_hi = x, y_for_decision
            bracket = (x_lo, y_lo, x_hi, y_hi)
            refit()
            if (x_hi - x_lo) < bracket_tol or m_is_stable():
                break
            x = 0.5 * (x_lo + x_hi)
            continue

        # Note: an exact tie (y_for_decision == 0.5) is NOT treated as "we've
        # found the crossing, stop" - with realistic pooled counts an exact
        # tie carries essentially no confidence (it can happen by pure
        # chance with a small pooled count, as in the tiny smoke test at
        # the bottom of this file), so it just breaks ">=" toward "up" and
        # the search keeps going; the bracket/M-stability criteria below
        # are what actually decide when there's enough evidence to stop.
        new_direction = "up" if y_for_decision >= 0.5 else "down"
        if direction is None:
            direction = new_direction
            x_prev, y_prev = x, y_for_decision
            refit()
            x = max(x + step, 0.0) if direction == "up" else max(x - step, 0.0)
            continue

        if new_direction == direction:
            # Still wrong about which side we're on - grow the step
            # (increasing pace) and keep pushing the same way, unless
            # we're already pinned at the x=0 floor and still want to go
            # lower, in which case 0 IS the answer for this direction and
            # there's nothing more to bracket.
            if direction == "down" and x <= 0.0 and x_prev <= 0.0:
                refit()
                break
            x_prev, y_prev = x, y_for_decision
            step *= 2.0
            refit()
            x = max(x + step, 0.0) if direction == "up" else max(x - step, 0.0)
            continue

        # Sign flipped relative to x_prev - the crossing is bracketed
        # between x_prev and this x.
        if direction == "up":
            bracket = (x_prev, y_prev, x, y_for_decision)
        else:
            bracket = (x, y_for_decision, x_prev, y_prev)
        refit()
        x_lo, y_lo, x_hi, y_hi = bracket
        x = 0.5 * (x_lo + x_hi)

    max_points_hit = points_used >= max_points
    if csv_logger is not None and max_points_hit:
        csv_logger.log(
            "search_cap_warning",
            iteration=iteration, phase="search",
            seeds_used=(
                f"max_points safety cap ({max_points}) reached without the "
                "bracket/M-stability criteria converging - using the best "
                "fit found so far."
            ),
        )

    if last_fit[0] is None:
        refit()
    M_hat, beta_hat, opt_result = last_fit
    M_hat_exp, beta_hat_exp, opt_result_exp = last_fit_exp

    result = {
        "M": M_hat,
        "beta": beta_hat,
        "converged": bool(opt_result.success),
        "points": point_records + list(extra_points or []),
        "note": (
            "Heuristic search estimate only - carries NO statistical "
            "guarantee on its own. Call certify_reward(M, ...) (or "
            "find_indifference_reward(..., verify=True)) for a rigorous, "
            "pre-registered confirmation at a single fixed x."
        ),
    }
    if fit_exponential:
        result["M_exp"] = M_hat_exp
        result["beta_exp"] = beta_hat_exp
        result["exp_converged"] = (
            bool(opt_result_exp.success) if opt_result_exp is not None else None
        )
    return result


def _neg_log_likelihood(params, xs, n_fixed, n_variable):
    """Negative log-likelihood of pooled Binomial counts at each x under
    P(goes for fixed) = sigmoid(-4*beta*(x - M)). beta is
    optimized in log-space (log_beta) so the unconstrained optimizer
    can't wander into beta <= 0, which would make the sigmoid non-
    monotonic in the wrong direction."""
    M, log_beta = params
    beta = np.exp(log_beta)
    nll = 0.0
    for x, nf, nv in zip(xs, n_fixed, n_variable):
        n = nf + nv
        if n == 0:
            continue
        z = 4.0 * beta * (x - M)
        # log(sigmoid(-z)) and log(sigmoid(z)), both computed stably via
        # logaddexp instead of forming exp(z) directly (avoids overflow
        # for large |z|).
        log_p_fixed = -np.logaddexp(0.0, z)
        log_p_variable = -np.logaddexp(0.0, -z)
        nll -= nf * log_p_fixed + nv * log_p_variable
    return float(nll)


def _fit_sigmoid_mle(xs, n_fixed, n_variable, m0):
    """Fit (M, beta) by maximizing the pooled-Binomial likelihood above.
    Returns (M_hat, beta_hat, scipy OptimizeResult) - beta_hat is always
    > 0 (see _neg_log_likelihood's log-space parameterization)."""
    xs = np.asarray(xs, dtype=float)
    n_fixed = np.asarray(n_fixed, dtype=float)
    n_variable = np.asarray(n_variable, dtype=float)

    x0 = [float(m0), 0.0]  # beta0 = exp(0) = 1.0
    res = minimize(
        _neg_log_likelihood, x0, args=(xs, n_fixed, n_variable), method="Nelder-Mead",
    )
    M_hat, log_beta_hat = res.x
    beta_hat = float(np.exp(log_beta_hat))
    return float(M_hat), beta_hat, res


def certify_reward(
    x_star,
    k_fixed,
    k_variable,
    g=4,
    value_fixed=1.0,
    incorrect_reward=0.0,
    lo=0.40,
    hi=0.60,
    alpha=0.05,
    target_n=200,
    max_runs=5000,
    stall_patience=5,
    hits_per_run=500,
    base_seed=10_000_000,
    train_kwargs=None,
    csv_logger=None,
    iteration=None,
    group_factory=None,
):
    """Phase 2 (rigorous, anytime-valid SEQUENTIAL test, WITH a statistical
    guarantee) - see module docstring. `train_kwargs` is forwarded
    verbatim to trainPPO.train() for every run - see
    find_indifference_reward() for individually-named-argument access.

    group_factory: optional `(g, difficulty, value) -> Group` callable -
    same meaning as find_candidate_reward's (see _build_groups'
    docstring). Defaults to plain MarginGroup(delta=1/k) when None.

    This is the actual VERIFICATION step: it checks, with (1-alpha)
    confidence (95% by default), whether P(goes for fixed) at x_star
    genuinely lies in (lo, hi) - default (0.40, 0.60) - rather than just
    producing a bigger estimate of it. It trains fresh from-scratch
    policies at the single fixed point x_star = max(x_star, 0.0) (never
    negative), one at a time, recomputing the running mean Y_hat and its
    anytime-valid confidence interval after each. Stops as soon as:
      - the CI sits entirely inside (lo, hi)      -> status="accepted"
      - the CI sits entirely outside (lo, hi)     -> status="rejected"
      - max_runs is reached without either        -> status="max_runs_reached"
      - stall_patience consecutive stalled runs   -> status="aborted_stalls"
    (a "stalled" run is one whose hits_per_run-episode eval went for
    NEITHER target at all - impossible now that every action falls in
    some group's block, so this should never actually trigger; kept only
    as a defensive fallback - excluded from the running mean entirely, not counted
    as evidence either way).

    max_runs is a SAFETY VALVE, not a realistic target - do not treat it
    as "how many runs this should take." Converging on a tight (lo, hi)
    band can genuinely take on the order of a couple hundred runs when
    the true probability at x_star sits close to lo or hi (the CI has to
    shrink to less than the distance to whichever boundary is nearer, and
    that shrinks slowly - roughly like a constant divided by n). The
    default of 5000 is meant to almost never actually be hit; if it is,
    that's a real signal x_star is sitting right at (or outside) the
    boundary of (lo, hi), not that the test needs more patience.

    target_n tunes HOW the confidence sequence's tightness is distributed
    across the run count (see _half_width's docstring) - it doesn't
    change the safety cap. Pick it close to the run count you actually
    expect to need; the default of 200 is a reasonable middle ground.

    Returns a dict: {"x", "status", "n_runs", "stalled_runs", "mean",
    "history", "y_hats", "n_fixed_total", "n_variable_total"}. "history"
    is a list of {"n", "mean", "ci_lo", "ci_hi"} dicts, one per
    non-stalled run, so you can see how the CI narrowed over the course
    of the test. "n_fixed_total"/"n_variable_total" are this batch's
    pooled hit counts across every non-stalled run - useful as one more
    (very informative) data point if a later search refit wants to reuse
    this batch's evidence without re-running it.
    """
    x_star = max(float(x_star), 0.0)
    train_kwargs = dict(train_kwargs or {})

    y_hats = []
    n_fixed_total = 0
    n_variable_total = 0
    stalled_count = 0
    consecutive_stalls = 0
    history = []
    seed_counter = 0
    status = "max_runs_reached"

    while len(y_hats) < max_runs:
        seed = base_seed + seed_counter
        seed_counter += 1
        hf, hv, misses = _train_and_get_hits(
            x_star, k_fixed, k_variable, g, value_fixed, incorrect_reward,
            hits_per_run, seed, train_kwargs,
            csv_logger=csv_logger, phase="certify", iteration=iteration,
            group_factory=group_factory,
        )
        denom = hf + hv

        if denom == 0:
            stalled_count += 1
            consecutive_stalls += 1
            if csv_logger is not None:
                csv_logger.log(
                    "certify_stall",
                    iteration=iteration, phase="certify", x=x_star, seed=seed,
                    certify_status="stalled",
                )
            if consecutive_stalls >= stall_patience:
                status = "aborted_stalls"
                break
            continue

        consecutive_stalls = 0
        n_fixed_total += hf
        n_variable_total += hv
        y_hats.append(hf / denom)
        n = len(y_hats)
        mean = float(np.mean(y_hats))
        hw = _half_width(n, alpha=alpha, n0=target_n)
        ci_lo, ci_hi = mean - hw, mean + hw
        history.append({"n": n, "mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi})
        print(f"[certify] n={n}  CI=({ci_lo:.4f}, {ci_hi:.4f})")
        if csv_logger is not None:
            csv_logger.log(
                "certify_run",
                iteration=iteration, phase="certify", x=x_star,
                certify_n=n, certify_mean=mean,
                certify_ci_lo=ci_lo, certify_ci_hi=ci_hi,
            )

        if ci_lo > lo and ci_hi < hi:
            status = "accepted"
            break
        if ci_hi < lo or ci_lo > hi:
            status = "rejected"
            break

    if csv_logger is not None:
        csv_logger.log(
            "certify_summary",
            iteration=iteration, phase="certify", x=x_star,
            certify_status=status,
            certify_n=len(y_hats),
            certify_mean=float(np.mean(y_hats)) if y_hats else "",
            n_fixed_total=n_fixed_total, n_variable_total=n_variable_total,
        )

    return {
        "x": x_star,
        "status": status,
        "n_runs": len(y_hats),
        "stalled_runs": stalled_count,
        "mean": float(np.mean(y_hats)) if y_hats else None,
        "history": history,
        "y_hats": y_hats,
        "n_fixed_total": n_fixed_total,
        "n_variable_total": n_variable_total,
    }


def find_indifference_reward(
    k_fixed,
    k_variable,
    verify=False,
    # --- search (phase 1) config ---
    search_range=None,
    hits_per_point=500,
    search_alpha=0.05,
    target_n_per_point=40,
    min_seeds_per_point=3,
    max_seeds_per_point=40,
    bracket_tol=None,
    m_stable_tol=None,
    m_stable_window=3,
    max_points=200,
    # --- certify (phase 2) config - only used if verify=True ---
    lo=0.40,
    hi=0.60,
    alpha=0.05,
    target_n=200,
    max_runs=5000,
    stall_patience=5,
    hits_per_certify_run=500,
    max_search_iterations=10,
    # --- shared env config ---
    g=4,
    value_fixed=1.0,
    incorrect_reward=0.0,
    base_seed=0,
    # --- logging config ---
    csv_path=None,
    fit_exponential=False,
    # --- group config ---
    group_factory=None,
    # --- normal training parameters (mirrors trainPPO.train()'s full
    # signature, minus `groups`/`seed` - both are set internally per run) ---
    n_envs=8,
    total_timesteps=200_000,
    label="reward_indifference",
    progress_bar=False,
    log_training_data=False,
    log_interval=None,
    print_final_summary=False,
    device="cpu",
    verbose=0,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=512,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    net_arch_pi=(64, 32),
    net_arch_vf=(64, 32),
    activation_fn=None,
    optimizer_class=None,
    weight_decay=0.0,
    actor_weight_decay=None,
    critic_weight_decay=None,
    policy_kwargs_extra=None,
    ppo_kwargs=None,
    # train()'s OWN built-in final eval - irrelevant here (per-group hit
    # counts come from hits_per_point/hits_per_certify_run's dedicated
    # evaluation instead), kept tiny so train() doesn't waste time
    # computing an aggregate number nothing here reads.
    eval_episodes=1,
    periodic_eval_freq=None,
    periodic_eval_episodes=100,
    weights_dir="weights",
    eval_logs_dir="eval_logs",
    save_model=False,
):
    """
    Estimate x* = M, the reward value for the k_variable-difficulty
    MarginGroup target (with the k_fixed-difficulty target's reward held
    at value_fixed) at which a freshly-trained PPO policy is genuinely
    indifferent between the two targets - see the module docstring for
    the full setup and the two-phase (search then certify) methodology.
    x is never negative (0 is allowed). Neither phase runs a
    predetermined number of training runs - both stop based on
    confidence/convergence, with generous safety caps (max_points,
    max_seeds_per_point, max_runs) only to guard against looping forever
    in a genuinely-undecidable case (e.g. the true probability sitting
    right at a boundary).

    Without verify, runs the heuristic search (find_candidate_reward)
    ONCE and returns its estimate - a solid, cheap number with no
    statistical guarantee attached; that's a deliberate default (this is
    the "solid estimate is fine" case).

    With verify=True, keeps searching AND certifying until a certify
    batch is actually ACCEPTED (P(goes for fixed) verified to lie
    in (lo, hi) with (1-alpha) confidence) - or until max_search_iterations
    attempts are exhausted, in which case it gives up and returns the
    best candidate found so far with certified=False rather than looping
    forever. Each iteration after the first reuses every previous
    iteration's pooled hit counts (both from the search and from earlier
    certify batches) when refitting the next candidate, but every
    certify batch's own accept/reject decision is still made from a
    fresh, from-scratch set of training runs - see the module docstring's
    "find_indifference_reward() ties both phases together" section for
    why that separation matters.

    The `alpha` passed in here is the FAMILY-WISE target for the whole
    verify loop, not the per-attempt alpha: internally, each individual
    certify_reward() call is run at alpha/max_search_iterations (see the
    module docstring's "alpha-splitting across attempts" note), so that
    the probability of the FINAL accepted result being a false accept
    stays bounded by `alpha` even in the worst case where every attempt up
    to max_search_iterations was needed - not just whichever attempt
    happened to succeed.

    Every training-related keyword argument here is passed straight
    through to trainPPO.train() for every run this function performs -
    the same knobs as any other script in this repo that calls train()
    directly. This includes the MODEL's own hyperparameters, not just the
    training-loop ones: net_arch_pi/net_arch_vf (model size), activation_fn
    (e.g. torch.nn.ReLU instead of SB3's default torch.nn.Tanh),
    optimizer_class (e.g. torch.optim.Adam/SGD instead of this project's
    default torch.optim.AdamW), and policy_kwargs_extra (any other
    ActorCriticPolicy keyword not otherwise exposed) - see trainPPO.train's
    own docstring for the full explanation of each.

    save_model defaults to False here (unlike trainPPO.train()'s own
    default of True): this function trains a large number of throwaway
    policies (every search seed, every certify run, every verify
    iteration) and never reloads any of them, so writing each one to
    weights_dir/ppo_{label}.zip would just repeatedly overwrite the same
    file for no benefit - pure wasted disk I/O on every single run. Pass
    save_model=True explicitly if you actually want the LAST run's model
    kept on disk.

    group_factory: optional `(g, difficulty, value) -> Group` callable
    controlling what `k_fixed`/`k_variable` actually build for each of
    the two targets - see _build_groups' docstring. Defaults to plain
    MarginGroup(delta=1/k) when None (this module's original,
    unconditional behavior). Passed straight through to every
    find_candidate_reward()/certify_reward() call this function makes
    (both the initial search and every verify-loop re-search/certify
    attempt) - the search/certify statistics themselves (the pooled
    Binomial counts, the sigmoid MLE fit, the anytime-valid confidence
    sequence) never look at what Group type produced them, so this is a
    complete swap: e.g. a `lambda g, k, value: AlternatingGroup(g=g, k=k,
    value=value)` runs this whole pipeline against AlternatingGroup
    instead, or a factory that unpacks a `(noise_scale, n)` tuple runs it
    against HeatmapGroup - `k_fixed`/`k_variable` don't have to be plain
    integers once a custom group_factory is given; they can be whatever
    shape that factory expects (see run_indifference_batch.py for
    worked examples of both).

    csv_path / fit_exponential - logging: the terminal now only prints
    the current x plus that run's target_rate_fixed during search, the
    running CI during certify, an occasional 1-line status note, and the
    final "M = ... beta = ..." (that last one is printed by this module's
    own __main__ block, not by this function). EVERYTHING else that used
    to be printed every run - hit_fixed/hit_variable/missed rates,
    chose_fixed/chose_variable counts, pooled per-point summaries, every
    seed's own y_hat, each refit's M/beta (and its convergence flag), the
    certify CI history, and each run's trained model's own weight L2 norm
    (`model_weight_norm` - a cheap post-training diagnostic; nothing in
    the search/certify math reads it) - is written instead to a single
    CSV at `csv_path` (default: `{eval_logs_dir}/{label}_run_log.csv`),
    one row per event, tagged by `record_type`. `fit_exponential=True`
    additionally fits (and logs to that same CSV) an exponential-tailed
    alternative to the sigmoid curve at every refit - see
    find_candidate_reward's docstring and _fit_exponential_mle for what
    that curve is; it's a comparison/diagnostic only and never changes
    which M/beta this function actually returns.

    Returns a dict:
        {
            "M": <final candidate's estimate>,
            "beta": <final candidate's fitted slope>,
            "search": <the LAST find_candidate_reward() call's full return dict>,
            "certify": <the LAST certify_reward() call's full return dict, or None>,
            "certified": <True/False/None>,
            "iterations": <list of {"search", "certify"} dicts, one per
                            search/certify round actually run>,
        }
    "certified" is None if verify=False, True iff the LAST certify
    attempt's status == "accepted", False otherwise (including if
    max_search_iterations was exhausted without an accept).
    """
    train_kwargs = dict(
        n_envs=n_envs,
        total_timesteps=total_timesteps,
        label=label,
        progress_bar=progress_bar,
        log_training_data=log_training_data,
        log_interval=log_interval,
        print_final_summary=print_final_summary,
        device=device,
        verbose=verbose,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        net_arch_pi=net_arch_pi,
        net_arch_vf=net_arch_vf,
        activation_fn=activation_fn,
        optimizer_class=optimizer_class,
        weight_decay=weight_decay,
        actor_weight_decay=actor_weight_decay,
        critic_weight_decay=critic_weight_decay,
        policy_kwargs_extra=policy_kwargs_extra,
        ppo_kwargs=ppo_kwargs,
        eval_episodes=eval_episodes,
        periodic_eval_freq=periodic_eval_freq,
        periodic_eval_episodes=periodic_eval_episodes,
        weights_dir=weights_dir,
        eval_logs_dir=eval_logs_dir,
        save_model=save_model,
        print_eval_summary=False,
    )

    if csv_path is None:
        csv_path = f"{eval_logs_dir}/{label}_run_log.csv"
    csv_logger = _RunCsvLogger(csv_path)
    print(f"[csv] logging every run's full detail -> {csv_path}")

    def run_search(base_seed_for_search, extra_points, iteration,
                    start_x_hint=None, initial_step_hint=None):
        return find_candidate_reward(
            k_fixed,
            k_variable,
            search_range=search_range,
            hits_per_point=hits_per_point,
            g=g,
            value_fixed=value_fixed,
            incorrect_reward=incorrect_reward,
            base_seed=base_seed_for_search,
            train_kwargs=train_kwargs,
            extra_points=extra_points,
            start_x_hint=start_x_hint,
            initial_step_hint=initial_step_hint,
            search_alpha=search_alpha,
            target_n_per_point=target_n_per_point,
            min_seeds_per_point=min_seeds_per_point,
            max_seeds_per_point=max_seeds_per_point,
            bracket_tol=bracket_tol,
            m_stable_tol=m_stable_tol,
            m_stable_window=m_stable_window,
            max_points=max_points,
            csv_logger=csv_logger,
            iteration=iteration,
            fit_exponential=fit_exponential,
            group_factory=group_factory,
        )

    extra_points = []  # pooled counts from earlier certify batches, if any
    search_result = run_search(base_seed, extra_points=None, iteration=1)

    result = {
        "M": search_result["M"],
        "beta": search_result["beta"],
        "search": search_result,
        "certify": None,
        "certified": None,
        "iterations": [],
    }

    if not verify:
        csv_logger.log(
            "final", final_M=result["M"], final_beta=result["beta"], certified="",
        )
        csv_logger.close()
        return result

    certify_seed_base = base_seed + 10_000_000
    # Bonferroni-style alpha-splitting: each certify attempt below tests at
    # alpha/max_search_iterations rather than the flat alpha, so that even
    # in the worst case (every attempt needed a retry, all the way to the
    # last one) the FAMILY-WISE false-accept probability across the whole
    # verify loop stays bounded by `alpha`, not inflated up to
    # ~1-(1-alpha)**max_search_iterations. See the module docstring for why
    # this matters even when (as is often the case) only one attempt is
    # actually needed - the guarantee has to be pre-registered for the
    # procedure as a whole, not decided after the fact based on how many
    # attempts happened to be used.
    certify_alpha = alpha / max_search_iterations
    for iteration in range(1, max_search_iterations + 1):
        # A DIFFERENT base_seed range than the search (or any prior
        # certify attempt) used, and entirely fresh training runs - never
        # reusing an earlier batch's own runs as evidence for THIS
        # batch's accept/reject decision (see module docstring).
        certify_result = certify_reward(
            search_result["M"],
            k_fixed,
            k_variable,
            g=g,
            value_fixed=value_fixed,
            incorrect_reward=incorrect_reward,
            lo=lo,
            hi=hi,
            alpha=certify_alpha,
            target_n=target_n,
            max_runs=max_runs,
            stall_patience=stall_patience,
            hits_per_run=hits_per_certify_run,
            base_seed=certify_seed_base,
            train_kwargs=train_kwargs,
            csv_logger=csv_logger,
            iteration=iteration,
            group_factory=group_factory,
        )
        certify_seed_base += 10_000_000

        result["search"] = search_result
        result["certify"] = certify_result
        result["M"] = search_result["M"]
        result["beta"] = search_result["beta"]
        result["iterations"].append({"search": search_result, "certify": certify_result})

        accepted = certify_result["status"] == "accepted"
        result["certified"] = accepted
        csv_logger.log(
            "verify_iteration",
            iteration=iteration, fitted_M=search_result["M"],
            fitted_beta=search_result["beta"],
            certify_status=certify_result["status"], certified=accepted,
        )
        if accepted:
            break
        if iteration == max_search_iterations:
            csv_logger.log(
                "verify_exhausted",
                iteration=iteration,
                seeds_used=(
                    "max_search_iterations exhausted without an accepted "
                    "certify batch - returning the best candidate found so "
                    "far (certified=False)."
                ),
            )
            break

        # Fold this certify batch's pooled counts in as one more (very
        # informative - certify batches run many episodes) data point,
        # and refit the candidate from everything seen so far.
        extra_points.append({
            "x": certify_result["x"],
            "n_fixed": certify_result["n_fixed_total"],
            "n_variable": certify_result["n_variable_total"],
        })

        # Cheap, no-new-training refit over every point collected across
        # every iteration so far (this certify batch included), purely
        # to seed the NEXT search's starting point/step close to where
        # M actually is - instead of re-walking x=value_fixed,
        # 2*value_fixed, ... from scratch every iteration, which just
        # re-trains fresh seeds at x values we already have strong
        # evidence are nowhere near M. This does NOT skip or shortcut
        # the search itself - find_candidate_reward still runs its full
        # doubling/bracket/stability procedure with entirely fresh
        # training runs starting from this point, so a wrong or stale
        # hint only costs a few extra points recovering from it (the
        # same doubling logic that already handles a bad starting guess
        # with no hint at all), never a worse or under-verified answer.
        hint_xs = np.array([p["x"] for p in extra_points])
        hint_nf = np.array([p["n_fixed"] for p in extra_points])
        hint_nv = np.array([p["n_variable"] for p in extra_points])
        hint_m0 = float(value_fixed) if value_fixed > 0 else float(np.mean(hint_xs))
        start_x_hint, _, _ = _fit_sigmoid_mle(hint_xs, hint_nf, hint_nv, hint_m0)
        start_x_hint = max(start_x_hint, 0.0)
        bracket_tol_effective = (
            bracket_tol if bracket_tol is not None
            else max(1e-3, 0.01 * max(value_fixed, 1e-6))
        )
        # A modest, not tiny, initial step: big enough that if the hint
        # is meaningfully off, the doubling walk still closes the gap in
        # very few points; small enough that a good hint doesn't waste
        # points on a first step sized for "no idea where M is."
        initial_step_hint = max(
            4.0 * bracket_tol_effective, 0.05 * max(start_x_hint, value_fixed, 1e-6),
        )
        print(
            f"[search] iteration {iteration + 1}: restarting near hinted "
            f"M={start_x_hint:.4f} (step={initial_step_hint:.4f}) instead of "
            f"re-walking from scratch"
        )

        search_result = run_search(
            base_seed + iteration * 20_000_000, extra_points=extra_points,
            iteration=iteration + 1,
            start_x_hint=start_x_hint, initial_step_hint=initial_step_hint,
        )

    csv_logger.log(
        "final", final_M=result["M"], final_beta=result["beta"],
        certified=result["certified"],
    )
    csv_logger.close()
    return result


if __name__ == "__main__":
    # Tiny smoke test - NOT a real search (far too little training/eval
    # data to trust the resulting M). Confirms the plumbing runs end to
    # end. Use find_indifference_reward() directly with real hyperparameters
    # for actual use - and note the smoke test overrides max_seeds_per_point/
    # max_points/max_runs down to tiny numbers purely so it finishes fast;
    # production use should leave those at their generous defaults.
    result = find_indifference_reward(
        k_fixed=3,
        k_variable=13,
        verify=True,
    )
    print("M =", result["M"], " beta =", result["beta"])
    print("certified =", result["certified"])
