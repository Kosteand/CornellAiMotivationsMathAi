"""
Plot p(x) = P(went for variable target) vs log(x) for each test's data
(from run_p_curve_experiments.py or, better, sample_p_curve_adaptive.py),
with a 95% confidence band, and print an interpretation.

Confidence band: PLAIN LINEAR INTERPOLATION of each sampled point's own
CI (the wider of a pooled Wilson interval and a seed-to-seed interval,
the latter shrunk toward a regional estimate borrowed from each point's
immediate neighbors when it has few of its own seeds - see
sample_p_curve_adaptive.py's module docstring, "Two mitigations for a
real observed overconfidence problem," for how that per-point CI is
built), NOT forced to be monotonically increasing.

An earlier version of this file chained points' CIs together using the
fact that p(x) is believed to be monotonic (p is strictly increasing in
reward, by this whole project's design) - "any earlier point's own lo is
also a valid lower bound for every x past it" - which forced env_lo and
env_hi to be non-decreasing everywhere. In practice, on real data, this
caused two problems: (1) it can only ever represent a STEP-LIKE band (the
bound between two points is literally constant across the whole gap,
by construction), which is a poor match for data that's plausibly
smooth/logistic rather than a genuine step function, and (2) because each
point's CI is independently computed at its own nominal 95% (not a
jointly-calibrated simultaneous band), propagating one point's bound onto
another can manufacture NEGATIVE WIDTH when two nearby, independently
noisy points disagree - a real, visible artifact once this pipeline had
enough closely-spaced points to run into it.

The fix here: drop the monotonicity-forcing entirely. Each sampled
point's own lo and hi are linearly interpolated to neighboring points'
own lo/hi (via np.interp) with NO cross-point propagation, so the band at
any x is a straight-line blend of its two nearest samples' own,
independently-valid intervals. Outside the sampled range, np.interp
clamps to the nearest endpoint's own value (flat extrapolation - the
same "we haven't measured out there, so don't invent more confidence"
behavior as before). This can't produce negative width: since lo_i <=
hi_i at every sampled point by construction, and lo/hi are each
interpolated separately, any point on the straight line between two
valid (lo, hi) pairs is itself a convex combination of two valid
"lo <= hi" facts, hence still lo <= hi everywhere in between. It's also
no longer artificially staircase-shaped - the band itself can be smooth,
matching a smooth true p(x) - and it can show real local structure (e.g.
a tighter interval flanked by looser ones) instead of ironing it away.

The tradeoff, made deliberately: this band is no longer guaranteed
non-decreasing. If two adjacent points disagree enough (a real sampling
fluke, or - more informatively - a genuine local irregularity worth
looking into), the plotted band can dip. That's treated as more honest
than silently forcing monotonicity onto data that doesn't yet support it.

A GP is still fit, but ONLY for the mean curve - i.e. to visualize the
overall SHAPE (is it logistic? linear? something else?), not for its
uncertainty. Its predicted mean is passed through a running-max so the
GP's own mean curve doesn't visibly violate the known monotonicity
(unlike the band above, this one number IS still forced monotonic - see
fit_gp_mean's docstring) - that step is unrelated to, and unchanged by,
the band no longer being forced.

The GP fit and the confidence band are two INDEPENDENT computations
still - the GP only ever sees each point's pooled mean and its own
per-point noise (alpha). The GP mean is clipped into the band
(clip_mean_to_envelope) purely so it never visibly contradicts the real,
data-derived interval at any x - a plain elementwise clip now, since the
band itself is already guaranteed lo <= hi everywhere (see above), no
extra monotonic tightening needed for that guarantee to hold.

log(x) is plotted directly as the x-axis's own numeric values (a normal
linear matplotlib axis over log(x)), NOT x plotted on a log-scaled axis -
those look superficially similar but are not the same thing: tick
spacing / the visual "distance" between points is linear in log(x)
either way, but a log-scaled x-axis still stores/reports the underlying
x values (so e.g. an axis label reading "0.5, 1, 2, 4" - still x), while
this plots the actual log(x) numbers (e.g. "-0.7, 0, 0.7, 1.4").

Shape diagnostic: fits SEVERAL candidate monotone, [0,1]-bounded curves
to log(x) - logistic, piecewise linear, probit, complementary log-log,
and log-log/Gompertz (see SHAPE_MODELS) - and reports which fits best.
EVERY candidate is fit the SAME way: maximizing the exact binomial log-
likelihood (including the log-C(N,k) combinatorial term) on each point's
own pooled (successes, failures) counts directly, via scipy.optimize -
not a least-squares fit on point means. Because every model uses the
identical likelihood, their AIC/BIC ARE now directly, validly comparable
against each other (this replaces an earlier version where the linear
alternative was fit via Gaussian-likelihood WLS instead, making its
AIC/BIC not comparable to the logistic fit's Binomial-likelihood one -
see fit_shape_diagnostic()'s docstring for the full history). The best
model is simply whichever has the lowest AIC; a small AIC gap (<2)
between the top two is called out as "ambiguous" rather than declaring a
confident winner. AIC/BIC/log-likelihood/params/standard errors for every
candidate are printed and included in summary.csv.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from scipy.stats import norm
from statsmodels.tools.numdiff import approx_hess
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from sample_p_curve_adaptive import point_ci

DATA_DIR = "p_curve_data"
PLOT_DIR = "p_curve_plots"
N_GRID = 400
MAX_INTERVAL_ROWS_TOTAL = 150  # hard cap on how many "confidence interval over an x
                               # interval" rows get printed to the terminal ACROSS THE
                               # WHOLE run of main() (all tests combined) - see
                               # print_interval_table()'s docstring for why this exists
                               # and what happens once it's hit.
PLOT_EXTRAPOLATION_FRAC = 0.12  # draw the curve/band this fraction of the sampled
                                # log(x) span PAST the lowest/highest sampled point on
                                # each side, purely so the plot visibly finishes leveling
                                # off into its flat tails instead of stopping abruptly
                                # right at the last data point. Safe to extend, not just
                                # cosmetic: interpolate_ci_band's own np.interp call
                                # already clamps to the nearest endpoint's own value
                                # outside the sampled range - so widening the grid
                                # doesn't invent any bound that isn't already implied by
                                # the real data. The GP mean curve genuinely IS
                                # extrapolating past its fitted range here, but it's
                                # clipped into that same real band (clip_mean_to_envelope),
                                # so it can't display anything the band doesn't support.


def per_point_summary(df):
    """One row per distinct x: pooled mean, this point's own CI (the
    same wider-of-Wilson-and-seed-SEM logic sample_p_curve_adaptive.py
    uses, including its neighbor-variance-shrinkage for the seed-SEM
    side - see that module's docstring), n_seeds, n_episodes_total, and
    total_success (pooled successes across every episode at this x - the
    raw binomial count fit_shape_diagnostic() needs for its statsmodels
    GLM fit, so it doesn't have to re-group the raw df itself). Sorted by
    x."""
    groups = [(x, sub.to_dict("records")) for x, sub in df.groupby("x")]
    groups.sort(key=lambda g: g[0])
    rows = []
    for i, (x, recs) in enumerate(groups):
        neighbor_y_hats = []
        if i > 0:
            neighbor_y_hats.extend(r["y_hat"] for r in groups[i - 1][1])
        if i < len(groups) - 1:
            neighbor_y_hats.extend(r["y_hat"] for r in groups[i + 1][1])
        lo, hi, n_seeds, n_total = point_ci(recs, neighbor_y_hats)
        mean_y = float(np.mean([r["y_hat"] for r in recs]))
        total_success = int(sum(r["chose_variable"] for r in recs))
        rows.append({
            "x": x, "log_x": np.log(x), "mean_y": mean_y,
            "lo": lo, "hi": hi, "n_seeds": n_seeds, "n_total": n_total,
            "total_success": total_success,
        })
    return pd.DataFrame(rows).sort_values("x").reset_index(drop=True)


def interpolate_ci_band(points_df, log_x_grid):
    """For each value in log_x_grid, linearly interpolate the sampled
    points' own lo and hi SEPARATELY (see module docstring for why this
    replaced the old monotonicity-chained envelope) - no cross-point
    propagation, so this band is NOT forced to be non-decreasing. Beyond
    the sampled range, np.interp clamps to the nearest endpoint's own
    value (flat extrapolation), matching the old behavior of not
    inventing extra confidence where nothing was measured.

    lo(x) <= hi(x) is still guaranteed at every grid point, forced or not:
    each sampled point has lo_i <= hi_i by construction (point_ci always
    returns a valid interval), and a straight line between two valid
    (lo, hi) pairs is a convex combination of two "lo <= hi" facts, which
    preserves the inequality everywhere along it - so no negative widths,
    without needing the old running-max/running-min machinery to get
    there."""
    xs = points_df["log_x"].to_numpy()
    los = points_df["lo"].to_numpy()
    his = points_df["hi"].to_numpy()
    order = np.argsort(xs)
    xs, los, his = xs[order], los[order], his[order]
    env_lo = np.interp(log_x_grid, xs, los)
    env_hi = np.interp(log_x_grid, xs, his)
    return env_lo, env_hi


def print_interval_table(test_name, points_df, row_budget):
    """Print each sampled point's own CI (the same per-point intervals
    that get linearly interpolated into the plotted band - see
    interpolate_ci_band) as an explicit table, one row per distinct
    sampled x-value. This is n_points rows for a test with n_points
    x-values - e.g. ~15-50 rows for a typical test at this pipeline's
    usual budgets, so potentially 100-300+ rows across all 7 TESTS in one
    run of main(), which is why `row_budget` exists: it's a mutable
    {"remaining": int} dict SHARED across every call within one run of
    main() (see MAX_INTERVAL_ROWS_TOTAL), so the total printed here is
    hard-capped regardless of how many tests/points there turn out to be.
    Once the shared budget is exhausted, remaining tests get a one-line
    note instead of their own table - never a silent, unbounded dump.

    (This used to print one row per GAP between points, using the
    monotonicity-chained bound valid across that whole gap - now that the
    band is just a linear interpolation between each point's own
    interval, with no chaining, the natural unit to report is each
    point's own measured interval, not a derived per-gap one.)"""
    xs = points_df["log_x"].to_numpy()
    x_raw = points_df["x"].to_numpy()
    los = points_df["lo"].to_numpy()
    his = points_df["hi"].to_numpy()
    order = np.argsort(xs)
    xs, x_raw, los, his = xs[order], x_raw[order], los[order], his[order]
    n_rows = len(xs)

    if row_budget["remaining"] <= 0:
        print(f"  [{test_name}] confidence-interval table SKIPPED - the shared "
              f"{MAX_INTERVAL_ROWS_TOTAL}-row print budget for this run was already used "
              f"up by earlier tests ({n_rows} rows would have been needed here).")
        return

    print(f"  [{test_name}] confidence interval (95%) at each sampled x "
          f"(band elsewhere = linear interpolation between these, not monotonic-forced):")
    n_printable = min(n_rows, row_budget["remaining"])
    for i in range(n_printable):
        print(f"    x = {x_raw[i]:<14.5g} (log(x) = {xs[i]:<10.4f}): p in "
              f"[{los[i]:.4f}, {his[i]:.4f}]  (width {his[i] - los[i]:.4f})")
    row_budget["remaining"] -= n_printable
    if n_printable < n_rows:
        print(f"    ... {n_rows - n_printable} more point(s) omitted - shared "
              f"{MAX_INTERVAL_ROWS_TOTAL}-row print budget for this run is exhausted.")


def clip_mean_to_envelope(mean, env_lo, env_hi):
    """Clip the GP mean curve into the plotted band, elementwise - purely
    so the mean (used for the printed crossing point/shape diagnostic,
    even though it's no longer drawn on the plot - see plot_and_interpret)
    never implies a value the real, data-derived interval doesn't support
    at that x. No monotonic tightening needed here: interpolate_ci_band
    already guarantees env_lo <= env_hi at every grid point by
    construction (see its docstring), so a plain np.clip is sufficient -
    unlike the old monotonicity-chained envelope, there's no "floor may
    exceed ceiling" edge case to guard against."""
    return np.clip(mean, env_lo, env_hi)


def fit_gp_mean(points_df):
    """GP fit purely for the mean-curve shape (see module docstring) -
    fit on the PER-POINT pooled average at each x (one observation per
    sampled x-value, NOT one per seed), with that point's own confidence
    interval supplied directly to the GP as its known noise variance
    (`alpha`), instead of asking the GP to infer noise from repeated raw
    observations.

    Why this changed (previous version fit on every individual seed's raw
    y_hat as if they were independent draws of the same underlying
    quantity): they aren't. Two seeds trained at the SAME x commonly land
    at different outcomes because of training-seed variance, not because
    they're independent samples of a shared distribution the GP should
    average over episode-by-episode - see the seed-to-seed variance
    discussion in sample_p_curve_adaptive.py's docstring and point_ci().
    Treating every seed's raw y_hat as its own i.i.d. GP observation lets
    an x-value with 100 seeds outweigh one with 4 seeds by sheer row
    count, and conflates within-seed episode noise with across-seed
    training noise as if they were the same source of error. The fix:
    collapse each x down to ONE point (its pooled/averaged y_hat) and let
    the GP's per-point `alpha` encode exactly how much to trust that
    average - using the SAME confidence interval (wider of pooled-Wilson
    and seed-to-seed SEM) already computed everywhere else in this
    pipeline via point_ci(), so "how much do we trust this point" is
    answered consistently everywhere, not re-derived a different way just
    for the GP.

    Both the target values and each point's CI are converted to LOGIT
    space before being handed to the GP (same reasoning as before: logit
    removes the [0, 1] boundary so the GP only ever fits an ordinary
    unconstrained curve, and the sigmoid transform back guarantees the
    plotted mean can't leave (0, 1)) - crucially, the CI is converted via
    logit too, not measured in probability space and then reused as if it
    applied in logit space. Those are different numbers: logit STRETCHES
    a given absolute interval width when the point is near 0 or 1 and
    compresses it near 0.5, so converting the interval's own endpoints
    (not just the point estimate) is what makes alpha reflect this
    point's actual noise level in the space the GP is actually fit in.

    A point's own interval [lo, hi] is a ~95% (+/-1.96 sigma) interval, so
    its logit-space half-width divided by 1.96 gives a logit-space sigma
    directly - squared, that's alpha for that point. A point pinned down
    by many consistent seeds has a tight interval -> tiny alpha -> the GP
    trusts it almost exactly; a point seen by only a couple of
    wildly-disagreeing seeds (or fewer than 2 seeds, where point_ci
    can't yet say more than "somewhere in [0, 1]") has a huge interval ->
    huge alpha -> the GP barely lets it move the curve at all - exactly
    the treatment each point's actual evidence deserves, and immune to
    the old "razor-tight point forces near-exact interpolation right next
    to a barely-sampled point with all its own noise ignored" failure
    mode, since alpha is now grounded in the same real, already-vetted CI
    logic used throughout the rest of this pipeline rather than raw
    per-seed scatter."""
    lo = points_df["lo"].to_numpy(dtype=float)
    hi = points_df["hi"].to_numpy(dtype=float)
    mean_y = points_df["mean_y"].to_numpy(dtype=float)
    n_total = points_df["n_total"].to_numpy(dtype=float)
    log_x = points_df["log_x"].to_numpy(dtype=float).reshape(-1, 1)

    # Floor away from the exact 0/1 boundary before taking any logit - a
    # proportion measured from n_total pooled episodes can't really be
    # resolved any finer than that anyway (same reasoning as before, just
    # applied per point using THAT point's own total episode count).
    eps = np.clip(1.0 / (2.0 * np.maximum(n_total, 1.0)), 1e-6, 0.5 - 1e-6)

    def logit(p):
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    logit_y = logit(mean_y)
    logit_lo = logit(lo)
    logit_hi = logit(hi)

    # lo/hi is a 95% interval, i.e. roughly the point estimate +/- 1.96
    # sigma in whatever space it's measured in - so its logit-space
    # half-width converts directly to a logit-space sigma this way.
    sigma_logit = (logit_hi - logit_lo) / (2.0 * 1.959963985)
    sigma_logit = np.clip(sigma_logit, 1e-3, None)  # numerical floor only - never rounds away real confidence
    alpha = sigma_logit ** 2

    # No WhiteKernel here: alpha already supplies each point's own known
    # noise variance directly, which is the whole point of this rewrite -
    # a global learned noise level would blur every point's very
    # different, already-known uncertainty back into one shared number.
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(3e-1, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                   n_restarts_optimizer=5, random_state=0)
    gp.fit(log_x, logit_y)

    # Extend the grid a bit past the sampled range on each side (see
    # PLOT_EXTRAPOLATION_FRAC's docstring) so the plotted curve/band
    # visibly finish leveling off instead of stopping abruptly at the
    # last data point.
    lo_x, hi_x = float(log_x.min()), float(log_x.max())
    span = hi_x - lo_x
    margin = PLOT_EXTRAPOLATION_FRAC * span if span > 0 else 0.5
    log_x_grid = np.linspace(lo_x - margin, hi_x + margin, N_GRID)
    logit_mean = gp.predict(log_x_grid.reshape(-1, 1))
    mean = 1.0 / (1.0 + np.exp(-logit_mean))  # back to (0, 1) - always, by construction
    # Enforce the known monotonicity on the DISPLAYED curve - p is
    # strictly increasing, so any dip is GP wiggle, not signal. Safe to
    # apply here (unlike on the raw/unbounded fit) since `mean` is
    # already guaranteed to be in (0, 1).
    mean = np.maximum.accumulate(mean)
    return log_x_grid, mean


def _logistic_fn(z, C, M):
    return expit(C * (z - M))


def _piecewise_linear_fn(z, x0_logx, slope):
    return np.clip(slope * (z - x0_logx), 0.0, 1.0)


def _probit_fn(z, C, M):
    return norm.cdf(C * (z - M))


def _cloglog_fn(z, C, M):
    # complementary log-log - ASYMMETRIC: rises slowly off y=0, then
    # approaches y=1 sharply. Useful if the transition looks front-loaded
    # (a long, flat-ish lead-in followed by a fast jump to 1).
    return -np.expm1(-np.exp(np.clip(C * (z - M), -50.0, 50.0)))


def _loglog_fn(z, C, M):
    # log-log (Gompertz) - the MIRROR of cloglog: rises sharply off y=0,
    # then approaches y=1 slowly. Useful if the transition looks
    # back-loaded (a fast initial jump followed by a long, flat-ish
    # settling into 1).
    return np.exp(-np.exp(np.clip(-C * (z - M), -50.0, 50.0)))


def _sigmoid_init(z, y, weights):
    """Shared initial guess/bounds for the three sigmoid-shaped models
    (logistic, probit, cloglog, loglog) - all parameterized the same way,
    (C, M) = (slope, midpoint-ish location in log(x))."""
    C0 = 4.0 / max(z.max() - z.min(), 1e-6)  # a starter slope that spans the sampled range
    M0 = float(np.average(z, weights=weights))
    margin = max(z.max() - z.min(), 1.0) * 2.0
    bounds = [(1e-6, 1000.0), (z.min() - margin, z.max() + margin)]
    return [C0, M0], bounds


def _piecewise_linear_init(z, y, weights):
    span = max(z.max() - z.min(), 1e-6)
    slope0 = 1.0 / span
    m0 = float(np.average(z, weights=weights))
    x0_0 = m0 - 0.5 / slope0  # start the ramp so it's centered on the weighted mean
    margin = span * 2.0
    bounds = [(z.min() - margin, z.max() + margin), (1e-6, 1000.0 / span)]
    return [x0_0, slope0], bounds


SHAPE_MODELS = [
    {"name": "logistic", "fn": _logistic_fn, "param_names": ["C", "M"], "init": _sigmoid_init,
     "describe": lambda p: f"slope C={p['C']:.4f}, midpoint log(x)={p['M']:.4f}"},
    #{"name": "piecewise_linear", "fn": _piecewise_linear_fn, "param_names": ["x0_logx", "slope"],
    # "init": _piecewise_linear_init,
    # "describe": lambda p: f"ramp starts log(x)={p['x0_logx']:.4f}, slope={p['slope']:.4f} "
    #                        f"(hits 1 at log(x)={p['x0_logx'] + 1.0 / p['slope']:.4f})"},
    {"name": "probit", "fn": _probit_fn, "param_names": ["C", "M"], "init": _sigmoid_init,
     "describe": lambda p: f"slope C={p['C']:.4f}, midpoint log(x)={p['M']:.4f}"},
    #{"name": "cloglog", "fn": _cloglog_fn, "param_names": ["C", "M"], "init": _sigmoid_init,
    # "describe": lambda p: f"slope C={p['C']:.4f}, location log(x)={p['M']:.4f} (asymmetric: slow-then-fast)"},
    #{"name": "loglog", "fn": _loglog_fn, "param_names": ["C", "M"], "init": _sigmoid_init,
    # "describe": lambda p: f"slope C={p['C']:.4f}, location log(x)={p['M']:.4f} (asymmetric: fast-then-slow)"},
]  # every candidate shape considered by fit_shape_diagnostic() - see its docstring.
   # Add a new one by giving it a model function (monotone, bounded [0,1]),
   # a param_names list, an init(z, y, weights) -> (x0_guess, bounds)
   # function, and a describe(params_dict) -> str for the printed summary.


def binomial_loglik(k, N, p):
    """The FULL binomial log-likelihood (including the log-C(N,k)
    combinatorial term, via scipy.special.gammaln) of observing `k`
    successes out of `N` trials at each point, given model probabilities
    `p` - not just the k*log(p) + (N-k)*log(1-p) part. Including the
    combinatorial term matters for getting an AIC/BIC that's on the
    standard, textbook binomial-likelihood scale (and thus directly
    comparable across every model in SHAPE_MODELS, all of which use this
    same function) - it doesn't affect which parameters maximize the
    likelihood (it doesn't depend on the model or its parameters at all),
    but it does affect the absolute log-likelihood/AIC/BIC values."""
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    return float(np.sum(gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1)
                         + k * np.log(p) + (N - k) * np.log(1.0 - p)))


def fit_binomial_mle(model_fn, x0_guess, bounds, z, k, N):
    """Fit `model_fn(z, *theta)` - any monotone curve bounded in [0, 1] -
    by maximizing the binomial_loglik() above via scipy.optimize.minimize
    (L-BFGS-B, so per-parameter bounds are respected directly - e.g.
    keeping a slope positive - without needing a reparameterization
    trick). Standard errors are approximated from the numerical Hessian
    of the negative log-likelihood at the optimum (statsmodels'
    approx_hess) - inverted to get the parameter covariance matrix, same
    idea as any MLE's asymptotic covariance, just computed by hand since
    these aren't statsmodels model objects. Returns
    (theta_hat, llf, std_errs_or_None, converged)."""
    def negloglik(theta):
        return -binomial_loglik(k, N, model_fn(z, *theta))

    res = minimize(negloglik, x0=np.asarray(x0_guess, dtype=float), bounds=bounds, method="L-BFGS-B")
    theta_hat = res.x
    llf = -float(res.fun)
    try:
        hess = approx_hess(theta_hat, negloglik)
        cov = np.linalg.inv(hess)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except Exception:
        se = None
    return theta_hat, llf, se, bool(res.success)


def fit_shape_diagnostic(points_df):
    """Fit every candidate shape in SHAPE_MODELS (logistic, piecewise
    linear, probit, cloglog, loglog - see that list, and each function's
    docstring, for what each shape looks like) to log(x), ALL via the
    exact same binomial maximum-likelihood procedure (fit_binomial_mle,
    on each point's own pooled successes/failures - not on its noisy
    mean). Because every model is fit under the identical likelihood,
    their AIC/BIC are directly, validly comparable this time (an earlier
    version fit the linear alternative via Gaussian-likelihood WLS
    instead, which meant its AIC/BIC lived on a different scale than the
    logistic fit's Binomial one - that mismatch is gone now that
    everything goes through fit_binomial_mle).

    AIC/BIC nobs convention: BIC's log(n) term uses n = the number of
    DISTINCT SAMPLED x-VALUES (i.e. len(points_df)) - matching what
    statsmodels' own GLM used for the logistic fit in the earlier version
    of this function (verified directly: statsmodels' GLM.bic_llf equals
    n_params*log(n_grouped_points) - 2*llf, NOT n_params*log(total
    episodes) - 2*llf) - so switching to a hand-rolled MLE here doesn't
    silently change what "BIC" means relative to what you've already seen
    from this file.

    Returns {"models": {name: {...}}, "best_model": name, "verdict": str}
    - each model's dict has params (by name), std_err (or None if the
    Hessian couldn't be inverted), log_likelihood, aic, bic, rmse
    (weighted, on the observed per-point MEANS - purely for an intuitive
    "how far off are the raw dots" number, NOT used to pick the winner),
    r2 (weighted "observed" R^2 - see below), pseudo_r2_mcfadden,
    n_params, and converged. "best_model" is whichever has the lowest
    AIC; "verdict" calls out an ambiguous top-two (AIC gap < 2, the usual
    rule-of-thumb threshold for "not clearly distinguishable") rather
    than declaring a falsely confident winner. Mostly-empty dict if there
    are too few points to fit (need >= 3 rows for a 2-parameter model).

    TWO DIFFERENT R^2-LIKE NUMBERS, both reported per model:

    - "r2": a weighted "observed" R^2, 1 - (weighted SS_resid /
      weighted SS_total) computed on the per-point MEANS (weights =
      each point's n_total, same weighting used everywhere else in this
      function) - intuitive (same 0..1-ish scale everyone's used to from
      OLS), but treats each point's mean as if it were a plain continuous
      observation with constant variance, which it isn't (a proportion's
      variance depends on n and on how close p is to 0 or 1) - use it for
      a quick gut-check, not as the rigorous fit statistic.
    - "pseudo_r2_mcfadden": 1 - llf_model/llf_null, where llf_null is the
      log-likelihood of the single constant-probability model (same
      pooled p for every point) - the standard binomial-likelihood analog
      of R^2, and consistent with the actual likelihood every model here
      is fit to maximize. This is the more statistically appropriate of
      the two; "r2" is the more intuitive one.
    """
    z = points_df["log_x"].to_numpy()
    y = points_df["mean_y"].to_numpy()
    n_total = points_df["n_total"].to_numpy().astype(float)
    total_success = points_df["total_success"].to_numpy().astype(float)

    result = {"models": {}, "best_model": None, "verdict": None}
    if len(points_df) < 3:
        return result

    n_points = len(points_df)  # BIC's "n" - see docstring above

    # Null model for McFadden's pseudo-R^2: one constant probability
    # (the overall pooled success rate) shared by every point - has to be
    # computed once, outside the per-model loop, since it doesn't depend
    # on the shape being tested.
    p_null = float(total_success.sum() / n_total.sum()) if n_total.sum() > 0 else 0.5
    llf_null = binomial_loglik(total_success, n_total, np.full_like(z, p_null))

    # Weighted "observed" R^2 also needs a shared baseline: the weighted
    # mean of the point MEANS (the "predict the average" null model on
    # the observed-proportion scale, analogous to OLS's SS_total).
    y_bar = float(np.average(y, weights=n_total))
    ss_tot = float(np.sum(n_total * (y - y_bar) ** 2))

    fits = {}
    for spec in SHAPE_MODELS:
        try:
            x0_guess, bounds = spec["init"](z, y, n_total)
            theta_hat, llf, se, converged = fit_binomial_mle(
                spec["fn"], x0_guess, bounds, z, total_success, n_total)
            fitted_p = spec["fn"](z, *theta_hat)
            ss_res = float(np.sum(n_total * (y - fitted_p) ** 2))
            rmse = float(np.sqrt(ss_res / n_total.sum()))
            r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else None
            pseudo_r2 = (1.0 - llf / llf_null) if llf_null != 0 else None
            n_params = len(theta_hat)
            params = {name: float(v) for name, v in zip(spec["param_names"], theta_hat)}
            fits[spec["name"]] = {
                "params": params,
                "std_err": ({name: float(s) for name, s in zip(spec["param_names"], se)}
                            if se is not None else None),
                "log_likelihood": llf, "aic": 2 * n_params - 2 * llf,
                "bic": n_params * np.log(n_points) - 2 * llf,
                "rmse": rmse, "r2": r2, "pseudo_r2_mcfadden": pseudo_r2,
                "n_params": n_params, "converged": converged,
                "describe": spec["describe"](params),
            }
        except Exception as e:
            fits[spec["name"]] = {"error": str(e)}
    result["models"] = fits

    valid = {name: f for name, f in fits.items() if "error" not in f}
    if valid:
        ranked = sorted(valid.items(), key=lambda kv: kv[1]["aic"])
        best_name, best = ranked[0]
        result["best_model"] = best_name
        if len(ranked) > 1:
            second_name, second = ranked[1]
            delta = second["aic"] - best["aic"]
            if delta > 2.0:
                result["verdict"] = f"{best_name} clearly best by AIC (ΔAIC={delta:.2f} over {second_name})"
            else:
                result["verdict"] = (f"{best_name} and {second_name} fit about equally well by AIC "
                                      f"(ΔAIC={delta:.2f}) - shape is ambiguous at this sample "
                                      f"density, or something else entirely (check the plot)")
        else:
            result["verdict"] = f"only {best_name} could be fit"
    return result


def find_crossing(log_x_grid, env_lo, env_hi, mean, level=0.5):
    """NOTE: since interpolate_ci_band() no longer forces env_lo/env_hi to
    be monotonic (see module docstring), either one could in principle
    cross `level` more than once if the underlying points are noisy
    enough locally. This picks the FIRST crossing (lowest x) found, same
    as before the band stopped being forced monotonic - a deliberate
    simplification, not a guarantee that it's the "right" one if the band
    genuinely wiggles across 0.5 more than once. `mean` itself is still
    forced monotonic (see fit_gp_mean), so x_star from `mean` always has
    at most one crossing regardless."""
    def _interp_crossing(arr):
        sign = np.sign(arr - level)
        flips = np.where(np.diff(sign) != 0)[0]
        if len(flips) == 0:
            return None
        i = flips[0]
        x0, x1 = log_x_grid[i], log_x_grid[i + 1]
        y0, y1 = arr[i], arr[i + 1]
        frac = (level - y0) / (y1 - y0) if y1 != y0 else 0.5
        return float(x0 + frac * (x1 - x0))  # returns log(x), not x

    x_star = _interp_crossing(mean)
    x_star_lo = _interp_crossing(env_hi)  # upper band crosses earlier
    x_star_hi = _interp_crossing(env_lo)  # lower band crosses later
    return x_star, x_star_lo, x_star_hi


def plot_and_interpret(test_name, df, out_dir=PLOT_DIR, interval_row_budget=None):
    if interval_row_budget is None:
        interval_row_budget = {"remaining": MAX_INTERVAL_ROWS_TOTAL}  # standalone-call default
    points_df = per_point_summary(df)
    log_x_grid, mean = fit_gp_mean(points_df)
    env_lo, env_hi = interpolate_ci_band(points_df, log_x_grid)
    mean = clip_mean_to_envelope(mean, env_lo, env_hi)
    max_gap = float(np.max(env_hi - env_lo))

    log_x_star, log_x_star_lo, log_x_star_hi = find_crossing(log_x_grid, env_lo, env_hi, mean)
    shape = fit_shape_diagnostic(points_df)

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Per earlier request, no per-seed scatter points or 0.5/crossing
    # reference lines are drawn - just the CI band. On top of it (per
    # this request), every fitted SHAPE_MODELS curve from
    # fit_shape_diagnostic() IS now drawn, each in its own color, evaluated
    # on the same log_x_grid as the CI band - a direct visual answer to
    # "does the winning-by-AIC model actually track the measured
    # uncertainty, and how do the candidates diverge from each other and
    # from the data outside the sampled range." The best-AIC model gets a
    # thicker line and its name starred in the legend; a model that failed
    # to fit is silently skipped (already reported as FAILED in the
    # printed summary).
    ax.fill_between(log_x_grid, env_lo, env_hi, color="crimson", alpha=0.35,
                     label="95% CI (interpolated per-point intervals)", zorder=2)

    model_curve_colors = {
        "logistic": "tab:blue", "piecewise_linear": "tab:orange",
        "probit": "tab:green", "cloglog": "tab:purple", "loglog": "tab:brown",
    }
    models = shape.get("models") or {}
    best_model_name = shape.get("best_model")

    # "How much time does each model spend outside the CI band" - computed
    # only over the ACTUALLY SAMPLED log(x) range (points_df's own min/max),
    # not the cosmetic extrapolation margin added purely so the plot looks
    # nicer past the last real point (PLOT_EXTRAPOLATION_FRAC) - the band
    # out there is just a flat clamp of the nearest real point, not a real
    # measurement, so "outside the CI" wouldn't mean much in that region.
    # This is a fraction of the sampled log(x) SPAN (the grid is evenly
    # spaced, so fraction-of-grid-points == fraction-of-span) where the
    # model's curve falls strictly outside [env_lo, env_hi] at that x - a
    # direct, quantitative answer to "does this model's curve wander
    # outside the measured uncertainty, and for how much of the range."
    lo_x_sampled = float(points_df["log_x"].min())
    hi_x_sampled = float(points_df["log_x"].max())
    in_sampled_range = (log_x_grid >= lo_x_sampled) & (log_x_grid <= hi_x_sampled)
    frac_outside_ci = {}

    for spec in SHAPE_MODELS:
        m = models.get(spec["name"])
        if not m or "error" in m:
            continue
        curve = spec["fn"](log_x_grid, *[m["params"][p] for p in spec["param_names"]])
        outside = (curve < env_lo) | (curve > env_hi)
        frac_outside_ci[spec["name"]] = (float(np.mean(outside[in_sampled_range]))
                                          if in_sampled_range.any() else None)
        is_best = spec["name"] == best_model_name
        label = f"{spec['name']}" + (" (best AIC)" if is_best else "")
        ax.plot(log_x_grid, curve, color=model_curve_colors.get(spec["name"]),
                 linewidth=2.5 if is_best else 1.3, alpha=1.0 if is_best else 0.8,
                 linestyle="-" if is_best else "--", label=label, zorder=4)

    ax.set_xlabel("log(x)  (x = variable target's reward value)")
    ax.set_ylabel("p = P(went for variable target)")
    ax.set_title(f"{test_name}\np vs log(x), {len(df)} seed observations across "
                 f"{len(points_df)} x-values")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{test_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\n=== {test_name} ===")
    print(f"  {len(df)} seed observations across {len(points_df)} distinct x-values "
          f"(x range: {points_df['x'].min():.5g} - {points_df['x'].max():.5g})")
    print(f"  worst confidence-band width anywhere in range: {max_gap:.4f} "
          f"({'meets' if max_gap <= 0.05 else 'does NOT meet'} the 0.05 target)")
    if log_x_star is not None:
        x_star = float(np.exp(log_x_star))
        if log_x_star_lo is not None and log_x_star_hi is not None:
            print(f"  indifference point: x* = {x_star:.4f} (log(x*)={log_x_star:.4f}), "
                  f"95% envelope CI x in [{np.exp(log_x_star_lo):.4f}, {np.exp(log_x_star_hi):.4f}]")
        else:
            print(f"  indifference point: x* = {x_star:.4f} (envelope CI doesn't bound it on both sides)")
    else:
        print("  WARNING: mean curve never crosses p=0.5 across the sampled range.")
    models = shape.get("models") or {}
    if models:
        print(f"  shape check (every model below fit by binomial MLE, so AIC/BIC ARE directly "
              f"comparable across all of them): {shape['verdict']}")
        for name in [m["name"] for m in SHAPE_MODELS]:  # fixed print order, not AIC-sorted
            m = models.get(name)
            if not m or "error" in m:
                if m and "error" in m:
                    print(f"    {name}: FIT FAILED ({m['error']})")
                continue
            star = " <- best AIC" if name == shape.get("best_model") else ""
            conv_note = "" if m["converged"] else " [optimizer did not report convergence]"
            r2_str = f"{m['r2']:.4f}" if m["r2"] is not None else "n/a"
            pr2_str = f"{m['pseudo_r2_mcfadden']:.4f}" if m["pseudo_r2_mcfadden"] is not None else "n/a"
            fo = frac_outside_ci.get(name)
            fo_str = f"{100.0 * fo:.1f}% of sampled range" if fo is not None else "n/a"
            print(f"    {name}: AIC={m['aic']:.2f}, BIC={m['bic']:.2f}, "
                  f"log-lik={m['log_likelihood']:.2f}, RMSE={m['rmse']:.4f}, "
                  f"R2={r2_str}, McFadden pseudo-R2={pr2_str}, "
                  f"outside 95% CI band: {fo_str} - "
                  f"{m['describe']}{star}{conv_note}")
    print_interval_table(test_name, points_df, interval_row_budget)
    print(f"  saved plot -> {out_path}")

    row = {
        "test": test_name, "n_obs": len(df), "n_x_values": len(points_df),
        "max_band_width": max_gap, "meets_target": max_gap <= 0.05,
        "x_star": float(np.exp(log_x_star)) if log_x_star is not None else None,
        "best_shape_model": shape.get("best_model"), "shape_verdict": shape.get("verdict"),
    }
    for name, m in models.items():
        if "error" in m:
            continue
        row[f"{name}_aic"] = m["aic"]
        row[f"{name}_bic"] = m["bic"]
        row[f"{name}_loglik"] = m["log_likelihood"]
        row[f"{name}_frac_outside_ci"] = frac_outside_ci.get(name)
        row[f"{name}_rmse"] = m["rmse"]
        row[f"{name}_r2"] = m["r2"]
        row[f"{name}_pseudo_r2"] = m["pseudo_r2_mcfadden"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--out-dir", default=PLOT_DIR)
    parser.add_argument("--tests", nargs="*", default=None)
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if args.tests:
        wanted = set(args.tests)
        csv_paths = [p for p in csv_paths if os.path.splitext(os.path.basename(p))[0] in wanted]
    if not csv_paths:
        raise SystemExit(f"No CSVs found in {args.data_dir!r} - run "
                          f"sample_p_curve_adaptive.py (or run_p_curve_experiments.py) first.")

    summaries = []
    interval_row_budget = {"remaining": MAX_INTERVAL_ROWS_TOTAL}  # shared across ALL
                                                                   # tests in this run -
                                                                   # see print_interval_table()
    for path in csv_paths:
        test_name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        summaries.append(plot_and_interpret(test_name, df, out_dir=args.out_dir,
                                             interval_row_budget=interval_row_budget))

    summary_df = pd.DataFrame(summaries)
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL TESTS")
    print("=" * 80)
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(summary_df)
    summary_path = os.path.join(args.out_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
