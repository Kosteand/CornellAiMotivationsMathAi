import math
import time

import numpy as np

from run_training import run_training


def _train_and_eval_point(x, target_hits, fixed_kwargs):
    """
    Runs one from-scratch training + eval call at left_reward=x, target_hits
    hits required. Returns (left, right, miss, stalled).

    stalled is inferred from run_training's own return: since miss_count
    tallies every episode that didn't reach a target, left+right+miss is
    always the true number of episodes run, and left+right < target_hits
    means run_eval_until_target_hits aborted early via its stall check.
    """
    kwargs = dict(fixed_kwargs)
    kwargs['left_reward'] = x
    kwargs['target_hits'] = target_hits
    kwargs.setdefault('save_weights', False)

    left, right, miss = run_training(**kwargs)
    stalled = (left + right) < target_hits
    return left, right, miss, stalled


def _fit_sigmoid_mle(xs, left_counts, total_counts):
    """
    Fits P(left) ~ sigmoid(4*beta*(x - M)) via maximum likelihood on
    pooled Binomial hit counts, rather than least-squares on per-point
    means. This is the fix for a real problem with the mean+variance
    approach: estimating a point's own variance from only 3 seeds is
    itself extremely noisy, so weighting by 1/variance can swing a
    point's influence by 1000x+ based on sampling luck in that variance
    estimate alone -- silently discarding genuinely informative points
    just because their 3 seeds happened to disagree.

    Here, each point's TOTAL (left_count, right_count) across however
    many seeds it has is used directly in a Binomial log-likelihood, and
    the (M, beta) that maximizes total log-likelihood across all points
    is returned. A point with disagreeing seeds contributes its pooled
    counts at full strength (proportional to its total trial count, same
    as any other point) rather than being crushed by an unreliable
    variance estimate -- the disagreement itself becomes evidence that
    P(left) is genuinely intermediate there, not a reason to distrust it.

    Note: this treats every seed at a given x as if it shared exactly the
    same P(left) (no seed-level random effect), which is technically a
    simplification -- it means the resulting log-likelihood shouldn't be
    read as a rigorous fit quality/confidence measure given the real
    between-seed variance this project has observed. It does NOT bias the
    point estimate of (M, beta) itself, which is the only thing this
    function is used for; certify_candidate is what carries the actual
    statistical guarantee, independent of anything from this search step.

    Deliberately avoids a scipy dependency, consistent with _fit_sigmoid.
    """
    xs = np.asarray(xs, dtype=np.float64)
    left_counts = np.asarray(left_counts, dtype=np.float64)
    total_counts = np.asarray(total_counts, dtype=np.float64)
    right_counts = total_counts - left_counts

    x_lo, x_hi = xs.min(), xs.max()
    pad = max((x_hi - x_lo) * 0.25, 1e-6)
    m_grid = np.linspace(x_lo - pad, x_hi + pad, 800)
    beta_grid = np.logspace(-6, 2, 800)

    best_ll = -np.inf
    best_m, best_beta = float(np.median(xs)), 1.0
    eps = 1e-12

    for m in m_grid:
        z = 4 * beta_grid[:, None] * (xs[None, :] - m)
        z = np.clip(z, -700, 700)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, eps, 1 - eps)
        ll = np.sum(left_counts[None, :] * np.log(p) + right_counts[None, :] * np.log(1 - p), axis=1)
        idx = np.argmax(ll)
        if ll[idx] > best_ll:
            best_ll = ll[idx]
            best_m = m
            best_beta = beta_grid[idx]

    return best_m, best_beta, best_ll


def _counts_from_seed_ratios(seed_ratios, hits_per_point):
    """
    Reconstructs exact (left_count, total_count) from a list of per-seed
    Y_hat ratios plus the known hits_per_point (target_hits) used to
    produce them. Exact as long as hits_per_point matches what was
    actually used -- Y_hat = left/hits_per_point, so left = round(Y_hat *
    hits_per_point) recovers the original integer exactly (mod floating
    point rounding when the CSV stored a rounded ratio, which is
    negligible here since values were saved to 4 decimal places).
    """
    left = sum(round(v * hits_per_point) for v in seed_ratios)
    total = len(seed_ratios) * hits_per_point
    return left, total



def _fit_sigmoid(xs, ys, weights=None):
    """
    Fits Y ~ sigmoid(4*beta*(x - M)) to (x, y) pairs via a grid search over
    (M, beta), minimizing WEIGHTED sum of squared error. Assumes Y is
    monotonically INCREASING in x (higher left_reward -> more likely to go
    left), so beta is searched over positive values only -- flip the sign
    here if your convention runs the other way.

    weights, if given, should be one non-negative weight per (x, y) pair,
    same length as xs/ys. Points with a larger weight pull the fit toward
    themselves harder. If None, every point is weighted equally (plain
    unweighted least squares, the original behavior).

    See _compute_point_weights for how to derive weights from per-seed
    data (inverse-variance-of-the-mean) -- that's the intended source of
    weights here, since a point's seeds, not its individual eval episodes,
    are the right unit of independent evidence for this fit.

    Deliberately avoids a scipy dependency (curve_fit) since this is a
    coarse, few-point fit for search purposes only; a grid search is more
    than sufficient and keeps this file dependency-free beyond numpy.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    if weights is None:
        weights = np.ones_like(xs)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    x_lo, x_hi = xs.min(), xs.max()
    pad = max((x_hi - x_lo) * 0.25, 1e-6)
    m_grid = np.linspace(x_lo - pad, x_hi + pad, 500)
    beta_grid = np.logspace(-6, 3, 500)  # broad range; true scale of x is unknown a priori

    best_sse = np.inf
    best_m, best_beta = float(np.median(xs)), 1.0

    for m in m_grid:
        z = 4 * beta_grid[:, None] * (xs[None, :] - m)
        z = np.clip(z, -700, 700)  # avoid overflow in exp
        preds = 1.0 / (1.0 + np.exp(-z))
        sse = np.sum(weights[None, :] * (preds - ys[None, :]) ** 2, axis=1)
        idx = np.argmin(sse)
        if sse[idx] < best_sse:
            best_sse = sse[idx]
            best_m = m
            best_beta = beta_grid[idx]

    return best_m, best_beta, best_sse


def _compute_point_weights(y_hats_by_seed, min_variance=1e-4):
    """
    Computes one inverse-variance-of-the-mean weight per point from that
    point's individual seed Y_hats.

    The unit of independent evidence here is the SEED, not the individual
    eval episode -- episodes within one seed are correlated (they all come
    from the same trained policy), so the real uncertainty this fit should
    respect is seed-to-seed disagreement, not raw episode count. A point
    whose seeds agree tightly should pull the fit hard; a point whose
    seeds are scattered across the whole range (bimodal commitment) is
    barely informative about the curve's location at that x and should
    only weakly influence the fit, even if it happens to have just as many
    total episodes as a tight point.

    weight = n_seeds / variance_across_seeds

    min_variance is a floor applied to the variance before dividing, so a
    point whose seeds happen to be identical (variance=0, e.g. all three
    seeds landed exactly on 0.000) doesn't get an infinite weight -- it's
    still capped at a large-but-finite value determined by min_variance.
    A point with only 1 valid seed has no estimate of its own variance;
    it falls back to weight = 1 (same as an unweighted point).
    """
    weights = []
    for seed_vals in y_hats_by_seed:
        n = len(seed_vals)
        if n <= 1:
            weights.append(1.0)
            continue
        var = float(np.var(seed_vals, ddof=1))
        var = max(var, min_variance)
        weights.append(n / var)
    return np.array(weights, dtype=np.float64)


def find_candidate_x_star(x_bounds, n_search_points=12, hits_per_point=100,
                            n_seeds_per_point=3, fixed_kwargs=None, verbose=True):
    """
    SEARCH phase -- heuristic, carries NO statistical guarantee on its own.

    Trains from scratch at several left_reward values spanning x_bounds.
    Each point runs n_seeds_per_point INDEPENDENT from-scratch trainings
    (default 3), each evaluated to hits_per_point hits (default 100); the
    point's Y_hat used for the sigmoid fit is the mean across those seeds.

    This addresses the two separate terms in Var[Y_hat] = Var_s[q(x,s)] +
    E_s[q(1-q)]/K: hits_per_point (K) only shrinks the within-agent term
    (how precisely you measure one trained agent's own left-probability),
    while n_seeds_per_point addresses the between-seed term (how much the
    trained POLICY ITSELF varies run to run at the same x -- e.g. if
    policies tend to fully commit left or right rather than landing near
    0.5, a single seed per point can make an otherwise-smooth sigmoid look
    like noisy jumps between 0 and 1). hits_per_point stays cheap to raise
    since eval is just forward passes; n_seeds_per_point is NOT cheap,
    since each seed is a full additional training run -- total cost here
    is n_search_points * n_seeds_per_point from-scratch trainings.

    Fits P(left) ~ sigmoid(4*beta*(x - M)) via MAXIMUM LIKELIHOOD on each
    point's pooled Binomial hit counts (summed left/right across its
    seeds), not least-squares on per-point means. This matters: estimating
    a point's own variance from only 3 seeds is itself extremely noisy,
    so a variance-based weight can swing a point's influence by 1000x+
    just from sampling luck in that variance estimate -- silently
    discarding genuinely informative points whose seeds happened to
    disagree. Under the MLE approach, a point's influence scales with its
    total trial count (same for every point here, since every point runs
    the same n_seeds_per_point x hits_per_point), regardless of whether
    its seeds agreed with each other -- disagreement becomes evidence
    that P(left) is genuinely intermediate there, not a reason to
    discount it. See _fit_sigmoid_mle for the full explanation and its
    one caveat (no seed-level random effect, so its log-likelihood isn't
    a rigorous confidence measure -- that's certify_candidate's job).
    Returns M as the candidate x*.

    IMPORTANT: none of these runs may be reused as certification evidence
    for the final CI (see certify_candidate) -- they're for navigation
    only, exactly as laid out in the "search vs certify" split.

    fixed_kwargs should contain every other run_training argument you want
    held constant across all points (right_reward, nUpdates, and critically
    max_steps == min_steps so step_decay never kicks in -- the estimand
    must be pinned to a fixed max_steps, not the training-time annealed
    one). Do not include left_reward, target_hits, or save_weights in it;
    those are set automatically per-call.
    """
    if fixed_kwargs is None:
        fixed_kwargs = {}

    x_lo, x_hi = x_bounds
    xs = np.linspace(x_lo, x_hi, n_search_points)
    y_hats = []
    y_hats_by_seed = []  # per valid point: list of that point's individual seed Y_hats
    left_counts = []     # per valid point: summed left hits across its seeds
    total_counts = []    # per valid point: summed total hits across its seeds

    total_runs = n_search_points * n_seeds_per_point
    run_counter = 0
    start_time = time.time()

    if verbose:
        print(f"[search] starting: {n_search_points} points x {n_seeds_per_point} "
              f"seeds = {total_runs} from-scratch runs total\n")

    for point_idx, x in enumerate(xs, start=1):
        seed_y_hats = []
        point_left = 0
        point_total = 0
        for seed_idx in range(n_seeds_per_point):
            run_counter += 1
            elapsed = time.time() - start_time
            if verbose:
                print(f"[search] point {point_idx}/{n_search_points} "
                      f"(x={x:.4g}), seed {seed_idx + 1}/{n_seeds_per_point} "
                      f"-- run {run_counter}/{total_runs} overall, "
                      f"elapsed {elapsed / 60:.1f} min")

            left, right, miss, stalled = _train_and_eval_point(x, hits_per_point, fixed_kwargs)
            if stalled or (left + right) == 0:
                if verbose:
                    print(f"[search] x={x:.4g} seed {seed_idx + 1}/{n_seeds_per_point}: "
                          f"stalled / no hits -- excluding this seed")
                continue
            y = left / (left + right)
            seed_y_hats.append(y)
            point_left += left
            point_total += (left + right)
            if verbose:
                print(f"[search] x={x:.4g} seed {seed_idx + 1}/{n_seeds_per_point}: "
                      f"Y_hat={y:.3f} ({left} left / {right} right / {miss} miss)")

        if len(seed_y_hats) == 0:
            if verbose:
                print(f"[search] x={x:.4g}: all seeds stalled / no hits -- excluding this point")
            y_hats.append(np.nan)
            continue

        point_mean = float(np.mean(seed_y_hats))
        y_hats.append(point_mean)
        y_hats_by_seed.append(seed_y_hats)
        left_counts.append(point_left)
        total_counts.append(point_total)
        if verbose:
            spread = f", per-seed spread={['%.3f' % v for v in seed_y_hats]}" if len(seed_y_hats) > 1 else ""
            print(f"[search] point {point_idx}/{n_search_points} done -- "
                  f"x={x:.4g}: mean Y_hat={point_mean:.3f} "
                  f"over {len(seed_y_hats)} seed(s){spread}\n")

    y_hats = np.array(y_hats)
    valid = ~np.isnan(y_hats)
    xs_valid, ys_valid = xs[valid], y_hats[valid]

    if len(xs_valid) < 3:
        raise RuntimeError(
            "Too few valid search points to fit a sigmoid -- widen x_bounds, "
            "increase hits_per_point/n_seeds_per_point, or check whether "
            "training is failing across this whole range."
        )

    if verbose:
        print(f"[search] pooled counts per point: "
              f"{[(int(l), int(t)) for l, t in zip(left_counts, total_counts)]}")

    M, beta, log_likelihood = _fit_sigmoid_mle(xs_valid, left_counts, total_counts)

    total_elapsed = time.time() - start_time
    if verbose:
        print(f"\n[search] === DONE === total elapsed {total_elapsed / 60:.1f} min "
              f"({run_counter} runs)")
        print(f"[search] fitted M (candidate x*) = {M:.4g}, beta = {beta:.4g}, "
              f"log-likelihood = {log_likelihood:.4g}")

    return {
        "x_star": M,
        "beta": beta,
        "log_likelihood": log_likelihood,
        "xs": xs_valid,
        "y_hats": ys_valid,
        "y_hats_by_seed": y_hats_by_seed,  # diagnostic: raw per-seed values, aligned with xs_valid
    }


def _half_width(n, s2, alpha, m=40.0):
    """Robbins normal-mixture confidence sequence -- valid at ALL n simultaneously."""
    return math.sqrt(2 * s2 * (n + m) / (n * n) * math.log(math.sqrt((n + m) / m) / alpha))


def certify_candidate(x_star, target_hits=100, lo=0.40, hi=0.60, alpha=0.05,
                        max_runs=300, min_runs=1,
                        fixed_kwargs=None, verbose=True):
    """
    CERTIFY phase -- fresh, independent, from-scratch runs at the single
    fixed point x_star, read out by a rule fixed in advance, using an
    anytime-valid confidence sequence for f_soft(x_star) = E[Y_hat], the
    estimand chosen at the very start of this project. This is where the
    actual statistical guarantee lives; nothing from the search phase
    feeds into it.

    Each run trains from scratch at x_star and evaluates via
    run_eval_until_target_hits (through run_training), giving
    Y_hat = left / (left + right) in [0, 1] (misses excluded from both
    numerator and denominator). Y_hat is used directly, as a continuous
    quantity -- NOT discretized into a binary "committed left/right"
    label. An earlier version of this function did discretize (via a
    fixed 0.2/0.8 threshold), which (a) silently switched the estimand
    from f_soft to a different quantity, f_hard = P(hard commitment), and
    (b) was demonstrably unsafe against this project's own data -- the
    x=54.55 batch had a seed at Y_hat=0.7, which falls inside the
    "ambiguous" zone that version would have raised an error on.

    THE VARIANCE BOUND -- and why this is provably valid, not just a
    reasonable-looking formula:
    Since each Y_hat is bounded in [0, 1], Hoeffding's lemma guarantees
    it is sub-Gaussian with variance proxy <= 1/4, for ANY distribution
    supported on [0, 1] -- no Bernoulli/hard-commitment assumption, no
    normality assumption, nothing estimated from the data. This is a
    FIXED, known-in-advance constant. Plugging that fixed constant (not
    an estimate) into the Robbins normal-mixture confidence sequence
    below preserves the martingale argument the sequence relies on, so
    the resulting (1-alpha) anytime-valid coverage guarantee is exact --
    see the accompanying proof for the full argument. An earlier version
    of this function instead estimated the variance from the sample
    (with ad hoc shrinkage bolted on after a failure this caused) --
    plugging an ESTIMATED variance into this formula breaks the proof
    outright, regardless of how the estimate is computed or regularized.
    Do not reintroduce that.

    Stops and returns accept=True as soon as the CI for f_soft(x_star) is
    fully inside (lo, hi). Also stops early with accept=False if the CI is
    fully OUTSIDE (lo, hi) -- this is a valid use of the same confidence
    sequence (it's still "the CI excludes/includes a region" at the same
    n), it just saves burning through max_runs when x_star is clearly not
    the midpoint.

    min_runs defaults to 1 and is NOT required for the guarantee to hold
    -- the fixed conservative variance already makes the interval far too
    wide to trigger any decision at small n (e.g. at n=2, the half-width
    is already >1, wider than the entire [0,1] range, so neither accept
    nor reject can fire regardless of the observed data). It exists purely
    as an optional efficiency knob if you want to skip printing/checking
    before some minimum n, not as a correctness safeguard.

    A stalled run (agent essentially never reaches any target at this x)
    doesn't yield a valid Y_hat and is excluded from the sample entirely
    rather than treated as evidence one way or the other. If 5 runs in a
    row stall, certification aborts (accept=None) rather than continuing
    indefinitely against what looks like a broken configuration at this x.
    NOTE: this means the guarantee below is technically conditional on
    "the training run doesn't stall" -- if stalling is itself correlated
    with which target the policy prefers, this is a real (if likely
    small) caveat on the guarantee's scope. It is not addressed by the
    proof below, which assumes the retained Y_hat's are themselves a
    clean i.i.d. sequence.

    fixed_kwargs: see find_candidate_x_star -- should be the SAME fixed
    settings used during search (right_reward, nUpdates, max_steps ==
    min_steps, etc.), so certification is evaluating the actual estimand
    the search was targeting.
    """
    if fixed_kwargs is None:
        fixed_kwargs = {}

    ys = []
    consecutive_stalls = 0
    mean, h = None, None
    n = 0
    attempt = 0
    start_time = time.time()

    if verbose:
        print(f"[certify] starting: x_star={x_star:.4g}, target interval=({lo}, {hi}), "
              f"alpha={alpha}, max_runs={max_runs}\n")

    while n < max_runs:
        attempt += 1
        elapsed = time.time() - start_time
        if verbose:
            print(f"[certify] attempt {attempt} (valid n so far={n}, "
                  f"max_runs={max_runs}), elapsed {elapsed / 60:.1f} min")

        left, right, miss, stalled = _train_and_eval_point(x_star, target_hits, fixed_kwargs)

        if stalled:
            consecutive_stalls += 1
            if verbose:
                print(f"[certify] run stalled ({left} left / {right} right / "
                      f"{miss} miss) -- excluded, not counted toward n")
            if consecutive_stalls >= 5:
                total_elapsed = time.time() - start_time
                if verbose:
                    print(f"\n[certify] === ABORTED === after {attempt} attempts, "
                          f"{total_elapsed / 60:.1f} min: 5 consecutive stalled runs")
                return {
                    "accept": None, "x": x_star, "runs": n, "ys": ys,
                    "reason": "aborted: 5 consecutive stalled runs -- agent "
                              "appears unable to reliably reach either "
                              "target at this x",
                }
            continue

        consecutive_stalls = 0
        n += 1
        y = left / (left + right)
        ys.append(y)
        if verbose:
            print(f"[certify] run {n}: Y_hat={y:.4f}")

        mean = sum(ys) / n
        if n >= 2:
            # FIXED, known-in-advance constant -- see docstring. NOT
            # estimated from ys. This is the entire fix.
            s2 = 0.25
            h = _half_width(n, s2, alpha)

            if verbose:
                print(f"[certify] n={n}: mean Y_hat={mean:.4f}, "
                      f"CI=({mean - h:.4f}, {mean + h:.4f})")

            ready_to_decide = n >= min_runs

            if ready_to_decide and mean - h > lo and mean + h < hi:
                total_elapsed = time.time() - start_time
                if verbose:
                    print(f"\n[certify] === ACCEPTED === x*={x_star:.4g} after "
                          f"{n} runs ({attempt} attempts), "
                          f"{total_elapsed / 60:.1f} min. "
                          f"f_soft={mean:.4f}, CI=({mean - h:.4f}, {mean + h:.4f})")
                return {"accept": True, "x": x_star, "f_soft": mean,
                        "ci": (mean - h, mean + h), "runs": n, "ys": ys}

            if ready_to_decide and (mean + h < lo or mean - h > hi):
                total_elapsed = time.time() - start_time
                if verbose:
                    print(f"\n[certify] === REJECTED === x*={x_star:.4g} after "
                          f"{n} runs ({attempt} attempts), "
                          f"{total_elapsed / 60:.1f} min. "
                          f"f_soft={mean:.4f}, CI=({mean - h:.4f}, {mean + h:.4f}) "
                          f"-- confidently outside ({lo}, {hi})")
                return {"accept": False, "x": x_star, "f_soft": mean,
                        "ci": (mean - h, mean + h), "runs": n, "ys": ys,
                        "reason": "CI confidently outside target interval"}

    total_elapsed = time.time() - start_time
    if verbose:
        print(f"\n[certify] === INCONCLUSIVE === x*={x_star:.4g} after "
              f"{n} runs ({attempt} attempts), {total_elapsed / 60:.1f} min. "
              f"max_runs={max_runs} reached without a decisive CI.")
    return {"accept": None, "x": x_star, "f_soft": mean,
            "ci": (mean - h, mean + h) if h is not None else None,
            "runs": n, "ys": ys,
            "reason": f"inconclusive after max_runs={max_runs}"}