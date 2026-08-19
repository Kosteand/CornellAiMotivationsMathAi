"""Group classes: ComplexityGroupBase (the abstract base) and its three
concrete subclasses, AlternatingGroup, MarginGroup, HeatmapGroup.

THIS is the file that defines the group classes and the abstract class
they inherit from - a TOP-LEVEL file (not under Utilities/), since
Utilities/ is reserved for the gym environment itself
(Utilities/bandit_env.py: BanditEnv, CsvLoggingWrapper, the legacy Proxy
classes) - group objects are a separate concern from the environment
that consumes them. Utilities/bandit_env.py imports the group classes
from here (`from groups import ...`) and re-exports them, so any
existing `from Utilities.bandit_env import MarginGroup` (etc.)
elsewhere in the codebase keeps working unchanged - this file is simply
the source of truth.

Direction/mechanism recap: instead of choosing a discrete index and
ENCODING it (the legacy Proxy classes in bandit_env.py), each group here
samples a continuous secret x (or vector of scores) and DERIVES the
correct discrete option from it - the agent observes that raw secret
directly (a "noiseless proxy" for the true label), and the group's
constructor parameters set how hard that secret is to decode.

Every concrete class below shares a common object shape, via
ComplexityGroupBase (see its own docstring for the exact contract):
constructible with its difficulty-defining parameters, exposes
`sample(rng)` (draws one episode's observation + rewarded option - this
is what BanditEnv actually calls), and exposes named metric methods
that turn the group's parameters into the math-side complexity metrics
the rest of this project (group_data.py, run_pipeline.py) records per
group:

    average_value()            - E[average entry] of the observation
    l_infinity_norm()          - E[largest entry] (E[L_infty norm])
    l1_norm()                  - E[L1 norm]
    l2_norm()                  - E[L2 norm]
    rms()                      - E[root-mean-square]
    standard_deviation()       - E[standard deviation]
    proxy_dimension_size()     - dimensionality of the observation
    num_options()               - number of options the model selects
                                   between
    p_proxy_correct()          - P(a naive/untrained read of the
                                   observation points to the option
                                   that actually gets rewarded)
    entropy_effective_rank()   - Shannon-entropy effective rank of the
                                   observation's covariance matrix
    participation_ratio()      - participation ratio of the
                                   observation's covariance matrix

Each of these is a thin, named accessor over one of the two aggregate,
Monte-Carlo-estimated (or overridden/exact) computations,
`expected_magnitude()` (avg/max/l1/l2/rms/std/sum, still available
directly if you want more than one at once) and
`effective_dimensionality()` (entropy_effective_rank/participation_
ratio) - see ComplexityGroupBase's docstring below for the full
estimation/caching/override contract they share.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Current mechanism: frequency-based complexity groups.
#
# Direction is reversed from the legacy proxies in bandit_env.py: instead of
# choosing an index and encoding it, each group samples a continuous secret x
# and *derives* the correct discrete option from it. The agent observes x
# directly (the "noiseless proxy" - no obfuscation, no inverse to solve).
# Difficulty comes from how high-frequency the x -> label mapping is, set
# by k, not from any algebraic secrecy.
# ---------------------------------------------------------------------------


DEFAULT_MC_SAMPLES = 200_000
DEFAULT_MC_SEED = 0

# Global cache, keyed on a group's PARAMETERS (via `_cache_key`) rather than
# on any particular group INSTANCE - per direct instruction: calling e.g.
# group.expected_magnitude() twice must return the exact same number (no
# re-randomizing), and two SEPARATELY CONSTRUCTED groups with identical
# parameters should share one estimate rather than each redoing the same
# Monte Carlo work. Keyed on (cache_key, metric_name, samples, seed).
_METRIC_CACHE: dict = {}


def _cached_metric(cache_key, metric_name, samples, seed, compute_fn):
    key = (cache_key, metric_name, samples, seed)
    if key not in _METRIC_CACHE:
        _METRIC_CACHE[key] = compute_fn()
    return _METRIC_CACHE[key]


def clear_metric_cache() -> None:
    """Empty the global metric-estimate cache. Mainly for tests that want
    a clean slate between runs with different DEFAULT_MC_SAMPLES/seed."""
    _METRIC_CACHE.clear()


def _effective_dim_from_covariance(cov: np.ndarray) -> dict:
    """Shannon-entropy effective rank (``exp(H)`` where ``H`` is the
    Shannon entropy, in nats, of the covariance matrix's eigenvalues
    normalized to sum to 1 - the "effective rank" of Roy & Vetterli 2007)
    AND the participation ratio (``(sum(eigenvalues))**2 /
    sum(eigenvalues**2)``, the simpler "effective dimensionality" measure
    used in neuroscience/dynamical-systems contexts) - both computed since
    it wasn't clear which is preferable. Both equal exactly 1 for a rank-1
    covariance and exactly d for a covariance proportional to the d x d
    identity; they only diverge on spectra in between (entropy's log
    weighting penalizes small-but-nonzero eigenvalues more gently).

    Eigenvalues are clipped to >= 0 first - a real, symmetric empirical
    covariance matrix is PSD in theory, but floating point can return tiny
    negative eigenvalues (~1e-12), which would make log()/squaring
    nonsensical otherwise."""
    eigenvalues = np.clip(np.linalg.eigvalsh(np.atleast_2d(cov)), 0.0, None)
    total = eigenvalues.sum()
    if total <= 0:
        return {"entropy_effective_rank": 0.0, "participation_ratio": 0.0}
    probs = eigenvalues[eigenvalues > 0] / total
    entropy = float(-np.sum(probs * np.log(probs)))
    return {
        "entropy_effective_rank": float(np.exp(entropy)),
        "participation_ratio": float(total ** 2 / np.sum(eigenvalues ** 2)),
    }


def _estimate_expected_magnitude(group: "ComplexityGroupBase", samples: int, seed: int) -> dict:
    """Monte Carlo E[avg]/E[max]/E[sum]/E[l1]/E[l2]/E[rms]/E[std] of a
    group's observation, from `samples` independent draws of
    `group._sample_batch`. `max` is E[largest entry] (== E[L_infty] under
    this project's own convention that every group's entries of interest
    are non-negative in the regimes actually used)."""
    rng = np.random.default_rng(seed)
    x, _ = group._sample_batch(rng, samples)
    x = np.atleast_2d(x)
    d = x.shape[1]
    sq = x ** 2
    sum_sq = sq.sum(axis=1)
    return {
        "avg": float(x.mean(axis=1).mean()),
        "max": float(x.max(axis=1).mean()),
        "sum": float(x.sum(axis=1).mean()),
        "l1": float(np.abs(x).sum(axis=1).mean()),
        "l2": float(np.sqrt(sum_sq).mean()),
        "rms": float(np.sqrt(sum_sq / d).mean()),
        "std": float(x.std(axis=1).mean()),
    }


def _estimate_effective_dimensionality(group: "ComplexityGroupBase", samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x, _ = group._sample_batch(rng, samples)
    cov = np.cov(np.atleast_2d(x), rowvar=False)
    return _effective_dim_from_covariance(cov)


def _estimate_p_proxy_correct(group: "ComplexityGroupBase", samples: int, seed: int) -> float:
    """Generic fallback used only when a subclass doesn't provide an exact
    formula via `_exact_p_proxy_correct` (see ComplexityGroupBase): P(the
    option with the largest ROW SUM, after reshaping the flattened
    observation into (g, observation_size // g), is the one that actually
    gets rewarded). Valid whenever observation_size divides evenly by g
    (true for every group in this project - each allocates an equal-size
    block of the observation per option); raises otherwise rather than
    silently assuming a layout that doesn't hold."""
    if group.observation_size % group.g != 0:
        raise NotImplementedError(
            f"{type(group).__name__} must override _exact_p_proxy_correct() "
            f"(or p_proxy_correct()) - its observation_size "
            f"({group.observation_size}) isn't an even multiple of g "
            f"({group.g}), so the generic per-option row-sum heuristic "
            f"doesn't apply."
        )
    rng = np.random.default_rng(seed)
    x, correct = group._sample_batch(rng, samples)
    per_option = x.reshape(samples, group.g, group.observation_size // group.g)
    predicted = np.argmax(per_option.sum(axis=2), axis=1)
    return float(np.mean(predicted == correct))


class ComplexityGroupBase(ABC):
    """Abstract base every group BanditEnv can use (AlternatingGroup,
    MarginGroup, HeatmapGroup) inherits from.

    Declares the primitives BanditEnv itself actually needs (`g`, `value`,
    `observation_size`, `observation_low`/`observation_high`, `sample`),
    plus two more primitives used ONLY by the default metric estimators
    below - never by BanditEnv or training code:

        _sample_batch(rng, n)  - a vectorized version of `sample`: draw
            `n` independent episodes at once, matching `sample`'s exact
            distribution, returning (observations, rewarded_options).
        _cache_key             - a hashable tuple identifying this
            group's PARAMETERS (kind + g + every difficulty/scale knob),
            deliberately excluding `value` (the reward), since none of
            the metrics below depend on it.

    On top of those, this class provides default, empirically-ESTIMATED
    implementations of every "how complex/big is this proxy" metric used
    across this project:

        num_options()              - trivial (self.g)
        proxy_dimension_size()     - trivial (self.observation_size)
        expected_magnitude()       - E[avg]/E[max]/E[sum]/E[l1]/E[l2]/
                                      E[rms]/E[std] of the observation
        effective_dimensionality() - Shannon-entropy effective rank AND
                                      participation ratio of the
                                      observation's covariance spectrum
        p_proxy_correct()          - P(a naive/untrained read of the
                                      observation points to the option
                                      that actually gets rewarded)

    Every estimated metric is GLOBALLY CACHED by `_cache_key` (see
    `_cached_metric`/`_METRIC_CACHE` above) - calling it twice, on the
    same or a different instance with equal parameters, returns the exact
    same number rather than re-randomizing.

    EXACT OVERRIDES (subclass-provided): p_proxy_correct checks
    `_exact_p_proxy_correct()` first - override that hook (return a float
    instead of the default None) when a cheap closed form exists (see
    MarginGroup, where it's exactly `1 - err`) instead of paying for Monte
    Carlo estimation of something already known exactly.

    INPUT OVERRIDES (caller-provided): every concrete subclass's
    constructor also accepts `expected_magnitude=`,
    `effective_dimensionality=`, `p_proxy_correct=` (all default None).
    Passing a value there is a standing instruction to use exactly that
    value for THIS instance - skipping estimation/caching entirely - e.g.
    when you already know the true value from a derivation not
    implemented here, or want to force a specific number for a
    comparison. Leave as None (the default) to estimate/cache normally.
    """

    g: int
    value: float

    @property
    @abstractmethod
    def observation_size(self) -> int:
        ...

    @property
    @abstractmethod
    def observation_low(self) -> npt.NDArray[np.float32]:
        ...

    @property
    @abstractmethod
    def observation_high(self) -> npt.NDArray[np.float32]:
        ...

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> tuple[npt.NDArray[np.float32], int]:
        """One episode: (observation, the option that gets rewarded).
        This is the method BanditEnv actually calls during training/eval -
        everything else on this class is for the metric estimators."""

    @abstractmethod
    def _sample_batch(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized: `n` independent episodes at once, matching
        `sample`'s exact distribution -> (observations, shape
        (n, observation_size); rewarded_options, shape (n,)). Internal -
        used only by the default Monte Carlo estimators above."""

    @property
    @abstractmethod
    def _cache_key(self) -> tuple:
        """See class docstring."""

    def _exact_p_proxy_correct(self) -> Optional[float]:
        """Override in a subclass with a cheap/exact formula for
        p_proxy_correct instead of needing Monte Carlo estimation. Return
        None (the default) to fall back to the generic estimator."""
        return None

    # -- trivial, non-estimated metrics --------------------------------

    def num_options(self) -> int:
        return self.g

    def proxy_dimension_size(self) -> int:
        return self.observation_size

    # -- estimated (or overridden, or cached) metrics ------------------

    def expected_magnitude(self, samples: int = DEFAULT_MC_SAMPLES,
                             seed: int = DEFAULT_MC_SEED) -> dict:
        if self._expected_magnitude_override is not None:
            return self._expected_magnitude_override
        return _cached_metric(
            self._cache_key, "expected_magnitude", samples, seed,
            lambda: _estimate_expected_magnitude(self, samples, seed),
        )

    def effective_dimensionality(self, samples: int = DEFAULT_MC_SAMPLES,
                                   seed: int = DEFAULT_MC_SEED) -> dict:
        if self._effective_dimensionality_override is not None:
            return self._effective_dimensionality_override
        return _cached_metric(
            self._cache_key, "effective_dimensionality", samples, seed,
            lambda: _estimate_effective_dimensionality(self, samples, seed),
        )

    def p_proxy_correct(self, samples: int = DEFAULT_MC_SAMPLES,
                          seed: int = DEFAULT_MC_SEED) -> float:
        if self._p_proxy_correct_override is not None:
            return self._p_proxy_correct_override
        exact = self._exact_p_proxy_correct()
        if exact is not None:
            return exact
        return _cached_metric(
            self._cache_key, "p_proxy_correct", samples, seed,
            lambda: _estimate_p_proxy_correct(self, samples, seed),
        )

    # -- named single-metric accessors ---------------------------------
    # Thin wrappers over expected_magnitude()/effective_dimensionality()
    # (each of which itself hits the same override/exact/cache resolution
    # above) - added so every metric in this project's checklist has its
    # own directly-callable, individually-named method instead of only
    # being reachable through the two aggregate dicts.

    def average_value(self, samples: int = DEFAULT_MC_SAMPLES,
                        seed: int = DEFAULT_MC_SEED) -> float:
        """E[average entry] of the observation."""
        return self.expected_magnitude(samples=samples, seed=seed)["avg"]

    def l_infinity_norm(self, samples: int = DEFAULT_MC_SAMPLES,
                          seed: int = DEFAULT_MC_SEED) -> float:
        """E[largest entry] of the observation (E[L_infty norm])."""
        return self.expected_magnitude(samples=samples, seed=seed)["max"]

    def l1_norm(self, samples: int = DEFAULT_MC_SAMPLES,
                 seed: int = DEFAULT_MC_SEED) -> float:
        """E[L1 norm] of the observation."""
        return self.expected_magnitude(samples=samples, seed=seed)["l1"]

    def l2_norm(self, samples: int = DEFAULT_MC_SAMPLES,
                 seed: int = DEFAULT_MC_SEED) -> float:
        """E[L2 norm] of the observation."""
        return self.expected_magnitude(samples=samples, seed=seed)["l2"]

    def rms(self, samples: int = DEFAULT_MC_SAMPLES,
             seed: int = DEFAULT_MC_SEED) -> float:
        """E[root-mean-square] of the observation."""
        return self.expected_magnitude(samples=samples, seed=seed)["rms"]

    def standard_deviation(self, samples: int = DEFAULT_MC_SAMPLES,
                             seed: int = DEFAULT_MC_SEED) -> float:
        """E[standard deviation] of the observation."""
        return self.expected_magnitude(samples=samples, seed=seed)["std"]

    def entropy_effective_rank(self, samples: int = DEFAULT_MC_SAMPLES,
                                 seed: int = DEFAULT_MC_SEED) -> float:
        """Shannon-entropy effective rank of the observation's
        covariance matrix."""
        return self.effective_dimensionality(samples=samples, seed=seed)["entropy_effective_rank"]

    def participation_ratio(self, samples: int = DEFAULT_MC_SAMPLES,
                              seed: int = DEFAULT_MC_SEED) -> float:
        """Participation ratio of the observation's covariance matrix."""
        return self.effective_dimensionality(samples=samples, seed=seed)["participation_ratio"]


class AlternatingGroup(ComplexityGroupBase):
    """
    One group of ``g`` options with a tunable decoding complexity ``k``.

    Each episode, a secret ``x ~ Uniform[0, 1)`` is drawn for this group.
    ``[0, 1)`` is partitioned into ``k * g`` equal-width bins, cyclically
    labeled ``0, 1, ..., g-1, 0, 1, ..., g-1, ...``, so:

        correct_option = floor(x * k * g) mod g

    As ``k`` increases, the label oscillates through all ``g`` options more
    times across the same domain - i.e. it becomes a higher-frequency
    function of ``x``. Gradient-based learners are well documented to learn
    low-frequency targets faster than high-frequency ones (spectral bias /
    F-Principle), so ``k`` is a continuous-ish difficulty dial rather than a
    hard discrete jump like modular arithmetic or cryptographic hardness.

    Parameters
    ----------
    g:
        Number of options in this group.
    k:
        Complexity dial: number of times the label cycles through
        ``0..g-1`` across ``[0, 1)``. ``k=1`` is the easy, piecewise-linear
        case; larger ``k`` is harder to learn.
    value:
        Reward given when the agent selects this group's correct option.
    expected_magnitude, effective_dimensionality, p_proxy_correct:
        Optional INPUT OVERRIDES - see ComplexityGroupBase's docstring.
        Left None (the default) to estimate/cache normally.
    """

    def __init__(self, g: int, k: int, value: float, *,
                 expected_magnitude: Optional[dict] = None,
                 effective_dimensionality: Optional[dict] = None,
                 p_proxy_correct: Optional[float] = None):
        if g < 2:
            raise ValueError("g must be at least 2")
        if k < 1:
            raise ValueError("k must be at least 1")

        self.g = int(g)
        self.k = int(k)
        self.value = float(value)

        self._expected_magnitude_override = expected_magnitude
        self._effective_dimensionality_override = effective_dimensionality
        self._p_proxy_correct_override = p_proxy_correct

    @property
    def observation_size(self) -> int:
        # The agent observes the raw secret x - one value, regardless of k.
        return 1

    @property
    def observation_low(self) -> npt.NDArray[np.float32]:
        return np.zeros(self.observation_size, dtype=np.float32)

    @property
    def observation_high(self) -> npt.NDArray[np.float32]:
        return np.ones(self.observation_size, dtype=np.float32)

    def sample(self, rng: np.random.Generator) -> tuple[npt.NDArray[np.float32], int]:
        """Draw this group's secret and the option it labels as correct."""
        x = float(rng.random())
        label = int(np.floor(x * self.k * self.g)) % self.g
        return np.asarray([x], dtype=np.float32), label

    def _sample_batch(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.random(n)
        labels = np.floor(x * self.k * self.g).astype(np.int64) % self.g
        return x.reshape(n, 1), labels

    @property
    def _cache_key(self) -> tuple:
        return ("alternating", self.k, self.g)

    def _exact_p_proxy_correct(self) -> Optional[float]:
        # The label is a DETERMINISTIC function of x - no noise mechanism
        # at all - so whatever "reading x" gives you and the rewarded
        # option are the same thing, always.
        return 1.0


class MarginGroup(ComplexityGroupBase):
    """
    One group of ``g`` options whose difficulty is a genuine margin (the
    gap between the correct option's score and the largest competing
    score), rather than a frequency/spectral-bias effect.

    Each episode:

    1. Draw ``g`` i.i.d. scores ``r_0, ..., r_{g-1} ~ Uniform[0, s)``.
    2. Pick the option the PROXY points to, ``proxy_label ~
       Uniform{0, ..., g-1}`` uniformly at random.
    3. Overwrite that option's score: ``x[proxy_label] =
       max(r_j for j != proxy_label) + delta``. Every other option keeps
       its random draw. ``x`` (the observation) always has its argmax at
       ``proxy_label`` - this step never changes.
    4. With probability ``err``, the REWARDED correct option is not
       ``proxy_label`` at all: it is instead drawn uniformly at random from
       the other ``g - 1`` options (never ``proxy_label`` itself when this
       triggers - see ``err`` below). Otherwise (probability ``1 - err``,
       including always when ``err == 0``), the rewarded correct option is
       exactly ``proxy_label``, same as before ``err`` existed.

    The observation is the length-``g`` vector ``x``, whose argmax is
    always ``proxy_label`` (the proxy's opinion) by construction - this
    does not depend on ``err``. The label RETURNED by ``sample`` (and
    hence the option the environment actually rewards) is ``proxy_label``
    with probability ``1 - err`` and a uniformly random OTHER option with
    probability ``err`` - i.e. ``err`` is exactly the per-episode
    probability that the true correct option is one the proxy does not
    point to. ``err = 0.0`` (the default, and the value implicitly used by
    every MarginGroup config in this project before ``err`` existed)
    recovers the original behavior exactly: the correct answer is always
    ``argmax(x)``, zero Bayes error, fully deterministic given the draw.

    The *margin* by which ``proxy_label`` wins - the amount ``x[proxy_
    label]`` exceeds the true max of the other ``g - 1`` genuinely random
    competitors - is exactly ``delta`` every episode, regardless of ``g``
    or ``err`` (``err`` only changes which option gets REWARDED, never the
    observation ``x`` itself or its distribution - see ``expected_
    magnitude`` callers, which are therefore unaffected by ``err``).

    This is different from the degenerate "zero vs. one nonzero" version:
    the wrong options are NOT a fixed known constant (they're real
    per-episode random draws on the same scale as the correct option), so
    there is an actual signal-detection problem - a classifier needs
    resolution/gain on the order of ``s / delta`` to separate the correct
    option's score from the random backdrop reliably. Margin theory
    (``||w|| ~ 1 / margin`` for a hard-margin linear separator on
    comparable-scale data) is the reason ``delta`` should track minimum
    required weight norm: shrinking ``delta`` relative to ``s`` shrinks
    the margin and grows the required norm, without changing the input
    dimension (always exactly ``g``, and every one of those ``g`` inputs
    is genuinely load-bearing - none of them can be dropped or padded).

    Parameters
    ----------
    g:
        Number of options (also the observation dimension - fixed,
        independent of ``k``).
    k:
        Difficulty dial: ``delta = s / k`` is the actual margin (how much
        the correct option's score exceeds the true maximum of the other
        ``g - 1`` random scores) - LARGER ``k`` = SMALLER margin = harder
        (matches this project's long-standing ``k``-as-difficulty
        convention from every sweep script, e.g. ``delta = 1/k`` at
        ``s=1``). ``k`` may be a non-integer float - it's just ``s /
        delta``, not a discrete index into anything. ``self.delta`` is
        still computed and stored (many internals read it directly), but
        is no longer a constructor parameter - pass ``k`` instead.
    value:
        Reward given when the agent selects this group's correct option.
    s:
        Upper bound (exclusive) of the per-option random score range,
        ``Uniform[0, s)``. Sets the scale of the "random backdrop" the
        margin has to clear. Defaults to 1.0.
    err:
        Per-episode probability that the option actually REWARDED as
        correct is NOT the one the proxy (``argmax(x)``) points to - i.e.
        the label-noise rate on top of an otherwise-unchanged observation.
        When this triggers, the rewarded option is drawn uniformly from
        the other ``g - 1`` options (so it is never ``proxy_label`` -
        "the correct option is one the proxy doesn't point to" is exactly
        what ``err`` controls). Must be in ``[0, 1]``. Defaults to 0.0,
        which recovers the original (pre-``err``) behavior exactly: every
        MarginGroup config anywhere in this project before ``err`` existed
        is implicitly ``err=0.0``. Does NOT change the observation ``x``
        or its distribution in any way - only which label ``sample``
        returns as correct - so ``expected_magnitude``/
        ``effective_dimensionality`` (derived purely from ``x``) are
        unaffected by ``err``; only ``p_proxy_correct`` (exactly
        ``1 - err`` - see ``_exact_p_proxy_correct`` below) and
        trained-network quantities (weight_norm, hit_rate, tau) depend on
        it.
    expected_magnitude, effective_dimensionality, p_proxy_correct:
        Optional INPUT OVERRIDES - see ComplexityGroupBase's docstring.
        Left None (the default) to estimate/cache normally (p_proxy_
        correct's default is actually the exact ``1 - err`` formula
        below, never Monte Carlo - see ``_exact_p_proxy_correct``).
    """

    def __init__(self, g: int, k: float, value: float, s: float = 1.0,
                 err: float = 0.0, *,
                 expected_magnitude: Optional[dict] = None,
                 effective_dimensionality: Optional[dict] = None,
                 p_proxy_correct: Optional[float] = None):
        if g < 2:
            raise ValueError("g must be at least 2")
        if k <= 0:
            raise ValueError("k must be positive")
        if s <= 0:
            raise ValueError("s must be positive")
        if not (0.0 <= err <= 1.0):
            raise ValueError("err must be in [0, 1]")

        self.g = int(g)
        self.k = float(k)
        self.s = float(s)
        self.err = float(err)
        self.delta = self.s / self.k  # derived - see `k` above
        self.value = float(value)

        self._expected_magnitude_override = expected_magnitude
        self._effective_dimensionality_override = effective_dimensionality
        self._p_proxy_correct_override = p_proxy_correct

    @property
    def observation_size(self) -> int:
        return self.g

    @property
    def observation_low(self) -> npt.NDArray[np.float32]:
        return np.zeros(self.g, dtype=np.float32)

    @property
    def observation_high(self) -> npt.NDArray[np.float32]:
        # The correct option's score can exceed s (it's max-of-others +
        # delta, and max-of-others can itself approach s), so the true
        # upper bound is s + delta, not s.
        return np.full(self.g, self.s + self.delta, dtype=np.float32)

    def sample(self, rng: np.random.Generator) -> tuple[npt.NDArray[np.float32], int]:
        """Draw this group's score vector and the option it labels correct.
        The returned label is ``proxy_label`` (== argmax(x)) with
        probability ``1 - err``, and a uniformly random OTHER option (never
        ``proxy_label``) with probability ``err`` - see ``err`` in the
        class docstring."""
        x = (rng.random(self.g) * self.s).astype(np.float32)
        proxy_label = int(rng.integers(self.g))

        others_max = float(np.max(np.delete(x, proxy_label))) if self.g > 1 else 0.0
        x[proxy_label] = np.float32(others_max + self.delta)

        if self.err > 0.0 and rng.random() < self.err:
            others = np.delete(np.arange(self.g), proxy_label)
            true_label = int(rng.choice(others))
        else:
            true_label = proxy_label

        return x, true_label

    def _sample_batch(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized `sample` - matches it exactly EXCEPT `err` is never
        applied here (see note below), since this batch method only ever
        feeds expected_magnitude/effective_dimensionality, which are
        properties of the observation `x` alone - `err` never changes `x`
        (see class docstring), only which label gets rewarded. The second
        return value here is `proxy_label` (== argmax(x)), not the
        err-adjusted true label - correct for THIS method's only two
        callers; p_proxy_correct uses the exact `1 - err` formula below
        instead of ever calling this method."""
        g, s, delta = self.g, self.s, self.delta
        x = rng.random((n, g)) * s
        proxy_label = rng.integers(0, g, size=n)

        x_others = x.copy()
        x_others[np.arange(n), proxy_label] = -np.inf
        others_max = x_others.max(axis=1)
        x[np.arange(n), proxy_label] = others_max + delta

        return x, proxy_label

    @property
    def _cache_key(self) -> tuple:
        return ("margin", self.k, self.s, self.err, self.g)

    def _exact_p_proxy_correct(self) -> Optional[float]:
        # argmax(x) == proxy_label ALWAYS by construction (see class
        # docstring) - the only way the proxy can be "wrong" about the
        # REWARDED option is err's relabeling, which happens with
        # probability err. Exact, no simulation needed.
        return 1.0 - self.err


class HeatmapGroup(ComplexityGroupBase):
    """
    One group of ``g`` options whose observation is ``f_out`` itself -
    the agent NEVER sees the clean recovered one-hot signal, only the
    heavily power-scaled intermediate array it would need to invert
    itself. That inversion is where the real difficulty lives (see
    below) - this is NOT a floating-point-rounding-noise mechanism.

    Each episode:

    1. Draw the correct option ``pos ~ Uniform{0, ..., g-1}``.
    2. Build a one-hot ``gradient`` (shape ``(g, 1)``) with a 1 at
       ``pos``.
    3. Draw ``heatmap_weights`` (shape ``(g, n)``) from
       ``rng.random()``, then row-normalize so every row's mean is
       exactly 1.
    4. Draw ``heatmap_noise`` (shape ``(g, n)``) from ``rng.random()``,
       row-center so every row's mean is exactly 0, then scale by
       ``noise_scale``.
    5. ``f_input = gradient * heatmap_weights + heatmap_noise``, then
       shift by ``+ noise_scale``.
    6. Raise the shifted input to powers ``1, 2, ..., n`` (one power per
       column) to get ``f_out`` - shape ``(g, n)``, flattened row-major
       to shape ``(g * n,)`` as the observation.

    Parameters
    ----------
    g:
        Number of options (the action-space size for this group - NOT
        the observation dimension, since the observation is the
        flattened ``(g, n)`` array).
    noise_scale:
        Scale of the per-row additive noise, and of the shift used to
        keep every power's base positive - also controls how far
        ``f_input`` sits from 1 before being raised to a power, which
        changes how large/spread-out ``f_out`` gets.
    n:
        Number of heatmap columns / highest power used when building
        ``f_out`` (powers ``1, 2, ..., n``) - the main difficulty knob:
        more columns at larger, more spread-out powers for the network
        to implicitly invert.
    value:
        Reward given when the agent selects this group's correct
        option.
    expected_magnitude, effective_dimensionality, p_proxy_correct:
        Optional INPUT OVERRIDES - see ComplexityGroupBase's docstring.
        Left None (the default) to estimate/cache normally.
    """

    def __init__(self, g: int, noise_scale: float, n: int, value: float, *,
                 expected_magnitude: Optional[dict] = None,
                 effective_dimensionality: Optional[dict] = None,
                 p_proxy_correct: Optional[float] = None):
        if g < 2:
            raise ValueError("g must be at least 2")
        if n < 1:
            raise ValueError("n must be at least 1")
        if noise_scale < 0:
            raise ValueError("noise_scale must be non-negative")

        self.g = int(g)
        self.noise_scale = float(noise_scale)
        self.n = int(n)
        self.value = float(value)

        self._expected_magnitude_override = expected_magnitude
        self._effective_dimensionality_override = effective_dimensionality
        self._p_proxy_correct_override = p_proxy_correct

    @property
    def observation_size(self) -> int:
        # f_out is (g, n), flattened - NOT g, since the agent sees the
        # full un-recovered array (see class docstring).
        return self.g * self.n

    @property
    def observation_low(self) -> npt.NDArray[np.float32]:
        # f_input_shifted is always > 0 (see class docstring), so every
        # power of it is too.
        return np.zeros(self.observation_size, dtype=np.float32)

    @property
    def observation_high(self) -> npt.NDArray[np.float32]:
        # f_input_shifted's own supremum: the target row's weight term
        # approaches (but never reaches) n as one raw draw -> 1 and the
        # rest -> 0, plus the noise term's own sub-noise_scale
        # contribution, plus the shift - so f_input_shifted < n +
        # 2 * noise_scale for every row, always. Raising that bound to
        # each column's own power gives a valid (generous, not tight)
        # upper bound per column, tiled across every row in the same
        # row-major flattening sample() uses.
        base_bound = self.n + 2.0 * self.noise_scale
        powers = np.arange(self.n) + 1
        col_high = base_bound ** powers  # shape (n,)
        return np.tile(col_high, self.g).astype(np.float32)

    def sample(self, rng: np.random.Generator) -> tuple[npt.NDArray[np.float32], int]:
        """Draw this group's ``f_out`` observation (flattened, shape
        ``(g * n,)``) and the option it labels correct. Internal math is
        done in float64 (matching ``rng.random()``'s own dtype and every
        other group's internal math here) - only the returned
        observation is cast to float32, same as
        ``AlternatingGroup``/``MarginGroup``."""
        g, n = self.g, self.n
        pos = int(rng.integers(g))

        gradient = np.zeros((g, 1), dtype=np.float64)
        gradient[pos] = 1.0

        heatmap_weights = rng.random((g, n))
        heatmap_weights = heatmap_weights / heatmap_weights.mean(axis=-1, keepdims=True)

        heatmap_noise = rng.random((g, n))
        heatmap_noise = heatmap_noise - heatmap_noise.mean(axis=-1, keepdims=True)
        heatmap_noise = heatmap_noise * self.noise_scale

        f_input = gradient * heatmap_weights + heatmap_noise
        f_input_shifted = f_input + self.noise_scale  # always > 0 - see docstring

        powers = np.arange(n) + 1
        f_out = f_input_shifted ** powers  # shape (g, n)

        return f_out.reshape(-1).astype(np.float32), pos

    def _sample_batch(self, rng: np.random.Generator, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized `sample` - `n_samples` independent episodes at once,
        matching `sample`'s exact per-episode distribution (correct row
        `pos` drawn uniformly per sample, not fixed - needed for anything
        that depends on per-COORDINATE structure, like a covariance
        matrix or the row-sum proxy-accuracy check, not just permutation-
        invariant aggregates)."""
        g, n, noise_scale = self.g, self.n, self.noise_scale
        pos = rng.integers(0, g, size=n_samples)

        weights = rng.random((n_samples, g, n))
        weights = weights / weights.mean(axis=2, keepdims=True)

        noise = rng.random((n_samples, g, n))
        noise = (noise - noise.mean(axis=2, keepdims=True)) * noise_scale

        gradient = np.zeros((n_samples, g, 1))
        gradient[np.arange(n_samples), pos, 0] = 1.0

        f_input = gradient * weights + noise
        f_input_shifted = f_input + noise_scale

        powers = np.arange(1, n + 1)
        f_out = f_input_shifted ** powers  # (n_samples, g, n)

        return f_out.reshape(n_samples, g * n), pos

    @property
    def _cache_key(self) -> tuple:
        return ("heatmap", self.noise_scale, self.n, self.g)

    # No _exact_p_proxy_correct override - the generic row-sum heuristic
    # in ComplexityGroupBase applies directly here (observation_size =
    # g*n, an exact multiple of g), and there's no closed form for it the
    # way MarginGroup has.
