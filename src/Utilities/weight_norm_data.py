"""Unified store for every weight-norm/tau sweep run in this project
(MarginGroup, HeatmapGroup), plus the machinery to compute "expected
magnitude" (E[average entry], E[largest entry], E[sum of entries], E[L1
norm], E[L2 norm], E[RMS], E[standard deviation] of the group's
observation proxy) for any group spec - whether or not that exact spec
has ever actually been trained.

Backing file: ``weight_norm_data.csv`` (repo root), one row per (group_type,
params) config actually run. Columns:
    group_type, k, s, err, delta, noise_scale, n, weight_decay,
    hit_rate, proxy_hit_rate, correct, proxy_correct, episodes, mean_reward,
    wn_policy_net_0, wn_policy_net_2, wn_action_net, weight_norm_actor,
    wn_value_net_0, wn_value_net_2, wn_value_net_out, weight_norm_critic,
    weight_norm_total,
    fit_status, fit_L, fit_A, fit_tau, fit_L_err, fit_A_err, fit_tau_err,
    r_squared, rmse, fit_points_used,
    source_file

``hit_rate``/``correct`` (original) measure how often the trained policy's
greedy action matches the option actually REWARDED that episode - with
``err > 0`` this is the noisy, possibly-mislabeled target. ``proxy_hit_rate``/
``proxy_correct`` (added 2026-08-13, alongside ``err``) instead measure how
often the policy's action matches the option the PROXY points to
(``argmax`` of the observation) - the "intended" target, regardless of
whether that episode's reward was actually assigned to it. For every row
with ``err == 0`` (every row in this file before the err sweep), these two
are mathematically IDENTICAL - the rewarded option and the proxy's option
are the exact same thing whenever err==0 - so every pre-existing row's
``proxy_hit_rate``/``proxy_correct`` were backfilled as exact copies of
``hit_rate``/``correct``, not estimates. Only genuinely diverges from
``hit_rate`` for ``err > 0`` rows.

A row's ``group_type`` determines which of (k, s, err, delta) / (noise_scale,
n) are populated (the other set is left blank):
    margin   -> k, s, err, delta all populated (s=1.0, err=0.0, delta=1/k
                is the plain, unscaled, error-free MarginGroup case - it is
                NOT a separate group_type, just this family's (s=1, err=0)
                point)
    heatmap  -> noise_scale, n populated (err is a MarginGroup-only concept
                - see Utilities/bandit_env.py's MarginGroup docstring - so
                it is always blank for heatmap rows)

``err`` (added 2026-08-13) is MarginGroup's per-episode label-noise rate:
the probability that the option actually REWARDED as correct is NOT the one
the proxy (argmax of the observation) points to. It does NOT change the
observation's distribution at all (see expected_magnitude below) - only
which label training saw as correct - so every row recorded before ``err``
existed is retroactively ``err=0.0`` (the code path is identical; ``err=0``
literally never triggers the label-noise branch), and every ``k``/``s``
formula elsewhere in this module that doesn't mention ``err`` is unaffected
by it for exactly that reason.

Every record's canonical identity is its ``spec`` tuple - the SAME
convention already used by run_indifference_batch.py's
mixed_group_factory:
    ("margin", k, s, err)
    ("heatmap", noise_scale, n)
(as of 2026-08-13, "margin_scaled" was renamed to plain "margin" - it was
always the same MarginGroup class/spec shape, just an unnecessary second
name for s != 1.0. Plain, unscaled, error-free margin is simply
("margin", k, 1.0, 0.0). Shorter margin specs - ("margin", k) or
("margin", k, s) - are still accepted by get_record/expected_magnitude as
shorthand for err=0.0, but the canonical/stored form is always the full
4-tuple.)

FUTURE SWEEPS: new weight-norm/tau sweep scripts should call
``append_record(...)`` (passing the fields below) instead of writing their
own bespoke per-sweep CSV. That keeps every run's data in ONE place with
ONE schema, instead of the fragmented per-sweep (per_layer_weight_norm_rerun,
margin_scaled_weight_norm_sweep, margin_scaled_weight_norm_sweep_small_s,
margin_scaled_weight_norm_sweep_mid_s, ...) files this project accumulated
before this module existed (those filenames predate the rename and are kept
as-is for provenance/history - only the group_type VALUES and spec tuples
were renamed, not old filenames).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, fields, asdict
from functools import lru_cache
from typing import Optional

import numpy as np

try:
    from Utilities.bandit_env import HeatmapGroup, MarginGroup
except ImportError:
    # Allow this module to still be imported as a standalone script (e.g.
    # `python3 weight_norm_data.py` from inside Utilities/) without the
    # `Utilities.` package prefix resolving - only spec_from_group needs
    # these classes, and it isn't used at import time.
    from bandit_env import HeatmapGroup, MarginGroup

_HERE = os.path.dirname(os.path.abspath(__file__))
UNIFIED_CSV_PATH = os.path.normpath(os.path.join(_HERE, "..", "weight_norm_data.csv"))

FIELDNAMES = [
    "group_type", "k", "s", "err", "delta", "noise_scale", "n", "weight_decay",
    "hit_rate", "proxy_hit_rate", "correct", "proxy_correct", "episodes", "mean_reward",
    "wn_policy_net_0", "wn_policy_net_2", "wn_action_net", "weight_norm_actor",
    "wn_value_net_0", "wn_value_net_2", "wn_value_net_out", "weight_norm_critic",
    "weight_norm_total",
    "fit_status", "fit_L", "fit_A", "fit_tau",
    "fit_L_err", "fit_A_err", "fit_tau_err", "r_squared", "rmse", "fit_points_used",
    "source_file",
]

# g is a hardcoded project-wide constant (every MarginGroup config in this
# project uses g=4) - see build_weight_norm_data.py / plot_M_fits'
# magnitude-hypothesis section for the derivation this depends on.
MARGIN_G = 4


def _f(x):
    """Parse a CSV field that may be '' (blank) into None, else float."""
    if x is None or x == "" or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(x)


def _i(x):
    v = _f(x)
    return None if v is None else int(v)


@dataclass
class WeightNormRecord:
    group_type: str          # "margin" | "heatmap"
    k: Optional[int]
    s: Optional[float]
    err: Optional[float]
    delta: Optional[float]
    noise_scale: Optional[float]
    n: Optional[int]
    weight_decay: Optional[float]
    hit_rate: Optional[float]
    proxy_hit_rate: Optional[float]
    correct: Optional[int]
    proxy_correct: Optional[int]
    episodes: Optional[int]
    mean_reward: Optional[float]
    wn_policy_net_0: Optional[float]
    wn_policy_net_2: Optional[float]
    wn_action_net: Optional[float]
    weight_norm_actor: Optional[float]
    wn_value_net_0: Optional[float]
    wn_value_net_2: Optional[float]
    wn_value_net_out: Optional[float]
    weight_norm_critic: Optional[float]
    weight_norm_total: Optional[float]
    fit_status: Optional[str]
    fit_L: Optional[float]
    fit_A: Optional[float]
    fit_tau: Optional[float]
    fit_L_err: Optional[float]
    fit_A_err: Optional[float]
    fit_tau_err: Optional[float]
    r_squared: Optional[float]
    rmse: Optional[float]
    fit_points_used: Optional[int]
    source_file: str = ""

    @property
    def spec(self):
        if self.group_type == "margin":
            s = self.s if self.s is not None else 1.0
            err = self.err if self.err is not None else 0.0
            return ("margin", self.k, s, err)
        if self.group_type == "heatmap":
            return ("heatmap", self.noise_scale, self.n)
        raise ValueError(f"unknown group_type {self.group_type!r}")

    @property
    def tau_diverged(self) -> bool:
        """Known curve_fit-divergence failure mode: fit_status='ok' but
        fit_tau is wildly wrong (billions+ or negative) - r_squared USUALLY
        collapses too, but not always (e.g. margin(13, 10.0) has
        fit_tau=-2.57e10 with r_squared=0.510, just above a bare
        r_squared<=0.5 cutoff - caught this the hard way: an earlier
        version of this filter used r_squared alone and missed it). So this
        checks BOTH: r_squared<=0.5, OR fit_tau itself being physically
        impossible (tau is a decay/growth timescale - it must be positive,
        and every genuine fit seen in this project stays under 1e6) - either
        condition alone is enough to flag divergence. Does NOT mean a given
        (k, s)/spec combo is inherently bad - per direct instruction
        (2026-08-12) these are random per-run occurrences, not
        deterministic properties of the spec."""
        if self.r_squared is None or self.fit_tau is None:
            return True
        return self.r_squared <= 0.5 or self.fit_tau <= 0 or abs(self.fit_tau) > 1e6

    def to_dict(self) -> dict:
        return asdict(self)


def _row_to_record(row: dict) -> WeightNormRecord:
    # err defaults to 0.0 (not None) for margin rows with a blank/missing
    # err field - every row recorded before err existed is retroactively
    # err=0.0 (see module docstring). Stays None for heatmap rows, which
    # don't have this field at all.
    _err_raw = _f(row.get("err"))
    err = _err_raw if _err_raw is not None else (0.0 if row["group_type"] == "margin" else None)

    hit_rate = _f(row.get("hit_rate"))
    correct = _i(row.get("correct"))
    # proxy_hit_rate/proxy_correct default to an exact copy of
    # hit_rate/correct when blank AND err == 0.0 - the two are
    # mathematically identical whenever err==0 (see module docstring), so
    # this is a backfill, not an estimate. Left as None if blank with
    # err != 0.0 (should never happen for a properly-written row) or for
    # heatmap rows (which have neither err nor proxy_hit_rate).
    _proxy_hr_raw = _f(row.get("proxy_hit_rate"))
    proxy_hit_rate = _proxy_hr_raw if _proxy_hr_raw is not None else (
        hit_rate if (err == 0.0 and row["group_type"] == "margin") else None)
    _proxy_correct_raw = _i(row.get("proxy_correct"))
    proxy_correct = _proxy_correct_raw if _proxy_correct_raw is not None else (
        correct if (err == 0.0 and row["group_type"] == "margin") else None)

    return WeightNormRecord(
        group_type=row["group_type"],
        k=_i(row.get("k")),
        s=_f(row.get("s")),
        err=err,
        delta=_f(row.get("delta")),
        noise_scale=_f(row.get("noise_scale")),
        n=_i(row.get("n")),
        weight_decay=_f(row.get("weight_decay")),
        hit_rate=hit_rate,
        proxy_hit_rate=proxy_hit_rate,
        correct=correct,
        proxy_correct=proxy_correct,
        episodes=_i(row.get("episodes")),
        mean_reward=_f(row.get("mean_reward")),
        wn_policy_net_0=_f(row.get("wn_policy_net_0")),
        wn_policy_net_2=_f(row.get("wn_policy_net_2")),
        wn_action_net=_f(row.get("wn_action_net")),
        weight_norm_actor=_f(row.get("weight_norm_actor")),
        wn_value_net_0=_f(row.get("wn_value_net_0")),
        wn_value_net_2=_f(row.get("wn_value_net_2")),
        wn_value_net_out=_f(row.get("wn_value_net_out")),
        weight_norm_critic=_f(row.get("weight_norm_critic")),
        weight_norm_total=_f(row.get("weight_norm_total")),
        fit_status=row.get("fit_status") or None,
        fit_L=_f(row.get("fit_L")),
        fit_A=_f(row.get("fit_A")),
        fit_tau=_f(row.get("fit_tau")),
        fit_L_err=_f(row.get("fit_L_err")),
        fit_A_err=_f(row.get("fit_A_err")),
        fit_tau_err=_f(row.get("fit_tau_err")),
        r_squared=_f(row.get("r_squared")),
        rmse=_f(row.get("rmse")),
        fit_points_used=_i(row.get("fit_points_used")),
        source_file=row.get("source_file", ""),
    )


def load_records(csv_path: str = UNIFIED_CSV_PATH) -> list[WeightNormRecord]:
    """Parse the unified CSV into a list of WeightNormRecord."""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return [_row_to_record(row) for row in csv.DictReader(f)]


# Loaded once at import time - "stores the data in python formats and
# works with it" per the request that created this module.
RECORDS: list[WeightNormRecord] = load_records()


def _build_index(records):
    index = {}
    for rec in records:
        spec = rec.spec
        if spec in index:
            raise ValueError(f"duplicate spec {spec} in weight_norm_data.csv")
        index[spec] = rec
    return index


_INDEX: dict = _build_index(RECORDS)


def normalize_spec(spec) -> tuple:
    """Canonicalize a margin spec to the full 4-tuple ("margin", k, s, err)
    - shorthand ("margin", k) and ("margin", k, s) are both accepted and
    normalized to s=1.0/err=0.0 for whichever trailing element(s) are
    missing. heatmap specs pass through unchanged (they don't have an err
    dimension)."""
    spec = tuple(spec)
    if spec[0] == "margin":
        k = spec[1]
        s = spec[2] if len(spec) > 2 else 1.0
        err = spec[3] if len(spec) > 3 else 0.0
        return ("margin", k, s, err)
    return spec


def get_record(spec) -> Optional[WeightNormRecord]:
    """Look up the WeightNormRecord for an exact spec tuple, e.g.
    ("margin", 13, 1.0, 0.0), ("margin", 13, 10.0, 0.3), ("heatmap", 1.0, 3).
    Shorthand margin specs (2- or 3-tuple) are accepted - see
    normalize_spec. Returns None if that exact config was never run."""
    return _INDEX.get(normalize_spec(spec))


def append_record(row: dict, csv_path: str = UNIFIED_CSV_PATH) -> WeightNormRecord:
    """Append one new row (dict with FIELDNAMES keys, missing keys treated
    as blank) to the unified CSV AND to the in-memory RECORDS/_INDEX, so a
    long-running process sees it immediately without reloading.

    FUTURE WEIGHT-NORM SWEEPS SHOULD CALL THIS instead of writing their
    own CSV, so every run lands in one unified place. Raises if the row's
    spec already exists (specs are supposed to be unique - re-running the
    same config should be a conscious decision, not an accidental
    duplicate)."""
    full_row = {f: row.get(f, "") for f in FIELDNAMES}
    rec = _row_to_record(full_row)
    if rec.spec in _INDEX:
        raise ValueError(f"spec {rec.spec} already exists in {csv_path}")

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(full_row)

    RECORDS.append(rec)
    _INDEX[rec.spec] = rec
    return rec


def remove_record(spec, csv_path: str = UNIFIED_CSV_PATH) -> bool:
    """Remove the row for an exact spec (see normalize_spec for accepted
    shorthand) from the unified CSV AND from RECORDS/_INDEX, by rewriting
    the CSV without that one row. Returns True if a row was actually
    removed, False if that spec wasn't present (a no-op, not an error).
    Used by upsert_record below - most callers should prefer that instead
    of calling this directly."""
    spec = normalize_spec(spec)
    if spec not in _INDEX:
        return False

    RECORDS[:] = [r for r in RECORDS if r.spec != spec]
    del _INDEX[spec]

    if os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(to_csv_rows(RECORDS))
    return True


def upsert_record(row: dict, csv_path: str = UNIFIED_CSV_PATH) -> WeightNormRecord:
    """Like append_record, but OVERWRITES any existing row for the same
    spec instead of raising - removes the old row (if present) via
    remove_record, then appends the new one. Use this instead of
    append_record when re-running a sweep is meant to replace that sweep's
    previous results (e.g. after changing what a sweep measures), rather
    than accumulate a second copy or fail outright."""
    # Build the spec straight from the row being written (the same way
    # _row_to_record's .spec property would), so this works even if the
    # OLD row (about to be removed) had slightly different optional
    # fields than the new one.
    new_spec = _row_to_record({f: row.get(f, "") for f in FIELDNAMES}).spec
    remove_record(new_spec, csv_path=csv_path)
    return append_record(row, csv_path=csv_path)


# ----------------------------------------------------------------------
# Expected-magnitude (E[average entry], E[largest entry], E[sum of
# entries] of the group's observation proxy) - available for ANY spec,
# not just ones that have actually been trained, since these are
# properties of the group's generative distribution, not of a trained
# network.
# ----------------------------------------------------------------------

def _margin_magnitude(k: int, s: float, g: int = MARGIN_G) -> dict:
    """Closed-form E[avg]/E[max]/E[sum] of MarginGroup's observation,
    derived from its exact generative process (see
    Utilities/bandit_env.py's MarginGroup.sample): g-1 backdrop draws ~
    Uniform[0, s), plus the correct entry = max(backdrop) + delta.
    delta = s / k; k is the "difficulty" knob, s the overall scale (s=1.0
    is the plain, unscaled case).

        E[avg] = s*(9/16 + 1/(4k))     (for g=4; see derivation notes)
        E[sum] = g * E[avg]            (== s*(9/4 + 1/k) when g=4)
        E[max] = s*(3/4 + 1/k)

    NOTE: sum_ratio == avg_ratio EXACTLY for any margin<->margin comparison
    (any s, any k), since g is a fixed project-wide constant (4) - this
    degeneracy does NOT hold once HeatmapGroup (whose n varies) is
    involved.
    """
    delta = s / k
    avg = 0.5625 * s + 0.25 * delta
    mx = 0.75 * s + delta
    total = g * avg
    return {"avg": avg, "max": mx, "sum": total}


@lru_cache(maxsize=256)
def _margin_extra_stats_mc(k: int, s: float, g: int = MARGIN_G,
                            samples: int = 500_000, seed: int = 0) -> tuple:
    """Monte Carlo E[L1]/E[L2]/E[RMS]/E[std] of MarginGroup's observation
    (added 2026-08-13, for plot_M_fits' proxy-magnitude-statistics check).
    No closed form is used here (unlike avg/max/sum in _margin_magnitude)
    because L2/RMS/std all involve squares of a dependent order statistic
    (the correct entry is max(backdrop) + delta, not i.i.d. with the rest),
    which doesn't have a clean closed form the way avg/max/sum do - so this
    simulates MarginGroup's EXACT generative process instead, the same way
    _heatmap_magnitude_mc already does for HeatmapGroup.

    Every entry here is >= 0 (backdrop ~ Uniform[0, s), correct = max(...)
    + delta with delta = s/k >= 0), so E[L1] == E[sum] exactly - this is
    NOT true for HeatmapGroup (see _heatmap_magnitude_mc), whose entries
    can go negative after the noise/shift/power steps.

    Cached per (k, s, g, samples, seed), same convention as
    _heatmap_magnitude_mc."""
    rng = np.random.default_rng(seed)
    delta = s / k

    backdrop = rng.random((samples, g - 1)) * s
    correct = backdrop.max(axis=1) + delta
    entries = np.concatenate([backdrop, correct[:, None]], axis=1)  # (samples, g)

    l1 = np.abs(entries).sum(axis=1)
    sq = entries ** 2
    sum_sq = sq.sum(axis=1)
    l2 = np.sqrt(sum_sq)
    rms = np.sqrt(sum_sq / g)
    std = entries.std(axis=1)

    return (float(l1.mean()), float(l2.mean()), float(rms.mean()), float(std.mean()))


@lru_cache(maxsize=256)
def _heatmap_magnitude_mc(noise_scale: float, n: int, g: int = MARGIN_G,
                           samples: int = 500_000, seed: int = 0) -> tuple:
    """Monte Carlo E[avg]/E[max]/E[sum]/E[L1]/E[L2]/E[RMS]/E[std] of
    HeatmapGroup's flattened f_out, following its EXACT generative process
    (Utilities/bandit_env.py HeatmapGroup.sample):
        1. gradient: one-hot (g,) - which row is "correct" doesn't affect
           the marginal distribution of entries, so we don't even need to
           sample it explicitly here (see below).
        2. heatmap_weights ~ Uniform[0,1) (g, n), then each row divided by
           its own mean (row mean forced to exactly 1).
        3. heatmap_noise ~ Uniform[0,1) (g, n), each row minus its own mean
           (row mean forced to exactly 0), scaled by noise_scale.
        4. f_input = gradient*heatmap_weights + heatmap_noise
        5. f_input_shifted = f_input + noise_scale
        6. f_out[:, j] = f_input_shifted[:, j] ** (j+1) for j = 0..n-1

    Cached (memoized) per (noise_scale, n, g, samples, seed) since this is
    somewhat expensive and the set of noise_scale/n values actually used
    in this project is small and gets looked up repeatedly.

    NOTE (added 2026-08-13, for the L1/L2/RMS/std extension below): unlike
    MarginGroup, HeatmapGroup's entries are NOT guaranteed >= 0 - noise is
    row-centered (mean forced to 0), so a row's minimum noise entries can
    go negative, and after the `+ noise_scale` shift and `** (j+1)` power
    steps an odd power can still land negative if noise_scale is small.
    So E[L1] (sum of |entries|) is NOT interchangeable with E[sum] (sum of
    entries) here, the way it is for MarginGroup.
    """
    rng = np.random.default_rng(seed)

    weights = rng.random((samples, g, n))
    weights = weights / weights.mean(axis=2, keepdims=True)

    noise = rng.random((samples, g, n))
    noise = (noise - noise.mean(axis=2, keepdims=True)) * noise_scale

    # gradient is one-hot per row-group; since every row of f_out is
    # generated identically in distribution regardless of whether it's
    # the "correct" row or not EXCEPT for the +weights term on the
    # correct row, we explicitly simulate one row as "correct" (gradient
    # contribution = weights) and the rest as "incorrect" (gradient
    # contribution = 0), matching the true row-mix exactly: 1 correct row
    # out of g per episode.
    f_input_correct = weights[:, 0, :] + noise[:, 0, :]          # 1 row/episode
    f_input_incorrect = noise[:, 1:, :]                          # g-1 rows/episode

    shifted_correct = f_input_correct + noise_scale
    shifted_incorrect = f_input_incorrect + noise_scale

    powers = np.arange(1, n + 1)
    f_out_correct = shifted_correct ** powers
    f_out_incorrect = shifted_incorrect ** powers

    # Combine correct (1 row) + incorrect (g-1 rows) into the full (g, n)
    # per-episode array's flattened stats.
    total_sum = f_out_correct.sum(axis=1) + f_out_incorrect.sum(axis=(1, 2))
    total_avg = total_sum / (g * n)
    if g > 1:
        total_max = np.maximum(f_out_correct.max(axis=1), f_out_incorrect.max(axis=(1, 2)))
    else:
        total_max = f_out_correct.max(axis=1)

    # L1/L2/RMS/std - added 2026-08-13. L1 uses abs() explicitly (see the
    # note above this function's Monte Carlo body: entries can be
    # negative here, unlike MarginGroup). L2/RMS are both derived from the
    # same sum-of-squares; std uses the identity var = E[x^2] - E[x]^2
    # (population/ddof=0, matching MarginGroup's entries.std(axis=1)).
    total_l1 = np.abs(f_out_correct).sum(axis=1) + np.abs(f_out_incorrect).sum(axis=(1, 2))
    sum_sq = (f_out_correct ** 2).sum(axis=1) + (f_out_incorrect ** 2).sum(axis=(1, 2))
    total_l2 = np.sqrt(sum_sq)
    total_rms = np.sqrt(sum_sq / (g * n))
    mean_sq = sum_sq / (g * n)
    total_var = np.clip(mean_sq - total_avg ** 2, 0, None)
    total_std = np.sqrt(total_var)

    return (
        float(total_avg.mean()), float(total_max.mean()), float(total_sum.mean()),
        float(total_l1.mean()), float(total_l2.mean()), float(total_rms.mean()),
        float(total_std.mean()),
    )


def expected_magnitude(spec, **mc_kwargs) -> dict:
    """Return {'avg': ..., 'max': ..., 'sum': ..., 'l1': ..., 'l2': ...,
    'rms': ..., 'std': ...} - seven measures of the "magnitude" of the
    group's observation proxy (E[average entry], E[largest entry], E[sum
    of entries], E[L1 norm], E[L2 norm], E[RMS], E[standard deviation]),
    for ANY spec (does not require the spec to have been trained).

    avg/max/sum use the closed-form MarginGroup formula (or a cached Monte
    Carlo estimate for HeatmapGroup) exactly as before this function grew
    l1/l2/rms/std (added 2026-08-13, for plot_M_fits' proxy-magnitude-
    statistics check) - those four are Monte Carlo for BOTH group types
    (see _margin_extra_stats_mc/_heatmap_magnitude_mc), since no closed
    form exists for L2/RMS/std of MarginGroup's observation (one entry is
    a dependent order statistic, not i.i.d. with the rest).

    For MarginGroup specifically, l1 == sum exactly (every entry is >= 0 -
    see _margin_extra_stats_mc's docstring) - this equivalence does NOT
    hold for HeatmapGroup, whose entries can go negative.

    Pass samples=/seed=/... through mc_kwargs to override the Monte Carlo
    defaults (applies to l1/l2/rms/std always, and to avg/max/sum too for
    heatmap specs). A margin spec's ``err`` element (if present, e.g.
    ("margin", k, s, err)) is IGNORED here on purpose - err only changes
    which label gets rewarded during training, never the observation's
    distribution (see MarginGroup.sample's docstring), so it has no effect
    on expected magnitude."""
    spec = tuple(spec)
    if spec[0] == "margin":
        k, s = spec[1], (spec[2] if len(spec) > 2 else 1.0)
        out = _margin_magnitude(k=k, s=s)
        l1, l2, rms, std = _margin_extra_stats_mc(k=k, s=s, **mc_kwargs)
        out.update({"l1": l1, "l2": l2, "rms": rms, "std": std})
        return out
    if spec[0] == "heatmap":
        avg, mx, total, l1, l2, rms, std = _heatmap_magnitude_mc(
            noise_scale=spec[1], n=spec[2], **mc_kwargs)
        return {"avg": avg, "max": mx, "sum": total,
                "l1": l1, "l2": l2, "rms": rms, "std": std}
    raise ValueError(f"unknown spec {spec!r}")


# ----------------------------------------------------------------------
# Format conversion helpers
# ----------------------------------------------------------------------

def to_dicts(records: Optional[list] = None) -> list[dict]:
    records = RECORDS if records is None else records
    return [r.to_dict() for r in records]


def to_dataframe(records: Optional[list] = None):
    import pandas as pd
    return pd.DataFrame(to_dicts(records))


def to_csv_rows(records: Optional[list] = None) -> list[dict]:
    """Rows as plain dicts with FIELDNAMES keys, ready for csv.DictWriter
    (spec/property fields are NOT included - only the raw stored columns)."""
    records = RECORDS if records is None else records
    return [{f: getattr(r, f) for f in FIELDNAMES} for r in records]


def by_group_type(group_type: str, records: Optional[list] = None) -> list[WeightNormRecord]:
    records = RECORDS if records is None else records
    return [r for r in records if r.group_type == group_type]


# ----------------------------------------------------------------------
# Additional per-spec metrics (added for the "shared machinery" project
# metric checklist - proxy dimension size, number of options, P(proxy
# points to the rewarded option), and effective-dimensionality of the
# proxy's covariance structure). Like expected_magnitude above, these are
# properties of a group's GENERATIVE PROCESS, not of a trained network -
# available for any spec, no training required.
# ----------------------------------------------------------------------

def spec_from_group(group) -> tuple:
    """Convert a live HeatmapGroup/MarginGroup INSTANCE (e.g. one of
    run_p_curve_experiments.py's TESTS[i]["fixed_group"](value) results)
    into the canonical spec tuple used everywhere else in this module.
    `value` (the reward) is never part of a spec - every metric in this
    module is a property of the group's OBSERVATION/target distribution,
    which none of these metrics below depend on.

    Margin's `k` is recovered as `s / delta` (the same relationship
    build_weight_norm_data.py uses in the other direction, `delta = 1/k`
    at s=1) - this can come out non-integer for a MarginGroup built with
    an arbitrary delta/s (e.g. run_p_curve_experiments.py's
    margin_vs_margin_harder_variable test uses delta=0.25 directly, never
    an explicit k), which is fine - k is stored/read as a float by every
    function below, never assumed to be an integer index into a sweep."""
    if isinstance(group, HeatmapGroup):
        return ("heatmap", group.noise_scale, group.n)
    if isinstance(group, MarginGroup):
        k = group.s / group.delta
        return ("margin", k, group.s, group.err)
    raise ValueError(f"unrecognized group type {type(group).__name__}")


def proxy_dimension_size(spec, g: int = MARGIN_G) -> int:
    """Dimensionality of the group's observation vector: g for margin
    (one scalar score per option), g*n for heatmap (n stacked
    power/heatmap columns per option) - see each class's
    `observation_size` property in Utilities/bandit_env.py, which this
    mirrors exactly without needing to construct an actual group
    instance."""
    spec = tuple(spec)
    if spec[0] == "margin":
        return g
    if spec[0] == "heatmap":
        n = spec[2]
        return g * n
    raise ValueError(f"unknown spec {spec!r}")


def num_options(spec, g: int = MARGIN_G) -> int:
    """Number of options the model selects between (the action space
    size for this one group, i.e. `group.g`). Every group in this project
    uses g=4 (MARGIN_G) - this is a real, explicit metric/column rather
    than an implicit assumption, and accepts an override in case that
    ever changes for a specific comparison."""
    return g


def p_proxy_correct(spec, g: int = MARGIN_G, **mc_kwargs) -> float:
    """P(the proxy - i.e. a naive, untrained read of the raw observation
    - points to the option that actually gets REWARDED that episode).

    margin: EXACT, no simulation needed. MarginGroup.sample always sets
        `argmax(x) == proxy_label` by construction (see its docstring/
        code) - the only source of "proxy wrong" is `err`, the explicit
        per-episode probability the reward is redirected to a different,
        uniformly random option. So P = 1 - err exactly.

    heatmap: NOT exact - there is no equivalent labeled parameter, and no
        closed form (the "correct" row's entries are `weights + noise`
        with the OTHER g-1 rows being noise-only, then every entry is
        shifted and raised to a column-specific integer power - the
        resulting joint distribution has no clean algebraic max/argmax
        probability). Estimated by Monte Carlo using the most natural
        generalization of "argmax(x)" to a multi-entry-per-option
        observation: reshape the flattened observation back into (g, n)
        and take the option with the largest ROW SUM (equivalently row
        mean, since n is shared across rows) - this is the simplest
        heuristic that reads every one of the g*n entries symmetrically,
        with no per-column reweighting. This is a DEFINITION choice (a
        different row-summary, e.g. max-of-row instead of sum-of-row,
        would give a different number) - documented here so it's an
        explicit, reproducible convention rather than a hidden one.
    """
    spec = tuple(spec)
    if spec[0] == "margin":
        err = spec[3] if len(spec) > 3 else 0.0
        return 1.0 - err
    if spec[0] == "heatmap":
        noise_scale, n = spec[1], spec[2]
        return _heatmap_proxy_accuracy_mc(noise_scale=noise_scale, n=n, g=g, **mc_kwargs)
    raise ValueError(f"unknown spec {spec!r}")


@lru_cache(maxsize=256)
def _heatmap_full_samples_key(noise_scale: float, n: int, g: int, samples: int, seed: int):
    """Cache key wrapper - see _heatmap_full_samples (returns numpy
    arrays, which aren't hashable, so the array-returning function itself
    can't be lru_cache'd directly; this cache holds a tuple of arrays
    instead, keyed on the same (noise_scale, n, g, samples, seed) inputs
    every other MC helper in this module uses)."""
    return _heatmap_full_samples(noise_scale, n, g, samples, seed)


def _heatmap_full_samples(noise_scale: float, n: int, g: int, samples: int, seed: int):
    """Simulate HeatmapGroup's EXACT generative process
    (Utilities/bandit_env.py HeatmapGroup.sample), keeping the FULL
    per-episode flattened (g*n,) observation vector (not just aggregate
    scalars, unlike _heatmap_magnitude_mc above) AND which row is
    "correct" - needed for anything that depends on per-COORDINATE
    structure (a covariance matrix, or checking whether a row-level
    heuristic picked the right row), not just permutation-invariant
    aggregates (sum/max/L1/L2/avg, which don't care which row was
    correct).

    Critically, `pos` (which row is correct) is drawn UNIFORMLY AT RANDOM
    per sample here, matching the real environment exactly - always
    fixing "row 0 is correct" (as _heatmap_magnitude_mc does, a valid
    shortcut for permutation-invariant aggregates only) would bias a
    covariance matrix or a per-row accuracy check, since it would make
    row 0 look systematically different from the other rows across the
    whole sample, an artifact that isn't present in the true generative
    process.

    Returns (f_out_flat, pos): f_out_flat has shape (samples, g*n); pos
    has shape (samples,), each entry in [0, g)."""
    rng = np.random.default_rng(seed)
    pos = rng.integers(0, g, size=samples)

    weights = rng.random((samples, g, n))
    weights = weights / weights.mean(axis=2, keepdims=True)

    noise = rng.random((samples, g, n))
    noise = (noise - noise.mean(axis=2, keepdims=True)) * noise_scale

    gradient = np.zeros((samples, g, 1))
    gradient[np.arange(samples), pos, 0] = 1.0

    f_input = gradient * weights + noise
    f_input_shifted = f_input + noise_scale

    powers = np.arange(1, n + 1)
    f_out = f_input_shifted ** powers  # (samples, g, n), broadcast over last axis

    return f_out.reshape(samples, g * n), pos


def _heatmap_proxy_accuracy_mc(noise_scale: float, n: int, g: int = MARGIN_G,
                                 samples: int = 200_000, seed: int = 0) -> float:
    """P(argmax of row-sums of the (g, n)-reshaped observation == the
    truly correct row), Monte Carlo estimated - see p_proxy_correct's
    docstring for the row-sum heuristic's definition/rationale."""
    f_out_flat, pos = _heatmap_full_samples_key(noise_scale, n, g, samples, seed)
    row_sums = f_out_flat.reshape(samples, g, n).sum(axis=2)
    predicted = np.argmax(row_sums, axis=1)
    return float(np.mean(predicted == pos))


def _margin_full_samples(k: float, s: float, g: int, samples: int, seed: int):
    """Simulate MarginGroup's EXACT generative process
    (Utilities/bandit_env.py MarginGroup.sample), keeping the FULL
    per-episode (g,) score vector AND which coordinate is
    `proxy_label` (== argmax(x) by construction) - needed for the
    covariance matrix below, which depends on per-coordinate structure,
    not just the permutation-invariant aggregates
    _margin_extra_stats_mc computes. `proxy_label` is drawn uniformly at
    random per sample (matching MarginGroup.sample exactly), not fixed to
    one coordinate, for the same reason _heatmap_full_samples randomizes
    `pos`."""
    rng = np.random.default_rng(seed)
    delta = s / k
    x = (rng.random((samples, g)) * s)
    proxy_label = rng.integers(0, g, size=samples)

    x_others = x.copy()
    x_others[np.arange(samples), proxy_label] = -np.inf
    others_max = x_others.max(axis=1)
    x[np.arange(samples), proxy_label] = others_max + delta

    return x, proxy_label


@lru_cache(maxsize=256)
def _margin_full_samples_key(k: float, s: float, g: int, samples: int, seed: int):
    return _margin_full_samples(k, s, g, samples, seed)


def _effective_dim_from_covariance(cov: np.ndarray) -> dict:
    """Shannon-entropy effective rank (`exp(H)` where `H` is the Shannon
    entropy, in nats, of the covariance matrix's eigenvalues normalized
    to sum to 1 - the "effective rank" of Roy & Vetterli 2007) AND the
    participation ratio (`(sum(eigenvalues))^2 / sum(eigenvalues^2)`, the
    simpler/more standard "effective dimensionality" measure used in
    neuroscience/dynamical-systems contexts - both computed since the
    request was unsure which is preferable). Both reduce to exactly 1 for
    a rank-1 covariance (all variance on one axis) and to exactly d for a
    covariance proportional to the d x d identity (variance spread
    perfectly evenly) - the two measures only diverge in HOW they
    penalize a spectrum that's neither of those extremes (entropy's log
    weighting penalizes small-but-nonzero eigenvalues more gently than
    participation ratio's sum-of-squares does).

    Eigenvalues are clipped to >= 0 before either computation - a
    symmetric empirical covariance matrix is PSD in theory, but np.linalg
    can return tiny negative eigenvalues (~1e-12) from floating-point
    error, which would otherwise make log()/squaring nonsensical."""
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = eigenvalues.sum()
    if total <= 0:
        return {"entropy_effective_rank": 0.0, "participation_ratio": 0.0}

    probs = eigenvalues[eigenvalues > 0] / total
    entropy = float(-np.sum(probs * np.log(probs)))
    entropy_effective_rank = float(np.exp(entropy))

    participation_ratio = float(total ** 2 / np.sum(eigenvalues ** 2))

    return {
        "entropy_effective_rank": entropy_effective_rank,
        "participation_ratio": participation_ratio,
    }


def effective_dimensionality(spec, g: int = MARGIN_G, samples: int = 200_000,
                               seed: int = 0) -> dict:
    """Effective dimensionality of the group's observation distribution,
    via the eigenvalue spectrum of its (Monte Carlo estimated) covariance
    matrix - see _effective_dim_from_covariance for the two measures
    returned (`entropy_effective_rank`, `participation_ratio`). This is a
    genuinely different notion of "complexity" than proxy_dimension_size
    (the RAW coordinate count, g or g*n) - a group can have a large raw
    dimension but a much smaller EFFECTIVE dimension if its coordinates
    are highly correlated (e.g. HeatmapGroup's n columns are all powers
    of the same underlying per-row signal, so they're far from
    independent)."""
    spec = tuple(spec)
    if spec[0] == "margin":
        k, s = spec[1], spec[2]
        x, _ = _margin_full_samples_key(k, s, g, samples, seed)
    elif spec[0] == "heatmap":
        noise_scale, n = spec[1], spec[2]
        x, _ = _heatmap_full_samples_key(noise_scale, n, g, samples, seed)
    else:
        raise ValueError(f"unknown spec {spec!r}")

    cov = np.cov(x, rowvar=False)
    return _effective_dim_from_covariance(np.atleast_2d(cov))
