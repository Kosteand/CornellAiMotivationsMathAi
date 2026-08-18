"""Unified store for run_indifference_batch.py results (M/beta per
fixed-vs-variable group-spec pair), filtered down to the region where
hit_rate is known-clean, plus the machinery to turn a stored (or new) pair
into the full set of ratios used for M-fitting (actor/layer2/action_net/
total weight_norm ratios, tau ratio, and avg/max/sum magnitude ratios) by
looking those up (or computing them) via Utilities.weight_norm_data.

Backing file: ``indifference_data.csv`` (repo root), one row per
(spec_fixed, spec_variable) comparison that SURVIVED the exclusion filter
below. Columns: spec_fixed, spec_variable, M, beta, certified, status,
label, source_file. spec_fixed/spec_variable are stored as their Python
repr (e.g. "('margin', 13, 10.0, 0.0)") and parsed back with
ast.literal_eval.

As of 2026-08-13, "margin_scaled" was renamed to plain "margin" - it was
always the same MarginGroup spec shape (k, s), just an unnecessary second
name for s != 1.0.

Also as of 2026-08-13, MarginGroup grew an ``err`` parameter (see
Utilities/bandit_env.py's MarginGroup docstring): the per-episode
probability that the REWARDED correct option is not the one the proxy
(argmax of the observation) points to. Every margin spec is now a 4-tuple
("margin", k, s, err); the plain/unscaled/error-free case is simply
s=1.0, err=0.0. Every margin spec recorded before ``err`` existed is
retroactively ``err=0.0`` (the underlying runs used the code path that IS
err=0.0, so this isn't an approximation). Shorthand ("margin", k) and
("margin", k, s) are still accepted by parse_spec and normalized to
err=0.0 (and s=1.0 for the 2-tuple case), but the canonical/stored form is
always the full 4-tuple.

EXCLUSION RULE (as of 2026-08-12, per direct instruction) - a pair is
dropped if EITHER side's spec is outside the region known to have clean
(hit_rate == 1.0, or very close) training:
    margin(k, s=1.0):   k > 25          (s=1.0 is the plain/unscaled case)
    margin(k, s=0.1):   k > 10
    margin(k, 0.1<s<1): k > 15
    margin(k, s>10):    always excluded
    heatmap(noise_scale, n):   n > 3 or noise_scale > 2.5
This rule does NOT (yet) have an err-based branch - err's effect on
hit_rate is an open question this project is actively measuring (see
run_err_weight_norm_sweep.py), not yet a characterized "clean region", so
is_excluded_spec ignores err entirely for now and only looks at (k, s).
See ``is_excluded_spec`` below for the exact, single-source-of-truth
implementation - apply it to any NEW batch summary before folding it in.

FUTURE BATCH SUMMARIES: call ``append_batch_summary_csv(path)`` to parse a
new run_indifference_batch.py summary CSV, apply the same exclusion rule,
and append the surviving rows to this module's CSV + in-memory RECORDS.
"""

from __future__ import annotations

import ast
import csv
import os
from dataclasses import dataclass, asdict
from typing import Optional

from Utilities import weight_norm_data as wnd

_HERE = os.path.dirname(os.path.abspath(__file__))
UNIFIED_CSV_PATH = os.path.normpath(os.path.join(_HERE, "..", "indifference_data.csv"))

FIELDNAMES = [
    "spec_fixed", "spec_variable", "M", "beta", "certified", "status",
    "label", "source_file",
]


def parse_spec(x):
    """Parse a k_fixed/k_variable cell from a run_indifference_batch.py
    summary CSV into a canonical spec tuple. Tagged cells look like
    "('margin', 13, 10.0, 0.0)" (parsed via ast.literal_eval); untagged
    cells are bare integers/floats from early batches that predate spec
    tagging, and are always plain (s=1.0, err=0.0) MarginGroup at that k.
    Shorter margin tuples - a 2-tuple ("margin", k) from before the
    2026-08-13 margin/margin_scaled unification, or a 3-tuple
    ("margin", k, s) from before ``err`` existed - are also accepted and
    normalized up to the full 4-tuple (s=1.0/err=0.0 filled in for
    whichever trailing element(s) are missing)."""
    if isinstance(x, tuple):
        spec = x
    else:
        s = str(x).strip()
        if s.startswith("("):
            spec = tuple(ast.literal_eval(s))
        else:
            return ("margin", int(float(s)), 1.0, 0.0)
    if spec[0] == "margin" and len(spec) < 4:
        k = spec[1]
        s_val = spec[2] if len(spec) > 2 else 1.0
        err = spec[3] if len(spec) > 3 else 0.0
        return ("margin", k, s_val, err)
    return spec


def is_excluded_spec(spec) -> bool:
    """Single source of truth for the hit-rate-based exclusion rule
    (2026-08-12). A spec (not a pair) is excluded if it falls outside the
    region known to train cleanly - see module docstring for the exact
    boundaries. NOTE (per direct instruction): scattered fit-divergence or
    hit_rate dips at specific, isolated (k, s) points elsewhere are random
    per-run bad luck, NOT evidence that a particular combo is
    deterministically risky - this rule only encodes the broad, confirmed
    regions, not point-by-point exclusions. Ignores err entirely (see
    module docstring) - it has no bearing on this (k, s)-only rule."""
    gt = spec[0]
    if gt == "margin":
        k = spec[1]
        s = spec[2] if len(spec) > 2 else 1.0
        if s == 1.0:
            return k > 25
        if s == 0.1:
            return k > 10
        if 0.1 < s < 1:
            return k > 15
        if s > 10:
            return True
        return False
    if gt == "heatmap":
        noise_scale, n = spec[1], spec[2]
        return n > 3 or noise_scale > 2.5
    raise ValueError(f"unknown group_type in spec {spec!r}")


@dataclass
class IndifferenceRecord:
    spec_fixed: tuple
    spec_variable: tuple
    M: float
    beta: Optional[float]
    certified: bool
    status: str
    label: str = ""
    source_file: str = ""

    @property
    def logM(self) -> float:
        import math
        return math.log(self.M)

    @property
    def is_heatmap(self) -> bool:
        """True if EITHER side is HeatmapGroup (covers both "half" pairs -
        one side heatmap, one side margin - and "full" pairs
        - both sides heatmap). Use is_full_heatmap/is_half_heatmap below to
        distinguish the two."""
        return self.spec_fixed[0] == "heatmap" or self.spec_variable[0] == "heatmap"

    @property
    def is_full_heatmap(self) -> bool:
        """True only when BOTH sides are HeatmapGroup (a pure
        heatmap<->heatmap comparison, e.g. testing noise_scale's effect in
        isolation)."""
        return self.spec_fixed[0] == "heatmap" and self.spec_variable[0] == "heatmap"

    @property
    def is_half_heatmap(self) -> bool:
        """True only when EXACTLY ONE side is HeatmapGroup (a cross-type
        margin <-> heatmap comparison)."""
        return self.is_heatmap and not self.is_full_heatmap

    def to_dict(self) -> dict:
        d = asdict(self)
        d["spec_fixed"] = repr(self.spec_fixed)
        d["spec_variable"] = repr(self.spec_variable)
        return d


def _row_to_record(row: dict) -> IndifferenceRecord:
    return IndifferenceRecord(
        spec_fixed=tuple(ast.literal_eval(row["spec_fixed"])),
        spec_variable=tuple(ast.literal_eval(row["spec_variable"])),
        M=float(row["M"]),
        beta=float(row["beta"]) if row.get("beta") not in (None, "") else None,
        certified=str(row.get("certified", "")).strip().lower() in ("true", "1"),
        status=row.get("status", ""),
        label=row.get("label", ""),
        source_file=row.get("source_file", ""),
    )


def load_records(csv_path: str = UNIFIED_CSV_PATH) -> list[IndifferenceRecord]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return [_row_to_record(row) for row in csv.DictReader(f)]


RECORDS: list[IndifferenceRecord] = load_records()


def append_batch_summary_csv(path: str, csv_path: str = UNIFIED_CSV_PATH,
                              source_file: Optional[str] = None) -> list[IndifferenceRecord]:
    """Parse a NEW run_indifference_batch.py summary CSV (columns
    k_fixed, k_variable, label, csv_path, M, beta, certified, status,
    error), apply is_excluded_spec to both sides of every pair, and append
    the surviving rows to indifference_data.csv + in-memory RECORDS.
    Returns the list of newly-added records."""
    import pandas as pd

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    source_file = source_file or os.path.basename(path)

    new_records = []
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for _, r in df.iterrows():
            spec_fixed = parse_spec(r["k_fixed"])
            spec_variable = parse_spec(r["k_variable"])
            if is_excluded_spec(spec_fixed) or is_excluded_spec(spec_variable):
                continue
            row = {
                "spec_fixed": repr(spec_fixed),
                "spec_variable": repr(spec_variable),
                "M": r["M"],
                "beta": r["beta"],
                "certified": r["certified"],
                "status": r["status"],
                "label": r.get("label", ""),
                "source_file": source_file,
            }
            writer.writerow(row)
            rec = _row_to_record(row)
            new_records.append(rec)

    RECORDS.extend(new_records)
    return new_records


# ----------------------------------------------------------------------
# Ratio construction - pulls weight_norm/tau from Utilities.weight_norm_data's
# lookup table, and avg/max/sum magnitude from its expected_magnitude()
# (closed-form for margin, Monte Carlo for heatmap).
# ----------------------------------------------------------------------

def compute_ratios(record: IndifferenceRecord, mc_kwargs: Optional[dict] = None) -> Optional[dict]:
    """Build the full ratio dict for one IndifferenceRecord:
    actor_ratio, layer2_ratio, action_net_ratio, total_ratio, tau_ratio,
    avg_ratio, max_ratio, sum_ratio, l1_ratio, l2_ratio, rms_ratio,
    std_ratio, M, logM, is_heatmap, is_full_heatmap,
    is_half_heatmap, plus the two
    specs. "ratio" always means variable / fixed (matching the
    variable-over-fixed convention used throughout this project's
    plot_M_fits data). l1_ratio/l2_ratio/rms_ratio/std_ratio (added
    2026-08-13, for plot_M_fits' proxy-magnitude-statistics check) come
    from Utilities.weight_norm_data.expected_magnitude()'s l1/l2/rms/std
    keys, the same way avg_ratio/max_ratio/sum_ratio already did.

    Magnitude ratios (avg/max/sum/l1/l2/rms/std) are ALWAYS computed - they
    only need the two specs themselves (expected_magnitude() works for any
    spec, trained or not), so they are NEVER None as long as
    record.spec_fixed/record.spec_variable are valid specs. Weight-norm/tau
    ratios (actor/layer2/action_net/total/tau) DO require an actual trained
    network on both sides; when either side's WeightNormRecord is missing
    (that exact spec was never trained in weight_norm_data.csv), those
    fields - and ONLY those fields - come back None; the row itself, and
    every magnitude field in it, is still returned. (Before 2026-08-13 this
    function returned None for the WHOLE row whenever weight-norm data was
    missing on either side, silently making magnitude ratios unavailable
    too even though nothing about computing them actually required
    training - that coupling was a bug relative to this module's own
    stated design and has been removed.)
    """
    mc_kwargs = mc_kwargs or {}
    fixed = wnd.get_record(record.spec_fixed)
    variable = wnd.get_record(record.spec_variable)

    mag_fixed = wnd.expected_magnitude(record.spec_fixed, **mc_kwargs)
    mag_variable = wnd.expected_magnitude(record.spec_variable, **mc_kwargs)

    def ratio(a, b):
        return None if (a is None or b is None or b == 0) else a / b

    if fixed is not None and variable is not None:
        actor_ratio = ratio(variable.weight_norm_actor, fixed.weight_norm_actor)
        layer2_ratio = ratio(variable.wn_policy_net_2, fixed.wn_policy_net_2)
        action_net_ratio = ratio(variable.wn_action_net, fixed.wn_action_net)
        total_ratio = ratio(variable.weight_norm_total, fixed.weight_norm_total)
        tau_ratio = ratio(variable.fit_tau, fixed.fit_tau)
        tau_diverged = fixed.tau_diverged or variable.tau_diverged
    else:
        # Weight-norm/tau data isn't available for (at least) one side -
        # those fields are None, but magnitude below is computed regardless.
        actor_ratio = layer2_ratio = action_net_ratio = total_ratio = tau_ratio = None
        tau_diverged = True  # "diverged" here really means "unavailable" - excluded by the same tau_diverged filter either way.

    return {
        "spec_fixed": record.spec_fixed,
        "spec_variable": record.spec_variable,
        "actor_ratio": actor_ratio,
        "layer2_ratio": layer2_ratio,
        "action_net_ratio": action_net_ratio,
        "total_ratio": total_ratio,
        "tau_ratio": tau_ratio,
        "tau_diverged": tau_diverged,
        "avg_ratio": ratio(mag_variable["avg"], mag_fixed["avg"]),
        "max_ratio": ratio(mag_variable["max"], mag_fixed["max"]),
        "sum_ratio": ratio(mag_variable["sum"], mag_fixed["sum"]),
        "l1_ratio": ratio(mag_variable["l1"], mag_fixed["l1"]),
        "l2_ratio": ratio(mag_variable["l2"], mag_fixed["l2"]),
        "rms_ratio": ratio(mag_variable["rms"], mag_fixed["rms"]),
        "std_ratio": ratio(mag_variable["std"], mag_fixed["std"]),
        "M": record.M,
        "logM": record.logM,
        "is_heatmap": record.is_heatmap,
        "is_full_heatmap": record.is_full_heatmap,
        "is_half_heatmap": record.is_half_heatmap,
        "label": record.label,
        "source_file": record.source_file,
    }


def all_ratios(records: Optional[list] = None, require_weight_norm: bool = True,
                mc_kwargs: Optional[dict] = None) -> list[dict]:
    """compute_ratios() for every stored record. compute_ratios() itself
    never returns None anymore (as of 2026-08-13 - see its docstring):
    magnitude ratios are always computable regardless of whether either
    side was actually trained. If require_weight_norm is True (default),
    rows whose weight-norm data ISN'T available on both sides (actor_ratio
    is None) are dropped - set to False to keep every row, including ones
    where only the magnitude ratios are populated and actor_ratio/
    layer2_ratio/action_net_ratio/total_ratio/tau_ratio are all None."""
    records = RECORDS if records is None else records
    out = [compute_ratios(r, mc_kwargs=mc_kwargs) for r in records]
    if require_weight_norm:
        out = [r for r in out if r is not None and r["actor_ratio"] is not None]
    return out


# ----------------------------------------------------------------------
# Format conversion helpers
# ----------------------------------------------------------------------

def to_dicts(records: Optional[list] = None) -> list[dict]:
    records = RECORDS if records is None else records
    return [r.to_dict() for r in records]


def to_dataframe(records: Optional[list] = None):
    import pandas as pd
    return pd.DataFrame(to_dicts(records))


def ratios_to_dataframe(records: Optional[list] = None, **kwargs):
    import pandas as pd
    return pd.DataFrame(all_ratios(records, **kwargs))
