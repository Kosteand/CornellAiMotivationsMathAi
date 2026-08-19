"""CSV-parsing functions for this project's two LEGACY, historical data
files - weight_norm_data.csv (per-trained-config weight norms/hit rates/
tau-decay fits) and indifference_data.csv (per-pair M/beta/certified/
status from the indifference search). Both files predate, and are now
superseded by, the group_data.csv / pair_data.csv / predictive_data.csv
three-file architecture (see group_data.py, pair_data.py,
build_predictive_data.py) - this module exists only so the OLD, already-
committed data in those two CSVs stays readable/parseable from code
(e.g. pair_data.py's migrate_from_legacy_indifference_data pulls
INDIFFERENCE_RECORDS from here), without keeping two separate legacy
modules around, and without mixing legacy CSV-parsing into Utilities/
(which is reserved for the environment itself - see
Utilities/bandit_env.py).

This is a pure CSV-parsing/reading file - the math-metric functions the
old weight_norm_data.py module also used to define (expected_magnitude,
effective_dimensionality, p_proxy_correct, proxy_dimension_size,
num_options, spec_from_group, and their Monte Carlo helpers) are NOT
carried over here: every one of those is now a method on the group
instances themselves (see groups.py's ComplexityGroupBase -
expected_magnitude()/effective_dimensionality()/p_proxy_correct()/
num_options()/proxy_dimension_size(), plus the named single-metric
accessors like l2_norm()/entropy_effective_rank()/etc.) - there's no
reason to keep a second, spec-tuple-based copy of that logic. Likewise,
the old indifference_data.py's compute_ratios/all_ratios (which built
ratios by calling those now-removed functions) are not carried over -
that role is now build_predictive_data.py's, operating on group_data.csv/
pair_data.csv rather than on live spec-tuple lookups.

The MARGIN_G constant the old modules defined is also gone - every
MarginGroup/HeatmapGroup config in this project uses g=4, and the one
place that still matters (converting an old, bare/no-g spec tuple into a
g-inclusive cache_key - see pair_data.py's _old_spec_to_cache_key) just
uses the literal 4 directly.
"""
from __future__ import annotations

import ast
import csv
import os
from dataclasses import dataclass, asdict
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# weight_norm_data_old.csv - per-trained-config weight norms / hit rates /
# tau-decay fits, keyed by a single g-inclusive `cache_key` column (same
# convention as group_data.csv - see group_data.py's module docstring).
# See the historical Utilities/weight_norm_data.py (now removed) for the
# full original design notes; this keeps only the parsing/storage half.
# ============================================================================

WEIGHT_NORM_CSV_PATH = os.path.join(_HERE, "weight_norm_data_old.csv")

WEIGHT_NORM_FIELDNAMES = [
    "cache_key", "weight_decay",
    "hit_rate", "proxy_hit_rate", "correct", "proxy_correct", "episodes", "mean_reward",
    "wn_policy_net_0", "wn_policy_net_2", "wn_action_net", "wn_actor",
    "wn_value_net_0", "wn_value_net_2", "wn_value_net_out", "wn_critic",
    "wn_whole_model",
    "fit_status", "fit_L", "fit_A", "fit_tau",
    "fit_L_err", "fit_A_err", "fit_tau_err", "r_squared", "rmse", "fit_points_used",
]


def _f(x):
    """Parse a CSV field that may be '' (blank) into None, else float."""
    if x is None or x == "":
        return None
    try:
        import numpy as np
        if isinstance(x, float) and np.isnan(x):
            return None
    except ImportError:
        pass
    return float(x)


def _i(x):
    v = _f(x)
    return None if v is None else int(v)


@dataclass
class WeightNormRecord:
    cache_key: tuple          # g-inclusive cache_key, e.g. ("margin", 1.0, 1.0, 0.0, 4)
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
    wn_actor: Optional[float]
    wn_value_net_0: Optional[float]
    wn_value_net_2: Optional[float]
    wn_value_net_out: Optional[float]
    wn_critic: Optional[float]
    wn_whole_model: Optional[float]
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

    @property
    def spec(self):
        """The BARE (no-g) legacy spec tuple this record's cache_key
        corresponds to - i.e. `self.cache_key` with the trailing `g`
        element dropped. This is the form legacy_csv_data's own
        indifference-data spec tuples use, and what
        normalize_weight_norm_spec/get_weight_norm_record key on."""
        if self.cache_key[0] == "margin":
            _, k, s, err, _g = self.cache_key
            return ("margin", k, s, err)
        if self.cache_key[0] == "heatmap":
            _, noise_scale, n, _g = self.cache_key
            return ("heatmap", noise_scale, n)
        raise ValueError(f"unknown cache_key {self.cache_key!r}")

    @property
    def tau_diverged(self) -> bool:
        """Known curve_fit-divergence failure mode: fit_status='ok' but
        fit_tau is wildly wrong. Checks BOTH r_squared<=0.5 and fit_tau
        being physically impossible (must be positive, every genuine fit
        in this project stays under 1e6) - either condition flags
        divergence."""
        if self.r_squared is None or self.fit_tau is None:
            return True
        return self.r_squared <= 0.5 or self.fit_tau <= 0 or abs(self.fit_tau) > 1e6

    def to_dict(self) -> dict:
        return asdict(self)


def _weight_norm_row_to_record(row: dict) -> WeightNormRecord:
    cache_key = ast.literal_eval(row["cache_key"])

    hit_rate = _f(row.get("hit_rate"))
    correct = _i(row.get("correct"))
    proxy_hit_rate = _f(row.get("proxy_hit_rate"))
    proxy_correct = _i(row.get("proxy_correct"))

    return WeightNormRecord(
        cache_key=cache_key,
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
        wn_actor=_f(row.get("wn_actor")),
        wn_value_net_0=_f(row.get("wn_value_net_0")),
        wn_value_net_2=_f(row.get("wn_value_net_2")),
        wn_value_net_out=_f(row.get("wn_value_net_out")),
        wn_critic=_f(row.get("wn_critic")),
        wn_whole_model=_f(row.get("wn_whole_model")),
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
    )


def load_weight_norm_records(csv_path: str = WEIGHT_NORM_CSV_PATH) -> list[WeightNormRecord]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return [_weight_norm_row_to_record(row) for row in csv.DictReader(f)]


WEIGHT_NORM_RECORDS: list[WeightNormRecord] = load_weight_norm_records()


def _build_weight_norm_index(records):
    index = {}
    for rec in records:
        spec = rec.spec
        if spec in index:
            raise ValueError(f"duplicate spec {spec} in weight_norm_data_old.csv")
        index[spec] = rec
    return index


_WEIGHT_NORM_INDEX: dict = _build_weight_norm_index(WEIGHT_NORM_RECORDS)


def normalize_weight_norm_spec(spec) -> tuple:
    """Canonicalize a legacy margin spec to the full 4-tuple ("margin",
    k, s, err) - shorthand ("margin", k) / ("margin", k, s) accepted and
    normalized. heatmap specs pass through unchanged."""
    spec = tuple(spec)
    if spec[0] == "margin":
        k = spec[1]
        s = spec[2] if len(spec) > 2 else 1.0
        err = spec[3] if len(spec) > 3 else 0.0
        return ("margin", k, s, err)
    return spec


def get_weight_norm_record(spec) -> Optional[WeightNormRecord]:
    """Look up the WeightNormRecord for an exact legacy spec tuple, e.g.
    ("margin", 13, 1.0, 0.0), ("heatmap", 1.0, 3). Returns None if that
    exact config was never run (or isn't in this legacy file)."""
    return _WEIGHT_NORM_INDEX.get(normalize_weight_norm_spec(spec))


def weight_norm_records_to_dicts(records: Optional[list] = None) -> list[dict]:
    records = WEIGHT_NORM_RECORDS if records is None else records
    return [r.to_dict() for r in records]


def weight_norm_records_to_dataframe(records: Optional[list] = None):
    import pandas as pd
    return pd.DataFrame(weight_norm_records_to_dicts(records))


# ============================================================================
# indifference_data.csv - per-(spec_fixed, spec_variable) M/beta/
# certified/status from the indifference search. See the historical
# Utilities/indifference_data.py (now removed) for the full original
# design notes; this keeps only the parsing/storage half (plus
# parse_spec/is_excluded_spec, which are needed to build NEW batch
# summaries the same way, not just to read the old file).
# ============================================================================

INDIFFERENCE_CSV_PATH = os.path.join(_HERE, "indifference_data.csv")

INDIFFERENCE_FIELDNAMES = [
    "spec_fixed", "spec_variable", "M", "beta", "certified", "status",
    "label", "source_file",
]


def parse_spec(x):
    """Parse a k_fixed/k_variable cell from a run_indifference_batch.py
    summary CSV into a canonical spec tuple. Tagged cells look like
    "('margin', 13, 10.0, 0.0)" (parsed via ast.literal_eval); untagged
    cells are bare integers/floats from early batches that predate spec
    tagging, and are always plain (s=1.0, err=0.0) MarginGroup at that k.
    Shorter margin tuples are also accepted and normalized up to the
    full 4-tuple."""
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
    """Single source of truth for the hit-rate-based exclusion rule used
    when building indifference_data.csv - a spec (not a pair) is
    excluded if it falls outside the region known to train cleanly:
        margin(k, s=1.0):   k > 25
        margin(k, s=0.1):    k > 10
        margin(k, 0.1<s<1):  k > 15
        margin(k, s>10):     always excluded
        heatmap(noise_scale, n): n > 3 or noise_scale > 2.5
    Ignores err entirely - this rule only looks at (k, s)."""
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
        return self.spec_fixed[0] == "heatmap" or self.spec_variable[0] == "heatmap"

    @property
    def is_full_heatmap(self) -> bool:
        return self.spec_fixed[0] == "heatmap" and self.spec_variable[0] == "heatmap"

    @property
    def is_half_heatmap(self) -> bool:
        return self.is_heatmap and not self.is_full_heatmap

    def to_dict(self) -> dict:
        d = asdict(self)
        d["spec_fixed"] = repr(self.spec_fixed)
        d["spec_variable"] = repr(self.spec_variable)
        return d


def _indifference_row_to_record(row: dict) -> IndifferenceRecord:
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


def load_indifference_records(csv_path: str = INDIFFERENCE_CSV_PATH) -> list[IndifferenceRecord]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return [_indifference_row_to_record(row) for row in csv.DictReader(f)]


INDIFFERENCE_RECORDS: list[IndifferenceRecord] = load_indifference_records()


def append_indifference_batch_summary_csv(path: str, csv_path: str = INDIFFERENCE_CSV_PATH,
                                            source_file: Optional[str] = None) -> list[IndifferenceRecord]:
    """Parse a NEW run_indifference_batch.py summary CSV (columns
    k_fixed, k_variable, label, csv_path, M, beta, certified, status,
    error), apply is_excluded_spec to both sides of every pair, and
    append the surviving rows to indifference_data.csv + in-memory
    INDIFFERENCE_RECORDS. Returns the list of newly-added records."""
    import pandas as pd

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    source_file = source_file or os.path.basename(path)

    new_records = []
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDIFFERENCE_FIELDNAMES)
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
            rec = _indifference_row_to_record(row)
            new_records.append(rec)

    INDIFFERENCE_RECORDS.extend(new_records)
    return new_records


def indifference_records_to_dicts(records: Optional[list] = None) -> list[dict]:
    records = INDIFFERENCE_RECORDS if records is None else records
    return [r.to_dict() for r in records]


def indifference_records_to_dataframe(records: Optional[list] = None):
    import pandas as pd
    return pd.DataFrame(indifference_records_to_dicts(records))
