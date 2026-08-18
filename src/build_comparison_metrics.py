"""Build a full metrics table for every fixed/variable comparison in
run_p_curve_experiments.py's TESTS - one row per (test, side), covering
every metric in the project's "shared machinery" metric checklist:

    proxy magnitude       - E[avg], E[Linfty]("max"), E[L1], E[L2],
                             E[RMS], E[std]  (Utilities/weight_norm_data.
                             py's expected_magnitude - already existed)
    proxy dimension size   - g (margin) / g*n (heatmap)   (ADDED - see
                             Utilities/weight_norm_data.py's
                             proxy_dimension_size)
    number of options      - g                            (ADDED - see
                             num_options; a constant 4 everywhere in this
                             project, but now an explicit column instead
                             of an implicit assumption)
    P(proxy -> correct)    - 1-err (margin, exact) / Monte Carlo row-sum
                             heuristic (heatmap)           (ADDED - see
                             p_proxy_correct)
    effective dimensionality - Shannon-entropy effective rank AND
                             participation ratio of the observation's
                             covariance spectrum             (ADDED - see
                             effective_dimensionality)
    actor weight norm       - weight_norm_actor, LOOKED UP from
                             weight_norm_data.csv by exact spec match.
                             NOT computed here - it requires an actual
                             trained model, and (see this script's
                             printed "MISSING actor weight norm" notes)
                             most of these TESTS' exact specs have never
                             been through one of the weight-norm sweeps,
                             so this will legitimately be blank for most
                             rows. This script does not train anything to
                             fill that gap - see
                             run_weight_diff_norm_replicates.py for the
                             one metric in the checklist that DOES need a
                             fresh training run (weight-difference norm).

For every metric, a RATIO column (variable / fixed) is also written -
this is what "(all of the above as ratios)" in the request refers to.

`value` (the reward) is irrelevant to every metric here except
weight_norm_actor's presence check - all these group properties are
independent of value (see each metric's docstring) - so every group is
built with value=1.0 purely to get an instantiated object to read specs
off of.

Output: eval_logs/comparison_metrics.csv, one row per (test, side).
"""
import csv
import os

import numpy as np

from run_p_curve_experiments import TESTS
from Utilities.weight_norm_data import (
    spec_from_group, proxy_dimension_size, num_options, p_proxy_correct,
    effective_dimensionality, expected_magnitude, get_record,
    MARGIN_G, _margin_full_samples_key,
)

OUT_PATH = "eval_logs/comparison_metrics.csv"

# Magnitude sub-keys, exactly as returned by expected_magnitude - user's
# checklist names in parentheses.
MAGNITUDE_KEYS = ["avg", "max", "l1", "l2", "rms", "std"]  # max == E[L_infty]

FIELDNAMES = [
    "test", "side", "group_type", "spec", "g",
] + [f"magnitude_{k}" for k in MAGNITUDE_KEYS] + [
    "proxy_dimension_size", "num_options", "p_proxy_correct",
    "entropy_effective_rank", "participation_ratio",
    "weight_norm_actor", "weight_norm_actor_available",
]

RATIO_FIELDNAMES = ["test"] + [f"magnitude_{k}_ratio" for k in MAGNITUDE_KEYS] + [
    "proxy_dimension_size_ratio", "num_options_ratio", "p_proxy_correct_ratio",
    "entropy_effective_rank_ratio", "participation_ratio_ratio",
    "weight_norm_actor_ratio", "weight_norm_actor_ratio_available",
]


def _magnitude_for_group(spec, g):
    """Like expected_magnitude, but actually correct for margin specs
    whose g != MARGIN_G (e.g. margin_vs_margin_more_options's g=8 side).

    expected_magnitude's margin path calls `_margin_magnitude`, whose
    E[avg]/E[max]/E[sum] closed forms are explicitly derived "for g=4"
    (see that function's docstring) and don't take a g argument at all -
    they'd be silently WRONG (not just imprecise) for a different g,
    since the formulas bake in "3 backdrop draws" specifically. heatmap's
    underlying MC functions DO already accept a g override correctly (see
    _heatmap_magnitude_mc's signature), so only margin needs a bypass
    here.

    For margin with g == MARGIN_G, this defers to expected_magnitude
    unchanged (same well-tested closed-form/MC hybrid, no behavior
    change). For margin with g != MARGIN_G, it computes every one of the
    six magnitude stats directly via Monte Carlo from
    _margin_full_samples_key (which IS g-general - it simulates
    MarginGroup.sample's actual process for any g), bypassing the g=4-only
    closed form entirely rather than returning a wrong number."""
    if spec[0] == "heatmap":
        return expected_magnitude(spec, g=g)
    if spec[0] != "margin":
        raise ValueError(f"unknown spec {spec!r}")
    if g == MARGIN_G:
        return expected_magnitude(spec)

    k, s = spec[1], spec[2]
    x, _ = _margin_full_samples_key(k, s, g, 200_000, 0)
    # Every entry is >= 0 by construction (backdrop ~ Uniform[0, s),
    # correct = max(backdrop) + delta >= 0) - same as
    # _margin_extra_stats_mc's g=4 case - so l1 == sum exactly here too.
    sq = x ** 2
    sum_sq = sq.sum(axis=1)
    return {
        "avg": float(x.mean(axis=1).mean()),
        "max": float(x.max(axis=1).mean()),
        "l1": float(np.abs(x).sum(axis=1).mean()),
        "l2": float(np.sqrt(sum_sq).mean()),
        "rms": float(np.sqrt(sum_sq / g).mean()),
        "std": float(x.std(axis=1).mean()),
    }


def _row_for_side(test_name, side, group):
    spec = spec_from_group(group)
    g = group.g  # NOT MARGIN_G's default - see _magnitude_for_group.

    magnitude = _magnitude_for_group(spec, g)
    ed = effective_dimensionality(spec, g=g)

    # weight_norm_data.csv's spec tuples (and every function in that
    # module that defaults g=MARGIN_G) assume g=4 for EVERY row in this
    # project - there is no g dimension in the stored schema at all. That
    # assumption breaks for margin_vs_margin_more_options's g=8 side: its
    # spec tuple is otherwise IDENTICAL to margin_vs_margin_baseline's
    # g=4 spec (same k/s/err), so a naive get_record(spec) lookup would
    # silently return the g=4 run's actor weight norm as if it were the
    # g=8 run's - wrong network, wrong result. Guard against that
    # explicitly: only trust a CSV lookup when this group's actual g
    # matches the constant every stored row assumes.
    if g == MARGIN_G:
        record = get_record(spec)
        wn_actor = record.weight_norm_actor if record is not None else None
    else:
        wn_actor = None

    row = {
        "test": test_name,
        "side": side,
        "group_type": spec[0],
        "spec": spec,
        "g": g,
        "proxy_dimension_size": proxy_dimension_size(spec, g=g),
        "num_options": g,
        "p_proxy_correct": p_proxy_correct(spec, g=g),
        "entropy_effective_rank": ed["entropy_effective_rank"],
        "participation_ratio": ed["participation_ratio"],
        "weight_norm_actor": wn_actor if wn_actor is not None else "",
        "weight_norm_actor_available": wn_actor is not None,
    }
    for k in MAGNITUDE_KEYS:
        row[f"magnitude_{k}"] = magnitude[k]
    return row


def _ratio_row(test_name, fixed_row, variable_row):
    def ratio(key):
        f_val, v_val = fixed_row[key], variable_row[key]
        if f_val in (None, "") or v_val in (None, ""):
            return ""
        if f_val == 0:
            return ""  # avoid division by zero - blank, not a misleading 0/inf
        return v_val / f_val

    row = {"test": test_name}
    for k in MAGNITUDE_KEYS:
        row[f"magnitude_{k}_ratio"] = ratio(f"magnitude_{k}")
    row["proxy_dimension_size_ratio"] = ratio("proxy_dimension_size")
    row["num_options_ratio"] = ratio("num_options")
    row["p_proxy_correct_ratio"] = ratio("p_proxy_correct")
    row["entropy_effective_rank_ratio"] = ratio("entropy_effective_rank")
    row["participation_ratio_ratio"] = ratio("participation_ratio")

    both_available = fixed_row["weight_norm_actor_available"] and variable_row["weight_norm_actor_available"]
    row["weight_norm_actor_ratio_available"] = both_available
    row["weight_norm_actor_ratio"] = (
        ratio("weight_norm_actor") if both_available else ""
    )
    return row


def build():
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    per_side_rows = []
    ratio_rows = []
    missing_actor_norm = []

    for test in TESTS:
        name = test["name"]
        fixed_group = test["fixed_group"](1.0)
        variable_group = test["variable_group"](1.0)

        fixed_row = _row_for_side(name, "fixed", fixed_group)
        variable_row = _row_for_side(name, "variable", variable_group)
        per_side_rows.append(fixed_row)
        per_side_rows.append(variable_row)
        ratio_rows.append(_ratio_row(name, fixed_row, variable_row))

        for side, row in (("fixed", fixed_row), ("variable", variable_row)):
            if not row["weight_norm_actor_available"]:
                missing_actor_norm.append(f"{name} ({side}): spec={row['spec']}")

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(per_side_rows)

    ratio_path = OUT_PATH.replace(".csv", "_ratios.csv")
    with open(ratio_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RATIO_FIELDNAMES)
        writer.writeheader()
        writer.writerows(ratio_rows)

    print(f"Wrote {len(per_side_rows)} rows -> {OUT_PATH}")
    print(f"Wrote {len(ratio_rows)} rows -> {ratio_path}")
    if missing_actor_norm:
        print(
            f"\nweight_norm_actor NOT available (spec never appears in "
            f"weight_norm_data.csv) for {len(missing_actor_norm)}/"
            f"{len(per_side_rows)} sides - these specs would need a "
            f"dedicated training run (e.g. via run_per_layer_weight_norm_"
            f"rerun.py's pattern, or Utilities/weight_norm_data.py's "
            f"append_record) before this column can be filled in:"
        )
        for line in missing_actor_norm:
            print(f"  - {line}")


if __name__ == "__main__":
    build()
