"""Build a full metrics table for every fixed/variable comparison in
run_p_curve_experiments.py's TESTS - one row per (test, side), covering
every metric in the project's "shared machinery" metric checklist:

    proxy magnitude         - E[avg], E[Linfty]("max"), E[L1], E[L2],
                              E[RMS], E[std]  (group.expected_magnitude(),
                              or the named accessors - see groups.py)
    proxy dimension size    - g (margin) / g*n (heatmap)
                              (group.proxy_dimension_size())
    number of options       - g (group.num_options())
    P(proxy -> correct)     - 1-err (margin, exact) / Monte Carlo row-sum
                              heuristic (heatmap)  (group.p_proxy_correct())
    effective dimensionality - Shannon-entropy effective rank AND
                              participation ratio of the observation's
                              covariance spectrum
                              (group.effective_dimensionality())
    actor weight norm        - wn_actor, LOOKED UP from group_data.csv by
                              exact cache_key match. NOT computed here -
                              it requires an actual trained model, and
                              (see this script's printed "MISSING actor
                              weight norm" notes) most of these TESTS'
                              exact specs have never been trained via
                              run_pipeline.py/run_my_comparisons.py, so
                              this will legitimately be blank for most
                              rows. This script does not train anything
                              to fill that gap - run run_my_comparisons.py
                              (or run_pipeline.run_comparisons) first if
                              you want wn_actor populated for these
                              specific TESTS entries.

For every metric, a RATIO column (variable / fixed) is also written -
this is what "(all of the above as ratios)" in the request refers to.
This script computes its OWN independent ratio table for run_p_curve_
experiments.py's TESTS specifically (rather than reading
predictive_data.csv - see build_predictive_data.py, which does the same
thing but for whatever's actually in pair_data.csv).

`value` (the reward) is irrelevant to every metric here except
wn_actor's presence check - all these group properties are independent
of value - so every group is built with value=1.0 purely to get an
instantiated object to read specs off of.

Every one of these metrics now comes directly off the group INSTANCE
(groups.py's ComplexityGroupBase methods) rather than off a spec-tuple
lookup keyed to an assumed, project-wide g - so there's no g-mismatch
risk the way there used to be when this script depended on the old,
g=4-only weight_norm_data.py formulas (margin_vs_margin_more_options's
g=8 side, for instance, is handled correctly with no special-casing).

Output: eval_logs/comparison_metrics.csv, one row per (test, side).
"""
import csv
import os
import sys

# 2026-08-19: this file moved from src/ into src/GPM/, and group_data.py
# (along with the rest of the run_my_comparisons.py dependency chain)
# moved from src/ into its own self-contained src/M_comparison_background/
# subfolder. Inserting that folder onto sys.path (computed from THIS
# FILE's own location) keeps `from data.group_data import ...` resolvable
# regardless of where this script is invoked from.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "M_comparison_background",
))

from run_p_curve_experiments import TESTS
from data.group_data import cache_key_str, get_group_row

# eval_logs/ stays at src/ top level (shared with the rest of the
# pipeline, not GPM-specific) - resolved relative to src/ (this file's
# parent's parent), not the current working directory.
OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_logs", "comparison_metrics.csv",
)

# Magnitude sub-keys, exactly as returned by expected_magnitude - user's
# checklist names in parentheses.
MAGNITUDE_KEYS = ["avg", "max", "l1", "l2", "rms", "std"]  # max == E[L_infty]

FIELDNAMES = [
    "test", "side", "cache_key", "g",
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


def _row_for_side(test_name, side, group):
    magnitude = group.expected_magnitude()
    ed = group.effective_dimensionality()

    stored = get_group_row(group._cache_key)
    wn_actor = None
    if stored is not None and stored.get("wn_actor", "") != "":
        wn_actor = float(stored["wn_actor"])

    row = {
        "test": test_name,
        "side": side,
        "cache_key": cache_key_str(group._cache_key),
        "g": group.g,
        "proxy_dimension_size": group.proxy_dimension_size(),
        "num_options": group.num_options(),
        "p_proxy_correct": group.p_proxy_correct(),
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
                missing_actor_norm.append(f"{name} ({side}): cache_key={row['cache_key']}")

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
            f"group_data.csv) for {len(missing_actor_norm)}/"
            f"{len(per_side_rows)} sides - these specs would need to be "
            f"trained via run_my_comparisons.py (or run_pipeline."
            f"run_comparisons) before this column can be filled in:"
        )
        for line in missing_actor_norm:
            print(f"  - {line}")


if __name__ == "__main__":
    build()
