"""File 1 of the three-file data architecture: one row per unique GROUP
(identified by its `_cache_key` - see groups.py's ComplexityGroupBase),
holding both the math-only metrics (expected magnitude, effective
dimensionality, proxy dimension size, num options, p_proxy_correct - all
available for ANY spec, no training required) and the training-derived
metrics (actor/critic weight norms, hit rate, etc.) for whichever specs
have actually been trained via the pipeline (run_pipeline.py).

Row identity is the group's `_cache_key` tuple itself - e.g.
"('margin', 1.0, 1.0, 0.0, 4)" - stored as ONE column (`cache_key`)
rather than a separate column per possible parameter (k/s/err/
noise_scale/n/g), most of which would be blank for any given row since a
row is only ever one group type at a time. The tuple already names the
group type as its first element and carries every parameter needed to
reconstruct/identify the group, so there's nothing a wider, mostly-empty
column layout would add.

Backing file: group_data.csv (repo root). One row per unique group
config. Re-upserting the same cache key overwrites that row (see
`upsert_group_row`) rather than duplicating it.
"""
from __future__ import annotations

import csv
import os
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
GROUP_DATA_CSV_PATH = os.path.join(_HERE, "group_data.csv")

# Weight-norm sub-groupings, matching run_shared_machinery_experiment.py's
# WEIGHT_GROUP_PREDICATES exactly (this file only needs the single-
# snapshot norm, not the before/after diff that module computes).
WEIGHT_NORM_GROUP_NAMES = (
    "actor", "critic", "whole_model",
    "policy_net_0", "policy_net_2", "action_net",
    "value_net_0", "value_net_2", "value_net_out",
)

FIELDNAMES = [
    "cache_key",
    "em_avg", "em_max", "em_sum", "em_l1", "em_l2", "em_rms", "em_std",
    "entropy_effective_rank", "participation_ratio",
    "num_options", "proxy_dimension_size", "p_proxy_correct",
] + [f"wn_{name}" for name in WEIGHT_NORM_GROUP_NAMES] + [
    "weight_decay",
    "hit_rate", "proxy_hit_rate", "correct", "proxy_correct",
    "episodes", "mean_reward",
    "fit_status", "fit_L", "fit_A", "fit_tau",
    "fit_L_err", "fit_A_err", "fit_tau_err", "r_squared", "rmse", "fit_points_used",
]

# legacy_csv_data.WeightNormRecord now uses the exact same field names as
# group_data.csv's columns (wn_actor/wn_critic/wn_whole_model, etc. - see
# legacy_csv_data.py's WeightNormRecord, itself parsed from
# weight_norm_data_old.csv's cache_key-based schema), so no renaming is
# needed anymore - kept as an empty dict (rather than removed outright)
# so _legacy_record_columns' lookup logic doesn't need to change if a
# future rename is ever needed again.
_LEGACY_FIELD_RENAMES: dict = {}

# Every legacy WeightNormRecord field that has a home in group_data.csv -
# i.e. everything except `cache_key` (identity field, already handled
# separately via cache_key_str/group._cache_key).
_LEGACY_FIELDS_TO_MERGE = [
    "weight_decay", "hit_rate", "proxy_hit_rate", "correct", "proxy_correct",
    "episodes", "mean_reward",
    "wn_policy_net_0", "wn_policy_net_2", "wn_action_net", "wn_actor",
    "wn_value_net_0", "wn_value_net_2", "wn_value_net_out", "wn_critic",
    "wn_whole_model",
    "fit_status", "fit_L", "fit_A", "fit_tau",
    "fit_L_err", "fit_A_err", "fit_tau_err", "r_squared", "rmse", "fit_points_used",
]


def cache_key_str(cache_key: tuple) -> str:
    """Canonical string form of a `group._cache_key` tuple, used as this
    file's row identity/join key (e.g. "('margin', 1.0, 1.0, 0.0, 4)") -
    plain `str()` is enough since this module never needs to parse it
    back into a tuple, only compare it against other groups' own
    `str(_cache_key)`."""
    return str(tuple(cache_key))


def _math_metric_columns(group, samples=None, seed=None) -> dict:
    """Every math-only metric this group's ComplexityGroupBase methods
    expose, using whichever combination of estimation/global-cache/
    input-override each method already resolves internally - this
    function never duplicates that logic, it just calls the public
    methods and flattens their results into this file's columns."""
    mc_kwargs = {}
    if samples is not None:
        mc_kwargs["samples"] = samples
    if seed is not None:
        mc_kwargs["seed"] = seed

    em = group.expected_magnitude(**mc_kwargs)
    ed = group.effective_dimensionality(**mc_kwargs)
    return {
        "em_avg": em["avg"], "em_max": em["max"], "em_sum": em["sum"],
        "em_l1": em["l1"], "em_l2": em["l2"], "em_rms": em["rms"], "em_std": em["std"],
        "entropy_effective_rank": ed["entropy_effective_rank"],
        "participation_ratio": ed["participation_ratio"],
        "num_options": group.num_options(),
        "proxy_dimension_size": group.proxy_dimension_size(),
        "p_proxy_correct": group.p_proxy_correct(**mc_kwargs),
    }


def weight_norms_from_model(model) -> dict:
    """L2 norm (sqrt of sum-of-squares over every element of every
    matched parameter tensor) of a SINGLE trained model's weights, broken
    down by the same named-parameter-prefix groupings
    run_shared_machinery_experiment.py's WEIGHT_GROUP_PREDICATES uses for
    its before/after DIFF - this is the un-differenced version (just the
    raw snapshot's own norm), which is what "individual training actor
    weight norm" means for a single group's own trained policy.

    `model` is a stable_baselines3.PPO instance (e.g. `TrainResult.model`
    from trainPPO.train()). Returns {"wn_actor": ..., "wn_critic": ...,
    "wn_whole_model": ..., "wn_policy_net_0": ..., ...} - keys match
    FIELDNAMES' wn_* columns exactly."""
    import torch

    predicates = {
        "whole_model": lambda name: True,
        "actor": lambda name: (
            name.startswith("mlp_extractor.policy_net.") or name.startswith("action_net.")
        ),
        "critic": lambda name: (
            name.startswith("mlp_extractor.value_net.") or name.startswith("value_net.")
        ),
        "policy_net_0": lambda name: name.startswith("mlp_extractor.policy_net.0."),
        "policy_net_2": lambda name: name.startswith("mlp_extractor.policy_net.2."),
        "action_net": lambda name: name.startswith("action_net."),
        "value_net_0": lambda name: name.startswith("mlp_extractor.value_net.0."),
        "value_net_2": lambda name: name.startswith("mlp_extractor.value_net.2."),
        "value_net_out": lambda name: name.startswith("value_net."),
    }

    params = list(model.policy.named_parameters())
    out = {}
    for name_key, predicate in predicates.items():
        total = 0.0
        for pname, p in params:
            if predicate(pname):
                total += float(torch.sum(p.detach() ** 2))
        out[f"wn_{name_key}"] = total ** 0.5
    return out


def build_group_row(group, training_result=None, weight_norms=None,
                     samples=None, seed=None) -> dict:
    """Assemble one full File-1 row dict for `group` (a live
    ComplexityGroupBase instance). Math-metric columns are always
    computed (cheap after the first call, thanks to groups.py's global
    metric cache). Training-derived columns
    (wn_*/hit_rate/correct/episodes/mean_reward) are populated only if
    `training_result` (a trainPPO.TrainResult) and/or `weight_norms`
    (a dict from weight_norms_from_model) are supplied - left blank
    otherwise, so a math-only row can be written before any training has
    happened for that spec."""
    row = {f: "" for f in FIELDNAMES}
    row["cache_key"] = cache_key_str(group._cache_key)
    row.update(_math_metric_columns(group, samples=samples, seed=seed))

    if weight_norms is not None:
        row.update(weight_norms)
    if training_result is not None:
        row["hit_rate"] = training_result.hit_rate
        row["correct"] = training_result.correct
        row["episodes"] = training_result.episodes
        row["mean_reward"] = training_result.mean_reward
    return row


def _legacy_record_columns(record) -> dict:
    """Flatten one legacy_csv_data.WeightNormRecord's fields into this
    file's columns (renaming weight_norm_actor/critic/total via
    _LEGACY_FIELD_RENAMES), for merging into a group_data.csv row."""
    out = {}
    for field in _LEGACY_FIELDS_TO_MERGE:
        value = getattr(record, field)
        column = _LEGACY_FIELD_RENAMES.get(field, field)
        out[column] = "" if value is None else value
    return out


def migrate_group_data_from_legacy(g: int = 4, samples=None, seed=None,
                                     csv_path: str = GROUP_DATA_CSV_PATH) -> int:
    """One-time (re-runnable) migration: for every group spec referenced
    anywhere in the legacy indifference_data.csv (both spec_fixed and
    spec_variable, across every row - see legacy_csv_data.
    INDIFFERENCE_RECORDS), build a live group instance, compute its math
    metrics, look up its matching row in the legacy weight_norm_data_old.csv
    (see legacy_csv_data.get_weight_norm_record) by exact spec match, and
    upsert the combined row (math metrics + whatever legacy training/fit
    data was found, or blank training columns if that exact spec was
    never trained in the legacy file) into group_data.csv.

    Returns the number of unique specs processed. Safe to re-run - each
    spec is upserted by its cache_key, so re-running just re-writes the
    same rows."""
    from legacy_csv_data import INDIFFERENCE_RECORDS, get_weight_norm_record
    from pair_data import _old_spec_to_cache_key
    from run_indifference_batch import mixed_group_factory

    specs = {}
    for rec in INDIFFERENCE_RECORDS:
        for spec in (rec.spec_fixed, rec.spec_variable):
            specs[cache_key_str(_old_spec_to_cache_key(spec, g=g))] = spec

    n = 0
    for spec in specs.values():
        group = mixed_group_factory(g, spec, 1.0)
        row = build_group_row(group, samples=samples, seed=seed)

        legacy_record = get_weight_norm_record(spec)
        if legacy_record is not None:
            row.update(_legacy_record_columns(legacy_record))

        upsert_group_row(row, csv_path=csv_path)
        n += 1
    return n


def load_group_rows(csv_path: str = GROUP_DATA_CSV_PATH) -> list[dict]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _write_all(rows: list[dict], csv_path: str = GROUP_DATA_CSV_PATH) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows({f: row.get(f, "") for f in FIELDNAMES} for row in rows)


def get_group_row(cache_key: tuple, csv_path: str = GROUP_DATA_CSV_PATH) -> Optional[dict]:
    """Look up the stored row for an exact `group._cache_key`, or None if
    that spec has never been written to group_data.csv."""
    target = cache_key_str(cache_key)
    for row in load_group_rows(csv_path):
        if row["cache_key"] == target:
            return row
    return None


def upsert_group_row(row: dict, csv_path: str = GROUP_DATA_CSV_PATH) -> None:
    """Write `row` (as built by build_group_row) into group_data.csv,
    REPLACING any existing row with the same cache_key - upserting is
    the right default here since the pipeline is expected to call this
    every time it touches a spec, e.g. to fill in training columns after
    a math-only row was written first."""
    rows = load_group_rows(csv_path)
    rows = [r for r in rows if r["cache_key"] != row["cache_key"]]
    rows.append({f: row.get(f, "") for f in FIELDNAMES})
    _write_all(rows, csv_path)
