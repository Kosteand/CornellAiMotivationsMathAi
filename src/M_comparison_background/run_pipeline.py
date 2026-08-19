"""Top-level pipeline: run a LIST OF COMPARISONS end to end and have all
relevant data land in the data files - the single-entry-point function
requested to replace hand-rolled per-sweep scripts.

Each entry in `comparisons` is a 3-tuple:
    (fixed_group, variable_group, hyperparameters)
where `fixed_group`/`variable_group` are live group instances (see
groups.py - MarginGroup/HeatmapGroup/AlternatingGroup, or any
other ComplexityGroupBase subclass), and `hyperparameters` is an
optional dict of PPO/model-size hyperparameters for training THAT PAIR's
two groups - pass `None` (or `{}`) to use DEFAULT_HYPERPARAMETERS below
unchanged.

For every unique group appearing across all comparisons (deduplicated by
`group._cache_key` - so a group reused across multiple comparisons is
only ever trained once), this:
  1. computes its math-only metrics (expected magnitude, effective
     dimensionality, proxy dimension size, num options, p_proxy_correct)
     via the group's own ComplexityGroupBase methods (estimated/cached/
     overridden exactly as that class already resolves - see groups.py),
  2. trains it (a single-group PPO run via trainPPO.train()) to get its
     individual actor/critic weight norms, and
  3. upserts the resulting row into group_data.py's
     group_data.csv (File 1).

2026-08-19: this pipeline does NOT write anything into pair_data.py's
pair_data.csv (File 2) - it never has an M/beta/certified/status to
give it (this pipeline only trains individual groups and computes their
math-only metrics; finding M is find_indifference_reward.py/
certify_possible_M_value.py's job entirely, via run_indifference_batch.py
or the legacy indifference_data.csv migration). Writing a pair_data.csv
row here used to happen anyway, as a blank "M not known yet" scaffold -
that produced permanent M-less rows in pair_data.csv/predictive_data.csv
for any comparison run through THIS pipeline alone (e.g. the two example
entries in run_my_comparisons.py's COMPARISONS list) that never actually
went through the indifference search, since nothing ever came back to
fill them in. Per direct request, a pair_data.csv row for a comparison
should only ever appear once that comparison has actually produced an M
value - so this pipeline now leaves pair_data.csv alone entirely, and a
pair only shows up there (and in predictive_data.csv) once it's gone
through the real M-finding pipeline.

WEIGHT-DIFFERENCE-NORM (the diff_* columns, computed by
run_shared_machinery_experiment.py's two-phase reward-switch protocol)
is explicitly OUT OF SCOPE for this pipeline for now, per direct
instruction ("wait on implementing the weight difference norm
calculation") - pair rows are written with those columns blank
(or carried forward from a prior run, never cleared), structurally ready
for a future call to fill them in. NOTE (per the same instruction): the
per-group training this pipeline already does for the "individual
training actor weight norm" is very likely reusable as PHASE 1 of that
future two-phase protocol (train group A alone, snapshot -> that's
exactly what run_shared_machinery_experiment.py's phase 1 already does)
rather than a second, separate training run - left as a documented
opportunity, not implemented here.

Usage - see run_my_comparisons.py for a ready-to-edit entry point that
calls this. Directly:

    from run_pipeline import run_comparisons
    from groups import MarginGroup, HeatmapGroup

    run_comparisons([
        (MarginGroup(g=4, k=1.0, value=1.0), MarginGroup(g=4, k=4.0, value=1.0), None),
        (HeatmapGroup(g=4, noise_scale=1.0, n=2, value=1.0),
         HeatmapGroup(g=4, noise_scale=1.0, n=6, value=1.0),
         {"total_timesteps": 500_000, "net_arch_pi": (128, 64), "net_arch_vf": (128, 64)}),
    ])
"""
from __future__ import annotations

import os
import sys

# 2026-08-19: data/ moved back out of M_comparison_background/ into its
# own top-level src/data/ (sibling of this file's own M_comparison_background/
# folder), per direct request - unlike groups.py/trainPPO.py/etc., which
# stay in M_comparison_background/, "data" is no longer this file's own
# sibling, so it needs its own sys.path insert (this file's grandparent,
# i.e. src/) rather than relying on a caller having already put
# M_comparison_background on sys.path (that insert only ever helped
# because data/ used to live *inside* M_comparison_background/ - it
# doesn't anymore).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groups import ComplexityGroupBase
from data.group_data import (
    build_group_row, cache_key_str, get_group_row, upsert_group_row, weight_norms_from_model,
)

# The pipeline's own default "model hyperparameters and size" - used for
# any comparison whose `hyperparameters` entry is None/{} and any key it
# doesn't itself set. This is a real, editable dict (not just "whatever
# trainPPO.train() defaults to") so the project's standard training
# budget/model size is visible and adjustable in ONE place; anything not
# listed here still falls back to trainPPO.train()'s own defaults.
DEFAULT_HYPERPARAMETERS: dict = {
    "total_timesteps": 200_000,   # this project's normal per-run budget
    "n_envs": 8,
    "n_steps": 512,
    "net_arch_pi": (64, 32),      # actor network size
    "net_arch_vf": (64, 32),      # critic network size
    "progress_bar": False,
    "print_eval_summary": False,
}


def _resolved_hyperparameters(hyperparameters: dict | None) -> dict:
    """DEFAULT_HYPERPARAMETERS, overridden per-key by whatever the caller
    actually supplied for this comparison (None/{} changes nothing)."""
    merged = dict(DEFAULT_HYPERPARAMETERS)
    merged.update(hyperparameters or {})
    return merged


def _dedupe_groups(comparisons):
    """Every distinct group instance across all comparisons, deduplicated
    by cache_key (first occurrence - and the hyperparameters it arrived
    with - wins). If the same spec appears in more than one comparison
    tuple, only the FIRST tuple's group instance and hyperparameters are
    actually used for training - this matches the "no variance, save the
    results" intent: two instances with the same params are supposed to
    be interchangeable, so training them twice with two different
    hyperparameter sets would be a contradiction, not a feature."""
    seen: dict[str, tuple["ComplexityGroupBase", dict]] = {}
    for fixed, variable, hyperparameters in comparisons:
        for group in (fixed, variable):
            key = cache_key_str(group._cache_key)
            seen.setdefault(key, (group, hyperparameters))
    return seen


def ensure_group_trained(group, hyperparameters=None, force_retrain=False) -> dict:
    """Make sure `group` has a row in group_data.csv with BOTH its math
    metrics and its training-derived weight norms populated, training it
    exactly once if (and only if) no such row already exists (or
    `force_retrain=True`). Returns the row that ended up stored.

    A pre-existing row missing only the training columns (e.g. written
    by a math-only caller before this pipeline touched that spec) is
    topped up with a fresh training run rather than left half-populated -
    "has this exact config already been trained" is judged by
    `wn_actor` being non-blank, not merely by the row existing at all."""
    key = group._cache_key
    existing = get_group_row(key)
    already_trained = existing is not None and existing.get("wn_actor", "") != ""

    if already_trained and not force_retrain:
        return existing

    import trainPPO

    resolved = dict(_resolved_hyperparameters(hyperparameters))
    label = resolved.pop("label", f"group_{cache_key_str(key)}")
    result = trainPPO.train(groups=[group], label=label, **resolved)
    norms = weight_norms_from_model(result.model)

    row = build_group_row(
        group, training_result=result, weight_norms=norms,
    )
    upsert_group_row(row)
    return row


def run_comparisons(comparisons, force_retrain=False) -> list[dict]:
    """Run the full pipeline over `comparisons` - a list of
    (fixed_group, variable_group, hyperparameters) 3-tuples:

        fixed_group, variable_group:
            Live ComplexityGroupBase instances (build them however you
            like - MarginGroup(...)/HeatmapGroup(...)/AlternatingGroup(...)
            from groups directly, or
            run_indifference_batch.mixed_group_factory for the old
            spec-tuple convention).
        hyperparameters:
            Optional dict of PPO hyperparameters/model size overrides
            (passed to trainPPO.train()) for training THIS comparison's
            two groups. None (or {}) uses DEFAULT_HYPERPARAMETERS above
            unchanged; any keys given here override just those keys.

    `force_retrain`: if True, retrain every group even if group_data.csv
    already has a fully-populated row for its exact spec (default False -
    "no variance" via the existing row is the whole point otherwise).

    Returns the list of group_data.csv rows (dicts) for every unique
    group touched, in dedup order. Populates group_data.csv (File 1) only
    - see this function's own docstring/the module docstring for why
    pair_data.csv (File 2) is deliberately left untouched here: this
    pipeline never has an M value to give it, so writing a row here
    would only ever be a permanent M-less scaffold. Run this comparison
    through find_indifference_reward.py/run_indifference_batch.py (or
    let the legacy indifference_data.csv migration pick it up) to get it
    an actual pair_data.csv row; call
    build_predictive_data.build_predictive_data() afterwards to
    (re)generate predictive_data.csv (File 3) from the updated inputs."""
    unique_groups = _dedupe_groups(comparisons)

    group_rows = []
    for group, hyperparameters in unique_groups.values():
        row = ensure_group_trained(
            group, hyperparameters=hyperparameters, force_retrain=force_retrain,
        )
        group_rows.append(row)

    return group_rows


if __name__ == "__main__":
    from groups import MarginGroup

    run_comparisons([
        (MarginGroup(g=4, k=1.0, value=1.0), MarginGroup(g=4, k=4.0, value=1.0), None),
    ])
