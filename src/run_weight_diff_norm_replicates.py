"""Run the "shared machinery" weight-difference-norm check (see
run_shared_machinery_experiment.py) 10 times for EVERY comparison stored
in Utilities/indifference_data.py's indifference_data.csv - i.e. every
(spec_fixed, spec_variable) pair with a certified M ("midpoint") that
survived find_indifference_reward.py/run_indifference_batch.py's
confidence-interval-based estimation - 5 replicates with that pair's
fixed side trained first (phase 1) / variable side second, and 5 with the
order reversed, "just in case order matters" per the request.

THIS IS NOT run_p_curve_experiments.py's TESTS list. TESTS is a small
(7-entry) illustrative set used for a completely different pipeline (the
GP-fit p(x)-vs-log(x) curves in sample_p_curve_adaptive.py /
plot_p_curve_results.py) and is NOT the source of truth for "which
comparisons exist" - indifference_data.csv is. Pulling comparisons from
TESTS would both miss most of what's actually in the stored data and
potentially include entries indifference_data.csv doesn't have (if TESTS
and the batch sweeps have ever drifted apart) - this script now reads the
comparisons list directly from indifference_data.csv's RECORDS instead,
so it's always exactly "every midpoint you did the confidence intervals
on," never a hand-maintained shadow copy of it.

Why order might matter: phase 1's group ends up trained "from scratch"
(random init), while phase 2's group is trained starting from whatever
phase 1 already converged to. If the two groups' difficulty/structure
differ, "group A learned first, then B retrained on top" is not
obviously symmetric with "B first, then A" - e.g. if A is much easier
than B, A-first might converge to a compact solution B's retraining barely
disturbs, while B-first might leave the network in a more complex state
that A's retraining barely disturbs instead. Running both orders (5 seeds
each, not just 1, since individual training runs are noisy - same
rationale run_p_curve_experiments.py uses for N_SEEDS_DENSE) is what lets
this be checked empirically instead of assumed away.

Both groups are given value=1.0 during their active phase - the actual
M/beta stored for a comparison is about finding the p(x) indifference
crossing, a different question than this experiment asks (which fully
separates the two groups' training into disjoint phases rather than
rewarding both simultaneously and tuning one's reward) - value=1.0 is
just a placeholder reward level to make each phase's active group
motivating; the group SPEC (not the reward value) is what this experiment
varies.

Reconstructing actual group objects from a stored spec tuple reuses
run_indifference_batch.py's own `mixed_group_factory` (not reimplemented
here) - the exact same ("margin", k, s, err) / ("heatmap", noise_scale,
n) -> Group dispatch already used to run comparisons in the first place -
at g=Utilities.weight_norm_data.MARGIN_G, the project-wide constant every
spec in indifference_data.csv implicitly assumes (specs never carry g
themselves - see Utilities/weight_norm_data.py's spec_from_group).

Output: eval_logs/weight_diff_norm_replicates.csv, one row per
(comparison, order, replicate). `spec_fixed`/`spec_variable`/`M`/`beta`/
`certified`/`status`/`label`/`source_file` are carried straight through
from the indifference_data.csv row so this CSV can be joined back to it
by (spec_fixed, spec_variable) once you decide how you want the two
merged. group1/group2 columns record which group was ACTUALLY trained
first (phase1) each row - group1 = the fixed-spec group when
order="fixed_first", group1 = the variable-spec group when
order="variable_first" - so every row's group1/diff_* columns line up
consistently with "whichever group went first," not with the comparison's
own fixed/variable labeling. A separate `fixed_group_phase` column
("phase1"/"phase2") makes it easy to regroup by the comparison's own
spec_fixed/spec_variable semantics afterwards.

Run:  python3 run_weight_diff_norm_replicates.py [--quick] [--limit N]
"""
import argparse
import csv
import os

from Utilities.indifference_data import RECORDS as INDIFFERENCE_RECORDS
from Utilities.weight_norm_data import MARGIN_G
from run_indifference_batch import mixed_group_factory
from run_shared_machinery_experiment import (
    run_shared_machinery_experiment, WEIGHT_GROUP_PREDICATES, _group_spec_dict,
)

OUT_PATH = "eval_logs/weight_diff_norm_replicates.csv"

N_REPLICATES_PER_ORDER = 5   # -> 10 total runs per comparison (5 + 5), per the request
PHASE_TIMESTEPS = 200_000    # matches this project's normal per-run budget
BASE_SEED = 0
INCORRECT_REWARD = 0.0       # matches every comparison in this project's convention

FIELDNAMES = [
    "spec_fixed", "spec_variable", "label", "M", "beta", "certified",
    "status", "source_file",
    "order", "replicate", "seed",
    "fixed_group_phase",  # "phase1" if the comparison's fixed side went first, else "phase2"
] + [f"group1_{k}" for k in ("type", "noise_scale", "n", "g", "value")] + [
    f"group2_{k}" for k in ("type", "noise_scale", "n", "g", "value")
] + [
    "incorrect_reward", "inactive_group_reward", "phase_timesteps",
    "switch_timestep", "end_timestep",
] + [f"diff_{g}" for g in WEIGHT_GROUP_PREDICATES]


def _spec_cols(prefix, group):
    raw = _group_spec_dict(prefix, group)
    return {
        f"{prefix}_type": raw[f"{prefix}_type"],
        f"{prefix}_noise_scale": raw[f"{prefix}_noise_scale"],
        f"{prefix}_n": raw[f"{prefix}_n"],
        f"{prefix}_g": raw[f"{prefix}_g"],
        f"{prefix}_value": raw[f"{prefix}_value"],
    }


def _comparison_label(record) -> str:
    """A human-readable identifier for one indifference_data.csv row -
    prefers its own `label` (populated for every row seen so far, e.g.
    "k_fixed1_k_variable2"), falling back to the raw spec pair for any
    row where label happens to be blank."""
    return record.label or f"{record.spec_fixed}__vs__{record.spec_variable}"


def run_all(records=None, phase_timesteps=PHASE_TIMESTEPS,
            n_replicates_per_order=N_REPLICATES_PER_ORDER, n_envs=8, n_steps=512):
    records = INDIFFERENCE_RECORDS if records is None else records

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    file_is_new = not os.path.exists(OUT_PATH)
    f = open(OUT_PATH, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if file_is_new:
        writer.writeheader()

    total_runs = len(records) * n_replicates_per_order * 2
    run_num = 0

    for record in records:
        name = _comparison_label(record)

        for order in ("fixed_first", "variable_first"):
            for replicate in range(n_replicates_per_order):
                run_num += 1
                seed = BASE_SEED + replicate  # same seed reused across BOTH orders,
                # so replicate i in "fixed_first" and replicate i in
                # "variable_first" differ ONLY in order, not in random
                # seed too - isolates order as the sole variable.

                # Fresh group instances every replicate (mixed_group_
                # factory builds a brand-new object each call, and
                # run_shared_machinery_experiment mutates-then-restores
                # .value anyway, but a fresh instance per run avoids any
                # possibility of state leaking between runs).
                fixed_group = mixed_group_factory(MARGIN_G, record.spec_fixed, 1.0)
                variable_group = mixed_group_factory(MARGIN_G, record.spec_variable, 1.0)

                if order == "fixed_first":
                    group1, group2 = fixed_group, variable_group
                    fixed_group_phase = "phase1"
                else:
                    group1, group2 = variable_group, fixed_group
                    fixed_group_phase = "phase2"

                result = run_shared_machinery_experiment(
                    group1,
                    group2,
                    incorrect_reward=INCORRECT_REWARD,
                    inactive_group_reward=0.0,
                    n_envs=n_envs,
                    phase_timesteps=phase_timesteps,
                    n_steps=n_steps,
                    label=f"wdn_{name}_{order}_{replicate}",
                    seed=seed,
                    save_checkpoints=False,
                    progress_bar=False,
                )

                row = {
                    "spec_fixed": record.spec_fixed,
                    "spec_variable": record.spec_variable,
                    "label": record.label,
                    "M": record.M,
                    "beta": record.beta,
                    "certified": record.certified,
                    "status": record.status,
                    "source_file": record.source_file,
                    "order": order,
                    "replicate": replicate,
                    "seed": seed,
                    "fixed_group_phase": fixed_group_phase,
                    "incorrect_reward": INCORRECT_REWARD,
                    "inactive_group_reward": 0.0,
                    "phase_timesteps": phase_timesteps,
                    "switch_timestep": result.switch_timestep,
                    "end_timestep": result.end_timestep,
                }
                row.update(_spec_cols("group1", group1))
                row.update(_spec_cols("group2", group2))
                for gname in WEIGHT_GROUP_PREDICATES:
                    row[f"diff_{gname}"] = result.diffs[gname]

                writer.writerow(row)
                f.flush()

                print(
                    f"[{run_num}/{total_runs}] {name} order={order} rep={replicate} "
                    f"seed={seed}: diff_whole_model={result.diffs['whole_model']:.4f} "
                    f"diff_actor={result.diffs['actor']:.4f} "
                    f"diff_critic={result.diffs['critic']:.4f}",
                    flush=True,  # stdout is FULLY buffered (not line-buffered) once
                    # redirected to a file/pipe instead of a terminal (e.g.
                    # `nohup ... > log.txt &`) - without this, `tail -f` on
                    # that log shows nothing for a long time even though
                    # runs are completing normally, since Python is just
                    # holding the printed lines in memory until the buffer
                    # fills. flush=True forces each line out immediately
                    # regardless of how the script is invoked.
                )

    f.close()
    print(f"\nWrote {total_runs} rows -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke-test settings (tiny phase_timesteps/n_envs/n_steps, "
             "1 replicate per order, first 2 comparisons only) instead of "
             "the real budget - verifies the whole script runs end to end "
             "without spending hours.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N comparisons from indifference_data.csv "
             "(in file order) instead of all of them. Useful for spot-"
             "checking a subset before committing to a full (potentially "
             "very long) run.",
    )
    args = parser.parse_args()

    if args.quick:
        run_all(
            records=INDIFFERENCE_RECORDS[:2],
            phase_timesteps=1000, n_replicates_per_order=1, n_envs=2, n_steps=64,
        )
    else:
        records = INDIFFERENCE_RECORDS[: args.limit] if args.limit else None
        run_all(records=records)
