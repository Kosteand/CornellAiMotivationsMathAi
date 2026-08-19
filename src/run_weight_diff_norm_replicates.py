"""Run the "shared machinery" weight-difference-norm check (see
run_shared_machinery_experiment.py) 20 times for EVERY comparison stored
in legacy_csv_data.py's indifference_data.csv - i.e. every
(spec_fixed, spec_variable) pair with a certified M ("midpoint") that
survived find_indifference_reward.py/run_indifference_batch.py's
confidence-interval-based estimation - 5 replicates with that pair's
fixed side trained first (phase 1) / variable side second, and 5 with the
order reversed, "just in case order matters" per the request - each of
those 10 runs done once with `phase2_inactive_reward=-1.0` (the OLD
group - whichever went first - is actively penalized with a reward of
-1, i.e. a penalty of 1, for still answering its own block once phase 2
starts). `phase1_inactive_reward` is always 0.0 - phase 1's "wrong
group" (whichever group hasn't gone first yet) is never penalized, only
ever left at 0 reward.

2026-08-19: PHASE2_INACTIVE_REWARDS previously also included 0.0 and
-0.1 (see eval_logs/weight_diff_norm_replicates.csv, which already has
those 1160 rows) - this run only adds the -1.0 condition on top of that,
so PHASE2_INACTIVE_REWARDS below is now just (-1.0,) rather than
re-running the two already-covered reward levels.

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
at g=4, the project-wide constant every spec in indifference_data.csv
implicitly assumes (specs never carry g themselves).

Output: eval_logs/weight_diff_norm_replicates.csv, one row per
(comparison, order, phase2_inactive_reward, replicate). `spec_fixed`/
`spec_variable`/`M`/`beta`/`certified`/`status`/`label`/`source_file` are
carried straight through from the indifference_data.csv row so this CSV
can be joined back to it by (spec_fixed, spec_variable) once you decide
how you want the two merged. group1/group2 columns record which group
was ACTUALLY trained first (phase1) each row - group1 = the fixed-spec
group when order="fixed_first", group1 = the variable-spec group when
order="variable_first" - so every row's group1/diff_* columns line up
consistently with "whichever group went first," not with the comparison's
own fixed/variable labeling. A separate `fixed_group_phase` column
("phase1"/"phase2") makes it easy to regroup by the comparison's own
spec_fixed/spec_variable semantics afterwards. `hit_rate_group1`/
`hit_rate_group2`/`choice_rate_group1`/`choice_rate_group2`/
`hit_rate_overall` are the post-run greedy-eval columns from
run_shared_machinery_experiment.py's `greedy_dual_evaluate()` - group2
here is always "whichever group trained SECOND" (see group1/group2 above),
so `hit_rate_group2` directly answers "did it switch to the new group, or
fail" for that row.

Run:  python3 run_weight_diff_norm_replicates.py [--quick] [--limit N]
"""
import argparse
import csv
import os
import sys

# 2026-08-19: data/ and run_indifference_batch.py moved into the
# self-contained M_comparison_background/ subfolder alongside the rest of
# the run_my_comparisons.py dependency chain. Inserting that folder onto
# sys.path keeps the imports below resolvable regardless of how this
# script ends up invoked.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "M_comparison_background"))

from data.legacy_csv_data import INDIFFERENCE_RECORDS
from run_indifference_batch import mixed_group_factory
from run_shared_machinery_experiment import (
    run_shared_machinery_experiment, WEIGHT_GROUP_PREDICATES, _group_spec_dict,
)

OUT_PATH = "eval_logs/weight_diff_norm_replicates.csv"

N_REPLICATES_PER_ORDER = 5   # -> 10 runs per comparison for this reward version
PHASE_TIMESTEPS = 200_000    # matches this project's normal per-run budget
BASE_SEED = 0
INCORRECT_REWARD = 0.0       # matches every comparison in this project's convention
PHASE1_INACTIVE_REWARD = 0.0  # always 0 - only phase 2's reward is varied
# 2026-08-19: per direct request, this run adds a THIRD reward version on
# top of the 0.0 and -0.1 versions already collected (see
# eval_logs/weight_diff_norm_replicates.csv) - the OLD group (whichever
# trained first) gets reward -1.0 (i.e. a penalty of 1) for still
# answering its own block once phase 2 starts. Only this one reward
# level is listed here so this run does 2 orders x N_REPLICATES_PER_ORDER
# per comparison (10 total runs/comparison at the default 5), NOT a
# re-run of the 0.0/-0.1 conditions.
PHASE2_INACTIVE_REWARDS = (-1.0,)

FIELDNAMES = [
    "spec_fixed", "spec_variable", "label", "M", "beta", "certified",
    "status", "source_file",
    "order", "replicate", "seed", "phase2_inactive_reward",
    "fixed_group_phase",  # "phase1" if the comparison's fixed side went first, else "phase2"
] + [f"group1_{k}" for k in ("type", "noise_scale", "n", "g", "value")] + [
    f"group2_{k}" for k in ("type", "noise_scale", "n", "g", "value")
] + [
    "incorrect_reward", "phase1_inactive_reward", "phase_timesteps",
    "switch_timestep", "end_timestep",
] + [f"diff_{g}" for g in WEIGHT_GROUP_PREDICATES] + [
    "hit_rate_overall", "hit_rate_group1", "hit_rate_group2",
    "choice_rate_group1", "choice_rate_group2", "eval_episodes",
]


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
            n_replicates_per_order=N_REPLICATES_PER_ORDER, n_envs=8, n_steps=512,
            phase2_inactive_rewards=PHASE2_INACTIVE_REWARDS, eval_episodes=500):
    records = INDIFFERENCE_RECORDS if records is None else records

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    file_is_new = not os.path.exists(OUT_PATH)
    f = open(OUT_PATH, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if file_is_new:
        writer.writeheader()

    total_runs = len(records) * n_replicates_per_order * 2 * len(phase2_inactive_rewards)
    run_num = 0

    for record in records:
        name = _comparison_label(record)

        for phase2_inactive_reward in phase2_inactive_rewards:
            for order in ("fixed_first", "variable_first"):
                for replicate in range(n_replicates_per_order):
                    run_num += 1
                    seed = BASE_SEED + replicate  # same seed reused across BOTH orders
                    # AND both phase2_inactive_reward versions, so replicate i
                    # differs from replicate i in any other (order,
                    # phase2_inactive_reward) combo ONLY in that one variable -
                    # isolates each dimension as the sole difference.

                    # Fresh group instances every replicate (mixed_group_
                    # factory builds a brand-new object each call, and
                    # run_shared_machinery_experiment mutates-then-restores
                    # .value anyway, but a fresh instance per run avoids any
                    # possibility of state leaking between runs).
                    fixed_group = mixed_group_factory(4, record.spec_fixed, 1.0)
                    variable_group = mixed_group_factory(4, record.spec_variable, 1.0)

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
                        phase1_inactive_reward=PHASE1_INACTIVE_REWARD,
                        phase2_inactive_reward=phase2_inactive_reward,
                        n_envs=n_envs,
                        phase_timesteps=phase_timesteps,
                        n_steps=n_steps,
                        label=f"wdn_{name}_{order}_pen{phase2_inactive_reward}_{replicate}",
                        seed=seed,
                        save_checkpoints=False,
                        progress_bar=False,
                        eval_episodes=eval_episodes,
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
                        "phase2_inactive_reward": phase2_inactive_reward,
                        "fixed_group_phase": fixed_group_phase,
                        "incorrect_reward": INCORRECT_REWARD,
                        "phase1_inactive_reward": PHASE1_INACTIVE_REWARD,
                        "phase_timesteps": phase_timesteps,
                        "switch_timestep": result.switch_timestep,
                        "end_timestep": result.end_timestep,
                    }
                    row.update(_spec_cols("group1", group1))
                    row.update(_spec_cols("group2", group2))
                    for gname in WEIGHT_GROUP_PREDICATES:
                        row[f"diff_{gname}"] = result.diffs[gname]
                    row["hit_rate_overall"] = result.hit_rates["hit_rate_overall"]
                    row["hit_rate_group1"] = result.hit_rates["hit_rate_group1"]
                    row["hit_rate_group2"] = result.hit_rates["hit_rate_group2"]
                    row["choice_rate_group1"] = result.hit_rates["choice_rate_group1"]
                    row["choice_rate_group2"] = result.hit_rates["choice_rate_group2"]
                    row["eval_episodes"] = result.hit_rates["episodes"]

                    writer.writerow(row)
                    f.flush()

                    print(
                        f"[{run_num}/{total_runs}] {name} order={order} "
                        f"phase2_inactive_reward={phase2_inactive_reward} rep={replicate} "
                        f"seed={seed}: diff_whole_model={result.diffs['whole_model']:.4f} "
                        f"diff_actor={result.diffs['actor']:.4f} "
                        f"diff_critic={result.diffs['critic']:.4f} "
                        f"hit_rate_group2={result.hit_rates['hit_rate_group2']:.3f} "
                        f"choice_rate_group2={result.hit_rates['choice_rate_group2']:.3f}",
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
