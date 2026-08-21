"""Entry point for the comparison pipeline (run_pipeline.run_comparisons).

Edit COMPARISONS below and run:

    python3 run_my_comparisons.py

Each entry is a 3-tuple:
    (fixed_group, variable_group, hyperparameters)

- fixed_group / variable_group: live group instances from groups.py
  (MarginGroup, HeatmapGroup, AlternatingGroup - or any other
  ComplexityGroupBase subclass).
- hyperparameters: a dict of PPO hyperparameters / model size overrides
  (n_envs, total_timesteps, net_arch_pi, net_arch_vf, learning_rate,
  etc. - anything trainPPO.train() accepts), or None to use
  run_pipeline.DEFAULT_HYPERPARAMETERS unchanged.

Running this:
  1. migrates every group spec referenced in the legacy
     indifference_data.csv into group_data.csv, pulling that spec's
     weight-norm/hit-rate/tau-fit data from the legacy
     weight_norm_data_old.csv wherever a matching trained run exists
     (safe/idempotent to re-run - see group_data.migrate_group_data_
     from_legacy);
  2. migrates the legacy indifference_data.csv results (M/beta/certified/
     status) into pair_data.csv (safe/idempotent to re-run - see
     pair_data.migrate_from_legacy_indifference_data);
  3. trains any group in COMPARISONS that hasn't already been trained
     (group_data.csv is checked first - see
     run_pipeline.ensure_group_trained), writing/updating group_data.csv
     (File 1) and pair_data.csv (File 2);
  4. regenerates predictive_data.csv (File 3) from the two - every pair
     row plus a ratio column per group-level metric (magnitude stats,
     effective dimensionality, p_proxy_correct, weight norms, tau fit,
     etc.) for every pair whose both sides have a group_data.csv row.

If predictive_data.csv still comes out with a header but no rows, it's
because pair_data.csv has no rows yet - that would only happen if the
legacy migration also found nothing (i.e. indifference_data.csv is
missing/empty) AND COMPARISONS is empty.
"""
import os
import sys

# 2026-08-19: every file this script needs (groups.py, run_pipeline.py,
# trainPPO.py, Utilities/, run_indifference_batch.py,
# find_indifference_reward.py, and data/) was consolidated into its own
# self-contained M_comparison_background/ subfolder, leaving only the
# things NOT strictly required to run this script (old one-off sweeps,
# graph-only scripts, etc.) at src/ top level alongside this entry point.
# Inserting that folder onto sys.path (computed from THIS FILE's own
# location, not the current working directory) keeps the imports below
# resolvable regardless of how this script is invoked.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "M_comparison_background"))

from groups import MarginGroup, HeatmapGroup
from run_pipeline import run_comparisons
from data.group_data import migrate_group_data_from_legacy
from data.pair_data import migrate_from_legacy_indifference_data
from data.build_predictive_data import build_predictive_data

COMPARISONS = [
    # --- the 7 comparisons graphed via GPM/run_p_curve_experiments.py's
    # TESTS - added so every one of those 7 has its "standard data" (math
    # metrics + trained actor/critic weight norms) in group_data.csv, the
    # same way the two comparisons below already do. Only 3 of these 7
    # actually need new training (margin k=1/err=0.2, margin k=5/s=5.0,
    # margin k=1/g=8 have never been trained) - the rest dedupe against
    # specs already in group_data.csv and are included anyway so this
    # list is a complete, one-to-one mirror of TESTS.

    # 1. margin_vs_margin_baseline (fixed==variable, same spec both
    #    sides - sanity check test, crossing should land at x=1.0)
    (
        MarginGroup(g=4, k=1.0, value=1.0),
        MarginGroup(g=4, k=1.0, value=1.0),
        None,
    ),

    # 2. margin_vs_margin_harder_variable - already trained; kept as-is.
    (
        MarginGroup(g=4, k=1.0, value=1.0),
        MarginGroup(g=4, k=4.0, value=1.0),
        None,
    ),

    # 3. margin_vs_heatmap (cross-family comparison)
    (
        MarginGroup(g=4, k=1.0, value=1.0),
        HeatmapGroup(g=4, noise_scale=1.0, n=3, value=1.0),
        None,
    ),

    # 4. heatmap_vs_heatmap_harder_variable - already trained; kept as-is
    #    (custom bigger model + longer budget, per its original entry).
    (
        HeatmapGroup(g=4, noise_scale=1.0, n=2, value=1.0),
        HeatmapGroup(g=4, noise_scale=1.0, n=6, value=1.0),
        {
            "total_timesteps": 500_000,
            "net_arch_pi": (128, 64),
            "net_arch_vf": (128, 64),
        },
    ),

    # 5. margin_vs_margin_label_noise - variable side (err=0.2) is new,
    #    never trained.
    (
        MarginGroup(g=4, k=1.0, value=1.0, err=0.0),
        MarginGroup(g=4, k=1.0, value=1.0, err=0.2),
        None,
    ),

    # 6. margin_vs_margin_scale_mismatch - variable side (k=5, s=5.0) is
    #    new, never trained.
    (
        MarginGroup(g=4, k=1.0, value=1.0, s=1.0),
        MarginGroup(g=4, k=5.0, value=1.0, s=5.0),
        None,
    ),

    # 7. margin_vs_margin_more_options - variable side (g=8) is new,
    #    never trained; note this is the one spec here that ISN'T g=4,
    #    so it won't be usable for anything that assumes the project-wide
    #    g=4 convention (e.g. the weight-diff-norm replicate pipeline).
    (
        MarginGroup(g=4, k=1.0, value=1.0),
        MarginGroup(g=8, k=1.0, value=1.0),
        None,
    ),

    # Add more (fixed_group, variable_group, hyperparameters) tuples here.
]


if __name__ == "__main__":
    n_groups_migrated = migrate_group_data_from_legacy()
    print(f"migrated {n_groups_migrated} legacy group spec(s) -> group_data.csv")

    n_pairs_migrated = migrate_from_legacy_indifference_data()
    print(f"migrated {n_pairs_migrated} legacy indifference row(s) -> pair_data.csv")

    group_rows = run_comparisons(COMPARISONS)

    print(f"\n{len(group_rows)} unique group(s) processed:")
    for row in group_rows:
        print(
            f"  {row['cache_key']}: wn_actor={row['wn_actor']} "
            f"p_proxy_correct={row['p_proxy_correct']}"
        )

    n = build_predictive_data()
    print(f"\nrebuilt predictive_data.csv ({n} rows)")
