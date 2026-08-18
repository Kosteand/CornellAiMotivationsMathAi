"""One-off build script: combines the four raw weight-norm-sweep CSVs into
the unified schema and writes weight_norm_data.csv. Run once now to produce
the unified file from this turn's uploads; future sweeps should append via
Utilities.weight_norm_data.append_record() instead of re-running this.
"""
import pandas as pd

UNIFIED_FIELDS = [
    "group_type", "k", "s", "delta", "noise_scale", "n", "weight_decay",
    "hit_rate", "correct", "episodes", "mean_reward",
    "wn_policy_net_0", "wn_policy_net_2", "wn_action_net", "weight_norm_actor",
    "wn_value_net_0", "wn_value_net_2", "wn_value_net_out", "weight_norm_critic",
    "weight_norm_total",
    "fit_status", "fit_L", "fit_A", "fit_tau",
    "fit_L_err", "fit_A_err", "fit_tau_err", "r_squared", "rmse", "fit_points_used",
    "source_file",
]

base = "/root/.claude/uploads/e8c7ffc3-a8bd-58e9-9b17-8bf54f626002"
per_layer = pd.read_csv(f"{base}/f4e5bed3-per_layer_weight_norm_rerun_7.csv")
small_s = pd.read_csv(f"{base}/bafd8c9a-margin_scaled_weight_norm_sweep_small_s_11.csv")
mid_s = pd.read_csv(f"{base}/428c5ec0-margin_scaled_weight_norm_sweep_mid_s_12.csv")
big_s = pd.read_csv(f"{base}/3a8bb6ab-margin_scaled_weight_norm_sweep_9.csv")

rows = []

# --- per_layer_weight_norm_rerun_7.csv: already has group_type, splits into
# margin (k only) and heatmap (noise_scale, n only) rows.
for _, r in per_layer.iterrows():
    row = {f: "" for f in UNIFIED_FIELDS}
    row["source_file"] = "per_layer_weight_norm_rerun_7.csv"
    if r["group_type"] == "margin":
        row["group_type"] = "margin"
        row["k"] = int(r["k"])
        row["s"] = 1.0
        row["delta"] = 1.0 / int(r["k"])
    elif r["group_type"] == "heatmap":
        row["group_type"] = "heatmap"
        row["noise_scale"] = float(r["noise_scale"])
        row["n"] = int(r["n"])
    else:
        raise ValueError(f"unexpected group_type {r['group_type']!r}")
    for f in ["weight_decay", "hit_rate", "correct", "episodes", "mean_reward",
              "wn_policy_net_0", "wn_policy_net_2", "wn_action_net", "weight_norm_actor",
              "wn_value_net_0", "wn_value_net_2", "wn_value_net_out", "weight_norm_critic",
              "weight_norm_total", "fit_status", "fit_L", "fit_A", "fit_tau",
              "fit_L_err", "fit_A_err", "fit_tau_err", "r_squared", "rmse", "fit_points_used"]:
        row[f] = r[f]
    rows.append(row)

# --- the three margin_scaled sweep files (filenames predate the
# 2026-08-13 margin/margin_scaled rename and are kept as-is for
# provenance): all share the same raw schema (k, s, delta, weight_decay,
# hit_rate, ..., fit_points_used), just different s ranges. group_type is
# "margin" here - margin_scaled was never a separate group_type, just an
# unnecessary second name for MarginGroup at s != 1.0.
for df, source in [(small_s, "margin_scaled_weight_norm_sweep_small_s_11.csv"),
                    (mid_s, "margin_scaled_weight_norm_sweep_mid_s_12.csv"),
                    (big_s, "margin_scaled_weight_norm_sweep_9.csv")]:
    for _, r in df.iterrows():
        row = {f: "" for f in UNIFIED_FIELDS}
        row["source_file"] = source
        row["group_type"] = "margin"
        row["k"] = int(r["k"])
        row["s"] = float(r["s"])
        row["delta"] = float(r["delta"])
        for f in ["weight_decay", "hit_rate", "correct", "episodes", "mean_reward",
                  "wn_policy_net_0", "wn_policy_net_2", "wn_action_net", "weight_norm_actor",
                  "wn_value_net_0", "wn_value_net_2", "wn_value_net_out", "weight_norm_critic",
                  "weight_norm_total", "fit_status", "fit_L", "fit_A", "fit_tau",
                  "fit_L_err", "fit_A_err", "fit_tau_err", "r_squared", "rmse", "fit_points_used"]:
            row[f] = r[f]
        rows.append(row)

out = pd.DataFrame(rows, columns=UNIFIED_FIELDS)
out.to_csv("/home/claude/repo_test/weight_norm_data.csv", index=False)
print(f"wrote {len(out)} rows -> weight_norm_data.csv")
print(out["group_type"].value_counts())
