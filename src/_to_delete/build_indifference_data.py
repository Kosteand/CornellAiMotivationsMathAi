"""One-off build script: combines the batch_summary CSVs into
indifference_data.csv, excluding rows whose fixed or variable spec falls
outside the hit-rate-clean regions established this project (see
legacy_csv_data.is_excluded_spec for the exact rule). Future
batch_summary runs should be folded in via
legacy_csv_data.append_indifference_batch_summary_csv(path) instead of
re-running this script from scratch. NOTE: this legacy data's role is
now superseded by pair_data.csv (see pair_data.py's
migrate_from_legacy_indifference_data) - this script is kept only for
historical reference.
"""
import ast
import pandas as pd

import sys
sys.path.insert(0, ".")
from legacy_csv_data import is_excluded_spec, parse_spec, INDIFFERENCE_FIELDNAMES as FIELDNAMES

base = "/root/.claude/uploads/e8c7ffc3-a8bd-58e9-9b17-8bf54f626002"
FILES = [
    ("batch_summary_2.csv", f"{base}/3a4b5144-batch_summary_2.csv"),
    ("batch_summary_3.csv", f"{base}/2c4fb5a6-batch_summary_3.csv"),
    ("batch_summary_6.csv", f"{base}/15f10512-batch_summary_6.csv"),
    ("batch_summary_8.csv", f"{base}/c3355d08-batch_summary_8.csv"),
    ("batch_summary_10.csv", f"{base}/6a483210-batch_summary_10.csv"),
    ("batch_summary_13.csv", f"{base}/7c0ab73c-batch_summary_13.csv"),
    ("batch_summary_14.csv", f"{base}/e44f3562-batch_summary_14.csv"),
]

rows = []
n_total = 0
n_excluded = 0
for name, path in FILES:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for _, r in df.iterrows():
        n_total += 1
        spec_fixed = parse_spec(r["k_fixed"])
        spec_variable = parse_spec(r["k_variable"])
        if is_excluded_spec(spec_fixed) or is_excluded_spec(spec_variable):
            n_excluded += 1
            continue
        rows.append({
            "spec_fixed": repr(spec_fixed),
            "spec_variable": repr(spec_variable),
            "M": r["M"],
            "beta": r["beta"],
            "certified": r["certified"],
            "status": r["status"],
            "label": r.get("label", ""),
            "source_file": name,
        })

out = pd.DataFrame(rows, columns=FIELDNAMES)
out.to_csv("/home/claude/repo_test/indifference_data.csv", index=False)
print(f"{n_total} total rows, {n_excluded} excluded, {len(out)} kept -> indifference_data.csv")
