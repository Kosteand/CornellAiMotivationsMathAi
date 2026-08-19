"""File 3 of the three-file data architecture: reads File 1
(group_data.csv, per-group math/training metrics) + File 2
(pair_data.csv, per-ordered-pair indifference-search + weight-diff-norm
results) and produces predictive_data.csv - every pair_data.csv row,
PLUS one ratio column per numeric group_data.csv metric (variable-side
value / fixed-side value), "creating a file with all of the relevant
predictive and actual information" per the request that specified this
three-file split.

Run directly (`python3 build_predictive_data.py`) to regenerate
predictive_data.csv from whatever's currently in group_data.csv/
pair_data.csv - this is pure derived data, safe to regenerate any time
either input file changes; nothing here reads or writes group_data.csv/
pair_data.csv, only predictive_data.csv.

IMPORTANT - if this produces an empty (or missing) predictive_data.csv,
it's because group_data.csv/pair_data.csv are themselves empty or don't
exist yet - this file has nothing to derive ratios FROM until something
has actually populated those two first. See the module docstring note at
the bottom of this file (or run_my_comparisons.py) for what to run.
"""
from __future__ import annotations

import csv
import os
import sys

# 2026-08-19: this file (and its CSV) moved from src/ into src/data/ - see
# group_data.py's matching comment. This insert matters here in
# particular since this file has its own __main__ block (can be run
# directly as `python3 data/build_predictive_data.py`, not just imported).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.group_data import GROUP_DATA_CSV_PATH, FIELDNAMES as GROUP_FIELDNAMES, load_group_rows
from data.pair_data import PAIR_DATA_CSV_PATH, FIELDNAMES as PAIR_FIELDNAMES, load_pair_rows

_HERE = os.path.dirname(os.path.abspath(__file__))
PREDICTIVE_DATA_CSV_PATH = os.path.join(_HERE, "predictive_data.csv")

# Every group_data.csv column a ratio is meaningful for - i.e. every
# numeric per-group metric EXCEPT `cache_key` itself, which identifies
# WHICH group a row is, not a magnitude/metric of it.
RATIO_METRIC_COLUMNS = [c for c in GROUP_FIELDNAMES if c != "cache_key"]


def _as_float(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build_predictive_rows(group_rows=None, pair_rows=None) -> tuple[list[dict], list[str]]:
    group_rows = load_group_rows() if group_rows is None else group_rows
    pair_rows = load_pair_rows() if pair_rows is None else pair_rows

    group_by_key = {row["cache_key"]: row for row in group_rows}

    out_fieldnames = list(pair_rows[0].keys()) if pair_rows else list(PAIR_FIELDNAMES)
    ratio_columns = [f"ratio_{col}" for col in RATIO_METRIC_COLUMNS]

    out_rows = []
    for pair in pair_rows:
        row = dict(pair)
        fixed = group_by_key.get(pair["spec_fixed"])
        variable = group_by_key.get(pair["spec_variable"])
        for col in RATIO_METRIC_COLUMNS:
            ratio_col = f"ratio_{col}"
            fixed_val = _as_float(fixed.get(col)) if fixed else None
            variable_val = _as_float(variable.get(col)) if variable else None
            if fixed_val is None or variable_val is None or fixed_val == 0:
                row[ratio_col] = ""
            else:
                row[ratio_col] = variable_val / fixed_val
        out_rows.append(row)

    fieldnames = out_fieldnames + ratio_columns
    return out_rows, fieldnames


def build_predictive_data(csv_path: str = PREDICTIVE_DATA_CSV_PATH) -> int:
    """Regenerate predictive_data.csv from the current group_data.csv +
    pair_data.csv. Returns the number of rows written. If pair_data.csv
    is empty/missing, still writes a HEADER-ONLY file (pair_data's own
    columns + every possible ratio_* column) rather than a truly empty
    file with no header at all - so opening it tells you the schema even
    before any real rows exist, and callers checking
    `len(load's DictReader) == 0` behave the same either way."""
    group_rows = load_group_rows()
    pair_rows = load_pair_rows()

    rows, fieldnames = build_predictive_rows(group_rows, pair_rows)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if not pair_rows:
        print(
            "NOTE: pair_data.csv has no rows yet, so predictive_data.csv "
            "was written with a header only, no data rows. Populate "
            "pair_data.csv first - e.g. run "
            "`python3 -c \"from pair_data import migrate_from_legacy_indifference_data as m; "
            "print(m())\"` to migrate the old indifference search results, "
            "and/or run run_my_comparisons.py to train new comparisons - "
            "then re-run this file."
        )
    return len(rows)


if __name__ == "__main__":
    n = build_predictive_data()
    print(f"wrote {n} rows -> {PREDICTIVE_DATA_CSV_PATH}")
