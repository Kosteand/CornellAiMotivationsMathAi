"""File 2 of the three-file data architecture: one row per ORDERED PAIR
of groups (fixed side, variable side) - indifference-search results
(M/beta/certified/status, from legacy_csv_data.py's migrated old
indifference data) plus columns for the eventual weight-difference-norm
breakdown (left blank - "wait on implementing the weight difference norm
calculation" per direct instruction; the column set matches
run_shared_machinery_experiment.py's WEIGHT_GROUP_PREDICATES exactly so
filling them in later is a pure data-population change, no schema
change).

Row identity = (spec_fixed, spec_variable) as `group._cache_key` STRINGS
(see group_data.py's cache_key_str) - the SAME join key group_data.csv
uses, so a pair row's two sides can be looked up there directly (this is
what build_predictive_data.py's ratio generator relies on).

Backing file: pair_data.csv (repo root).
"""
from __future__ import annotations

import csv
import os
from typing import Optional

from group_data import cache_key_str

_HERE = os.path.dirname(os.path.abspath(__file__))
PAIR_DATA_CSV_PATH = os.path.join(_HERE, "pair_data.csv")

# Every MarginGroup/HeatmapGroup config in this project uses g=4 - see
# groups.py's MarginGroup/HeatmapGroup docstrings. No shared constant for
# this (removed) - the literal 4 is used directly wherever it's needed
# below, which is only when converting an OLD, bare (no-g) spec tuple
# (from the legacy indifference data - see legacy_csv_data.py) into a
# g-inclusive cache_key, since new group instances always carry their
# own real `g`.

# Matches run_shared_machinery_experiment.py's WEIGHT_GROUP_PREDICATES
# keys exactly - kept as a plain tuple here (rather than importing that
# module, which pulls in torch/stable_baselines3 just for a list of
# strings) since only the NAMES are needed for this file's column list.
WEIGHT_DIFF_GROUP_NAMES = (
    "whole_model", "actor", "critic",
    "policy_net_0", "policy_net_2", "action_net",
    "value_net_0", "value_net_2", "value_net_out",
)

FIELDNAMES = [
    "spec_fixed", "spec_variable", "label", "M", "beta", "certified", "status",
] + [f"diff_{name}" for name in WEIGHT_DIFF_GROUP_NAMES] + [
    "source_file",
]


def _old_spec_to_cache_key(spec: tuple, g: int = 4) -> tuple:
    """Convert one of the legacy (no-g) spec tuples - ("margin", k, s,
    err) or ("heatmap", noise_scale, n) - into the g-inclusive cache_key
    tuple groups.py's MarginGroup/HeatmapGroup._cache_key now produces,
    at the project-wide g=4 every one of those old records implicitly
    assumed.

    IMPORTANT: every element is cast to the exact type MarginGroup's/
    HeatmapGroup's own __init__ casts it to (k/s/err -> float, noise_scale
    -> float, n -> int) - legacy spec tuples often store k as a bare int
    (e.g. ("margin", 1, 1.0, 0.0)), but a live MarginGroup(k=1, ...)
    instance's `_cache_key` always has k as a FLOAT (`self.k =
    float(k)`). Without this cast, `str(("margin", 1, 1.0, 0.0, 4))` !=
    `str(("margin", 1.0, 1.0, 0.0, 4))` - two representations of the
    exact same group - and every join against group_data.csv (which is
    always keyed by the live-instance cache_key) would silently fail to
    match for any legacy spec with an integer k."""
    spec = tuple(spec)
    if spec[0] == "margin":
        k = float(spec[1])
        s = float(spec[2]) if len(spec) > 2 else 1.0
        err = float(spec[3]) if len(spec) > 3 else 0.0
        return ("margin", k, s, err, g)
    if spec[0] == "heatmap":
        noise_scale, n = float(spec[1]), int(spec[2])
        return ("heatmap", noise_scale, n, g)
    raise ValueError(f"unrecognized spec {spec!r}")


def pair_row_from_group_instances(group_fixed, group_variable, *, M=None, beta=None,
                                    certified=None, status="", label="",
                                    source_file="") -> dict:
    """Build one File-2 row from two live group instances (the ordered
    fixed/variable pair) plus whatever indifference-search fields are
    known - M/beta/certified/status default to blank (a pure scaffold
    row, e.g. from run_pipeline.py before any indifference search has run
    for that pair) and diff_* is ALWAYS blank here (weight-diff-norm
    population is a separate, not-yet-implemented step)."""
    row = {f: "" for f in FIELDNAMES}
    row["spec_fixed"] = cache_key_str(group_fixed._cache_key)
    row["spec_variable"] = cache_key_str(group_variable._cache_key)
    row["label"] = label
    row["M"] = "" if M is None else M
    row["beta"] = "" if beta is None else beta
    row["certified"] = "" if certified is None else certified
    row["status"] = status
    row["source_file"] = source_file
    return row


def load_pair_rows(csv_path: str = PAIR_DATA_CSV_PATH) -> list[dict]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _write_all(rows: list[dict], csv_path: str = PAIR_DATA_CSV_PATH) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows({f: row.get(f, "") for f in FIELDNAMES} for row in rows)


def get_pair_row(spec_fixed_key: str, spec_variable_key: str,
                  csv_path: str = PAIR_DATA_CSV_PATH) -> Optional[dict]:
    for row in load_pair_rows(csv_path):
        if row["spec_fixed"] == spec_fixed_key and row["spec_variable"] == spec_variable_key:
            return row
    return None


def upsert_pair_row(row: dict, csv_path: str = PAIR_DATA_CSV_PATH,
                     preserve_existing_diffs: bool = True) -> None:
    """Write `row` into pair_data.csv, replacing any existing row for the
    same (spec_fixed, spec_variable) pair. If `preserve_existing_diffs`
    (default True) and a prior row for this pair already had non-blank
    diff_* values (e.g. from a future weight-diff-norm run), those are
    carried forward onto the new row instead of being blanked out by a
    later scaffold-only upsert (e.g. re-running the pipeline for the same
    comparisons shouldn't erase weight-diff-norm results computed since
    the last run)."""
    rows = load_pair_rows(csv_path)
    existing = None
    kept = []
    for r in rows:
        if r["spec_fixed"] == row["spec_fixed"] and r["spec_variable"] == row["spec_variable"]:
            existing = r
            continue
        kept.append(r)

    new_row = {f: row.get(f, "") for f in FIELDNAMES}
    if preserve_existing_diffs and existing is not None:
        for name in WEIGHT_DIFF_GROUP_NAMES:
            col = f"diff_{name}"
            if new_row.get(col, "") == "" and existing.get(col, "") != "":
                new_row[col] = existing[col]
        # Also preserve M/beta/certified/status if the new row didn't
        # supply them but an existing row already had them recorded.
        for col in ("M", "beta", "certified", "status", "label", "source_file"):
            if new_row.get(col, "") == "" and existing.get(col, "") != "":
                new_row[col] = existing[col]

    kept.append(new_row)
    _write_all(kept, csv_path)


def migrate_from_legacy_indifference_data(g: int = 4,
                                            source_file: str = "indifference_data.csv",
                                            csv_path: str = PAIR_DATA_CSV_PATH) -> int:
    """One-time migration: read every row of the legacy indifference data
    (M/beta/certified/status per (spec_fixed, spec_variable), specs
    bare/no-g - see legacy_csv_data.py) and upsert an equivalent row into
    pair_data.csv, with specs converted to g-inclusive cache-key strings
    (see _old_spec_to_cache_key) and diff_* left blank. Returns the
    number of rows migrated. Safe to re-run - each row is upserted by its
    (spec_fixed, spec_variable) cache-key pair, so running this twice
    just re-writes the same rows rather than duplicating them."""
    from legacy_csv_data import INDIFFERENCE_RECORDS

    n = 0
    for rec in INDIFFERENCE_RECORDS:
        row = {f: "" for f in FIELDNAMES}
        row["spec_fixed"] = cache_key_str(_old_spec_to_cache_key(rec.spec_fixed, g=g))
        row["spec_variable"] = cache_key_str(_old_spec_to_cache_key(rec.spec_variable, g=g))
        row["label"] = rec.label
        row["M"] = rec.M
        row["beta"] = "" if rec.beta is None else rec.beta
        row["certified"] = rec.certified
        row["status"] = rec.status
        row["source_file"] = source_file
        upsert_pair_row(row, csv_path=csv_path)
        n += 1
    return n
