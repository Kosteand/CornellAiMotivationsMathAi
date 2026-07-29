import csv

import numpy as np

from midpoint_search import _fit_sigmoid_mle, _counts_from_seed_ratios

HITS_PER_POINT = 100  # must match whatever target_hits/hits_per_point was actually used

if __name__ == "__main__":
    xs = []
    y_hats = []
    left_counts = []
    total_counts = []

    with open("search_results_so_far.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            x = float(row[0])
            seed_vals = [float(v) for v in row[1].split(";") if v != ""]
            if len(seed_vals) == 0:
                continue  # a point with no valid seeds contributes nothing

            xs.append(x)
            y_hats.append(float(np.mean(seed_vals)))

            left, total = _counts_from_seed_ratios(seed_vals, HITS_PER_POINT)
            left_counts.append(left)
            total_counts.append(total)

    xs = np.array(xs)
    y_hats = np.array(y_hats)

    print(f"Points: {list(xs)}")
    print(f"Y_hat means: {[round(y, 3) for y in y_hats]}")
    print(f"Pooled counts (left/total): {list(zip(left_counts, total_counts))}")

    M, beta, log_likelihood = _fit_sigmoid_mle(xs, left_counts, total_counts)

    print(f"\nCandidate x* (M) = {M:.4g}")
    print(f"Fitted beta      = {beta:.4g}")
    print(f"Log-likelihood   = {log_likelihood:.4g}")