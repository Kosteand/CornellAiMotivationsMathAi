import csv
import os
import time

import numpy as np

from midpoint_search import _train_and_eval_point

XS = [125, 130]
N_SEEDS_PER_POINT = 10
HITS_PER_POINT = 100
OUTPUT_CSV = "search_results_wide_anchors2.csv"

FIXED_KWARGS = {
    "right_reward": 10,
    "nUpdates": 5000,
    "nStepsPerUpdate": 512,
    "max_steps": 500,
    "min_steps": 500,
    "step_penalty": 0.0,
    "early_stop_patience": 200,
    "early_stop_min_updates": 500,
}


def already_done():
    """Reads whatever's already in the output CSV so a restart can resume mid-point."""
    done = {}
    if not os.path.exists(OUTPUT_CSV):
        return done
    with open(OUTPUT_CSV, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            x = float(row[0])
            seeds = [float(v) for v in row[1].split(";") if v != ""]
            done[x] = seeds
    return done


if __name__ == "__main__":
    progress = already_done()
    for x in XS:
        progress.setdefault(x, [])

    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w") as f:
            f.write("x,seeds\n")

    total_runs = sum(N_SEEDS_PER_POINT - len(progress[x]) for x in XS)
    run_counter = 0
    start_time = time.time()

    print(f"[wide-anchors] starting: {XS} x up to {N_SEEDS_PER_POINT} seeds each "
          f"= {total_runs} from-scratch runs remaining\n")

    for point_idx, x in enumerate(XS, start=1):
        seeds_needed = N_SEEDS_PER_POINT - len(progress[x])
        if seeds_needed <= 0:
            print(f"[wide-anchors] x={x} already has {len(progress[x])} seeds -- skipping\n")
            continue

        for seed_idx in range(seeds_needed):
            run_counter += 1
            elapsed = time.time() - start_time
            current_seed_num = len(progress[x]) + 1
            print(f"[wide-anchors] point {point_idx}/{len(XS)} (x={x}), "
                  f"seed {current_seed_num}/{N_SEEDS_PER_POINT} "
                  f"-- run {run_counter}/{total_runs} overall, "
                  f"elapsed {elapsed / 60:.1f} min")

            left, right, miss, stalled = _train_and_eval_point(x, HITS_PER_POINT, FIXED_KWARGS)
            if stalled or (left + right) == 0:
                print(f"[wide-anchors] x={x} seed {current_seed_num}: "
                      f"stalled / no hits -- excluding this seed, will retry on resume")
                continue

            y = left / (left + right)
            progress[x].append(y)
            print(f"[wide-anchors] x={x} seed {current_seed_num}: "
                  f"Y_hat={y:.3f} ({left} left / {right} right / {miss} miss)")

            # Rewrite the whole file after every single seed -- not just every
            # point -- since each point now takes ~10x longer to fully finish.
            # Iterate over ALL recorded x values (not just the current XS
            # list), so re-running this script with a different XS later
            # can't silently drop previously-collected points from the file.
            with open(OUTPUT_CSV, "w") as f:
                f.write("x,seeds\n")
                for xi in sorted(progress.keys()):
                    if progress[xi]:
                        seeds_str = ";".join(f"{v:.4f}" for v in progress[xi])
                        f.write(f"{xi},{seeds_str}\n")

        point_mean = float(np.mean(progress[x])) if progress[x] else float("nan")
        print(f"[wide-anchors] point {point_idx}/{len(XS)} done -- x={x}: "
              f"mean Y_hat={point_mean:.3f} over {len(progress[x])} seed(s)\n")

    total_elapsed = time.time() - start_time
    print(f"\n[wide-anchors] === DONE === total elapsed {total_elapsed / 60:.1f} min "
          f"({run_counter} runs). Results in {OUTPUT_CSV}.")