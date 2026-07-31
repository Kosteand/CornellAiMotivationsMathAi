"""
Phase 2: re-runs the best candidates from a completed run_optuna_search.py
search, a few seeds each, reporting both the primary (speed+stability)
score and the secondary either_target_pct side by side -- so you pick
the final setting by looking at the actual tradeoff, not a formula.

NOTE: this version reads its candidates from TOP_CANDIDATES_FROM_LOG
(hardcoded below, transcribed from an actual terminal log) rather than
from the live optuna_study.db -- that database lost its trial data after
the search completed (checked directly: the study now shows a single
stray RUNNING trial with no value, not the 20 that actually finished).
If you run a future search and the database stays intact, you can
restore the original database-driven load_top_candidates() by querying
optuna.load_study(...) and filtering on TrialState.COMPLETE instead.
"""

from run_training import run_training
from run_optuna_search import TrialProgressTracker, FIXED_KWARGS, TARGET_NUPDATES

N_TOP_CANDIDATES = 5
N_SEEDS_PER_CANDIDATE = 2

# Same fixed settings as the search phase, except weights actually saved
# (in case you want to keep the winner). nUpdates/lrDecayHorizon/
# entropyDecayHorizon are already TARGET_NUPDATES via FIXED_KWARGS --
# search already runs at the real target budget, so there's no separate
# "reduced vs. full" distinction to reconcile here this time.
CONFIRM_KWARGS = dict(FIXED_KWARGS)
CONFIRM_KWARGS["save_weights"] = True


# The live optuna_study.db lost its trial data (checked directly -- the
# study now shows only 1 stray RUNNING trial with no value, not the 20
# that actually completed). Rather than depend on that file, the top 5
# candidates are transcribed VERBATIM from the terminal log of the run
# that actually happened -- exact float values, trial numbers, and scores,
# so this is equivalent to what load_top_candidates() would have returned
# had the database been intact.
TOP_CANDIDATES_FROM_LOG = [
    # (trial_number, params, score_for_reference)
    (15, {"actorLr": 0.0047134491951152544, "criticLr": 0.000612447061450724}, 0.3106971136512426),
    (2,  {"actorLr": 0.001439948636756358,  "criticLr": 0.004797134999884008}, 0.3200309636940867),
    (12, {"actorLr": 2.6685690130619545e-05, "criticLr": 0.002319499556125058}, 0.34566298416827135),
    (10, {"actorLr": 3.850433136585328e-05,  "criticLr": 0.0031250775107881674}, 0.3783664424286274),
    (0,  {"actorLr": 0.002452369296438102,   "criticLr": 0.014009621654327294}, 0.3873350814823653),
]


def load_top_candidates():
    return [(t, params) for t, params, _score in TOP_CANDIDATES_FROM_LOG]


class DummyTrial:
    """TrialProgressTracker only needs .report()/.should_prune(); give it
    inert versions so it just accumulates, no pruning, during confirmation."""
    def report(self, value, step):
        pass

    def should_prune(self):
        return False


if __name__ == "__main__":
    candidates = load_top_candidates()
    if not candidates:
        print("No completed trials found in the study yet -- run "
              "run_optuna_search.py first (partial progress is fine).")
        raise SystemExit(1)

    results = []

    for trial_number, params in candidates:
        actorLr = params["actorLr"]
        criticLr = params["criticLr"]  # independent now, not derived

        for seed_idx in range(N_SEEDS_PER_CANDIDATE):
            print(f"\n=== confirming trial {trial_number}, seed {seed_idx + 1}/{N_SEEDS_PER_CANDIDATE} "
                  f"=== actorLr={actorLr:.2e}, criticLr={criticLr:.2e}")

            tracker = TrialProgressTracker(DummyTrial(), TARGET_NUPDATES)

            try:
                left_count, right_count, miss_count = run_training(
                    criticLr=criticLr,
                    actorLr=actorLr,
                    lstmLr=criticLr,  # mirrors run_optuna_search.py's behavior
                    lstmLrFloor=CONFIRM_KWARGS["criticLrFloor"],
                    progress_callback=tracker,
                    progress_callback_interval=1,
                    **CONFIRM_KWARGS,
                )
            except Exception as e:
                print(f"trial {trial_number} seed {seed_idx}: crashed during "
                      f"confirmation -- {type(e).__name__}: {e}")
                continue

            either_pct = 100.0 * (left_count + right_count) / max(left_count + right_count + miss_count, 1)

            results.append({
                "trial_number": trial_number,
                "seed": seed_idx,
                "actorLr": actorLr,
                "criticLr": criticLr,
                "composite_score": tracker.final_score(),
                "either_target_pct": either_pct,
                "left_count": left_count,
                "right_count": right_count,
                "miss_count": miss_count,
            })

    print("\n\n=== CONFIRMATION RESULTS (sorted by composite score, lower = faster/more stable) ===")
    for r in sorted(results, key=lambda r: r["composite_score"]):
        print(f"trial {r['trial_number']} seed {r['seed']}: "
              f"composite={r['composite_score']:.4f}  "
              f"either_target_pct={r['either_target_pct']:.1f}%  "
              f"(left={r['left_count']}, right={r['right_count']}, miss={r['miss_count']})  "
              f"actorLr={r['actorLr']:.2e} criticLr={r['criticLr']:.2e}")