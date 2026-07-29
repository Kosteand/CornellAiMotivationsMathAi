"""
Phase 2: takes the best candidates found by run_optuna_search.py and
re-runs each a few seeds, reporting both the primary (speed+stability)
score and the secondary either_target_pct side by side -- so you pick
the final setting by looking at the actual tradeoff, not a formula.

Run this after run_optuna_search.py has completed (or been interrupted --
partial results are fine, since this just reads whatever's in the study
so far).
"""

import optuna

from run_training import run_training
from run_optuna_search import (
    TrialProgressTracker, FIXED_KWARGS, STORAGE, STUDY_NAME, TARGET_NUPDATES,
)

N_TOP_CANDIDATES = 5
N_SEEDS_PER_CANDIDATE = 2

# Same fixed settings as the search phase, except weights actually saved
# (in case you want to keep the winner). nUpdates/lrDecayHorizon/
# entropyDecayHorizon are already TARGET_NUPDATES via FIXED_KWARGS --
# search already runs at the real target budget, so there's no separate
# "reduced vs. full" distinction to reconcile here this time.
CONFIRM_KWARGS = dict(FIXED_KWARGS)
CONFIRM_KWARGS["save_weights"] = True


def load_top_candidates():
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top = sorted(completed, key=lambda t: t.value)[:N_TOP_CANDIDATES]
    return [(t.number, t.params) for t in top]


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