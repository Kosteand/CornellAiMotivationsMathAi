"""
LR search targeting QUICK CONVERGENCE TO A HIGH HIT RATE within 2000
updates. This replaces the earlier entropy/grad-norm-spike-based
objective entirely -- that proxy turned out to correlate poorly (even
somewhat inversely, on the last batch of data) with actual task
performance, and in one case ranked a completely-stalled policy (0/100
eval hits) as the single best trial, because a policy can confidently
converge to a low-entropy, stable, USELESS behavior just as easily as a
good one. Entropy/spikes measured HOW the policy trains; they never
checked WHETHER it works.

Run this first. Once it finishes, look at the top few trials (by value,
ascending -- lower is better) and hand them to run_optuna_confirm.py for
multi-seed confirmation before picking a final choice.

Launch the dashboard (in a separate terminal, any time during or after
the search) with:
    optuna-dashboard sqlite:///optuna_study.db

RESUMING: this study is stored in a SQLite file (STORAGE below), not
memory, so completed trials are never lost even if the process is killed
or crashes -- just rerun this script and it picks up in the same study.
Note that `study.optimize(objective, n_trials=N_TRIALS)` runs N_TRIALS
NEW trials on top of whatever's already recorded, not N_TRIALS total --
adjust N_TRIALS accordingly when resuming after an interruption.

*** CRITICAL: if you ever change TrialProgressTracker's scoring formula
OR anything about the underlying TRAINING DYNAMICS every trial
experiences, you MUST also change STUDY_NAME to a fresh value.
MedianPruner compares a trial's intermediate report at step N against the
median of ALL PRIOR trials' reports at step N, regardless of which code
version -- or training regime -- produced those old numbers; reusing a
study across incompatible versions makes old and new trials numerically
incomparable. This has bitten this search repeatedly already (see the
retired lr_decay_search and lr_search_2000_convergence_v1/v2 studies) --
when in doubt, use a new STUDY_NAME. ***

THE METRIC -- hit-rate regret:
At every update, compute a ROLLING hit rate (hits / total episodes) over
a trailing window of updates (HIT_RATE_WINDOW), smoothing out the noise
from only a handful of episodes finishing per update. The per-update
"deficit" is max(0, TARGET_HIT_RATE - rolling_hit_rate) -- how far below
the target rate you currently are. The score is the AVERAGE deficit
across the whole run. This single continuous quantity captures both
"quick" and "high" at once:
  - Converging quickly means the deficit shrinks toward 0 early and
    stays there -- little accumulates over the run.
  - Converging to a HIGH final rate means the deficit approaches 0 at
    all, rather than plateauing at some mediocre level (e.g. a policy
    stuck at a real 36% hit rate accumulates a large deficit for the
    ENTIRE run, scoring correctly worse than one that reaches 84% and
    holds it, even though neither ever regresses).
  - A stalled policy (~0% hit rate throughout) sits near the maximum
    possible deficit the whole time -- automatically scored as close to
    the worst possible outcome, with no special-case check required.
TARGET_HIT_RATE=1.0 is the TRUE ceiling, not the ~80% observed in past
(worse-tuned) runs -- capping the target at 80% would cap the metric's
ability to reward improvement beyond it (deficit hits exactly 0 past the
target, so 82% and 98% would score identically). Setting it to 1.0 means
the search keeps looking for something better than what's been seen so
far, rather than treating past best-known performance as good enough.

WHY WEIGHT DECAY IS NOT SEARCHED HERE: the earlier search's own data
showed weight decay was consistently unhelpful, and it separately
confounds this project's Kolmogorov-complexity question (L2 decay
incentivizes low-norm solutions, which could bias the agent toward
whichever target admits a lower-norm policy for reasons unrelated to the
actual reward tradeoff). Fixed at exactly 0.0 for actor, critic, and LSTM.

WHY criticLr IS INDEPENDENT OF actorLr: with weight decay removed from
the search, there's room to let criticLr vary independently without the
space becoming too large to explore efficiently. lstmLr remains tied to
criticLr by explicit request (that ratio is meant to stay fixed for this
round of experimentation).
"""

from collections import deque

import optuna

from run_training import run_training

# --- Search settings --------------------------------------------------
TARGET_NUPDATES = 2000
DECAY_COOLDOWN_FRACTION = 0.9   # lrDecayHorizon/entropyDecayHorizon are this
                                 # fraction of TARGET_NUPDATES, not
                                 # TARGET_NUPDATES itself -- reaching the LR
                                 # floor / zero entropy WITH updates left
                                 # gives the policy time to settle into
                                 # weights consistent with that regime.
N_TRIALS = 20                    # worst-case-safe for a 9hr floor even with
                                 # ZERO pruning credited; pruning will very
                                 # likely make this finish somewhat under 9hr
STORAGE = "sqlite:///optuna_study.db"
STUDY_NAME = "lr_search_2000_hitrate_v2"  # bumped from v1 -- TARGET_HIT_RATE
                                            # changed from 0.80 to 1.0, which
                                            # changes every trial's deficit
                                            # calculation (a trial previously
                                            # scoring exactly 0 past 80% now
                                            # gets nonzero regret unless it's
                                            # truly near 100%)

# --- Objective: hit-rate regret --------------------------------------------
TARGET_HIT_RATE = 1.0     # the TRUE ceiling, not the ~80% observed so far --
                           # 80% was just where past (worse-tuned) runs happened
                           # to land, not a known hard limit. Capping the target
                           # there would cap the metric's ability to reward
                           # further improvement once a trial exceeds it (deficit
                           # hits exactly 0, so 82% and 98% would score identically).
                           # Setting the target to 1.0 means every trial keeps
                           # being rewarded for getting closer to 100%, with no
                           # artificial ceiling on how good "good" can be.
HIT_RATE_WINDOW = 100     # rolling window size, in UPDATES, for smoothing
                           # the per-update hit-rate estimate

DIVERGENCE_PENALTY = 100.0  # returned for a crashed/diverged/stalled trial --
                             # far above any value a genuinely-evaluated trial
                             # could produce (max possible regret is now
                             # TARGET_HIT_RATE itself, i.e. 1.0)

# Fixed settings. actor/critic/lstm weight decay fixed at 0.0 (not
# searched -- see module docstring); lrDecayHorizon/entropyDecayHorizon
# set to DECAY_COOLDOWN_FRACTION (90%) of TARGET_NUPDATES.
FIXED_KWARGS = dict(
    criticLrFloor=3e-5,
    actorLrFloor=1e-5,
    nUpdates=TARGET_NUPDATES,
    lrDecayHorizon=int(TARGET_NUPDATES * DECAY_COOLDOWN_FRACTION),
    entropyDecayHorizon=int(TARGET_NUPDATES * DECAY_COOLDOWN_FRACTION),
    nStepsPerUpdate=512,
    ppo_epochs=4,
    clip_eps=0.2,
    gamma=0.99,
    lam=0.95,
    beginEntropy=0.15,
    endEntropy=0.0,
    step_penalty=0.0,
    left_reward=1000,
    right_reward=10,
    max_steps=500,
    min_steps=500,
    step_decay=20,
    useCProfiler=False,
    useTorchProfiler=False,
    validate_args_flag_param=True,
    check_for_NaN_errors=False,
    load_weights=False,
    save_weights=False,
    target_hits=100,
    actor_weight_decay=0.0,
    critic_weight_decay=0.0,
    lstm_weight_decay=0.0,
)


class TrialProgressTracker:
    """
    Accumulates per-update hit-rate-regret during one run_training() call
    via the progress_callback hook, and reports/prunes through Optuna as
    it goes. See module docstring for the metric's full rationale.
    """

    def __init__(self, trial, nupdates):
        self.trial = trial
        self.nupdates = nupdates
        self.window = deque(maxlen=HIT_RATE_WINDOW)  # each entry: (hits, total) for one update
        self.total_deficit = 0.0
        self.updates_seen = 0

    def __call__(self, sample_phase, metrics):
        hits = metrics["left_count"] + metrics["right_count"]
        total = hits + metrics["miss_count"]
        self.window.append((hits, total))
        self.updates_seen += 1

        window_hits = sum(h for h, t in self.window)
        window_total = sum(t for h, t in self.window)
        rolling_hit_rate = (window_hits / window_total) if window_total > 0 else 0.0

        deficit = max(0.0, TARGET_HIT_RATE - rolling_hit_rate)
        self.total_deficit += deficit

        # Report the RUNNING AVERAGE deficit so far, so Optuna's pruner can
        # compare trajectories on equal footing across different step counts.
        # Lower is better throughout (this study's direction is "minimize").
        self.trial.report(self.total_deficit / self.updates_seen, step=sample_phase)

        if self.trial.should_prune():
            raise optuna.TrialPruned()

    def final_score(self):
        return self.total_deficit / max(self.updates_seen, 1)


def objective(trial):
    # Ranges are centered on the CURRENT defaults (actorLr=1e-4,
    # criticLr=3e-4). Log-scale (log=True) -- exponential spacing, not
    # linear. Upper bound = 50x current; lower bound = 5x below current
    # (asymmetric on purpose -- undershooting the LR is just slow, not
    # unstable, so there's less reason to constrain how low it can go).
    actorLr = trial.suggest_float("actorLr", 2e-5, 5e-3, log=True)      # current default: 1e-4 (5x below / 50x above)
    criticLr = trial.suggest_float("criticLr", 6e-5, 1.5e-2, log=True)  # current default: 3e-4 (5x below / 50x above), independent, not derived

    tracker = TrialProgressTracker(trial, TARGET_NUPDATES)

    try:
        left_count, right_count, miss_count = run_training(
            criticLr=criticLr,
            actorLr=actorLr,
            lstmLr=criticLr,  # tied to criticLr by explicit request -- see module docstring
            lstmLrFloor=FIXED_KWARGS["criticLrFloor"],
            progress_callback=tracker,
            progress_callback_interval=1,
            **FIXED_KWARGS,
        )
    except optuna.TrialPruned:
        raise  # let Optuna handle it -- do not swallow this
    except Exception as e:
        print(f"[objective] trial {trial.number} crashed with params "
              f"{trial.params} -- treating as a failed/diverged trial. "
              f"Error: {type(e).__name__}: {e}")
        return DIVERGENCE_PENALTY

    # Explicit stall check, kept for clarity even though the regret metric
    # already naturally scores a stalled trial near its own worst case
    # (see module docstring) -- this makes stalled trials UNAMBIGUOUSLY
    # the worst (100.0, same as a crash), not just "coincidentally close
    # to the natural ceiling of ~0.80."
    if left_count + right_count < FIXED_KWARGS["target_hits"]:
        print(f"[objective] trial {trial.number} eval stalled "
              f"({left_count + right_count}/{FIXED_KWARGS['target_hits']} hits) "
              f"-- treating as a failed trial.")
        return DIVERGENCE_PENALTY

    print(f"[objective] trial {trial.number}: final eval hit rate = "
          f"{100.0*(left_count+right_count)/(left_count+right_count+miss_count):.1f}% "
          f"(left={left_count}, right={right_count}, miss={miss_count})")

    return tracker.final_score()


if __name__ == "__main__":
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=50, n_startup_trials=5),
    )

    study.optimize(objective, n_trials=N_TRIALS)

    print("\n=== TOP 5 TRIALS (lower = quicker convergence to a higher hit rate) ===")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    for t in sorted(completed, key=lambda t: t.value)[:5]:
        print(f"trial {t.number}: value={t.value:.4f}  params={t.params}")