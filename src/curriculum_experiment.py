"""
Distance/reward curriculum for the no-walls branch.

Map: a wall-free corridor from (0,0) to (200,10). The agent spawns at
(100,5) and never moves. The right target sits at (105,5) worth
RIGHT_REWARD (10) for the entire experiment and is never touched. The left
target starts at (95,5) (distance 5 from spawn) worth 10, and this script
is what moves it and re-values it over time.

Procedure (as specified for this branch):

  1. left_reward is continuously nudged, in effect at every distance, for
     the entire run -- there is no separate search phase. Every single
     time an episode ends by hitting the LEFT target, left_reward is
     immediately decreased by REWARD_NUDGE_PCT percent; every single time
     an episode ends by hitting the RIGHT target, left_reward is
     immediately increased by REWARD_NUDGE_PCT percent. If several hits of
     either kind land in the same update (multiple envs finishing at
     once), the nudge is applied once per individual hit (compounding),
     not once per update. This is a continuous online control loop, not a
     discrete search -- there's no separate "tuning attempt," no gallop/
     bisect, and no give-up/fallback: left_reward just keeps drifting
     toward whatever value balances the split, for as long as training
     runs, at every distance. It's clamped to [LEFT_REWARD_FLOOR,
     LEFT_REWARD_CEILING] purely as a safety rail against the two
     irreversible floating-point states (exact 0.0 and inf) a sustained
     one-sided policy collapse could otherwise drive it into -- see the
     comment above those constants for why that specifically happens.
  2. Distance advances once, on the 50-update rolling window: (a) the
     left/right split is in [MARGIN_LOW, MARGIN_HIGH]% (default
     [49, 51]%), (b) the rolling hit rate (non-miss rate) is at least
     REQUIRED_HIT_PCT percent (default 100 -- i.e. zero misses in the
     window), AND (c) at least MOVES_BEFORE_DISTANCE_CHANGE updates (default
     50) have passed since the distance last changed. Condition (c) exists
     because the window is only 50 updates wide -- immediately after a
     distance change, the window can still contain stale episodes that
     finished under the OLD distance. Waiting MOVES_BEFORE_DISTANCE_CHANGE
     updates (== WINDOW_SIZE) guarantees that by the time the window is
     trusted, every entry in it postdates the change.
  3. When distance advances, the left target moves one unit further from
     spawn; left_reward is NOT reset -- whatever continuous value it has
     drifted to carries over as the starting point for the new distance,
     and the same continuous per-hit nudging keeps running unchanged.
  4. Repeat until the left target has been walked out to distance 25.

This script does NOT re-run training from scratch for each distance --
it's all one continuous run_training() call. The map and reward are
mutated mid-training via run_training's progress_callback hook (see that
function's docstring in run_training.py): returning a dict of env
attributes from the callback applies them to the live vector env via
env.set_attr(), exactly like run_training already does internally for
max_steps. The curriculum controller below IS that callback.

Because this stops training via a raised exception once distance 25 is
resolved (see CurriculumComplete), it deliberately runs with
save_weights=False -- the trained weights aren't the point of this
experiment, the per-distance left_reward values are (see RESULTS_CSV).
"""
import os
from collections import deque

import numpy as np

from run_training import run_training


# =========================================================================
# USER CONFIG -- everything you're likely to want to tune lives here.
# =========================================================================

# -- Map / task definition (this branch's experiment design) ------------
SPAWN_X = 100
RIGHT_X = 105
RIGHT_REWARD = 10.0
START_DISTANCE = 5        # matches the map spec: left target starts at (95,5)
END_DISTANCE = 25
INITIAL_LEFT_REWARD = 10.0

# -- Reward-nudge controller ---------------------------------------------
# How much left_reward moves, in percent, on EVERY individual left-target
# hit (decrease) or right-target hit (increase). E.g. 0.05 means each hit
# multiplies left_reward by (1 - 0.05/100) or (1 + 0.05/100).
REWARD_NUDGE_PCT = 0.05

# Safety bounds only -- NOT meant to meaningfully constrain the value in
# normal operation, just to keep it out of the two irreversible regions
# the pure multiplicative nudge can otherwise drift into during a
# sustained one-sided policy collapse:
#   - LEFT_REWARD_FLOOR: without a floor, enough consecutive left-hit
#     nudges compound left_value down until it underflows to exactly
#     0.0 in floating point. Once that happens it's a dead state forever
#     -- 0.0 * (1 +/- pct) is still 0.0, so no future nudge (in either
#     direction) can ever move it again.
#   - LEFT_REWARD_CEILING: the mirror-image problem on the other side --
#     enough consecutive right-hit nudges compound left_value up until
#     it overflows to inf, which is equally a dead state (inf * anything
#     positive stays inf).
# Both are set many orders of magnitude away from INITIAL_LEFT_REWARD
# (10) so they never interfere with the actual reward-balancing dynamics
# -- they only exist to stop left_value from ever reaching literal 0 or
# inf.
LEFT_REWARD_FLOOR = 1e-10
LEFT_REWARD_CEILING = 1e10

# -- Distance-advance requirement -----------------------------------------
# Minimum rolling non-miss (hit) rate, in percent, required (alongside the
# split being in range -- see MARGIN_LOW/MARGIN_HIGH below) before
# distance is allowed to advance. 100 means zero misses anywhere in the
# current 50-update window.
REQUIRED_HIT_PCT = 30.0

# -- Training hyperparameters ---------------------------------------------
# Entropy bonus: held flat (no decay) rather than annealed, since the task
# itself keeps changing underneath the policy (distance/reward moves) --
# annealing exploration to near-zero would fight against needing to keep
# adapting. Set equal to disable decay entirely (begin == end makes
# run_training's entropy decay term zero regardless of horizon).
BEGIN_ENTROPY = 0.10
END_ENTROPY = 0.10

# Learning rates passed straight through to run_training (which no longer
# decays LR at all -- see run_training.py -- so these are the exact LR
# used for the entire run, not just a starting point). Lowered to 1/10th
# of run_training's own defaults (0.0003 / 0.0001 / 0.0003) after the
# policy repeatedly diverged/collapsed to one side even with the
# reward-value floor/ceiling above in place.
CRITIC_LR = 0.0003 / 10   # 3e-5
ACTOR_LR = 0.0001 / 10    # 1e-5
LSTM_LR = 0.0003 / 10     # 3e-5

# -- Verbosity -------------------------------------------------------------
# A heartbeat status line (current distance/value/window) is printed every
# HEARTBEAT_INTERVAL updates regardless of whether anything changed, so a
# long stretch of training between changes still shows progress instead of
# going silent.
HEARTBEAT_INTERVAL = 250


# =========================================================================
# Internal constants -- govern how the 50-update rolling window/trust
# check works. Not meant to be casually tuned like the knobs above, but
# kept as named constants (rather than magic numbers) for readability.
# =========================================================================

WINDOW_SIZE = 50                     # "50-update rolling average"
MOVES_BEFORE_DISTANCE_CHANGE = 50    # don't advance distance until this many
                                      # updates have passed since the last
                                      # distance change (NOT gated by reward
                                      # nudges -- those apply continuously
                                      # regardless of this counter)
MARGIN_LOW, MARGIN_HIGH = 45, 55

RESULTS_CSV = "eval_logs/distance_reward_curriculum.csv"
REWARD_LOG_CSV = "eval_logs/distance_reward_nudges.csv"


class CurriculumComplete(Exception):
    """Raised from on_update() once the left target has been walked all
    the way out to END_DISTANCE, to unwind out of run_training()'s update
    loop. Caught in main() -- not an error."""
    pass


class DistanceRewardCurriculum:
    """
    Stateful progress_callback for run_training(). Pass an instance's
    on_update method as progress_callback (with progress_callback_interval
    left at its default of 1 -- this needs every update to keep an
    accurate rolling window and to apply reward nudges as soon as hits
    happen).
    """

    def __init__(self):
        self.distance = START_DISTANCE
        self.left_value = INITIAL_LEFT_REWARD

        self.window = deque(maxlen=WINDOW_SIZE)
        self.moves_since_distance_change = 0

        # One entry per finished distance, in order -- this is what lets
        # main() print a full distance -> left_reward summary at the end
        # without having to re-read the CSV back in.
        self.results = []

        os.makedirs("eval_logs", exist_ok=True)
        with open(RESULTS_CSV, "w") as f:
            f.write("distance,left_x,left_reward,final_left_pct,final_hit_pct\n")
        with open(REWARD_LOG_CSV, "w") as f:
            f.write("update,distance,left_value,left_count,right_count,miss_count\n")

        print(f"[curriculum] starting at distance {self.distance} (left target at "
              f"x={self._left_x()}), left_reward={self.left_value:.4f}, right_reward="
              f"{RIGHT_REWARD} fixed at x={RIGHT_X}. Target: distance {END_DISTANCE}. "
              f"reward_nudge_pct={REWARD_NUDGE_PCT}%, required_hit_pct={REQUIRED_HIT_PCT}%.")

    # -- helpers ----------------------------------------------------------

    def _left_x(self):
        return SPAWN_X - self.distance

    def _env_updates(self):
        # NOTE: these keys must match MazeEnv's ACTUAL attribute names
        # (set in SpaceEnv.py's __init__ from its targetCords/targetAwards
        # constructor params), not a generic/guessed name -- env.set_attr()
        # just does setattr(sub_env, key, value), so a mismatched key
        # silently creates an unused new attribute instead of updating
        # anything step()/getObs() actually reads.
        return {
            "targetCords": np.array([[self._left_x(), 5], [RIGHT_X, 5]]),
            "targetAwards": np.array([self.left_value, RIGHT_REWARD]),
        }

    def _window_counts(self):
        """Raw (left, right, miss, total) sums over the current rolling
        window -- the actual counts behind every left%/hit% figure."""
        left_sum = sum(l for l, r, m in self.window)
        right_sum = sum(r for l, r, m in self.window)
        miss_sum = sum(m for l, r, m in self.window)
        return left_sum, right_sum, miss_sum, left_sum + right_sum + miss_sum

    def _window_str(self, left_sum, right_sum, miss_sum, total):
        hits = left_sum + right_sum
        left_pct = (left_sum / hits * 100) if hits > 0 else 0.0
        hit_pct = (hits / total * 100) if total > 0 else 0.0
        return (f"window[n={len(self.window)}/{WINDOW_SIZE}]: "
                f"left={left_sum} right={right_sum} miss={miss_sum} (of {total} episodes) "
                f"-> left%={left_pct:.2f} hit%={hit_pct:.2f}")

    def _print_heartbeat(self, samplePhase):
        trust_state = ("trusted" if self.moves_since_distance_change >= MOVES_BEFORE_DISTANCE_CHANGE
                        else f"buffering ({self.moves_since_distance_change}/{MOVES_BEFORE_DISTANCE_CHANGE})")
        if len(self.window) == 0:
            window_part = "window empty so far"
        else:
            window_part = self._window_str(*self._window_counts())
        print(f"[curriculum] update {samplePhase}: distance {self.distance} (x={self._left_x()}), "
              f"left_reward={self.left_value:.4f}, distance_trust={trust_state} | {window_part}")

    def _reset_for_new_distance(self):
        self.window.clear()
        self.moves_since_distance_change = 0

    # -- the callback itself ----------------------------------------------

    def on_update(self, samplePhase, metrics):
        left_count = metrics["left_count"]
        right_count = metrics["right_count"]
        miss_count = metrics["miss_count"]

        # -- continuous per-hit reward nudge, in effect at every distance,
        # applied immediately (before the next update) as soon as hits are
        # tallied this update. Compounds if multiple hits of either kind
        # landed in the same update.
        if left_count > 0:
            self.left_value *= (1 - REWARD_NUDGE_PCT / 100) ** left_count
        if right_count > 0:
            self.left_value *= (1 + REWARD_NUDGE_PCT / 100) ** right_count
        # Safety clamp only -- see LEFT_REWARD_FLOOR/LEFT_REWARD_CEILING
        # above. Keeps left_value out of the irreversible 0/inf regions
        # without meaningfully constraining normal operation.
        self.left_value = min(max(self.left_value, LEFT_REWARD_FLOOR), LEFT_REWARD_CEILING)
        value_changed = (left_count > 0 or right_count > 0)

        # Logged every single update (like update_info.csv logs every
        # update's loss/entropy), not just updates where left_value
        # actually changed -- this is the continuous, always-on record of
        # left_value alongside the update number, regardless of whether
        # this particular update had any hits.
        with open(REWARD_LOG_CSV, "a") as f:
            f.write(f"{samplePhase},{self.distance},{self.left_value},"
                    f"{left_count},{right_count},{miss_count}\n")

        # -- rolling window / distance-advance bookkeeping
        self.window.append((left_count, right_count, miss_count))
        self.moves_since_distance_change += 1

        if HEARTBEAT_INTERVAL and samplePhase % HEARTBEAT_INTERVAL == 0:
            self._print_heartbeat(samplePhase)

        if len(self.window) < WINDOW_SIZE:
            return self._env_updates() if value_changed else None

        left_sum, right_sum, miss_sum, total = self._window_counts()
        if total == 0:
            return self._env_updates() if value_changed else None  # defensive only

        hits = left_sum + right_sum
        left_pct = (left_sum / hits * 100) if hits > 0 else 0.0
        hit_pct = hits / total * 100

        split_ok = MARGIN_LOW <= left_pct <= MARGIN_HIGH
        hit_ok = hit_pct >= REQUIRED_HIT_PCT
        distance_trusted = self.moves_since_distance_change >= MOVES_BEFORE_DISTANCE_CHANGE

        if split_ok and hit_ok and distance_trusted:
            window_str = self._window_str(left_sum, right_sum, miss_sum, total)
            print(f"[curriculum] update {samplePhase}: distance {self.distance} (x={self._left_x()}) "
                  f"READY TO ADVANCE at left_reward={self.left_value:.4f} -- {window_str}")
            return self._advance_distance(left_pct=left_pct, hit_pct=hit_pct)

        return self._env_updates() if value_changed else None

    def _advance_distance(self, left_pct, hit_pct):
        with open(RESULTS_CSV, "a") as f:
            f.write(f"{self.distance},{self._left_x()},{self.left_value},"
                    f"{left_pct:.3f},{hit_pct:.3f}\n")
        self.results.append({
            "distance": self.distance,
            "left_x": self._left_x(),
            "left_reward": self.left_value,
            "left_pct": left_pct,
            "hit_pct": hit_pct,
        })
        print(f"[curriculum] === distance {self.distance} (left target at x={self._left_x()}) "
              f"DONE -> left_reward={self.left_value:.4f} (left%={left_pct:.2f}, "
              f"hit%={hit_pct:.2f}) ===")

        if self.distance >= END_DISTANCE:
            raise CurriculumComplete()

        next_distance = self.distance + 1
        print(f"[curriculum] advancing to distance {next_distance} (left target moves from "
              f"x={self._left_x()} to x={SPAWN_X - next_distance}), carrying over "
              f"left_reward={self.left_value:.4f} -- the continuous nudge keeps running "
              f"unchanged at the new distance.")

        self.distance = next_distance
        self._reset_for_new_distance()
        return self._env_updates()


def main():
    curriculum = DistanceRewardCurriculum()
    try:
        run_training(
            left_reward=INITIAL_LEFT_REWARD,
            right_reward=RIGHT_REWARD,
            criticLr=CRITIC_LR,
            actorLr=ACTOR_LR,
            lstmLr=LSTM_LR,
            beginEntropy=BEGIN_ENTROPY,
            endEntropy=END_ENTROPY,
            nUpdates=10_000_000,       # effectively unbounded -- the
                                        # curriculum stops us via
                                        # CurriculumComplete once distance
                                        # 25 is resolved
            progress_callback=curriculum.on_update,
            progress_callback_interval=1,
            save_weights=False,
        )
    except CurriculumComplete:
        print(f"\nCurriculum finished: walked the left target out to distance {END_DISTANCE}. "
              f"Per-distance results written to {RESULTS_CSV}.\n")
        print_summary(curriculum.results)


def print_summary(results):
    """Final distance -> left_reward table, printed to the terminal so you
    don't have to go open the CSV to see the answer."""
    print(f"{'distance':>8}  {'left_x':>6}  {'left_reward':>12}  {'left%':>6}  {'hit%':>6}")
    for r in results:
        print(f"{r['distance']:>8}  {r['left_x']:>6}  {r['left_reward']:>12.4f}  "
              f"{r['left_pct']:>6.2f}  {r['hit_pct']:>6.2f}")


if __name__ == "__main__":
    main()
