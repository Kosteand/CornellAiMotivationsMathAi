"""
Distance/reward curriculum for the no-walls branch.

Map: a wall-free corridor from (0,0) to (200,10). The agent spawns at
(100,5) and never moves. The right target sits at (105,5) worth
RIGHT_REWARD (10) for the entire experiment and is never touched. The left
target starts at (95,5) (distance 5 from spawn) worth 10, and this script
is what moves it and re-values it over time.

Procedure (as specified for this branch):

  1. Train at the current distance until, on a 50-update rolling window,
     the agent hits SOME target 100% of the time (zero misses in the
     window) AND the left/right split among those hits is in [49, 51]%
     for the left target (equivalently the right target, since with zero
     misses left% + right% == 100%).
  2. Once that holds, move the left target one spot further from spawn
     (distance += 1) and record the left_reward value that achieved the
     50/50 split at the PREVIOUS distance.
  3. After EVERY change (moving the target OR tweaking its reward value),
     don't trust/act on the 50-update rolling window until at least 100
     updates have passed since that change -- the window itself is only
     50 wide, but the first 50 updates after any change may still contain
     stale episodes finished under the old configuration, so we wait for
     a full extra window's worth of buffer before reading it.
  4. If the split has drifted, adjust left_reward using a "gallop, then
     bisect" search: first nudge is 5% of the current value in the
     direction that should pull the split back toward 50/50; if that
     still undershoots, double the step and keep going (exponential
     search) until the split crosses to the other side of 50%, then
     binary-search between the two bracketing values. This deliberately
     avoids a big first jump (which would waste time re-exploring), while
     still converging quickly once a bracket is found.
  5. If tuning at a given distance doesn't converge within
     MAX_ATTEMPTS_PER_DISTANCE tries (noise, or a genuinely hard case),
     give up gracefully: print a warning, fall back to whichever tried
     value came closest to a 50/50 split, and move on to the next
     distance anyway rather than stalling the whole experiment.
  6. Repeat until the left target has been walked out to distance 25.

This script does NOT re-run training from scratch for each distance/value
-- it's all one continuous run_training() call. The map and reward are
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


# ---- Experiment configuration -----------------------------------------

SPAWN_X = 100
RIGHT_X = 105
RIGHT_REWARD = 10.0

START_DISTANCE = 5     # matches the map spec: left target starts at (95,5)
END_DISTANCE = 25
INITIAL_LEFT_REWARD = 10.0

WINDOW_SIZE = 50                # "50-update rolling average"
MOVES_BEFORE_TRUST = 100        # don't read/act on the window until this
                                 # many updates have passed since the last
                                 # change (distance move OR value tweak)
MARGIN_LOW, MARGIN_HIGH = 49, 51

FIRST_STEP_FRACTION = 0.05      # first nudge after any change = 5% of value
MAX_ATTEMPTS_PER_DISTANCE = 20
LEFT_REWARD_FLOOR = 0.1

RESULTS_CSV = "eval_logs/distance_reward_curriculum.csv"
ATTEMPTS_CSV = "eval_logs/distance_reward_attempts.csv"

# Verbosity: a heartbeat status line (current distance/value/window) is
# printed every HEARTBEAT_INTERVAL updates regardless of whether anything
# changed, so a long stretch of training between changes still shows
# progress instead of going silent. The "waiting on misses" state (split
# already OK, just waiting for stale misses to clear the window) is
# printed every WAIT_PRINT_INTERVAL checks rather than every single one,
# since it can otherwise repeat every update for a long time.
HEARTBEAT_INTERVAL = 250
WAIT_PRINT_INTERVAL = 10

# Collapse/runaway guard: if the gallop phase doubles its step this many
# times in a row without EVER crossing to the other side of 50/50, that's
# not "the right value is just further away than expected" -- a step that
# keeps doubling GALLOP_WARNING_DOUBLINGS times has already multiplied the
# reward by 2**GALLOP_WARNING_DOUBLINGS (256x at the default of 8), and if
# that still isn't enough to pull even one episode the other way, the
# policy has most likely stopped exploring that action entirely (e.g. it
# collapsed to a fully deterministic policy and there's no exploration
# left to ever notice the reward changed). Printed once per distance so
# it doesn't spam every subsequent doubling.
GALLOP_WARNING_DOUBLINGS = 8

# Entropy bonus: held flat (no decay) rather than annealed, since the task
# itself keeps changing underneath the policy (distance/reward moves) --
# annealing exploration to near-zero would fight against needing to keep
# adapting. Set equal to disable decay entirely (see run_training's
# entropy formula: begin == end makes the decay term zero regardless of
# horizon). Tweak freely; these are read fresh each run.
BEGIN_ENTROPY = 0.05
END_ENTROPY = 0.05

# Learning rate: also held effectively flat, for the same reason as
# entropy above (nUpdates=10_000_000 is just an unbounded-run placeholder,
# not a real decay horizon -- decoupling it from LR_DECAY_HORIZON makes
# that intentional rather than an accident of how large nUpdates happens
# to be set to).
LR_DECAY_HORIZON = 1_000_000


class CurriculumComplete(Exception):
    """Raised from on_update() once the left target has been walked all
    the way out to END_DISTANCE (converged or given up on), to unwind out
    of run_training()'s update loop. Caught in main() -- not an error."""
    pass


class DistanceRewardCurriculum:
    """
    Stateful progress_callback for run_training(). Pass an instance's
    on_update method as progress_callback (with progress_callback_interval
    left at its default of 1 -- this needs every update to keep an
    accurate rolling window).
    """

    def __init__(self):
        self.distance = START_DISTANCE
        self.left_value = INITIAL_LEFT_REWARD

        self.window = deque(maxlen=WINDOW_SIZE)
        self.moves_since_change = 0

        # Gallop/bisect search state for the reward tuning at the CURRENT
        # distance. Reset every time we advance to a new distance.
        self.stage = "idle"        # "idle" -> "gallop" -> "bisect"
        self.direction = None      # +1 (increasing helps) or -1 (decreasing helps)
        self.step = None
        self.under_value = None    # a left_value known to give left% < 49
        self.over_value = None     # a left_value known to give left% > 51
        self.attempts = 0
        self.best_value = self.left_value
        self.best_diff = float("inf")
        self.best_left_pct = None
        self.best_hit_pct = None

        # See GALLOP_WARNING_DOUBLINGS above.
        self.gallop_doublings = 0
        self._gallop_warning_printed = False

        # One entry per finished distance, in order -- this is what lets
        # main() print a full distance -> left_reward summary at the end
        # without having to re-read the CSV back in.
        self.results = []

        # Counter for throttling the "waiting on misses" print (see
        # WAIT_PRINT_INTERVAL above).
        self._wait_print_count = 0

        os.makedirs("eval_logs", exist_ok=True)
        with open(RESULTS_CSV, "w") as f:
            f.write("distance,left_x,left_reward,converged,attempts,final_left_pct,final_hit_pct\n")
        with open(ATTEMPTS_CSV, "w") as f:
            f.write("update,distance,left_value,left_pct,hit_pct,miss_count_in_window\n")

        print(f"[curriculum] starting at distance {self.distance} (left target at "
              f"x={self._left_x()}), left_reward={self.left_value:.4f}, right_reward="
              f"{RIGHT_REWARD} fixed at x={RIGHT_X}. Target: distance {END_DISTANCE}.")

    # -- helpers ----------------------------------------------------------

    def _left_x(self):
        return SPAWN_X - self.distance

    def _env_updates(self):
        return {
            "target_coords": np.array([[self._left_x(), 5], [RIGHT_X, 5]]),
            "target_awards": np.array([self.left_value, RIGHT_REWARD]),
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
        trust_state = ("trusted" if self.moves_since_change >= MOVES_BEFORE_TRUST
                        else f"buffering ({self.moves_since_change}/{MOVES_BEFORE_TRUST})")
        if len(self.window) == 0:
            window_part = "window empty so far"
        else:
            window_part = self._window_str(*self._window_counts())
        print(f"[curriculum] update {samplePhase}: distance {self.distance} (x={self._left_x()}), "
              f"left_reward={self.left_value:.4f}, stage={self.stage}, "
              f"attempt {self.attempts}/{MAX_ATTEMPTS_PER_DISTANCE}, "
              f"moves_since_change={self.moves_since_change} [{trust_state}] | {window_part}")

    def _reset_for_new_distance(self):
        self.window.clear()
        self.moves_since_change = 0
        self.stage = "idle"
        self.direction = None
        self.step = None
        self.under_value = None
        self.over_value = None
        self.attempts = 0
        self.best_value = self.left_value
        self.best_diff = float("inf")
        self.best_left_pct = None
        self.best_hit_pct = None
        self._wait_print_count = 0
        self.gallop_doublings = 0
        self._gallop_warning_printed = False

    # -- the callback itself ----------------------------------------------

    def on_update(self, samplePhase, metrics):
        self.window.append((metrics["left_count"], metrics["right_count"], metrics["miss_count"]))
        self.moves_since_change += 1

        if HEARTBEAT_INTERVAL and samplePhase % HEARTBEAT_INTERVAL == 0:
            self._print_heartbeat(samplePhase)

        if self.moves_since_change < MOVES_BEFORE_TRUST:
            return None

        left_sum, right_sum, miss_sum, total = self._window_counts()
        if total == 0:
            return None  # defensive only -- shouldn't happen this far in

        hits = left_sum + right_sum
        left_pct = (left_sum / hits * 100) if hits > 0 else 0.0
        hit_pct = hits / total * 100
        hit_ok = (miss_sum == 0)
        window_str = self._window_str(left_sum, right_sum, miss_sum, total)

        with open(ATTEMPTS_CSV, "a") as f:
            f.write(f"{samplePhase},{self.distance},{self.left_value},{left_pct:.3f},"
                    f"{hit_pct:.3f},{miss_sum}\n")

        diff = abs(left_pct - 50)
        if diff < self.best_diff:
            self.best_diff = diff
            self.best_value = self.left_value
            self.best_left_pct = left_pct
            self.best_hit_pct = hit_pct

        split_ok = MARGIN_LOW <= left_pct <= MARGIN_HIGH
        success = hit_ok and split_ok

        if success:
            print(f"[curriculum] update {samplePhase}: distance {self.distance} (x={self._left_x()}) "
                  f"CONVERGED at left_reward={self.left_value:.4f} -- {window_str}")
            return self._advance_distance(converged=True, left_pct=left_pct, hit_pct=hit_pct)

        if split_ok and not hit_ok:
            # The split already looks right -- the remaining misses just
            # haven't cleared the window yet. This isn't something
            # left_reward tuning can fix, so wait for more training rather
            # than spending a search attempt on it.
            self._wait_print_count += 1
            if self._wait_print_count % WAIT_PRINT_INTERVAL == 1:
                print(f"[curriculum] update {samplePhase}: distance {self.distance} (x={self._left_x()}) "
                      f"split already OK at left_reward={self.left_value:.4f}, waiting for stale "
                      f"misses to clear -- {window_str}")
            return None

        if self.attempts >= MAX_ATTEMPTS_PER_DISTANCE:
            print(f"[curriculum] update {samplePhase}: distance {self.distance} (left target at "
                  f"x={self._left_x()}): gave up after {self.attempts} tuning attempts without "
                  f"landing in [{MARGIN_LOW}, {MARGIN_HIGH}]%. Last try: left_reward="
                  f"{self.left_value:.4f} -- {window_str}. Falling back to the closest value "
                  f"tried: left_reward={self.best_value:.4f} (left%={self.best_left_pct:.2f}, "
                  f"hit%={self.best_hit_pct:.2f}). Moving on to the next distance anyway.")
            self.left_value = self.best_value
            return self._advance_distance(converged=False, left_pct=self.best_left_pct,
                                           hit_pct=self.best_hit_pct)

        old_value = self.left_value
        old_stage = self.stage
        self._tweak(left_pct)
        self.attempts += 1
        self.moves_since_change = 0
        print(f"[curriculum] update {samplePhase}: distance {self.distance} (x={self._left_x()}) "
              f"attempt {self.attempts}/{MAX_ATTEMPTS_PER_DISTANCE} [{old_stage}->{self.stage}] "
              f"-- {window_str} | left_reward {old_value:.4f} -> {self.left_value:.4f}")
        return self._env_updates()

    def _tweak(self, left_pct):
        """Advance the gallop/bisect search by one step given the latest
        left_pct reading, updating self.left_value in place."""
        need_direction = 1 if left_pct < MARGIN_LOW else -1  # +1 == increase left_value helps

        if self.stage == "idle":
            self.direction = need_direction
            self.step = max(FIRST_STEP_FRACTION * self.left_value, 1e-6)
            if self.direction > 0:
                self.under_value = self.left_value
            else:
                self.over_value = self.left_value
            self.stage = "gallop"
            self.gallop_doublings = 0
            new_value = self.left_value + self.direction * self.step

        elif self.stage == "gallop":
            crossed = (need_direction != self.direction)
            if crossed:
                if self.direction > 0:
                    self.over_value = self.left_value
                else:
                    self.under_value = self.left_value
                self.stage = "bisect"
                new_value = (self.under_value + self.over_value) / 2
            else:
                if self.direction > 0:
                    self.under_value = self.left_value
                else:
                    self.over_value = self.left_value
                self.step *= 2
                self.gallop_doublings += 1
                new_value = self.left_value + self.direction * self.step

                if (self.gallop_doublings >= GALLOP_WARNING_DOUBLINGS
                        and not self._gallop_warning_printed):
                    self._gallop_warning_printed = True
                    growth = 2 ** self.gallop_doublings
                    print(f"[curriculum] WARNING: distance {self.distance} -- the reward gallop "
                          f"has doubled {self.gallop_doublings} times ({growth}x growth) without "
                          f"EVER crossing 50/50. This is very unlikely to mean 'the right value "
                          f"is just further away' -- it usually means the policy has stopped "
                          f"exploring the other action entirely (e.g. it collapsed to a fully "
                          f"deterministic policy) and no reward value will bring it back. Check "
                          f"for a collapsed/deterministic policy (e.g. one side's episode count "
                          f"exploding while the other stays at 0) before waiting out the "
                          f"remaining attempts.")

        else:  # bisect
            if need_direction > 0:
                self.under_value = self.left_value
            else:
                self.over_value = self.left_value
            new_value = (self.under_value + self.over_value) / 2

        self.left_value = max(new_value, LEFT_REWARD_FLOOR)

    def _advance_distance(self, converged, left_pct, hit_pct):
        with open(RESULTS_CSV, "a") as f:
            f.write(f"{self.distance},{self._left_x()},{self.left_value},{converged},"
                    f"{self.attempts},{left_pct:.3f},{hit_pct:.3f}\n")
        self.results.append({
            "distance": self.distance,
            "left_x": self._left_x(),
            "left_reward": self.left_value,
            "converged": converged,
            "attempts": self.attempts,
            "left_pct": left_pct,
            "hit_pct": hit_pct,
        })
        print(f"[curriculum] === distance {self.distance} (left target at x={self._left_x()}) "
              f"DONE -> left_reward={self.left_value:.4f} (converged={converged}, "
              f"attempts={self.attempts}, left%={left_pct:.2f}, hit%={hit_pct:.2f}) ===")

        if self.distance >= END_DISTANCE:
            raise CurriculumComplete()

        next_distance = self.distance + 1
        print(f"[curriculum] advancing to distance {next_distance} (left target moves from "
              f"x={self._left_x()} to x={SPAWN_X - next_distance}), carrying over "
              f"left_reward={self.left_value:.4f} as the starting point for the new search.")

        self.distance = next_distance
        self._reset_for_new_distance()
        return self._env_updates()


def main():
    curriculum = DistanceRewardCurriculum()
    try:
        run_training(
            left_reward=INITIAL_LEFT_REWARD,
            right_reward=RIGHT_REWARD,
            beginEntropy=BEGIN_ENTROPY,
            endEntropy=END_ENTROPY,
            lrDecayHorizon=LR_DECAY_HORIZON,
            entropyDecayHorizon=LR_DECAY_HORIZON,  # moot since begin==end
                                                     # entropy zeroes the
                                                     # decay term anyway,
                                                     # but set explicitly
                                                     # for clarity
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
    don't have to go open the CSV to see the answer. Also flags any
    distance that only got there via the give-up/best-effort fallback
    (converged=False) rather than a genuine 50/50 landing."""
    print(f"{'distance':>8}  {'left_x':>6}  {'left_reward':>12}  {'converged':>9}  "
          f"{'left%':>6}  {'hit%':>6}  {'attempts':>8}")
    for r in results:
        flag = "" if r["converged"] else "  <-- gave up, best-effort value"
        print(f"{r['distance']:>8}  {r['left_x']:>6}  {r['left_reward']:>12.4f}  "
              f"{str(r['converged']):>9}  {r['left_pct']:>6.2f}  {r['hit_pct']:>6.2f}  "
              f"{r['attempts']:>8}{flag}")


if __name__ == "__main__":
    main()