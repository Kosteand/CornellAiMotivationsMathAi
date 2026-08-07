import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import os
import sys
import inspect
import csv
import re
from collections import deque

from Utilities.SpaceEnv import *
from Utilities.HeatMap import *
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from functools import partial
from Utilities.MultiHeatMap import *
import cProfile
import pstats
import io
from PPO import *


useProfiler = False 
device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Running RL Training on: {device.upper()}")

#DI you want to train and create new weeights or use old weights
saveWeights = True
load_weights = True

# `python experiment2.py --force` (or -f) bypasses the skip-if-already-trained check below
# and retrains + overwrites the existing weights/plots for that exact label. Off by default
# so a plain `python experiment2.py` still resumes/skips like before.
FORCE_RETRAIN = "--force" in sys.argv or "-f" in sys.argv

# `python experiment2.py --record` (or -r) turns on writing to the results spreadsheet. Off
# by default so a plain `python experiment2.py` (e.g. a quick one-off/debug run) doesn't spam
# a new numbered CSV every time.
RECORD_RESULTS = "--record" in sys.argv or "-r" in sys.argv

TRAJ_DIR = "trajectory_plots"   # spawn/path/target visualizations go in their own folder
os.makedirs(TRAJ_DIR, exist_ok=True)
TRAJ_EPISODES = 100             # keep roughly the last N completed episodes (deque, approx)

PLOTS_DIR = "training_plots"    # hit-rate/value/entropy curves go here instead of loose in src/
os.makedirs(PLOTS_DIR, exist_ok=True)

LOG_DIR = "logs"                # diagnostic text logs (not plots) -- kept out of the way
os.makedirs(LOG_DIR, exist_ok=True)

# Spreadsheet of experiment results -- one row per finished trainAgent() run, appended as it
# goes (not rewritten), so a killed sweep keeps whatever rows it already produced. Plain CSV
# rather than .xlsx: no extra dependency (openpyxl isn't installed in the container), and CSV
# opens fine in Excel/Sheets/LibreOffice. A fresh numbered file is picked ONCE per script
# invocation (not per trainAgent call) -- every `python experiment2.py --record` gets its own
# experiment_results{N}.csv instead of everything piling into one growing file.
def _nextResultsCsvPath():
    existing = [f for f in os.listdir(".") if re.fullmatch(r"experiment_results\d*\.csv", f)]
    nums = [int(m.group(1)) if m.group(1) else 1
            for m in (re.fullmatch(r"experiment_results(\d*)\.csv", f) for f in existing)]
    nextNum = max(nums, default=0) + 1
    return "experiment_results.csv" if nextNum == 1 else f"experiment_results{nextNum}.csv"

RESULTS_CSV = _nextResultsCsvPath()
if RECORD_RESULTS:
    print(f"Recording results to: {RESULTS_CSV}")
RESULTS_CSV_COLUMNS = [
    "label", "runIndex", "nUpdates", "target0_award", "target1_award",
    "n_episodes_all", "timeout_all", "target0_all", "target1_all",
    "n_episodes_last100", "timeout_last100", "target0_last100", "target1_last100",
]

def _appendResultRow(row: dict):
    """Append one run's results as a row, writing the header only if the file is new.
    No-op unless --record/-r was passed -- see RECORD_RESULTS above."""
    if not RECORD_RESULTS:
        return
    file_exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


 # 3. Create Environment with Rendering
low = np.array([0, 0])
high = np.array([100, 100])
num_cols = high[0] - low[0] + 1
num_rows = high[1] - low[1] + 1
walls = np.zeros((num_cols, num_rows), dtype=bool)

spawn = np.array([5, 5])
target_coords = np.array([[35, 40], [70,20]])
# RQ: target 0 gives ZERO reward, target 1 gives 10, but BOTH are tracked by an N=1 map (same
# decode difficulty for each) -- sharper than the earlier double-reward test: does the agent
# still learn/visit target 0 at all when it has literally no payoff, or does it purely chase
# target 1? Isolates reward-driven preference from decode-difficulty preference.
target_awards = np.array([12, 12])

def _labelEnvContext():
    # same kwargs SpaceEnv.buildMaps hands to a heatmap partial, so a map built here
    # matches the one the env builds -- lets us read its real toString() for the label.
    return {'lowLeft': low, 'topRight': high, 'targetCords': target_coords,
            'spawn': spawn, 'targetAwards': target_awards}

def _fsSafe(s: str) -> str:
    """Collapse anything not [A-Za-z0-9._-] to '_' so a toString() is filename-safe."""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s).strip("_")

def _fmtNum(x) -> str:
    """Compact, filename-safe number for hyperparameter tags, e.g. 0.00005 -> '5e-05',
    2048 -> '2048'. Avoids literal '.' in the label (legal in filenames, but 'p' reads
    cleaner at a glance across many similarly-named runs)."""
    s = f"{x:g}" if isinstance(x, float) else str(x)
    return s.replace(".", "p")

def mapLabel(m) -> str:
    """Descriptive, unique, filesystem-safe label for a heatmap entry.

    Builds the map from the module env context (exactly as buildMaps does) and reads its
    toString(), which recurses through wrappers (MultiPolynomialInverse -> TargetWrap ->
    OptimalActionTarget) and encodes N + target index -- so nested configs get readable,
    distinct names. Falls back to a keyword-derived name if the map can't be built.
    """
    rng_state = np.random.get_state()   # building a map draws random weights; don't perturb training RNG
    try:
        if isinstance(m, partial):
            inst = m(**_labelEnvContext())
        elif isinstance(m, type):
            sig = inspect.signature(m)
            kwargs = {k: v for k, v in _labelEnvContext().items() if k in sig.parameters}
            inst = m(**kwargs)
        else:
            inst = m
        label = describeMap(inst)
        if label:
            return _fsSafe(label)
    except Exception:
        pass
    finally:
        np.random.set_state(rng_state)
    return _mapLabelFallback(m)

def _mapLabelFallback(m) -> str:
    """Keyword-derived label used only if a map can't be instantiated for its toString()."""
    if isinstance(m, partial):
        parts = [getattr(m.func, "__name__", str(m.func))]
        inner = m.keywords.get("inner_map_type")
        if inner is None and m.args:
            inner = m.args[0]
        if inner is not None:
            parts.append(mapLabel(inner))
        noise = m.keywords.get("noiseLevel")
        if noise is not None:
            parts.append(f"n{int(round(noise * 100))}")
        offset = m.keywords.get("offset")
        if offset is not None:
            parts.append(f"o{int(offset[0])}_{int(offset[1])}")
        targetIndex = m.keywords.get("targetIndex")
        if targetIndex is not None:
            parts.append(f"t{int(targetIndex)}")
        component = m.keywords.get("component")
        if component is not None:
            parts.append(f"c{int(component)}")
        N = m.keywords.get("N")
        if N is not None:
            parts.append(f"Nc{int(N)}")
        return "_".join(parts)
    if isinstance(m, type):
        ts = getattr(m, "toString", None)
        if ts is not None:
            try:
                return ts()
            except TypeError:
                pass
        return m.__name__
    return describeMap(m)

def makeEnv(mapTypes, stepPenalty, timeoutPenalty):

    def _init():
        # Create your specific env
        env = MazeEnv(low, high, spawn, target_awards, target_coords, heatMapTypes=mapTypes,
                       walls=walls, stepPenalty=stepPenalty, timeoutPenalty=timeoutPenalty)
        return env
    return _init



def trainAgent(mapsTypes, runIndex=None):
    """runIndex: set when doing repeated runs of an otherwise-IDENTICAL config (no seeding
    anywhere in this pipeline, so identical configs still land on genuinely different learned
    policies run to run -- this exists purely to keep repeats from colliding on the same
    label/weights, not because the config actually differs)."""
    hiddenDim = 128   # named here (not a literal in the PPOAgent(...) call below) so it can
                      # also feed the architecture tag in the label, below.

    #Defining core constants (moved above the label so they can feed the hyperparameter tag)
    criticLr = 0.0003  # reverted: the critic-LR-down experiment (3e-4 -> 1e-4) didn't change
                        # the plateau shape at N=10, so no reason to keep it off the config that
                        # actually produced the best N=7 result.
    actorLr = 0.00005
    nUpdates = 40  # back down from 75 for this sweep -- timeoutPenalty=-3 is new anyway (no
                   # existing data has it), so nothing strictly matches old runs either way;
                   # accepting all 8 pairs run fresh rather than trying to reuse old data.
    nStepsPerUpdate = 1024
    stepPenalty = 0
    timeoutPenalty = 0

    gamma = 0.99
    lam = 0.95
    beginEntropy = 0.05
    endEntropy = 0.000
    entropyBonus = beginEntropy

    # Skip-if-done: resume a killed sweep at the next untrained noise level.
    # A finished agent leaves weights tagged "UV" (just trained) which testValidity
    # later renames to "pass"/"fail" — if any of those exist, this level is done.
    # target_awards is part of the label too: identical maps with a different reward split
    # are a genuinely different experiment (e.g. equal reward vs one target paying double),
    # and must not collide on the same weight files / get silently skipped. The architecture
    # tag (MLP.describe) is part of it for the same reason: a changed network (layer count,
    # hidden width) is trained-from-scratch-incompatible with an old checkpoint, so it must
    # not collide with -- or be silently skipped in favor of -- weights saved under a
    # different architecture. Same logic now extends to the training hyperparameters
    # themselves (nUpdates, nStepsPerUpdate, entropy schedule, LRs): these have been getting
    # tuned run to run, and a changed hyperparameter set trains a genuinely different agent
    # that must not collide with -- or get silently skipped in favor of -- weights saved
    # under a different configuration.
    hpTag = (f"_hp_nu{_fmtNum(nUpdates)}_ns{_fmtNum(nStepsPerUpdate)}"
             f"_ent{_fmtNum(beginEntropy)}-{_fmtNum(endEntropy)}"
             f"_lrA{_fmtNum(actorLr)}-lrC{_fmtNum(criticLr)}"
             f"_g{_fmtNum(gamma)}-l{_fmtNum(lam)}"
             f"_sp{_fmtNum(stepPenalty)}_tp{_fmtNum(timeoutPenalty)}")
    label = ("".join(mapLabel(m) for m in mapsTypes)
             + "_awards" + "_".join(str(int(a)) for a in target_awards)
             + "_" + MLP.describe(hiddenDim)
             + hpTag
             + ("" if runIndex is None else f"_run{runIndex}"))
    already_trained = any(os.path.isfile(f"weights/actor_weights{label}{suf}.h5") for suf in ("UV", "pass", "fail"))
    if already_trained and not FORCE_RETRAIN:
        print(f"Skipping already-trained agent: {label}")
        return
    if already_trained and FORCE_RETRAIN:
        print(f"--force: retraining and overwriting existing weights for: {label}")

    nEnvs = 22

    env = gym.vector.SyncVectorEnv([makeEnv(mapsTypes, stepPenalty, timeoutPenalty) for _ in range(nEnvs)])
    #print("single obs space:", env.single_observation_space.shape)   # you say (1,)
    #print("n_envs:", env.num_envs)
    obsShape = env.single_observation_space.shape[0]
    actionShape = env.single_action_space.n



    # ent_coeff starts at beginEntropy -- overwritten every update by the entropyBonus anneal
    # schedule below, before agent.train_step is ever called, so this initial value is really
    # just documentation of where the schedule starts (not a separate, independent setting).
    agent = PPOAgent(obs_dim=obsShape,hidden_dim=hiddenDim ,out=actionShape, device=device, critic_lr=criticLr, actor_lr=actorLr, ent_coeff=beginEntropy,n_steps=nStepsPerUpdate)

    # Vector-specific wrapper
    envWrapper = gym.wrappers.vector.RecordEpisodeStatistics(env, buffer_length=10000)
    criticLosses = []
    actorLosses = []
    entropies = []
    hitTarget = []
    valueEstimates = []   # mean critic V per update
    current_max_steps = 250
    min_steps = 250
    step_decay = 15

    # Arena-size curriculum: start small (1/4 of the 100x100 "OG" arena) and grow to full
    # size by the end of training, instead of always drawing from the full random range.
    # Linear in samplePhase/nUpdates (nUpdates-agnostic), same style as the entropy anneal
    # below -- unlike the max_steps curriculum above, which uses a step_decay/%100 schedule
    # that's effectively a no-op at the nUpdates values actually used lately (100 doesn't
    # divide evenly into a 60-150 update run).
    startArenaSize = 25
    endArenaSize = 100
    resetOptions = {
    "randomSpawn": True,
    "randomSize": True,
    "randomTargetCoords": True,
    "max_steps": current_max_steps,
    "maxArenaSize": startArenaSize + 1   # +1: np_random.integers() upper bound is exclusive
}

    # Trajectory tracking for the visualizer: spawn/size/targets all randomize every episode
    # (resetOptions above), so raw (x,y) isn't comparable across episodes -- episode_paths
    # records each env's CURRENT episode as absolute coords; completed_episodes (capped at
    # TRAJ_EPISODES, oldest dropped first) holds the ~last N finished episodes for plotting,
    # each normalized relative to its own spawn at plot time.
    episode_paths = [[] for _ in range(nEnvs)]
    episode_targets = [None] * nEnvs
    completed_episodes = deque(maxlen=TRAJ_EPISODES)

    buffer = RolloutBuffer()

    # Diagnostic log (not a plot): how often the policy SAMPLES a different action than its
    # own argmax, per update. High/rising values in a region mean the near-tied-logits +
    # nonzero-entropy-floor combo is producing real decision noise -- e.g. the back-and-forth
    # dithering seen in some timeout trajectories -- rather than a deliberate, confident policy.
    mismatch_log_path = os.path.join(LOG_DIR, "action_mismatch_" + label + ".csv")
    mismatch_log = open(mismatch_log_path, "w")
    mismatch_log.write("update,ent_coeff,measured_entropy,mismatch_rate\n")

    for samplePhase in tqdm(range(nUpdates)):
        # floor at endEntropy (was hard-coded 0.05, which overrode the anneal target
        # and kept the entropy bonus high enough to swamp the sparse reward signal)
        entropyBonus = max(endEntropy, beginEntropy - (samplePhase*2 / nUpdates) * (beginEntropy - endEntropy))
        # This schedule was being computed every update but never actually applied -- PPOAgent
        # was built once with a hardcoded ent_coeff=0.01 and that never changed, so the anneal
        # had zero effect on training. Actually wire it in.
        agent.ent_coeff = entropyBonus

        # Same fraction-based anneal style as entropyBonus above, growing startArenaSize ->
        # endArenaSize over the run, but ramping over (nUpdates - plateauUpdates) so the last
        # plateauUpdates updates train at full size instead of only just reaching it on the
        # final update. Read by the NEXT reset() call (either the samplePhase==0 initial
        # reset, or the end-of-update "new buffer" reset below), so episodes sampled during
        # update `samplePhase` use the size computed here.
        plateauUpdates = 10
        arenaSizeFrac = min(1.0, samplePhase / max(nUpdates - plateauUpdates, 1))
        current_max_arena_size = int(round(startArenaSize + (endArenaSize - startArenaSize) * arenaSizeFrac))
        resetOptions["maxArenaSize"] = current_max_arena_size + 1   # +1: integers() upper bound is exclusive
        # Printed only on the two milestones (start, and the update where it first reaches
        # full size), not every update -- makes the actual schedule directly visible in the
        # console instead of something you have to trust from reading the formula.
        if samplePhase == 0:
            print(f"Arena curriculum: starting at max size {current_max_arena_size} "
                  f"(update 0), reaching full size {endArenaSize} by update "
                  f"{nUpdates - plateauUpdates} of {nUpdates}")
        elif current_max_arena_size == endArenaSize and _prev_max_arena_size != endArenaSize:
            print(f"Arena curriculum: reached full size {endArenaSize} at update {samplePhase}")
        _prev_max_arena_size = current_max_arena_size

        if samplePhase % 100 == 0 and current_max_steps > min_steps:
            current_max_steps -= step_decay

            env.set_attr("max_steps", current_max_steps) 
            # New step val in all 4 envs
            print(f"Tightening the clock! New Max Steps: {current_max_steps}")
        # anneal gamma if needed:gamma = 0.95 + 0.04 * (current_max_steps - min_steps) / (1000 - min_steps)
            
        epValuePreds = torch.zeros(nStepsPerUpdate, nEnvs, device=device)
        epRewards = torch.zeros(nStepsPerUpdate, nEnvs, device=device)
        epActionLogProbs = torch.zeros(nStepsPerUpdate, nEnvs, device=device)
        masks = torch.zeros(nStepsPerUpdate, nEnvs, device=device)
        sampled_vs_greedy_mismatches = 0   # how many (step, env) pairs sampled a non-argmax action

        if(samplePhase==0):
            states,__ = envWrapper.reset(options=resetOptions)
            for i in range(nEnvs):
                episode_paths[i] = [env.envs[i].coords.copy()]
                episode_targets[i] = env.envs[i].targetCords.copy()
        prev_done = np.zeros(nEnvs, dtype=bool)
        for step in range(nStepsPerUpdate):
            cur_states = states
            actions, actionLogProbs, stateValuePreds, greedyActions = agent.take_action(states, return_greedy=True)
            sampled_vs_greedy_mismatches += int((actions != greedyActions).sum().item())
            nextStates, rewards, terminated, truncated,infos = envWrapper.step(
            actions.cpu().numpy())

            # Buffer boundary = a NEW buffer next: truncate every episode still running (i.e.
            # not just-terminated) so GAE bootstraps from V(s); the envs are reset after this
            # loop to start fresh episodes. Truncation only ever happens here, never in the env.
            if step == nStepsPerUpdate - 1:
                terminated = np.asarray(terminated, dtype=bool)
                truncated = np.asarray(truncated, dtype=bool) | ~terminated

            step_next = np.array(nextStates, dtype=np.float64).copy()

            finals = infos.get("final_observation", None)

            if finals is not None:
                mask = infos.get("_final_observation",
                    np.array([o is not None for o in finals]))
                for i in np.where(mask)[0]:
                    step_next[i] = finals[i]




            '''
            if truncated.any():
                print("version:", gym.__version__)
                print("infos keys:", list(infos.keys()))
                i = int(np.where(truncated)[0][0])
                print("next_states[i]:", nextStates[i])
                if "final_observation" in infos:
                    print("final_obs[i]:", infos["final_observation"][i])'''
            # Only real episode endings (target hit OR out-of-steps termination) count toward
            # the hit rate. The buffer-boundary truncation above is an incomplete episode and
            # is deliberately NOT counted.
            term_mask = np.asarray(terminated, dtype=bool)
            # "targetHit" can be absent from infos entirely: SpaceEnv.reset() returns info={},
            # and any env whose autoreset fired THIS step contributes that empty dict instead
            # of a step() result -- if that happens to every env this step, the key vanishes
            # from the aggregated infos altogether. Only ever read where term_mask gates it
            # (a real termination), which is where it's reliably present.
            th = np.asarray(infos.get("targetHit", np.full(nEnvs, -1)))
            if term_mask.any():
                for i in np.where(term_mask)[0]:
                    hitTarget.append(int(th[i]))

            # Trajectory tracking. If env i was done last step, gymnasium's autoreset already
            # fired inside THIS step() call -- env.envs[i].coords is the fresh spawn, and the
            # given action was a no-op for it (see the `valid = ~prev_done` masking below), so
            # start a new path here rather than appending. Otherwise append the real step.
            for i in range(nEnvs):
                if prev_done[i]:
                    episode_paths[i] = [env.envs[i].coords.copy()]
                    episode_targets[i] = env.envs[i].targetCords.copy()
                else:
                    episode_paths[i].append(env.envs[i].coords.copy())
                if term_mask[i]:   # real ending only (hit or timeout) -- matches hitTarget's gating
                    completed_episodes.append({
                        "path": np.array(episode_paths[i]),
                        "targets": episode_targets[i].copy(),
                        "hit": int(th[i]),
                    })

            valid = ~prev_done
            done = np.logical_or(terminated, truncated)
            buffer.append({
                "states":      np.asarray(cur_states, dtype=np.float64),  # PRE-step obs
                "actions":     actions.cpu().numpy(),
                "log_probs":   actionLogProbs.cpu().numpy(),
                "values":      stateValuePreds.cpu().numpy(),
                "rewards":     np.asarray(rewards, dtype=np.float32),
                "next_states": step_next,
                "terminated":  np.asarray(terminated, dtype=np.float32),
                "truncated":   np.asarray(truncated, dtype=np.float32),
                "valid":       valid.astype(np.float32),
            })


        

            prev_done = done
            states = nextStates


            
            dones = torch.tensor(
            [term or trunc for term, trunc in zip(terminated, truncated)],
            dtype=torch.float32, device=device)
        
        
            
            epValuePreds[step] = torch.squeeze(stateValuePreds)
            epRewards[step] = torch.tensor(rewards, device=device)
            epActionLogProbs[step] = actionLogProbs
            
            
            
            masks[step] = torch.tensor([not (term or trunc) for term, trunc in zip(terminated, truncated)])

            
            # MAYBE DELETE TODO
        #epRewards = (epRewards - epRewards.mean()) / (epRewards.std() + 1e-8)

        # New buffer -> fresh episodes: the boundary truncation above cut every still-running
        # episode, so reset all envs now to start new ones (max_steps kept in sync in case
        # it's ever annealed).
        resetOptions["max_steps"] = current_max_steps
        states, __ = envWrapper.reset(options=resetOptions)
        # Re-seed trajectory tracking to match: prev_done gets zeroed at the top of the next
        # update's loop (line ~225), so without this, the tracker treats the next step as a
        # normal continuation and APPENDS the new episode's fresh spawn onto whatever was left
        # in episode_paths[i] from before this reset -- splicing two unrelated episodes into
        # one path (matplotlib then draws a straight "teleport" line across the gap).
        for i in range(nEnvs):
            episode_paths[i] = [env.envs[i].coords.copy()]
            episode_targets[i] = env.envs[i].targetCords.copy()

        # update the actor and critic networks
        results = agent.train_step(buffer)

        # results.entropy is the actual measured policy entropy (EMA-tracked in train_step),
        # distinct from entropyBonus (the coefficient/weight applied to it in the loss) --
        # previously computed by train_step and silently discarded every update.
        if results.entropy is not None:
            entropies.append(results.entropy)

        mismatch_rate = sampled_vs_greedy_mismatches / (nStepsPerUpdate * nEnvs)
        mismatch_log.write(f"{samplePhase},{entropyBonus:.6f},"
                            f"{results.entropy if results.entropy is not None else ''},"
                            f"{mismatch_rate:.6f}\n")
        mismatch_log.flush()

        vals = np.array(buffer["values"])          # [T, N]
        valueEstimates.append(float(vals.mean()))
        buffer = RolloutBuffer()

    """Stuff for that profiler"""
    if(useProfiler):
        profiler.disable()
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)
        print(stream.getvalue())
        profiler.dump_stats("./visualize/profile_output.prof")

    """ plot the results """

    # %matplotlib inline
    

    rolling_length = 20
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
    fig.suptitle(
        f"Training plots for {agent.__class__.__name__} in the 2d space \n \
                (n_envs={nEnvs}, n_steps_per_update={nStepsPerUpdate})"
    )




    n_targets = len(target_coords)
    window = 100

    outcomes = np.asarray(hitTarget, dtype=int)
    if outcomes.size == 0:
        print("no completed episodes to plot")
    else:
        E = outcomes.size
        # rolling hit-rate per target: at episode e, mean over the trailing `window` episodes
        per_target_curves = np.zeros((n_targets, E))
        any_curve = np.zeros(E)
        timeout_curve = np.zeros(E)
        for e in range(E):
            lo = max(0, e - window + 1)
            w = outcomes[lo:e + 1]
            for t in range(n_targets):
                per_target_curves[t, e] = np.mean(w == t)
            any_curve[e] = np.mean(w >= 0)
            timeout_curve[e] = np.mean(w < 0)   # outcomes==-1 -> ran out of steps, no hit

        # Miss breakdown: is the "any target" shortfall mostly timeouts (exploration/pathing
        # failure) or something else? Printed for the whole run AND for just the most recent
        # window, since early-training randomness otherwise drags the overall average down.
        def _breakdown(w):
            n = w.size
            return (f"timeout {np.mean(w<0):.1%}  " +
                    "  ".join(f"target{t} {np.mean(w==t):.1%}" for t in range(n_targets)) +
                    f"  (n={n})")
        print(f"Outcome breakdown, all {E} completed episodes: {_breakdown(outcomes)}")
        print(f"Outcome breakdown, last {min(window, E)} episodes: {_breakdown(outcomes[-window:])}")

        # One row in the results spreadsheet per finished run -- same numbers as the two
        # breakdown lines above, just machine-readable across runs instead of console-only.
        def _rates(w):
            return (int(w.size), float(np.mean(w < 0)),
                    float(np.mean(w == 0)), float(np.mean(w == 1)))
        n_all, timeout_all, target0_all, target1_all = _rates(outcomes)
        n_last, timeout_last, target0_last, target1_last = _rates(outcomes[-window:])
        _appendResultRow({
            "label": label,
            "runIndex": "" if runIndex is None else runIndex,
            "nUpdates": nUpdates,
            "target0_award": int(target_awards[0]),
            "target1_award": int(target_awards[1]) if len(target_awards) > 1 else "",
            "n_episodes_all": n_all,
            "timeout_all": timeout_all,
            "target0_all": target0_all,
            "target1_all": target1_all,
            "n_episodes_last100": n_last,
            "timeout_last100": timeout_last,
            "target0_last100": target0_last,
            "target1_last100": target1_last,
        })

        fig, ax = plt.subplots(figsize=(10, 5))
        for t in range(n_targets):
            ax.plot(per_target_curves[t], label=f"target {t}")
        ax.plot(any_curve, label="any target", color="black", linestyle="--", linewidth=2)
        ax.plot(timeout_curve, label="timeout", color="red", linestyle=":", linewidth=2)
        ax.set_title(f"Rolling hit rate (trailing {window} episodes)\n{label}", fontsize=9)
        ax.set_xlabel("Episode")
        ax.set_ylabel("P(outcome)")
        ax.set_ylim(0, 1)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "hit_rate_curve"+label+".png"))
        plt.close(fig)



    if len(valueEstimates) > 0:
        ve = np.asarray(valueEstimates, dtype=np.float64)
        roll = 20
        ve_ma = np.convolve(ve, np.ones(roll)/roll, mode="valid") if ve.size >= roll else ve
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ve_ma)
        ax.set_title(f"Mean critic value estimate V(s) (rolling {roll})\n{label}", fontsize=9)
        ax.set_xlabel("Update")
        ax.set_ylabel("mean V(s)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "value_estimate"+label+".png"))
        plt.close(fig)



    if len(entropies) > 0:
        ent = np.asarray(entropies, dtype=np.float64)
        roll = 20
        ent_ma = np.convolve(ent, np.ones(roll)/roll, mode="valid") if ent.size >= roll else ent
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ent_ma, label="measured policy entropy (rolling)")
        ax.set_title(f"Policy entropy (rolling {roll})\n{label}", fontsize=9)
        ax.set_xlabel("Update")
        ax.set_ylabel("entropy")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "entropy_curve"+label+".png"))
        plt.close(fig)



    # Trajectory visualizer: spawn, agent path, and the two targets (colored distinctly) for
    # ~the last TRAJ_EPISODES completed episodes. Everything is plotted RELATIVE TO SPAWN,
    # since spawn/arena size/target positions all randomize every episode (resetOptions) --
    # raw (x,y) isn't comparable across episodes, but "offset from spawn" is.
    if len(completed_episodes) == 0:
        print("no completed episodes to plot trajectories for")
    else:
        from matplotlib.lines import Line2D
        from matplotlib.colors import ListedColormap
        from matplotlib.collections import LineCollection
        target_colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]  # by target index

        def _color_for(hit):
            return "gray" if hit < 0 else target_colors[hit % len(target_colors)]

        # Optimal-action backdrop: OptimalActionTarget.map(coords) is the ground-truth greedy
        # action (nearest target, ignoring walls -- SpaceEnv.reset always zeroes self.walls for
        # randomSize runs, so "nearest target by L2" IS the true shortest-path action here).
        # Shading every grid cell by that action lets a path be checked visually against what
        # the agent's observation actually encodes, distinct from the target-index colors above.
        # Distinct HUES, not light/dark shades of the same hue: paired shades (e.g. light vs
        # dark blue for +Y/-Y) collapse toward the same washed-out pastel once alpha-blended
        # over white (checked: #a6cee3 and #1f78b4 land within ~50/255 of each other at
        # alpha=0.35) while the legend swatches render at full opacity -- so the legend looks
        # clearly 4-way distinct but the actual backdrop doesn't, and a real per-cell action
        # change reads as "no visible difference". Four separate hues survive alpha dilution.
        _action_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]  # 0:+Y 1:-Y 2:+X 3:-X
        _action_alpha = 0.45
        _action_cmap = ListedColormap(_action_colors)
        _action_labels = ["+Y (up)", "-Y (down)", "+X (right)", "-X (left)"]

        def _plot_action_backdrop(ax, spawn_abs, targets_abs, path_rel, targets_rel, pad=2):
            xy = np.concatenate([path_rel, targets_rel], axis=0)
            x0, x1 = int(np.floor(xy[:, 0].min())) - pad, int(np.ceil(xy[:, 0].max())) + pad
            y0, y1 = int(np.floor(xy[:, 1].min())) - pad, int(np.ceil(xy[:, 1].max())) + pad
            xs = np.arange(x0, x1 + 1)
            ys = np.arange(y0, y1 + 1)
            oracle = OptimalActionTarget(targetCords=targets_abs,
                                          lowLeft=np.array([x0, y0]), topRight=np.array([x1, y1]))
            field = np.zeros((len(ys), len(xs)))
            for yi, y in enumerate(ys):
                for xi, x in enumerate(xs):
                    field[yi, xi] = oracle.map(np.array([x + spawn_abs[0], y + spawn_abs[1]]))
            # interpolation="nearest": imshow's default antialiasing blends colors across
            # cell edges, making a genuinely sharp per-cell action boundary look like it
            # "leaks"/fades smoothly between regions -- nearest keeps every cell's edge crisp,
            # matching the fact that OptimalActionTarget.map() is a hard per-cell decision.
            ax.imshow(field, origin="lower",
                      extent=[x0 - 0.5, x1 + 0.5, y0 - 0.5, y1 + 0.5],
                      cmap=_action_cmap, vmin=0, vmax=3, alpha=_action_alpha, zorder=0, aspect="auto",
                      interpolation="nearest")

        # 1) Aggregate overlay: every episode's path + target positions, relative to its own
        # spawn, drawn semi-transparent so density/clustering is visible.
        fig, ax = plt.subplots(figsize=(8, 8))
        for ep in completed_episodes:
            spawn = ep["path"][0]
            path_rel = ep["path"] - spawn
            targets_rel = ep["targets"] - spawn
            style = "--" if ep["hit"] < 0 else "-"
            ax.plot(path_rel[:, 0], path_rel[:, 1], color=_color_for(ep["hit"]),
                     alpha=0.25, linewidth=1, linestyle=style)
            for t, tc in enumerate(targets_rel):
                ax.scatter(*tc, color=target_colors[t % len(target_colors)],
                           alpha=0.15, s=40, marker="*", zorder=2)
        ax.scatter(0, 0, color="black", marker="x", s=80, zorder=3)
        legend_elems = [Line2D([0], [0], marker="x", color="black", linestyle="", label="spawn")]
        for t in range(len(target_coords)):
            legend_elems.append(Line2D([0], [0], marker="*", color=target_colors[t % len(target_colors)],
                                        linestyle="", label=f"target {t}"))
        legend_elems.append(Line2D([0], [0], color="gray", linestyle="--", label="timeout path"))
        ax.legend(handles=legend_elems, loc="best", fontsize=8)
        ax.set_title(f"Agent trajectories, relative to spawn (last ~{len(completed_episodes)} "
                     f"episodes)\n{label}", fontsize=9)
        ax.set_xlabel("x - spawn_x")
        ax.set_ylabel("y - spawn_y")
        # adjustable="box" (not "datalim"): datalim EXPANDS the shorter axis's view limits to
        # force 1:1 scaling, which for a mostly-vertical-or-horizontal path pads in a ton of
        # empty space and visually strands the spawn marker/path far from the actual data --
        # looks like a "broken" disconnected trajectory even though the underlying coords are
        # fine. "box" instead reshapes the axes' physical box, keeping the view tight to data.
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.savefig(os.path.join(TRAJ_DIR, "trajectories_overlay_" + label + ".png"))
        plt.close(fig)

        # 2) One individual graph per episode (all ~TRAJ_EPISODES kept, oldest to newest),
        # in its own per-run subfolder so ~100 small files don't clutter TRAJ_DIR.
        episode_dir = os.path.join(TRAJ_DIR, label)
        os.makedirs(episode_dir, exist_ok=True)
        # alpha matched to the backdrop's alpha so the legend swatches look like what's
        # actually on the plot, instead of full-opacity reference colors that visually promise
        # more contrast than the (alpha-blended) backdrop actually delivers.
        action_legend_elems = [Line2D([0], [0], marker="s", color=_action_colors[a], linestyle="",
                                       markersize=10, alpha=_action_alpha,
                                       label=f"optimal: {_action_labels[a]}")
                                for a in range(4)]
        for k, ep in enumerate(completed_episodes):
            spawn = ep["path"][0]
            path_rel = ep["path"] - spawn
            targets_rel = ep["targets"] - spawn
            n_steps = len(path_rel) - 1
            fig, axx = plt.subplots(figsize=(5, 5))
            _plot_action_backdrop(axx, spawn, ep["targets"], path_rel, targets_rel)
            # Colored by step index (dark->light), not a flat color: the LINE only shows
            # where the agent went, not how long it spent there -- an agent dithering back and
            # forth over the same few cells draws the exact same short-looking segment as a
            # single clean pass, silently hiding hundreds of wasted steps. A time gradient
            # makes revisited cells show mixed colors instead of looking identical to a
            # straight, fast traversal.
            segments = np.stack([path_rel[:-1], path_rel[1:]], axis=1)
            lc = LineCollection(segments, cmap="plasma", zorder=1, linewidth=1.5)
            lc.set_array(np.arange(len(segments)))
            axx.add_collection(lc)
            for t, tc in enumerate(targets_rel):
                axx.scatter(*tc, color=target_colors[t % len(target_colors)], s=80, marker="*", zorder=2)
            axx.scatter(0, 0, color="black", marker="x", s=60, zorder=3)
            outcome = f"hit target {ep['hit']}" if ep["hit"] >= 0 else "timeout"
            axx.set_title(f"episode {k}: {outcome} ({n_steps} steps, dark->light over time)", fontsize=8)
            axx.set_xlabel("x - spawn_x")
            axx.set_ylabel("y - spawn_y")
            axx.set_aspect("equal", adjustable="box")
            axx.legend(handles=action_legend_elems, loc="upper left", fontsize=6,
                       bbox_to_anchor=(1.02, 1.0))
            plt.tight_layout()
            plt.savefig(os.path.join(episode_dir, f"episode_{k:03d}.png"))
            plt.close(fig)



    if not os.path.exists("weights"):
        os.mkdir("weights")
    actor_weights_path = "weights/actor_weights"+label+"UV"+".h5"
    critic_weights_path = "weights/critic_weights"+label+"UV"+".h5"
    """ save network weights """
    torch.save(agent.actor.state_dict(), actor_weights_path)
    torch.save(agent.critic.state_dict(), critic_weights_path)

    mismatch_log.close()
    print(f"action mismatch log: {mismatch_log_path}")

    env.close()

def _targetPoly(N, targetIndex=None):
    """One MultiPolynomialInverse(N) channel block. targetIndex=None -> OptimalActionTarget
    points at whichever target is nearest (i.e. "both"); an index -> TargetWrap pins it to
    that one target."""
    inner = OptimalActionTarget if targetIndex is None else \
        partial(TargetWrap, inner_map_type=OptimalActionTarget, targetIndex=targetIndex)
    return partial(MultiPolynomialInverse, N=N, map=inner)

def _channelCount(entry):
    """How many observation channels one heatMapTypes entry contributes (mirrors the
    obs_size computation in SpaceEnv.__init__)."""
    target = entry.func if isinstance(entry, partial) else entry
    if isinstance(target, type) and issubclass(target, MultiHeatMappable) and isinstance(entry, partial):
        n = entry.keywords.get("N")
        if n is None and entry.args:
            n = entry.args[0]
        return int(n)
    return 1

def padToChannels(mapsList, total):
    """Pad a heatMapTypes list with ZeroMap entries (always read 0) until it contributes
    exactly `total` observation channels, so every config in a comparison sweep gives the
    actor the same input size regardless of how many real channels it uses."""
    used = sum(_channelCount(m) for m in mapsList)
    if used > total:
        raise ValueError(f"{mapsList} already uses {used} channels > target {total}")
    return mapsList + [ZeroMap] * (total - used)

# N=10 showed the same fast-rise-then-noisy-plateau as N=7 with none of the usual suspects
# (decode SNR, critic LR, exploration noise) explaining it -- back off to N=7, and pin it to
# target 0 only (TargetWrap via targetIndex=0) instead of nearest-of-both, to isolate
# single-target pursuit from the target-selection behavior seen in the boundary-defect episodes.
##trainAgent([_targetPoly(7, targetIndex=0)])

# Asymmetric decode difficulty per target: an N=1 map pinned to target 0 (coarse signal) and
# a separate N=7 map pinned to target 1 (fine signal), both fed to the same agent at once --
# does the agent still reliably pursue target 0 despite its much noisier/harder-to-decode
# channel, or does it end up favoring target 1 just because that signal is cleaner?
#trainAgent([_targetPoly(1, targetIndex=0), _targetPoly(7, targetIndex=1)])

# Single N=1 map pointing at only target 0 -- no signal about target 1 at all. Sanity check:
# can the agent still learn to reliably reach the one target it actually gets information
# about, in isolation, with the simplest possible (N=1) encoding?

# Repeat runs: same config repeated 10x (not a sweep -- target_awards fixed at [12,12] every
# time). No seeding anywhere in this pipeline, so identical configs still land on genuinely
# Reward sweep: target 0 fixed at 12, target 1 stepping 12 -> 32 by 4 (12, 16, 20, 24, 28, 32),
# for both the N=1-vs-7 and N=3-vs-7 decode-difficulty pairings. Each award value already makes
# each award value x 10 repeats (no seeding anywhere in this pipeline, so identical configs
# still land on genuinely different learned policies -- 10 repeats per point denoises the
# reward-preference curve instead of judging each award value off a single noisy run).
# runIndex keeps each repeat's weights/plots/label from colliding. Results land in RESULTS_CSV,
# one row per run -- 6 award values x 10 repeats x 2 N-pairings = 120 runs total.
def runCrossoverSearch(a, b, award_range, vary=1, repeats=5):
    """Denser, INFORMED search for the target-preference crossover reward (the target1_award
    at which preference flips from target0 to target1) for the (N=a -> target0, N=b ->
    target1) decode-difficulty pairing. `award_range` should already be centered on where the
    crossover is expected -- e.g. from the earlier coarse sweep (award step 4, range 12->32):
    N=1-vs-7 crossover ~= 18.8, N=3-vs-7 crossover ~= 14.2. This refines those estimates with
    step-1 resolution instead of re-searching the whole 12->32 range blind. target0 stays
    fixed at 12; `repeats` repeats per award value, same denoising rationale as before.

    Reusable for future (a, b) pairs too -- just pass a new pairing and an award_range
    centered on wherever that pairing's crossover is expected to land.

    vary=1 (default): target0 stays fixed at 12, target1 sweeps award_range -- finds how much
    MORE reward the harder-to-decode b needs to overcome a's decode-ease advantage.
    vary=0: target1 stays fixed at 12, target0 sweeps award_range instead -- the same
    crossover question approached from the other side (how much LESS reward can the
    easier-to-decode a tolerate before losing its natural preference).
    """
    global target_awards
    for award in award_range:
        target_awards = np.array([12, award]) if vary == 1 else np.array([award, 12])
        for run_idx in range(repeats):
            trainAgent([_targetPoly(a, targetIndex=0), _targetPoly(b, targetIndex=1)], runIndex=run_idx)


# Registry of REAL, measured crossover points (target1_award at which preference flips,
# target0 fixed at 12) -- from completed runCrossoverSearch sweeps. This replaces guessing
# each new pair's search window by hand (gap * fudge-factor): _estimateCrossoverCenter fits a
# line through (decode_gap, crossover-12) over whatever's in here and evaluates it at the new
# pair's gap. With only 2 points that line IS just what the manual estimates already were --
# the payoff is that adding a 3rd, 4th, ... real point (as sweeps finish) automatically
# sharpens every future estimate instead of re-guessing from scratch each time.
KNOWN_CROSSOVERS = {
    (1, 7): 18.8,
    (3, 7): 14.2,
}

def _estimateCrossoverCenter(a, b):
    """Fit of (decode_gap = b-a) -> (crossover - 12) over KNOWN_CROSSOVERS, evaluated at this
    pair's gap. Forced through the origin (gap=0 -> excess=0), NOT a free 2-parameter line --
    at gap=0, a and b have identical decode difficulty, so the crossover MUST be exactly 12
    (no reward bias needed for a symmetric pairing). With only 2-3 real data points an
    unconstrained line isn't well-determined and can extrapolate below 12 at small gaps
    (checked: it did -- gap=1 gave ~7.3, which doesn't make sense), so this uses the known
    origin as a real constraint instead of a free-floating intercept."""
    gaps = np.array([bb - aa for (aa, bb) in KNOWN_CROSSOVERS.keys()], dtype=float)
    excess = np.array([v - 12 for v in KNOWN_CROSSOVERS.values()], dtype=float)
    slope = float(np.sum(gaps * excess) / np.sum(gaps ** 2))   # least-squares through origin
    return 12 + slope * (b - a)

def systematicCrossoverSweep(pairs, n_points=5, step=1, repeats=5):
    """For each (a,b) in `pairs`: estimate its crossover center via _estimateCrossoverCenter
    and run runCrossoverSearch over `n_points` award values around it (step resolution `step`,
    `repeats` runs per award value). Skips any pair already in KNOWN_CROSSOVERS -- that's real
    completed data, not something to re-run. Prints the estimated center and search range for
    each pair before running it, so the plan is visible before the (potentially long) sweep
    actually starts."""
    for a, b in pairs:
        if (a, b) in KNOWN_CROSSOVERS:
            print(f"Skipping ({a},{b}): already have real crossover data ({KNOWN_CROSSOVERS[(a, b)]})")
            continue
        center = _estimateCrossoverCenter(a, b)
        lo = max(1, int(round(center)) - n_points // 2)
        print(f"({a},{b}): estimated crossover ~{center:.1f}, "
              f"searching target1_award {lo}-{lo + (n_points - 1) * step}")
        runCrossoverSearch(a, b, range(lo, lo + n_points * step, step), repeats=repeats)


# Budget cap: ~100 runs for now. 8 pairs x 4 award points x 3 repeats = 96 runs. Pairs chosen
# to span decode-gap 1 through 5 with more than one anchor per gap where possible (so we can
# check whether crossover really only depends on the gap b-a, or also on a/b individually) --
# includes the 4 originally-requested pairs (1,6)/(1,5)/(3,6)/(3,5) plus 4 more for spread.
_small_pairs = [
    (1, 6),  # gap 5
    (2, 7),  # gap 5, different anchor
    (1, 5),  # gap 4
    (2, 6),  # gap 4, different anchor
    (3, 6),  # gap 3
    (4, 7),  # gap 3, different anchor
    (3, 5),  # gap 2
    (5, 6),  # gap 1
]
systematicCrossoverSweep(_small_pairs, n_points=4, repeats=3)


target_awards = [12,12]
trainAgent([_targetPoly(7)])
