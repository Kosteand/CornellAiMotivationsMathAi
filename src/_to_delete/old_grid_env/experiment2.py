import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import os
import inspect

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


 # 3. Create Environment with Rendering
low = np.array([0, 0])
high = np.array([100, 100])
num_cols = high[0] - low[0] + 1
num_rows = high[1] - low[1] + 1
walls = np.zeros((num_cols, num_rows), dtype=bool)

spawn = np.array([5, 5])
target_coords = np.array([[35, 40], [70,20]])
# RQ: target 1 gives DOUBLE the reward of target 0, but BOTH are tracked by an N=1 map (same
# decode difficulty for each) -- isolates reward-driven preference from decode-difficulty
# preference (complements the earlier equal-reward/different-N experiments).
target_awards = np.array([20, 10])

def _labelEnvContext():
    # same kwargs SpaceEnv.buildMaps hands to a heatmap partial, so a map built here
    # matches the one the env builds -- lets us read its real toString() for the label.
    return {'lowLeft': low, 'topRight': high, 'targetCords': target_coords,
            'spawn': spawn, 'targetAwards': target_awards}

def _fsSafe(s: str) -> str:
    """Collapse anything not [A-Za-z0-9._-] to '_' so a toString() is filename-safe."""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s).strip("_")

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

def makeEnv(mapTypes):

    def _init():
        # Create your specific env
        env = MazeEnv(low, high, spawn, target_awards, target_coords, heatMapTypes=mapTypes, walls=walls)
        return env
    return _init



def trainAgent(mapsTypes):
    # Skip-if-done: resume a killed sweep at the next untrained noise level.
    # A finished agent leaves weights tagged "UV" (just trained) which testValidity
    # later renames to "pass"/"fail" — if any of those exist, this level is done.
    # target_awards is part of the label too: identical maps with a different reward split
    # are a genuinely different experiment (e.g. equal reward vs one target paying double),
    # and must not collide on the same weight files / get silently skipped.
    label = "".join(mapLabel(m) for m in mapsTypes) + "_awards" + "_".join(str(int(a)) for a in target_awards)
    if any(os.path.isfile(f"weights/actor_weights{label}{suf}.h5") for suf in ("UV", "pass", "fail")):
        print(f"Skipping already-trained agent: {label}")
        return

    nEnvs = 22

    env = gym.vector.SyncVectorEnv([makeEnv(mapsTypes) for _ in range(nEnvs)])
    #print("single obs space:", env.single_observation_space.shape)   # you say (1,)
    #print("n_envs:", env.num_envs) 
    obsShape = env.single_observation_space.shape[0]
    actionShape = env.single_action_space.n

    #Defining core constants
    criticLr = 0.0001
    actorLr = 0.00005
    nUpdates = 500
    nStepsPerUpdate = 256

    gamma = 0.99
    lam = 0.95
    beginEntropy = 0.05
    endEntropy = 0.005
    entropyBonus = beginEntropy



    agent = PPOAgent(obs_dim=obsShape,hidden_dim=64 ,out=actionShape, device=device, critic_lr=criticLr, actor_lr=actorLr, ent_coeff=0.01,n_steps=nStepsPerUpdate)

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
    resetOptions = {
    "randomSpawn": True,
    "randomSize": True, 
    "randomTargetCoords": True,
    "max_steps": current_max_steps
}
        
   
    buffer = RolloutBuffer()
    for samplePhase in tqdm(range(nUpdates)):
        # floor at endEntropy (was hard-coded 0.05, which overrode the anneal target
        # and kept the entropy bonus high enough to swamp the sparse reward signal)
        entropyBonus = max(endEntropy, beginEntropy - (samplePhase*2 / nUpdates) * (beginEntropy - endEntropy))
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

        if(samplePhase==0):
            states,__ = envWrapper.reset(options=resetOptions)
        prev_done = np.zeros(nEnvs, dtype=bool)        
        for step in range(nStepsPerUpdate):
            cur_states = states
            actions, actionLogProbs, stateValuePreds= agent.take_action(states)
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
            if term_mask.any():
                th = np.asarray(infos["targetHit"])
                for i in np.where(term_mask)[0]:
                    hitTarget.append(int(th[i]))
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

        # update the actor and critic networks
        results = agent.train_step(buffer)



        vals = np.array(buffer["values"])          # [T, N]
        valueEstimates.append(float(vals.mean()))
        buffer = RolloutBuffer()

        # log the losses and entropy

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
        for e in range(E):
            lo = max(0, e - window + 1)
            w = outcomes[lo:e + 1]
            for t in range(n_targets):
                per_target_curves[t, e] = np.mean(w == t)
            any_curve[e] = np.mean(w >= 0)

        fig, ax = plt.subplots(figsize=(10, 5))
        for t in range(n_targets):
            ax.plot(per_target_curves[t], label=f"target {t}")
        ax.plot(any_curve, label="any target", color="black", linestyle="--", linewidth=2)
        ax.set_title(f"Rolling hit rate (trailing {window} episodes)\n{label}", fontsize=9)
        ax.set_xlabel("Episode")
        ax.set_ylabel("P(hit)")
        ax.set_ylim(0, 1)
        ax.legend()
        plt.tight_layout()
        plt.savefig("hit_rate_curve"+label+".png")
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
        plt.savefig("value_estimate"+label+".png")
        plt.close(fig)





    if not os.path.exists("weights"):
        os.mkdir("weights")
    actor_weights_path = "weights/actor_weights"+label+"UV"+".h5"
    critic_weights_path = "weights/critic_weights"+label+"UV"+".h5"
    """ save network weights """
    torch.save(agent.actor.state_dict(), actor_weights_path)
    torch.save(agent.critic.state_dict(), critic_weights_path)
    
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

# Both targets tracked by their own N=1 map (equal decode difficulty); target 1 pays double
# (target_awards=[10,20] above). Preference between them should be reward-driven, not
# decode-difficulty-driven.
trainAgent([_targetPoly(1, targetIndex=0), _targetPoly(1, targetIndex=1)])

