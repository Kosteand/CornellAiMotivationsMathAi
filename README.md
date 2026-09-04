# Measuring Proxy Preference in RL Agents

An environment for studying **goal misgeneralization** — when an agent learns to pursue a
proxy that correlated with the true objective during training, then fails or behaves
unpredictably once conditions change.

The standard example is a maze agent trained where the goal always sits in one corner. It may
learn "go to that corner" rather than "find the goal," and the two are indistinguishable until
you move the goal. The difficulty in studying this is that in a rich visual environment you
cannot enumerate the proxies an agent *might* have learned — wall textures, lighting, position
priors, arbitrary pixel statistics are all candidates.

This environment makes the proxy set small and known by construction.

## The setup

A grid world with two targets. The agent's observation is not pixels but a short vector of
scalars, where each block of channels encodes the direction toward one target under a chosen
encoding. Because the encodings are specified rather than learned from images, the set of
available proxies is exactly the set you put in.

Two knobs control the experiment:

- **Encoding difficulty** — how hard each target's signal is to decode from its channels
- **Reward** — how much each target pays on contact

Sweeping both measures the **reward premium**: how much extra reward a harder-to-decode target
must offer before the agent will pursue it instead of the easier one. That converts a
qualitative question ("which proxy did it learn?") into a scalar with error bars.

---

## Module reference

### `Utilities/HeatMap.py` — base signal types

Every observation channel ultimately comes from a `HeatMappable`: an object mapping a grid
coordinate to a scalar. The protocol is deliberately small — `map(coords)` returns the value,
`getRange()` returns the `(low, high)` used for normalization, `toString()` produces a
filesystem-safe identifier for run labelling.

Concrete maps fall into two groups.

**Field maps** assign a value to every cell based on geometry: `DistanceTarget` (L2 distance to
the nearest target), `ManhattanDistanceTarget`, `LInftyDistanceTarget`, and simple gradients
like `xAscending`. These produce smooth potential fields an agent can descend.

**Oracle maps** return the ground-truth answer rather than a distance. `OptimalActionTarget`
returns the index of the greedy action toward the nearest target — the strongest possible
signal, used as the inner map that harder encodings are built on top of. Because the arena has
no walls during randomized runs, nearest-by-L2 genuinely is the shortest-path action, so the
oracle is exact rather than approximate.

**Wrappers** compose these. `TargetWrap(inner, targetIndex)` pins a map to one specific target
instead of the nearest, which is what makes two-target competition possible.
`DirectionWrap(inner, offset)` samples the inner map at a neighbouring cell, turning a scalar
field into a directional reading. `NoiseWrap(inner, noiseLevel)` adds spatially-cached noise so
the same cell reads consistently within an episode.

Wrappers are passed as `functools.partial` objects rather than instances, so the environment can
construct them with arena context (bounds, target positions, rewards) at build time.

### `Utilities/MultiHeatMap.py` — multi-channel polynomial encodings

Where `HeatMap.py` produces one scalar per map, `MultiHeatMappable` produces several. This is
where decode difficulty is manufactured.

`MultiPolynomialInverse` wraps an inner map and emits `k` channels. For a true value `v` from
the inner map, each channel carries

```
c_i = (v * w_i + n_i + 1) ^ p
```

with per-episode random weights `w` (normalized to mean 1), noise `n` (normalized to mean 0),
and exponent `p`. The `+1` keeps the base positive so fractional and high powers stay
well-defined; values are clipped at a small floor as a guard.

The two knobs are independent:

- **`p` (degree)** controls nonlinearity. Recovering `v` requires an inverse of order `p`, so
  higher `p` means a harder decode. It need not be an integer — fractional degrees give a
  continuous difficulty dial.
- **`k` (channels)** controls redundancy. More channels mean more noisy views of the same value,
  so noise can be averaged down.

Because weights are mean-1 and noise is mean-0 across channels, a closed-form inverse exists:
take the `p`-th root of each channel, average, subtract 1. This holds exactly only to first
order — the nonlinearity breaks the cancellation, and residual error grows with noise scale.
That inverse is used as an *oracle* to measure how much value information survives a given
random draw, not as something the network is expected to reproduce.

`getRange()` propagates the inner map's range analytically through the polynomial rather than
sampling, so normalization constants stay correct after every re-roll. `pregen()` re-draws `w`
and `n`, and is called on each environment reset.

### `Utilities/SpaceEnv.py` — the Gymnasium environment

`MazeEnv` is a discrete grid world with four actions, a configurable set of targets with
per-target rewards, and optional walls.

**Observation assembly.** `buildMaps()` instantiates every entry in `heatMapTypes`, passing each
the arena context it needs, then computes normalization constants. Single-channel maps get their
own affine transform; a multi-channel block gets **one shared transform across all its
channels**. That distinction matters — normalizing each channel separately would destroy the
shared structure the closed-form inverse depends on, while leaving them unnormalized lets a
high-exponent channel dominate the network's gradients. `getObs()` concatenates every map's
output and applies the cached transform in one operation.

**Randomization.** `reset()` optionally re-draws arena bounds, spawn position, and target
positions, then rebuilds all maps — which re-rolls the encoding parameters. The agent never sees
the same encoding twice and cannot memorize a particular weight draw.

**Reward structure.** Reaching a target terminates the episode and pays that target's award;
every other step costs a small penalty. Running out of steps truncates. The distinction between
these two endings is load-bearing downstream — see the implementation notes.

**Visualization.** `visualize(mapIndex)` renders any heatmap as a field with the agent, walls,
and targets overlaid, which is the fastest way to confirm a new encoding does what you intended
before spending compute on it.

### `PPO.py` — the learning algorithm

Written from scratch rather than imported, mainly so the correctness details below are visible
and fixable.

**`MLP`** — a 3-hidden-layer tanh network, used identically for actor and critic. Its
`describe()` classmethod produces an architecture tag used in run labels, so a changed
architecture cannot silently collide with an incompatible checkpoint.

**`RolloutBuffer`** — stores one on-policy rollout as parallel lists: states, actions, log-probs,
values, rewards, next states, termination flags, and a validity mask. It enforces a strict key
set on append (a typo raises rather than silently dropping data) and marks itself stale after
use, since PPO cannot reuse a batch across policy updates.

**`compute_gaes`** — generalized advantage estimation over `[T, N]` tensors, running backward
through time. It uses **two distinct masks**, which is the subtlety most implementations miss:
the value bootstrap is gated on termination only (a truncated episode still has future value),
while the advantage recursion is gated on termination *or* truncation (advantage must not flow
across an episode boundary either way).

**`PPOAgent`** — actor and critic with separate Adam optimizers and exponential LR decay.
`take_action` samples from a categorical policy and optionally returns the greedy action too, so
the gap between sampled and argmax choices can be logged as a measure of how much of a rollout
is genuine exploration versus decision noise. `train_step` computes advantages, flattens
`[T, N]` into a flat batch, drops invalid transitions, normalizes advantages, then runs several
epochs of shuffled minibatches with the clipped surrogate objective, an entropy bonus, and
gradient-norm clipping. Everything runs in float64.

### `decoderTest.py` — supervised difficulty probes

Ranking encodings by difficulty requires measuring difficulty independently of the RL agent.
This script strips away the environment entirely: draw values, encode them, and ask how hard
they are to recover.

For each configuration it reports:

- **Linear probe R²** on the raw channels — establishes whether the encoding is genuinely
  nonlinear or trivially readable. Deliberately weak; a linear fit on a handful of raw channels
  has no capacity to fake success.
- **MLP validation R² and MAE** — what a trained network achieves.
- **Weight norm** — an MDL-flavoured stand-in for description length.
- **Epochs-to-threshold** — how many epochs a fixed architecture needs to reach a target
  accuracy.
- **A "genuine" flag** — true when the MLP beats the linear baseline by a margin, i.e. the
  encoding really does require nonlinear decoding.

A finding worth recording: **epochs-to-threshold is by far the best-behaved measure.** Across a
degree sweep it spans roughly 500x, while MLP R² sits saturated near 1.0 throughout and carries
almost no information. Weight norm is non-monotone in degree and should be treated with
suspicion.

### `experiment2.py` — the experiment driver

Ties everything together. `_targetPoly(N, targetIndex, maxDegree)` builds one encoded target
block; `trainAgent(mapsTypes)` trains a single agent on a list of them.

Run labels are generated from the full configuration — encodings, rewards, architecture, and
every training hyperparameter — so runs differing in any respect cannot collide on the same
weight files or be silently skipped as already-trained.

Sweep helpers (`runFixedWindowSweep`, `runCrossoverSearch`) vary target rewards across a range
with repeats per point, since identical configurations still land on genuinely different
policies without seeding. A design note learned the hard way: the reward window must be the
**same for every configuration**. An earlier version centred each window on where the crossover
was expected, which quietly encoded the hypothesis into the sampling.

Training uses an arena-size curriculum (small early, full size by the end) and an annealed
entropy bonus. Outputs include rolling hit-rate curves, critic value estimates, policy entropy,
per-episode trajectory plots drawn against an optimal-action backdrop, and a results CSV with
one row per run.

---

## Implementation notes

A few things that were not obvious and took real debugging.

**Truncation vs. termination.** Reaching a target ends the episode with no future value; running
out of steps does not. GAE must bootstrap from `V(s)` on truncation but not on termination, and
must cut the advantage chain on both. Getting this wrong silently destroys the critic — the loss
stays flat, entropy never drops, and the policy remains uniform.

**Autoreset dead steps.** Under Gymnasium's vectorized autoreset, the step following an episode
end returns the reset observation and ignores the action supplied. Recording it as a normal
transition injects meaningless `(state, action, reward)` tuples into the buffer; these are
tracked with a validity mask and dropped before the update.

**Channel conditioning.** At high exponents the channels span several orders of magnitude. Each
block is normalized by a shared affine transform computed from its analytic range — shared
rather than per-channel, which keeps the decode structure intact while fixing the dynamic range.

**float64 throughout.** High-degree channels lose small-value resolution in float32.

## Running it

```bash
pip install gymnasium torch numpy matplotlib tqdm

python experiment2.py            # train; skips configs with existing weights
python experiment2.py --force    # retrain and overwrite
python experiment2.py --record   # append results to a CSV
python decoderTest.py            # supervised difficulty probes
```

Outputs land in `training_plots/`, `trajectory_plots/`, `logs/`, and `weights/`.

## Status

Active research in the Cornell Math+AI Lab. Results are preliminary and the analysis is still
moving, so specific premium numbers should be treated as provisional.
