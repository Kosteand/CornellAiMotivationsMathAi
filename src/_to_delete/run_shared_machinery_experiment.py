"""Measure "shared machinery" between two groups via a reward-switch
weight-drift experiment.

Motivation
----------
The p-curve / indifference-ratio work (run_p_curve_experiments.py etc.)
measures how much MORE reward one group's difficulty costs relative to
another's, by finding where the agent is indifferent between them when
BOTH are simultaneously rewarded. This script asks a different question:
when a single policy is trained on group 1 alone and then RETRAINED on
group 2 alone (same network, same optimizer state, no reset), how much do
the weights actually have to move to go from "good at group 1" to "good
at group 2"? A small |w2 - w1| means the two groups' difficulty structure
shares a lot of machinery in the network (little has to change); a large
one means they need mostly disjoint solutions.

Protocol
--------
Train ONE PPO model on a two-group BanditEnv (group 1, group 2) for 2x
the usual timestep budget, split into two back-to-back phases:

  Phase 1 (~first half, PHASE_TIMESTEPS each by default): group 1 keeps
      its normal reward value; group 2's reward is forced down to
      `phase1_inactive_reward` (0.0 by default) - the agent has no
      incentive to solve anything but group 1 this phase.
  Phase 2 (~second half): the roles flip - group 2 gets its normal
      reward value back, group 1's reward is forced to
      `phase2_inactive_reward` (a SEPARATE parameter from phase 1's, also
      0.0 by default - see "Gradient-starvation seam" below).

Both groups are observed and both action blocks exist in every episode
throughout - exactly like the normal two-group comparison env - only the
REWARD attached to each group's block changes across the phase boundary.
This means the agent still sees group 2's (unrewarded) secrets during
phase 1 and vice versa, so any drift is about what the policy/value nets
choose to DO with that structure, not about the observation distribution
changing.

Why this needs no manual "round to the nearest update" arithmetic:
SB3's OnPolicyAlgorithm.learn() only ever stops after a COMPLETE rollout
(`n_steps * n_envs` env-steps collected, one gradient update run over all
of it) - see its `while self.num_timesteps < total_timesteps: collect
_rollouts(); train()` loop. So calling
`model.learn(total_timesteps=X, reset_num_timesteps=False)` on the SAME
model always leaves `model.num_timesteps` sitting on an exact multiple of
`n_steps * n_envs` when it returns - it can run a few extra env-steps past
X (up to `n_steps*n_envs - 1`) to finish the update it's partway through,
but it never stops mid-update. Requesting `phase_timesteps=200_000`
therefore naturally lands the switch "after the gradient update, no
overlap between the two segments inside the same update" for free,
without this script computing its own rounding - `switch_timestep` in the
result just reports whatever `model.num_timesteps` actually was.

One subtlety this script gets right on purpose (and got wrong in an
earlier draft): SB3's `_setup_learn` does
`total_timesteps += self.num_timesteps` internally whenever
`reset_num_timesteps=False` - i.e. the `total_timesteps` ARGUMENT you pass
means "how many MORE steps to run from here", not an absolute target
(confirmed by reading stable_baselines3/common/base_class.py directly).
So phase 2's call passes `phase_timesteps` again, unchanged - passing
`switch_timestep + phase_timesteps` would double-count switch_timestep
(SB3 would add num_timesteps to it a second time) and run phase 2 for
roughly twice as long as intended.

Does anything decay/schedule across the phase boundary, so it'd matter
whether phase 2 is "continued" vs called as a brand-new run?
    - The two calls run on the SAME `model` object - same network
      weights, same Adam/AdamW optimizer state (running moment
      estimates), never reinitialized. This continuity is the entire
      point of the experiment: it's what makes |w2 - w1| measure "how far
      did an already-group-1-trained network have to move," rather than
      the difference between two independently-initialized networks. If
      you instead ran phase 2 as a genuinely separate `trainPPO.train()`
      call, you'd get a fresh random init and a reset optimizer, and
      |w2 - w1| would mostly reflect random-init noise, not transfer.
    - `learning_rate`/`clip_range`/`clip_range_vf` are the only PPO
      hyperparameters SB3 supports as decaying SCHEDULES (functions of
      `progress_remaining`); every other hyperparameter here (gamma,
      gae_lambda, ent_coef, vf_coef, max_grad_norm, n_epochs, batch_size)
      is always a plain constant, call-to-call or not. This script passes
      `learning_rate`/`clip_range` as plain floats, which SB3 turns into
      CONSTANT schedules - so today, with the defaults, it makes no
      difference whether phase 2 is "continued" or "restarted" as far as
      these values go; they're identical at every single step either way.
      This WOULD start to matter if a caller ever passed a callable/
      schedule for `learning_rate` or `clip_range`: each `learn()` call's
      `progress_remaining` is computed against THAT call's own
      `total_timesteps` (phase_timesteps, not the full run), so two
      back-to-back calls decay separately per-phase rather than smoothly
      across the full 2*phase_timesteps run the way one single
      `learn(total_timesteps=2*phase_timesteps)` call would. Not an issue
      today; flagged here so it isn't a surprise if schedules get added
      later.

Does the phase's inactive reward (`phase1_inactive_reward`/
`phase2_inactive_reward`) affect the inactive group's CORRECT and
INCORRECT answers (it should, and now does)?
    Yes. BanditEnv originally had one global `incorrect_reward` shared by
    every group's block, independent of any group's own `.value` - so
    the first version of this script only zeroed out the inactive group's
    reward for actually landing on its label, while a wrong-but-in-that-
    block guess still paid the ordinary global incorrect_reward. That
    would have undermined the gradient-starvation seam (a nonzero/
    negative phase reward needs to penalize EVERY action toward the
    inactive group, not just the lucky/unlucky correct one).
    Utilities/bandit_env.py's BanditEnv now also accepts a per-group
    SEQUENCE for `incorrect_reward` (kept by reference, not copied, so
    external mutation is visible to every env instance sharing it - the
    same trick already used for `.value`); this script builds one shared
    `incorrect_rewards = [ir, ir]` list and, each phase, sets BOTH the
    inactive group's `.value` AND its `incorrect_rewards[i]` entry to
    that phase's inactive reward, so every action in the inactive group's
    block - correct or not - gets the same reward. Passing a plain float
    to BanditEnv still works exactly as before (fully backward-
    compatible) for every other script in this repo.

Is this restricted to exactly two groups, and is it easy to set up for a
given pair (it should be)?
    Yes to both. `run_shared_machinery_experiment(group1, group2, ...)`
    takes the two groups as explicit positional arguments (not a list),
    so it's structurally impossible to pass more or fewer than two - no
    manual length-checking needed. Setting it up for a specific comparison
    is exactly as easy as building the fixed_group/variable_group pair in
    one of run_p_curve_experiments.py's TESTS entries: construct two
    AlternatingGroup instances (typically two HeatmapGroup calls) and pass
    them straight in, e.g. the __main__ block below reuses
    heatmap_vs_heatmap_harder_variable's exact (g=4, noise_scale=1.0,
    n=2 vs n=6) pairing with no extra plumbing.

Weight snapshots
-----------------
w1: every parameter tensor in `model.policy`, cloned immediately after
    phase 1 ends (right when the switch happens).
w2: the same, cloned at the very end of phase 2.

|w2 - w1| is reported as an L2 norm - sqrt(sum of squares across every
element of every tensor in the group) - the same convention
run_per_layer_weight_norm_rerun.py's layer_norm() uses for weight norms,
just applied to the (w2 - w1) difference instead of to a single
snapshot's raw weights. Reported over the same groupings that script uses
for weight_norm_actor / weight_norm_critic / per-layer:

    whole_model    - every parameter in model.policy
    actor          - mlp_extractor.policy_net (both hidden layers) +
                     action_net
    critic         - mlp_extractor.value_net (both hidden layers) +
                     value_net
    policy_net_0   - actor's first hidden layer only (obs -> 64)
    policy_net_2   - actor's second hidden layer only (64 -> 32)
    action_net     - actor's output head only (32 -> g)
    value_net_0    - critic's first hidden layer only (obs -> 64)
    value_net_2    - critic's second hidden layer only (64 -> 32)
    value_net_out  - critic's output head only (32 -> 1)

"intermediate layers" (per the request) means policy_net_0/policy_net_2/
value_net_0/value_net_2 - the hidden layers, as opposed to the
action_net/value_net_out heads.

Gradient-starvation seam
-------------------------
Each phase's "off" group gets its own reward, via two SEPARATE
parameters rather than one shared `inactive_group_reward`:
`phase1_inactive_reward` (group2's reward while group1 is being trained -
the "wrong group" during phase 1) and `phase2_inactive_reward` (group1's
reward while group2 is being trained - the "old group" during phase 2).
Both default to 0.0 (matching the original single-`inactive_group_reward`
behavior), but a caller can now push `phase2_inactive_reward` negative
(e.g. -0.1) to actively penalize the OLD group's actions once training has
moved on to the new one, without touching phase 1's inactive reward at
all - this is exactly the "gradient starvation" seam this module used to
just document without wiring up per-phase.

Post-run hit-rate eval
-----------------------
After phase 2 ends (and group1/group2 are restored to their original
reward values), the function runs a `greedy_dual_evaluate()` pass - the
same greedy-accuracy convention `trainPPO.evaluate()` uses elsewhere in
this repo, applied to the actual two-group env this experiment already
trained on (a single-group eval isn't possible here without changing the
observation size the network was trained against - see
`greedy_dual_evaluate()`'s docstring). This directly answers "did the
model actually switch to the new group, or fail" - not just an aggregate
hit rate, but SEPARATE accuracy conditioned on which group's action block
the agent chose to answer in (`hit_rate_group1`/`hit_rate_group2`), plus
how often it chose each block at all (`choice_rate_group1`/
`choice_rate_group2`) - a model that's still mostly answering group1's
block (or answering group2's block but getting it wrong) has failed to
switch, even if its raw reward looks fine.

Run:  python3 run_shared_machinery_experiment.py
"""
import csv
import os
import sys
from dataclasses import dataclass, field

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 2026-08-19: Utilities/ and trainPPO.py moved into the self-contained
# M_comparison_background/ subfolder alongside the rest of the
# run_my_comparisons.py dependency chain. Inserting that folder onto
# sys.path keeps the imports below resolvable regardless of how this
# script ends up invoked.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "M_comparison_background"))

from Utilities.bandit_env import BanditEnv, HeatmapGroup, MarginGroup
from trainPPO import make_env


# --- weight groupings, mirroring run_per_layer_weight_norm_rerun.py's
# wn_policy_net_0 / wn_policy_net_2 / wn_action_net / wn_value_net_0 /
# wn_value_net_2 / wn_value_net_out / weight_norm_actor / weight_norm_
# critic / weight_norm_total breakdown, applied to a DIFFERENCE of two
# snapshots instead of a single snapshot. Prefixes match
# model.policy.named_parameters() exactly (confirmed via
# apply_split_weight_decay in trainPPO.py and by direct inspection in
# run_per_layer_weight_norm_rerun.py).
WEIGHT_GROUP_PREDICATES = {
    "whole_model": lambda name: True,
    "actor": lambda name: (
        name.startswith("mlp_extractor.policy_net.") or name.startswith("action_net.")
    ),
    "critic": lambda name: (
        name.startswith("mlp_extractor.value_net.") or name.startswith("value_net.")
    ),
    "policy_net_0": lambda name: name.startswith("mlp_extractor.policy_net.0."),
    "policy_net_2": lambda name: name.startswith("mlp_extractor.policy_net.2."),
    "action_net": lambda name: name.startswith("action_net."),
    "value_net_0": lambda name: name.startswith("mlp_extractor.value_net.0."),
    "value_net_2": lambda name: name.startswith("mlp_extractor.value_net.2."),
    "value_net_out": lambda name: name.startswith("value_net."),
}

# "Intermediate layers" = the hidden layers of both sub-networks, as
# opposed to their output heads (action_net / value_net_out).
INTERMEDIATE_LAYER_GROUPS = (
    "policy_net_0", "policy_net_2", "value_net_0", "value_net_2",
)

SUMMARY_FIELDNAMES = [
    "group1_type", "group1_noise_scale", "group1_n", "group1_g", "group1_value",
    "group2_type", "group2_noise_scale", "group2_n", "group2_g", "group2_value",
    "incorrect_reward", "phase1_inactive_reward", "phase2_inactive_reward",
    "phase_timesteps", "switch_timestep", "end_timestep",
] + [f"diff_{g}" for g in WEIGHT_GROUP_PREDICATES] + [f"n_params_{g}" for g in WEIGHT_GROUP_PREDICATES] + [
    "hit_rate_overall", "hit_rate_group1", "hit_rate_group2",
    "choice_rate_group1", "choice_rate_group2", "eval_episodes",
]


def _snapshot_params(policy):
    """Deep-copy every parameter tensor in `policy`, keyed by the exact
    name `named_parameters()` reports (e.g. 'action_net.weight',
    'mlp_extractor.policy_net.0.bias')."""
    return {name: p.detach().clone() for name, p in policy.named_parameters()}


def diff_norm(before: dict, after: dict, predicate) -> tuple[float, int]:
    """L2 norm of (after - before) restricted to parameter names matching
    `predicate`, plus the number of scalar elements that went into it -
    same "sum of squares over every element of every matched tensor, then
    sqrt" convention as run_per_layer_weight_norm_rerun.py's layer_norm(),
    just applied to a difference instead of a raw snapshot."""
    total = 0.0
    n_params = 0
    for name, p_before in before.items():
        if not predicate(name):
            continue
        p_after = after[name]
        total += float(torch.sum((p_after - p_before) ** 2))
        n_params += p_before.numel()
    return total ** 0.5, n_params


def greedy_dual_evaluate(model, group1, group2, incorrect_reward, episodes=500):
    """Greedy accuracy of `model` on the SAME two-group BanditEnv it was
    trained on (group1, group2, at whatever `.value` each currently
    holds - callers should restore both to their real/original values
    before calling this, exactly as `run_shared_machinery_experiment`
    does). This can't be simplified to a single-group eval (e.g. reusing
    `trainPPO.evaluate(model, [group2], ...)` alone to check "did it learn
    group2") because the POLICY NETWORK's input layer was built for the
    concatenated two-group observation size - dropping to a one-group env
    would change the observation shape and the model couldn't even run.

    Returns a dict:
      hit_rate_overall:   fraction of all `episodes` answered correctly
                          (matches trainPPO.evaluate()'s convention).
      hit_rate_group1/2:  accuracy CONDITIONED on the agent having chosen
                          that group's action block that episode (i.e.
                          "when it went for group N, how often was it
                          right") - 0.0 (not NaN) if the agent never chose
                          that block in any of the `episodes` episodes, so
                          this is always a plain float, never None/NaN.
      choice_rate_group1/2: fraction of all episodes where the agent chose
                          that group's action block at all (whether or not
                          it got the label right) - lets you see behavioral
                          collapse (e.g. "still answering group1's block
                          90% of the time") separately from accuracy.
      episodes:           the `episodes` argument, echoed back for
                          convenience when writing summary rows.

    "did the model actually switch to the new group, or just fail" reads
    directly off hit_rate_group2 (and choice_rate_group2) after phase 2 -
    high choice_rate_group2 with low hit_rate_group2 means it started
    answering the new group's block but got the label wrong; low
    choice_rate_group2 means it never even switched to trying."""
    env = BanditEnv(groups=[group1, group2], incorrect_reward=incorrect_reward)

    correct = 0
    chosen_counts = [0, 0]
    correct_counts = [0, 0]
    for _ in range(episodes):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        correct += int(info["correct"])
        chosen_group = info["chosen_group"]
        if chosen_group is not None:
            chosen_counts[chosen_group] += 1
            if info["matched_group"] == chosen_group:
                correct_counts[chosen_group] += 1

    return {
        "hit_rate_overall": correct / episodes,
        "hit_rate_group1": (correct_counts[0] / chosen_counts[0]) if chosen_counts[0] else 0.0,
        "hit_rate_group2": (correct_counts[1] / chosen_counts[1]) if chosen_counts[1] else 0.0,
        "choice_rate_group1": chosen_counts[0] / episodes,
        "choice_rate_group2": chosen_counts[1] / episodes,
        "episodes": episodes,
    }


@dataclass
class SharedMachineryResult:
    model: PPO
    switch_timestep: int
    end_timestep: int
    w1: dict = field(repr=False)
    w2: dict = field(repr=False)
    diffs: dict[str, float] = field(default_factory=dict)
    n_params: dict[str, int] = field(default_factory=dict)
    hit_rates: dict[str, float] = field(default_factory=dict)


def run_shared_machinery_experiment(
    group1,
    group2,
    incorrect_reward: float = 0.0,
    phase1_inactive_reward: float = 0.0,
    phase2_inactive_reward: float = 0.0,
    n_envs: int = 8,
    phase_timesteps: int = 200_000,
    label: str = "shared_machinery",
    device: str = "cpu",
    verbose: int = 0,
    seed: int | None = None,
    learning_rate: float = 3e-4,
    n_steps: int = 512,
    batch_size: int = 512,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    net_arch_pi=(64, 32),
    net_arch_vf=(64, 32),
    weights_dir: str = "weights",
    save_checkpoints: bool = True,
    progress_bar: bool = False,
    eval_episodes: int = 500,
) -> SharedMachineryResult:
    """Run the two-phase reward-switch training protocol described in this
    module's docstring and return the |w2 - w1| weight-drift breakdown.

    `group1`/`group2` are AlternatingGroup instances (e.g. two HeatmapGroup
    configs, the same way run_p_curve_experiments.py's TESTS build a
    fixed_group/variable_group pair) - their `.value` attributes ARE
    mutated during the run (phase 1: group1 at its original value, group2
    forced to `inactive_group_reward`; phase 2: reversed) and are restored
    to their original values before this function returns, so the same
    objects can be reused afterwards (e.g. for a follow-up eval) without
    surprises.

    `phase_timesteps` is the target length of EACH phase (so total
    training is ~2 * phase_timesteps, per the "400,000 update... 2x the
    usual" request with the default 200,000 usual budget) - the actual
    number of env-steps run in each phase will be phase_timesteps rounded
    UP to the next full `n_steps * n_envs` rollout (see module docstring),
    reported exactly via `switch_timestep`/`end_timestep`.

    `phase1_inactive_reward`/`phase2_inactive_reward`: the reward given to
    whichever group is "off" during phase 1 / phase 2 respectively (both
    default 0.0, matching this function's original single-
    `inactive_group_reward` behavior). Kept as two separate parameters
    (not one shared value) specifically so a caller can penalize the OLD
    group during phase 2 (e.g. `phase2_inactive_reward=-0.1`) without
    also changing phase 1's inactive reward - see the module docstring's
    "Gradient-starvation seam" section.
    """
    original_value1 = group1.value
    original_value2 = group2.value
    groups = [group1, group2]

    # A single shared, MUTABLE list, passed by reference into every one of
    # the n_envs BanditEnv copies below (BanditEnv.__init__ keeps this
    # exact object rather than copying it when given a sequence - see
    # Utilities/bandit_env.py). This is what lets the "inactive" group's
    # WRONG-answer reward be flipped to that phase's inactive reward too
    # (not just its `.value`/correct-answer reward) with a single assignment
    # below, immediately visible to every parallel env - the same
    # broadcast-via-shared-reference trick already used for `group.value`.
    incorrect_rewards = [incorrect_reward, incorrect_reward]

    vec_env = DummyVecEnv(
        [make_env(groups, incorrect_rewards) for _ in range(n_envs)]
    )

    policy_kwargs = dict(
        net_arch=dict(pi=list(net_arch_pi), vf=list(net_arch_vf)),
        optimizer_class=torch.optim.AdamW,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        device=device,
        verbose=verbose,
        seed=seed,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
    )

    if save_checkpoints:
        os.makedirs(weights_dir, exist_ok=True)

    # --- Phase 1: only group 1 rewarded ---
    # Group 1 keeps its normal correct/incorrect rewards; group 2 is
    # "inactive" - BOTH its correct-answer reward (`.value`) AND its
    # incorrect-answer reward (`incorrect_rewards[1]`) are forced to
    # `phase1_inactive_reward`, so picking anything in group 2's block (its
    # label or not) gives the same signal. This matters once
    # `inactive_group_reward` is a nonzero/negative "gradient starvation"
    # value: without also overriding the incorrect-answer reward, an
    # inactive group would still hand out the normal incorrect_reward for
    # 3 out of every g wrong-but-in-block guesses, diluting the intended
    # penalty.
    group1.value = original_value1
    group2.value = phase1_inactive_reward
    incorrect_rewards[0] = incorrect_reward
    incorrect_rewards[1] = phase1_inactive_reward
    model.learn(
        total_timesteps=phase_timesteps,
        reset_num_timesteps=True,
        progress_bar=progress_bar,
    )
    switch_timestep = model.num_timesteps

    w1 = _snapshot_params(model.policy)
    if save_checkpoints:
        model.save(f"{weights_dir}/ppo_{label}_w1")

    # --- Phase 2: only group 2 rewarded ---
    # NOTE ON total_timesteps WITH reset_num_timesteps=False: SB3's
    # _setup_learn ADDS the model's current `num_timesteps` to whatever
    # `total_timesteps` you pass it (`total_timesteps += self.num_
    # timesteps`) before looping - i.e. the argument means "how many MORE
    # steps to run from here", not an absolute target. So this call passes
    # `phase_timesteps` again (not `switch_timestep + phase_timesteps`,
    # which would double-count switch_timestep and run phase 2 roughly
    # twice as long as intended - confirmed by inspecting
    # stable_baselines3/common/base_class.py's _setup_learn and verified
    # against a real run's num_timesteps here).
    group1.value = phase2_inactive_reward
    group2.value = original_value2
    incorrect_rewards[0] = phase2_inactive_reward
    incorrect_rewards[1] = incorrect_reward
    model.learn(
        total_timesteps=phase_timesteps,
        reset_num_timesteps=False,
        progress_bar=progress_bar,
    )
    end_timestep = model.num_timesteps

    w2 = _snapshot_params(model.policy)
    if save_checkpoints:
        model.save(f"{weights_dir}/ppo_{label}_w2")

    # Restore original values so the same group objects (and, since
    # `incorrect_rewards` is only referenced by this run's envs, that list
    # too - though nothing outside this function keeps a reference to it)
    # behave normally if the caller reuses group1/group2 afterwards (e.g.
    # for a fresh eval env) - and so the post-run hit-rate eval right below
    # measures accuracy under the REAL reward structure, not whatever
    # phase 2 happened to end on.
    group1.value = original_value1
    group2.value = original_value2
    incorrect_rewards[0] = incorrect_reward
    incorrect_rewards[1] = incorrect_reward

    diffs = {}
    n_params = {}
    for group_name, predicate in WEIGHT_GROUP_PREDICATES.items():
        norm, count = diff_norm(w1, w2, predicate)
        diffs[group_name] = norm
        n_params[group_name] = count

    hit_rates = greedy_dual_evaluate(
        model, group1, group2, incorrect_reward, episodes=eval_episodes,
    )

    return SharedMachineryResult(
        model=model,
        switch_timestep=switch_timestep,
        end_timestep=end_timestep,
        w1=w1,
        w2=w2,
        diffs=diffs,
        n_params=n_params,
        hit_rates=hit_rates,
    )


def _group_spec_dict(prefix: str, group) -> dict:
    """Flatten a group's identifying spec into `{prefix}_type`/
    `{prefix}_noise_scale`/`{prefix}_n`/`{prefix}_g`/`{prefix}_value`
    columns for the summary CSV - blank (not 0) for whichever fields
    don't apply to this group's type, same "blank rather than a
    misleading 0" convention as run_per_layer_weight_norm_rerun.py."""
    if isinstance(group, HeatmapGroup):
        return {
            f"{prefix}_type": "heatmap",
            f"{prefix}_noise_scale": group.noise_scale,
            f"{prefix}_n": group.n,
            f"{prefix}_g": group.g,
            f"{prefix}_value": group.value,
        }
    if isinstance(group, MarginGroup):
        return {
            f"{prefix}_type": "margin",
            f"{prefix}_noise_scale": "",
            f"{prefix}_n": "",
            f"{prefix}_g": group.g,
            f"{prefix}_value": group.value,
        }
    return {
        f"{prefix}_type": type(group).__name__,
        f"{prefix}_noise_scale": "",
        f"{prefix}_n": "",
        f"{prefix}_g": getattr(group, "g", ""),
        f"{prefix}_value": group.value,
    }


def append_summary_row(
    csv_path: str,
    group1,
    group2,
    incorrect_reward: float,
    phase1_inactive_reward: float,
    phase2_inactive_reward: float,
    phase_timesteps: int,
    result: SharedMachineryResult,
) -> None:
    """Append one row to `csv_path` (writing the header first if the file
    doesn't exist yet), same one-row-at-a-time append convention as every
    other sweep script in this repo so nothing already-completed is lost
    if a longer sweep gets killed early."""
    row = {
        **_group_spec_dict("group1", group1),
        **_group_spec_dict("group2", group2),
        "incorrect_reward": incorrect_reward,
        "phase1_inactive_reward": phase1_inactive_reward,
        "phase2_inactive_reward": phase2_inactive_reward,
        "phase_timesteps": phase_timesteps,
        "switch_timestep": result.switch_timestep,
        "end_timestep": result.end_timestep,
    }
    for group_name in WEIGHT_GROUP_PREDICATES:
        row[f"diff_{group_name}"] = result.diffs[group_name]
        row[f"n_params_{group_name}"] = result.n_params[group_name]
    row["hit_rate_overall"] = result.hit_rates["hit_rate_overall"]
    row["hit_rate_group1"] = result.hit_rates["hit_rate_group1"]
    row["hit_rate_group2"] = result.hit_rates["hit_rate_group2"]
    row["choice_rate_group1"] = result.hit_rates["choice_rate_group1"]
    row["choice_rate_group2"] = result.hit_rates["choice_rate_group2"]
    row["eval_episodes"] = result.hit_rates["episodes"]

    file_is_new = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        if file_is_new:
            writer.writeheader()
        writer.writerow(row)


def print_result_summary(result: SharedMachineryResult) -> None:
    print(f"switch_timestep={result.switch_timestep}  end_timestep={result.end_timestep}")
    print(f"{'group':<14}{'|w2-w1|':>12}{'n_params':>12}")
    for group_name in WEIGHT_GROUP_PREDICATES:
        tag = " (intermediate)" if group_name in INTERMEDIATE_LAYER_GROUPS else ""
        print(
            f"{group_name:<14}{result.diffs[group_name]:>12.4f}"
            f"{result.n_params[group_name]:>12d}{tag}"
        )
    hr = result.hit_rates
    print(
        f"\nhit_rate_overall={hr['hit_rate_overall']:.3f}  "
        f"(over {hr['episodes']} eval episodes)"
    )
    print(
        f"  group1: hit_rate={hr['hit_rate_group1']:.3f}  "
        f"choice_rate={hr['choice_rate_group1']:.3f}"
    )
    print(
        f"  group2: hit_rate={hr['hit_rate_group2']:.3f}  "
        f"choice_rate={hr['choice_rate_group2']:.3f}"
    )


if __name__ == "__main__":
    # Example run: the same pairing as run_p_curve_experiments.py's
    # heatmap_vs_heatmap_harder_variable test (g=4, noise_scale=1.0,
    # n=2 vs n=6) - one HeatmapGroup with 3x more columns/powers to
    # invert than the other. See _group_spec_dict for how each group's
    # spec is captured in the summary row regardless of which pairing is
    # used here.
    group1 = HeatmapGroup(g=4, noise_scale=1.0, n=2, value=1.0)
    group2 = HeatmapGroup(g=4, noise_scale=1.0, n=6, value=1.0)

    result = run_shared_machinery_experiment(
        group1,
        group2,
        incorrect_reward=0.0,
        phase1_inactive_reward=0.0,
        phase2_inactive_reward=0.0,  # pass e.g. -0.1 to penalize the OLD group in phase 2
        phase_timesteps=200_000,
        label="shared_machinery_heatmap_n2_vs_n6",
        progress_bar=True,
    )

    print_result_summary(result)
    append_summary_row(
        "eval_logs/shared_machinery_summary.csv",
        group1,
        group2,
        incorrect_reward=0.0,
        phase1_inactive_reward=0.0,
        phase2_inactive_reward=0.0,
        phase_timesteps=200_000,
        result=result,
    )
    print("Appended -> eval_logs/shared_machinery_summary.csv")
