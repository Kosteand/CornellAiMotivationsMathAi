"""PPO training on BanditEnv, the frequency-complexity bandit environment.

Why PPO: same rationale as the maze version this file replaced - a clipped
objective with minibatch reuse is far more sample-efficient and robust
under sparse/binary reward than a hand-rolled A2C.

This trains directly on Utilities.bandit_env.BanditEnv: each episode is a
single (secret x per group) -> (chosen action) -> (reward) step, so
"n_steps per update" here just means "how many independent bandit episodes
get batched into one gradient estimate" - there's no multi-step trajectory
within an episode the way there was for MazeEnv.

`train()` takes every hyperparameter and config value as an argument -
nothing is hardcoded at module scope. This file defines `train()` only;
see run_trainPPO.py for a runnable example that calls it.
"""
import csv
import os
from dataclasses import dataclass, field

import numpy as np
import gymnasium as gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from Utilities.bandit_env import BanditEnv, ComplexityGroup, CsvLoggingWrapper


@dataclass
class TrainResult:
    """Returned by train(). `model` is what earlier versions of this
    function returned directly - existing callers that only used the
    model (e.g. `model = train(...)`) need one small update to
    `result = train(...); model = result.model`, but nothing here breaks
    silently: unpacking a TrainResult where a PPO model was expected raises
    immediately rather than behaving unexpectedly."""

    model: PPO
    correct: int
    episodes: int
    mean_reward: float
    # Only populated if train() was called with periodic_eval_freq set:
    # a list of (timestep, hit_rate) pairs from the DURING-training greedy
    # evals PeriodicEvalCallback ran. Empty otherwise.
    periodic_eval_history: list = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.correct / self.episodes


def make_env(groups, incorrect_reward):
    def _init():
        return BanditEnv(groups=groups, incorrect_reward=incorrect_reward)
    return _init


def apply_split_weight_decay(model, actor_weight_decay, critic_weight_decay, learning_rate):
    """
    Rebuild model.policy.optimizer with separate weight_decay for the actor
    (pi) and critic (vf) parameters.

    SB3's ActorCriticPolicy builds ONE optimizer over ALL of its parameters
    (`self.optimizer_class(self.parameters(), lr=..., **optimizer_kwargs)`),
    so `optimizer_kwargs=dict(weight_decay=...)` applies the same decay to
    everything - there's no actor-vs-critic split available through the
    normal PPO/policy_kwargs interface. This works around that using
    PyTorch's own per-parameter-group support (no library changes needed):
    build two explicit param groups, one per network, each with its own
    `weight_decay`, and swap that in as `model.policy.optimizer`. SB3 only
    ever touches `optimizer.param_groups` afterwards (to update `lr` each
    rollout), so a differently-constructed Adam is a drop-in replacement.

    Any parameters that aren't part of the actor or critic sub-networks
    (e.g. a non-trivial features extractor, if one is ever configured)
    are still included, in their own group using `critic_weight_decay` as
    a fallback, so nothing silently stops training.
    """
    policy = model.policy

    actor_params = list(policy.mlp_extractor.policy_net.parameters()) + list(
        policy.action_net.parameters()
    )
    critic_params = list(policy.mlp_extractor.value_net.parameters()) + list(
        policy.value_net.parameters()
    )

    covered_ids = {id(p) for p in actor_params} | {id(p) for p in critic_params}
    other_params = [p for p in policy.parameters() if id(p) not in covered_ids]

    param_groups = [
        {"params": actor_params, "weight_decay": actor_weight_decay},
        {"params": critic_params, "weight_decay": critic_weight_decay},
    ]
    if other_params:
        param_groups.append(
            {"params": other_params, "weight_decay": critic_weight_decay}
        )

    policy.optimizer = policy.optimizer_class(
        param_groups, lr=learning_rate, **policy.optimizer_kwargs
    )


class BanditDataLoggerCallback(BaseCallback):
    """
    Records every training transition's secret(s), true label(s), and
    chosen action - without touching disk during training.

    SB3's PPO calls `env.step()` on a DummyVecEnv, so a per-step file write
    (like CsvLoggingWrapper does) would mean one flush() syscall per
    transition per env - across many envs and timesteps that easily dwarfs
    a training run lasting only a few seconds. This callback instead
    buffers rows as plain Python lists in memory (cheap) and writes ONE csv
    in a single burst in `_on_training_end`, so the added cost is close to
    zero regardless of how many timesteps you train for.

    Reads straight out of PPO's rollout collection loop via `self.locals`:
    - `obs_tensor` is the observation *used to pick the action* (the x
      values), captured before env.step() is called.
    - `actions` and `infos` come from that same env.step() call, so they
      line up with `obs_tensor` row-for-row.
    """

    def __init__(self, csv_path: str, groups, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.num_groups = len(groups)
        # One column per observation slot within each group, not one per
        # group - a group like MarginGroup contributes observation_size > 1
        # columns, so num_groups alone isn't enough to name/align columns.
        self._x_columns = [
            f"x_{i}_{j}"
            for i, group in enumerate(groups)
            for j in range(group.observation_size)
        ]
        self._rows: list[list] = []

    def _on_step(self) -> bool:
        obs = self.locals["obs_tensor"].detach().cpu().numpy()
        actions = np.asarray(self.locals["actions"]).reshape(-1)
        infos = self.locals["infos"]

        for env_idx in range(obs.shape[0]):
            info = infos[env_idx]
            row = (
                list(obs[env_idx])
                + list(info["labels"])
                + [
                    int(actions[env_idx]),
                    bool(info["correct"]),
                    info["matched_group"],
                ]
            )
            self._rows.append(row)

        return True

    def _on_training_end(self) -> None:
        fieldnames = (
            self._x_columns
            + [f"label_{i}" for i in range(self.num_groups)]
            + ["action", "correct", "matched_group"]
        )
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            writer.writerows(self._rows)


def evaluate(model, groups, incorrect_reward, episodes, log_csv=None):
    """
    Greedy accuracy over `episodes` single-step bandit episodes.

    Returns (correct, episodes, mean_reward). Random-guess baseline for a
    single group of size g is ~1/g; with multiple groups sharing the same
    action space it's higher because a single action can satisfy more than
    one group's label.
    """
    env = BanditEnv(groups=groups, incorrect_reward=incorrect_reward)
    if log_csv is not None:
        env = CsvLoggingWrapper(env, csv_path=log_csv)

    correct = 0
    total_reward = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        correct += int(info["correct"])
        total_reward += reward

    if log_csv is not None:
        env.close()

    return correct, episodes, total_reward / episodes


class PeriodicEvalCallback(BaseCallback):
    """
    Runs a deterministic (greedy) accuracy evaluation - the same measure
    `evaluate()` computes once at the very end of training - every
    `eval_freq` environment timesteps DURING training, and records
    (timestep, hit_rate) pairs in memory (self.history), never touching
    disk.

    Why this exists: BanditDataLoggerCallback's per-episode 'correct'
    column reflects PPO's STOCHASTIC rollout action (sampled from the
    current policy distribution - PPO needs that sampling for exploration
    and its log-prob objective), not the greedy argmax action the trained
    model is actually judged on at eval time. A learning-curve fit to
    that stochastic training signal systematically undershoots the greedy
    eval curve, because sampling noise lowers apparent accuracy even once
    the policy has essentially converged to a good greedy policy - hence
    the fitted asymptote (L) reading noticeably below the real final
    hit_rate. Fitting to periodic GREEDY eval accuracy instead closes
    that gap, since it's measuring the exact same thing hit_rate is, just
    at multiple points during training instead of only at the end.

    Each periodic check calls `evaluate()` on a fresh BanditEnv - it never
    touches the training rollout's own vec_env/buffer, so this adds
    compute (eval_episodes extra episodes every eval_freq timesteps) but
    zero interference with training itself.

    eval_freq is checked against self.num_timesteps (SB3's running total
    across all n_envs, incremented by n_envs every _on_step call) via a
    while-loop rather than `% eval_freq == 0`, since num_timesteps can
    jump by more than 1 between calls and would otherwise skip a
    boundary; this also handles eval_freq itself being smaller than
    n_envs correctly (rare, but not assumed away).
    """

    def __init__(self, groups, incorrect_reward, eval_freq, eval_episodes, verbose=0):
        super().__init__(verbose)
        self.groups = groups
        self.incorrect_reward = incorrect_reward
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self._next_eval = eval_freq
        self.history: list[tuple[int, float]] = []

    def _on_step(self) -> bool:
        while self.num_timesteps >= self._next_eval:
            correct, n, _mean_reward = evaluate(
                self.model, self.groups, self.incorrect_reward, self.eval_episodes
            )
            self.history.append((self._next_eval, correct / n))
            self._next_eval += self.eval_freq
        return True


def train(
    groups,
    # --- environment config ---
    incorrect_reward=0.0,
    n_envs=8,
    # --- training loop config ---
    total_timesteps=200_000,
    label="bandit",
    progress_bar=True,
    log_training_data=True,
    log_interval=1,
    print_final_summary=False,
    # --- PPO hyperparameters ---
    device="cpu",
    verbose=1,
    seed=None,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=512,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    net_arch_pi=(64, 32),
    net_arch_vf=(64, 32),
    activation_fn=None,
    optimizer_class=None,
    weight_decay=0.0,
    actor_weight_decay=None,
    critic_weight_decay=None,
    policy_kwargs_extra=None,
    ppo_kwargs=None,
    # --- evaluation config ---
    eval_episodes=500,
    periodic_eval_freq=None,
    periodic_eval_episodes=100,
    # --- output config ---
    weights_dir="weights",
    eval_logs_dir="eval_logs",
    save_model=True,
    print_eval_summary=True,
):
    """
    Train PPO on BanditEnv.

    Parameters
    ----------
    groups:
        List of ComplexityGroup instances passed to BanditEnv - defines the
        group sizes (g), complexities (k), and reward values for this run.
    incorrect_reward:
        Reward given when the chosen action doesn't match any group's
        label. Passed straight through to BanditEnv.
    n_envs:
        Number of BanditEnv copies run in parallel via DummyVecEnv.
    total_timesteps:
        Total env steps across all n_envs. Since each BanditEnv episode is
        exactly one step, this is also the total number of independent
        bandit episodes sampled over the whole run.
    label:
        Used to name the saved model (`weights_dir/ppo_{label}.zip`, only
        written if `save_model=True`) and the CSV logs
        (`eval_logs_dir/ppo_{label}_train_log.csv` and
        `ppo_{label}_eval.csv`).
    progress_bar:
        Passed to model.learn(). Requires `tqdm` and `rich`
        (`pip install "stable-baselines3[extra]"`) - set False to avoid
        that dependency.
    log_training_data:
        If True, attach BanditDataLoggerCallback and write every training
        transition's x/label/action to `eval_logs_dir/ppo_{label}_train_log.csv`.
        Measured to add ~0% overhead even at tens of thousands of
        timesteps, since it buffers in memory and writes once at the end.
    log_interval:
        How often (in training iterations/rollouts, i.e. every n_steps *
        n_envs timesteps) SB3 prints its stats box (the "----- | time/...
        | -----" table) during training. 1 = every iteration (SB3's
        default), 5 = every 5th, etc. Set to None to suppress that
        periodic printing entirely - combine with `print_final_summary`
        for a single box at the very end instead of one per iteration.
        Has no effect if `verbose=0` (which disables that box - and every
        other SB3 console message - regardless of this setting).
    print_final_summary:
        If True, force one extra print of the stats box after training
        finishes, showing the final iteration's numbers even if
        `log_interval` suppressed it during the loop. Also gated by
        `verbose >= 1` - if verbose=0, nothing prints regardless of this.
    device, verbose, seed, learning_rate, n_steps, batch_size, n_epochs,
    gamma, gae_lambda, clip_range, ent_coef, vf_coef, max_grad_norm:
        Passed straight through to stable_baselines3.PPO.
    net_arch_pi, net_arch_vf:
        Hidden layer sizes for the actor (`pi`) and critic (`vf`) MLPs,
        passed as `policy_kwargs=dict(net_arch=dict(pi=..., vf=...))`.
        This is "model size" - bigger tuples = more capacity.
    activation_fn:
        The nonlinearity between hidden layers, as a `torch.nn.Module`
        CLASS (not an instance) - e.g. `torch.nn.Tanh` (SB3's own
        MlpPolicy default - what every run before this parameter existed
        used, so `None` here means "let SB3 pick its default" rather than
        pinning a specific class, which is the same thing), `torch.nn.ReLU`,
        `torch.nn.LeakyReLU`, `torch.nn.GELU`. Passed straight through as
        `policy_kwargs["activation_fn"]`.
    optimizer_class:
        The `torch.optim` optimizer CLASS (not an instance) - e.g.
        `torch.optim.AdamW` (this project's own default - see the comment
        above the policy_kwargs construction below for why AdamW over
        SB3's own Adam default), `torch.optim.Adam`, `torch.optim.SGD`,
        `torch.optim.RMSprop`. `None` means "use this project's AdamW
        default", matching every run before this parameter existed.
        Passed straight through as `policy_kwargs["optimizer_class"]`.
        Whichever class ends up here also determines what
        `apply_split_weight_decay` rebuilds the optimizer AS, if
        actor_weight_decay/critic_weight_decay triggers that rebuild - no
        separate wiring needed for that path, since it always reads back
        `policy.optimizer_class` rather than hardcoding AdamW itself.
    weight_decay:
        L2 penalty applied by the Adam optimizer. SB3 normally uses ONE
        optimizer over every policy parameter, so this is the default
        decay for both actor and critic.
    actor_weight_decay, critic_weight_decay:
        If either is set (not None), the actor's and critic's parameters
        get their OWN weight_decay instead of sharing `weight_decay` -
        whichever one is left as None falls back to `weight_decay`. This
        rebuilds the optimizer with separate parameter groups (see
        `apply_split_weight_decay`); if both are None, the model keeps
        SB3's normal single optimizer with `weight_decay` applied
        uniformly (no rebuild, no behavior change from before).
    policy_kwargs_extra:
        Optional dict of any additional `policy_kwargs` keys not already
        covered above (e.g. `ortho_init`, `log_std_init`,
        `optimizer_kwargs` overrides beyond weight_decay, `share_features_
        extractor`) - merged into the internally-built policy_kwargs dict
        (net_arch/activation_fn/optimizer_class/optimizer_kwargs) without
        needing this function's signature to enumerate every possible
        ActorCriticPolicy argument. Keys here OVERRIDE the internally-built
        ones on conflict, so e.g. `policy_kwargs_extra=dict(optimizer_class=
        torch.optim.SGD)` would win over the `optimizer_class` parameter
        above if both were set (there's no reason to set both, but this
        documents which one wins if you do).
    ppo_kwargs:
        Optional dict of any additional stable_baselines3.PPO keyword
        arguments not already covered above (e.g. `target_kl`,
        `use_sde`, `tensorboard_log`) - merged in without needing this
        function's signature to enumerate every possible PPO argument.
    eval_episodes:
        Number of fresh episodes used for the post-training greedy
        accuracy evaluation.
    periodic_eval_freq:
        If set (not None), attach a PeriodicEvalCallback that runs a
        greedy accuracy eval every `periodic_eval_freq` timesteps DURING
        training (in addition to the usual one at the very end) and
        records the results in-memory only - see PeriodicEvalCallback's
        docstring for why you'd want this instead of/alongside
        log_training_data. None (default) skips this entirely - no
        added compute, no behavior change from before this existed.
    save_model:
        If True (default, matches prior behavior), write the trained
        model to `weights_dir/ppo_{label}.zip` and print the
        `saved -> ...` line. Set False to skip this entirely - useful for
        scripts (like find_indifference_reward.py) that train hundreds or
        thousands of throwaway models and never reload any of them, where
        saving every single run just burns disk I/O for no benefit.
    print_eval_summary:
        If True (default, matches prior behavior), print the
        "PPO greedy eval: x/y correct (...%), mean reward ..." line after
        the built-in final evaluation. Set False to suppress it - useful
        for callers (like find_indifference_reward.py) that run their own
        separate evaluation and print/log their own summary instead, where
        this line would just be noise repeated on every one of many runs.
    periodic_eval_episodes:
        Episodes per periodic eval (only used if periodic_eval_freq is
        set). Kept smaller than eval_episodes by default since this runs
        many times per training run (e.g. 200 times over 200_000
        timesteps at periodic_eval_freq=1000) - each extra episode here
        multiplies by however many checkpoints there are, unlike
        eval_episodes which only pays that cost once.
    weights_dir:
        Directory the trained model checkpoint is saved to.
    eval_logs_dir:
        Directory the training-data and evaluation CSV logs are written to.

    Returns
    -------
    A TrainResult with `.model` (the trained stable_baselines3.PPO model),
    `.correct`/`.episodes`/`.mean_reward` from the post-training eval, and
    `.hit_rate` (`correct / episodes`).
    """
    vec_env = DummyVecEnv(
        [make_env(groups, incorrect_reward) for _ in range(n_envs)]
    )

    split_weight_decay = (
        actor_weight_decay is not None or critic_weight_decay is not None
    )

    # AdamW instead of SB3's default Adam: Adam's weight_decay is really an
    # L2 penalty folded into the gradient (so it interacts with the
    # adaptive per-parameter learning rates), whereas AdamW decouples decay
    # from the gradient update entirely - the standard choice whenever
    # weight_decay is actually meant to shrink weights toward zero at a
    # controlled rate, which is exactly what upcoming weight-decay sweeps
    # need. With weight_decay=0.0 (the default) AdamW is identical to Adam,
    # so this is a no-op for every run so far. optimizer_class=None keeps
    # this exact default; pass a different torch.optim class (e.g.
    # torch.optim.Adam, torch.optim.SGD) to override it per run.
    #
    # activation_fn=None likewise keeps SB3's own MlpPolicy default
    # (torch.nn.Tanh) - what every run before this parameter existed used -
    # rather than this module pinning a class of its own; pass e.g.
    # torch.nn.ReLU to override it per run.
    policy_kwargs = dict(
        net_arch=dict(pi=list(net_arch_pi), vf=list(net_arch_vf)),
        optimizer_class=optimizer_class if optimizer_class is not None else torch.optim.AdamW,
    )
    if activation_fn is not None:
        policy_kwargs["activation_fn"] = activation_fn
    if not split_weight_decay and weight_decay != 0.0:
        # No actor/critic split requested - apply uniformly via SB3's
        # normal optimizer_kwargs path, no optimizer rebuild needed.
        policy_kwargs["optimizer_kwargs"] = dict(weight_decay=weight_decay)
    if policy_kwargs_extra:
        # Extra keys OVERRIDE whatever was built above on conflict - see
        # this parameter's own docstring for why.
        policy_kwargs.update(policy_kwargs_extra)

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
        **(ppo_kwargs or {}),
    )

    if split_weight_decay:
        apply_split_weight_decay(
            model,
            actor_weight_decay=(
                actor_weight_decay if actor_weight_decay is not None else weight_decay
            ),
            critic_weight_decay=(
                critic_weight_decay if critic_weight_decay is not None else weight_decay
            ),
            learning_rate=learning_rate,
        )

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(eval_logs_dir, exist_ok=True)

    callbacks = []
    train_log_path = None
    if log_training_data:
        train_log_path = f"{eval_logs_dir}/ppo_{label}_train_log.csv"
        callbacks.append(
            BanditDataLoggerCallback(csv_path=train_log_path, groups=groups)
        )

    periodic_eval_callback = None
    if periodic_eval_freq is not None:
        periodic_eval_callback = PeriodicEvalCallback(
            groups=groups,
            incorrect_reward=incorrect_reward,
            eval_freq=periodic_eval_freq,
            eval_episodes=periodic_eval_episodes,
        )
        callbacks.append(periodic_eval_callback)

    if not callbacks:
        callback = None
    elif len(callbacks) == 1:
        callback = callbacks[0]
    else:
        callback = CallbackList(callbacks)

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=progress_bar,
        callback=callback,
        log_interval=log_interval,
    )

    if print_final_summary and verbose >= 1:
        # PPO's train() records fresh stats into model.logger on every
        # iteration regardless of log_interval - only the periodic *dump*
        # (print) was skipped. dump_logs() dumps whatever's currently
        # recorded, i.e. the last iteration's numbers, so this prints one
        # final box without needing log_interval=1 during the run.
        model.dump_logs(iteration=model._n_updates)

    if train_log_path is not None:
        print(f"training data -> {train_log_path}")

    model_path = f"{weights_dir}/ppo_{label}"
    if save_model:
        model.save(model_path)
        print(f"saved -> {model_path}.zip")

    correct, n, mean_reward = evaluate(
        model,
        groups,
        incorrect_reward,
        eval_episodes,
        log_csv=f"{eval_logs_dir}/ppo_{label}_eval.csv",
    )
    if print_eval_summary:
        print(
            f"PPO greedy eval: {correct}/{n} correct "
            f"({correct / n:.1%}), mean reward {mean_reward:.3f}"
        )
    return TrainResult(
        model=model,
        correct=correct,
        episodes=n,
        mean_reward=mean_reward,
        periodic_eval_history=(
            periodic_eval_callback.history if periodic_eval_callback is not None else []
        ),
    )
