import os
import io
import cProfile
import pstats
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from numba import njit

from Utilities.SpaceEnv import MazeEnv


# MazeEnv's action space is Discrete(4): direction = action, with
# even/odd choosing +1/-1 and //2 choosing the y-axis vs x-axis (see
# SpaceEnv.py's step()). Since the two targets sit at [0,5] (left) and
# [10,5] (right), increasing x moves toward the right target.
# NOTE: up/down here assumes increasing y is "up" -- flip if your
# convention runs the other way, this doesn't affect training itself,
# only the label written to episode_info.csv.
ACTION_NAMES = {0: "up", 1: "down", 2: "right", 3: "left"}


@njit
def compute_gae(rewards, masks, value_preds, gamma, lam, T, n_envs):
    gamma = np.float32(gamma)
    lam = np.float32(lam)
    advantages = np.zeros((T, n_envs), dtype=np.float32)
    gae = np.zeros(n_envs, dtype=np.float32)
    for t in range(T - 2, -1, -1):
        td_error = rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
        gae = td_error + gamma * lam * masks[t] * gae
        advantages[t] = gae
    return advantages


class ActoCritic(nn.Module):
    def __init__(
        self,
        n_features: np.int_,
        n_actions: np.int_,
        device: torch.device,
        critic_lr: np.float32,
        actor_lr: np.float32,
        n_envs: int,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        validate_args_flag: bool = True,
        actor_weight_decay: float = 0.0,
        critic_weight_decay: float = 0.0,
        lstm_weight_decay: float = 0.0,
        lstm_lr: np.float32 = None,
    ) -> None:

        super().__init__()
        self.device = device
        self.n_envs = n_envs
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.validate_args_flag = validate_args_flag

        if lstm_lr is None:
            lstm_lr = critic_lr  # preserves the original shared-schedule behavior if not given

        self.lstm = nn.LSTM(n_features, self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True).to(device)

        critic_layers = [
            nn.Linear(self.lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ]

        actor_layers = [
            nn.Linear(self.lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        ]

        self.critic = nn.Sequential(*critic_layers).to(self.device)
        self.actor = nn.Sequential(*actor_layers).to(self.device)

        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=critic_lr, weight_decay=critic_weight_decay)
        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=actor_lr, weight_decay=actor_weight_decay)
        self.lstm_optim = optim.AdamW(self.lstm.parameters(), lr=lstm_lr, weight_decay=lstm_weight_decay)

    def forward(self, x: np.ndarray, lstmState=None):
        x = torch.Tensor(x).to(next(self.parameters()).device)
        x, lstmState = self.lstm(x.unsqueeze(1), lstmState)
        x = x.squeeze(1)
        state_values = self.critic(x)
        action_logits_vec = self.actor(x)
        return (state_values, action_logits_vec, lstmState)

    def select_action(self, x: np.ndarray, lstmState=None):
        stateValues, action_logits, lstmState = self.forward(x, lstmState)
        if torch.isnan(action_logits).any():
            print(f"NaN in action_logits during rollout")
        action_pd = torch.distributions.Categorical(logits=action_logits, validate_args=self.validate_args_flag)
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()
        return (actions, action_log_probs, stateValues, entropy, lstmState)

    def get_losses(self, states, rewards, action_log_probs_old, value_preds_old,
                    entropy_old, masks, gamma, lam, ent_coef, device, actions, clip_eps,
                    initial_lstm_state=None):
        T = len(rewards)
        n_envs_actual = states.shape[1]

        states_seq = states.permute(1, 0, 2)
        x, _ = self.lstm(states_seq, initial_lstm_state)
        x = x.permute(1, 0, 2)
        if torch.isnan(x).any():
            print(f"NaN detected in LSTM output, skipping update")
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        x_flat = x.reshape(T * n_envs_actual, self.lstm_hidden_size)

        value_preds = self.critic(x_flat).reshape(T, n_envs_actual)
        action_logits = self.actor(x_flat).reshape(T, n_envs_actual, -1)
        action_pd = torch.distributions.Categorical(logits=action_logits, validate_args=self.validate_args_flag)
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()

        rewards_np = rewards.detach().cpu().numpy()
        masks_np = masks.detach().cpu().numpy()
        value_preds_np = value_preds.detach().cpu().numpy()

        advantages_np = compute_gae(rewards_np, masks_np, value_preds_np, gamma, lam, T, n_envs_actual)

        advantages = torch.from_numpy(advantages_np).to(device)
        returns = advantages + value_preds.detach()

        critic_loss = ((returns - value_preds).pow(2) * masks).sum() / masks.sum()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(action_log_probs - action_log_probs_old.detach())
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        actor_loss = -(
            torch.min(ratio * advantages.detach(), clipped_ratio * advantages.detach()) * masks
        ).sum() / masks.sum() - ent_coef * (entropy * masks).sum() / masks.sum()

        return critic_loss, actor_loss, entropy

    def update_parameters(self, critic_loss, actor_loss):
        self.critic_optim.zero_grad(set_to_none=True)
        self.actor_optim.zero_grad(set_to_none=True)
        self.lstm_optim.zero_grad(set_to_none=True)

        (critic_loss + actor_loss).backward()

        # clip_grad_norm_ returns the PRE-clipping norm -- capturing these
        # lets us see how often/how hard clipping is actually engaging,
        # rather than guessing.
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        lstm_grad_norm = nn.utils.clip_grad_norm_(self.lstm.parameters(), max_norm=0.5)

        self.critic_optim.step()
        self.actor_optim.step()
        self.lstm_optim.step()

        return (float(critic_grad_norm), float(actor_grad_norm), float(lstm_grad_norm))


def run_eval_until_target_hits(agent, low, high, spawn, target_awards, target_coords,
                                 step_penalty, target_hits=100,
                                 stall_check_interval=100, stall_hit_rate=0.10):
    """
    Runs fresh eval episodes with an already-trained agent until `target_hits`
    episodes have reached EITHER target (left or right). Misses (timeouts
    that reach neither target) don't count toward target_hits, but ARE
    tallied and returned separately.

    Safety abort ("stall check"): every `stall_check_interval` episodes
    attempted (default every 100), if the number of hits so far is below
    `stall_hit_rate` (default 10%) of episodes attempted, evaluation stops
    early rather than potentially running forever on an agent that's
    essentially never reaching a target. E.g. with the defaults: stop if
    fewer than 10 hits after 100 episodes, fewer than 20 after 200, etc.

    Returns (left_count, right_count, miss_count, total_episodes, stalled).
    If stalled is True, left_count + right_count will be less than
    target_hits -- the caller should treat this run as inconclusive rather
    than a valid sample of the agent's true left/right split.
    """
    left_count = 0
    right_count = 0
    miss_count = 0
    total_episodes = 0
    stalled = False

    while left_count + right_count < target_hits:
        evalEnv = MazeEnv(low, high, spawn, target_awards, target_coords,
                          heatMapTypes=[], vision_range=1, use_ray_scans=False,
                          step_penalty=step_penalty)
        evalResetOptions = {
            "randomSpawn": False,
            "randomSize": False,
            "randomTargetCoords": False,
            "max_steps": 500
        }
        obs, info = evalEnv.reset(options=evalResetOptions)
        done = False
        lstmStateEval = None
        terminated = False

        with torch.no_grad():
            while not done:
                actions, _, _, _, lstmStateEval = agent.select_action(obs[None, :], lstmStateEval)
                action = actions.item()
                obs, reward, terminated, truncated, info = evalEnv.step(action)
                done = terminated or truncated

        total_episodes += 1
        if terminated and evalEnv.last_target_hit == 0:
            left_count += 1
        elif terminated and evalEnv.last_target_hit == 1:
            right_count += 1
        else:
            miss_count += 1

        evalEnv.close()

        if total_episodes % stall_check_interval == 0:
            hits_so_far = left_count + right_count
            required_hits = stall_hit_rate * total_episodes
            if hits_so_far < required_hits:
                stalled = True
                break

    return left_count, right_count, miss_count, total_episodes, stalled


def run_training(
    # core PPO / optimization hyperparameters
    criticLr: float = 0.0003,
    actorLr: float = 0.0001,
    criticLrFloor: float = 3e-5,
    actorLrFloor: float = 1e-5,
    lstmLr: float = 0.0003,
    lstmLrFloor: float = 3e-5,
    nUpdates: int = 5000,
    lrDecayHorizon: int = None,
    entropyDecayHorizon: int = None,
    nStepsPerUpdate: int = 512,
    ppo_epochs: int = 4,
    clip_eps: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    beginEntropy: float = 0.15,
    endEntropy: float = 0.05,
    # environment / reward shaping
    step_penalty: float = -0.1,
    left_reward: float = 1000,
    right_reward: float = 10,
    max_steps: int = 500,
    min_steps: int = 500,
    step_decay: int = 20,
    # run settings / debug flags
    useCProfiler: bool = False,
    useTorchProfiler: bool = False,
    validate_args_flag_param: bool = True,
    check_for_NaN_errors: bool = False,
    load_weights: bool = False,
    save_weights: bool = True,
    target_hits: int = 100,
    early_stop_patience: int = 0,
    early_stop_min_updates: int = 500,
    actor_weight_decay: float = 0.0,
    critic_weight_decay: float = 0.0,
    lstm_weight_decay: float = 0.0,
    progress_callback: object = None,
    progress_callback_interval: int = 1,
) -> tuple[int, int, int]:
    """
    Runs the full PPO training loop (identical in behavior to the original
    train.py script) with the given hyperparameters/settings, then runs
    post-training eval episodes until `target_hits` episodes have reached
    a target (see run_eval_until_target_hits for the stall-abort behavior).

    criticLrFloor and actorLrFloor are the linear-decay floors: criticLr
    decays linearly to criticLrFloor over lrDecayHorizon, while actorLr
    decays linearly to actorLrFloor. lstmLr/lstmLrFloor give the LSTM
    optimizer its OWN independent learning rate and decay schedule,
    rather than sharing the critic's (as it did previously) -- their
    defaults (0.0003 / 3e-5) are exactly the critic's current defaults,
    so leaving them unset reproduces the original shared-schedule
    behavior exactly.

    lrDecayHorizon and entropyDecayHorizon decouple "how many updates
    does THIS run actually execute" (nUpdates) from "over how many
    updates do the LR and entropy schedules unfold," respectively. Both
    default to None, which falls back to nUpdates -- this is deliberately
    a pure substitution, not a reshaped formula: the LR formulas still
    ramp toward 0 (clamped at the floor, same as always) and the entropy
    formula still has its original factor of 2 (reaching endEntropy at
    HALF of entropyDecayHorizon, same as always) -- only the denominator
    that used to be hardcoded to nUpdates is now independently settable.
    This means the default (leaving both unset) reproduces the exact
    original numeric behavior with no approximation: it's the same
    formula with the same value substituted in, not a redesigned formula
    that happens to land close to the old one. (An earlier version of
    this parameter unified LR and entropy under one shared horizon with a
    reshaped formula -- that version's default did NOT exactly reproduce
    the original LR schedule, only the original entropy schedule, because
    the two formulas had different implicit shapes to begin with. Keeping
    them as two separate parameters, with the original formula shapes
    untouched, avoids that mismatch entirely.)

    These matter for a REDUCED-budget run meant to be a faithful PREFIX
    of a longer intended run (e.g. a cheap hyperparameter search using
    nUpdates=1000 to approximate the first 1000 updates of a real
    nUpdates=5000 run): if the horizons were tied to nUpdates itself, a
    reduced-budget run would anneal both LR and entropy proportionally
    faster in absolute update-count terms than the real run it's meant to
    approximate, so "this LR looked fine over 1000 updates" could
    silently fail to transfer to "this LR is fine over 5000 updates,"
    since the two runs were never actually on the same decay trajectory
    to begin with. Setting lrDecayHorizon=entropyDecayHorizon=5000
    explicitly while nUpdates stays at the reduced search budget makes
    the reduced run's updates a true prefix of what the full run would
    actually do.

    All of the original script's side effects are preserved
    (eval_logs/episode_info.csv, eval_logs/eval_results.txt, the saved weights,
    and eval_logs/update_info.csv, read later by plot_returns.py to
    produce training_plots.png -- this function itself does not plot
    anything). The only thing this function returns is:

        (left_count, right_count, miss_count)

    left_count + right_count will equal target_hits unless eval stalled
    (see run_eval_until_target_hits) -- in that case it will be less, and
    the caller should treat the run as inconclusive rather than a valid
    left/right split.

    load_weights, if True, loads the existing weight files (actor/critic/
    lstm) into the agent right after it's constructed, before training
    begins -- training then proceeds as normal on top of those loaded
    weights.

    save_weights, if True (the default), saves the trained actor/critic/
    lstm weights to disk at the end of the run, as the original script
    always did. If False, that save is skipped.

    early_stop_patience, if > 0, enables early stopping: training tracks
    the last early_stop_patience completed episode outcomes across ALL
    envs combined -- hits AND misses alike -- and stops training early
    (proceeding straight to the post-training eval, as if nUpdates had
    been reached normally) once ALL of those outcomes are the exact same
    target, with zero misses and zero hits to the other target anywhere
    in that window. Misses are deliberately included rather than skipped:
    a single miss slipping into an otherwise one-sided streak means the
    policy hasn't fully collapsed to one deterministic path yet, so it
    resets the streak rather than being ignored -- this makes the trigger
    stricter/more conservative, only firing on genuine convergence rather
    than "hasn't gone the other way recently." early_stop_min_updates is a
    floor on how many updates must run first, so an early lucky streak
    right at the start of training (before the policy has learned
    anything real) can't trigger a spurious early stop. Disabled by
    default (early_stop_patience=0) to preserve the original script's
    behavior exactly when not explicitly opted into.

    actor_weight_decay, critic_weight_decay, lstm_weight_decay: L2 weight
    decay passed directly to each component's AdamW optimizer (the actor
    and critic each get their own; the LSTM has always used its own
    separate optimizer -- see lstm_optim -- so it gets its own decay
    setting too, independent of the critic's, even though it currently
    shares the critic's learning-rate schedule). All three default to 0.0,
    exactly matching the original script's hardcoded weight_decay=0 on
    every optimizer -- so leaving these unset changes nothing.

    progress_callback, if given, is called every progress_callback_interval
    updates as progress_callback(samplePhase, metrics), where metrics is a
    dict with this update's critic_loss, actor_loss, entropy,
    critic_grad_norm, actor_grad_norm, lstm_grad_norm (all floats -- the
    same values written to update_info.csv this update), plus
    left_count/right_count/miss_count (ints, tallied from THIS update's
    finished episodes only). This exists so an external caller (e.g. an
    Optuna objective function) can inspect training as it happens and
    decide to abort early -- do this by raising optuna.TrialPruned() (or
    any other exception) from inside the callback; it will propagate up
    through run_training() uncaught, skipping the post-training eval and
    weight-saving entirely (which is the point: a pruned trial shouldn't
    pay for either). A normal (non-raising) callback has no effect on
    training whatsoever -- it's called purely for its side effects on
    whatever object owns it (e.g. accumulating history, calling
    trial.report()), never using its return value.
    """
    if lrDecayHorizon is None:
        lrDecayHorizon = nUpdates
    if entropyDecayHorizon is None:
        entropyDecayHorizon = nUpdates

    if useCProfiler:
        profiler = cProfile.Profile()
        profiler.enable()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    print(f"Running RL Training on: {device.upper()}")

    actor_weights_path = "weights/actor_weights.h5"
    critic_weights_path = "weights/critic_weights.h5"
    lstm_weights_path = "weights/lstm_weights.h5"

    low = np.array([0, 0])
    high = np.array([10, 10])
    spawn = np.array([7, 5])
    target_coords = np.array([[0, 5], [10, 5]])
    target_awards = np.array([left_reward, right_reward])

    def makeEnv():
        def _init():
            heatMapTypes = []
            env = MazeEnv(low, high, spawn, target_awards, target_coords,
                          heatMapTypes=heatMapTypes, vision_range=1, use_ray_scans=False,
                          step_penalty=step_penalty)
            return env
        return _init

    nEnvs = 22
    env = gym.vector.SyncVectorEnv([makeEnv() for _ in range(nEnvs)])

    obsShape = env.single_observation_space.shape[0]
    actionShape = env.single_action_space.n

    entropyBonus = beginEntropy
    lstmState = None

    agent = ActoCritic(obsShape, actionShape, device, criticLr, actorLr, nEnvs, 256, 2,
                        validate_args_flag=validate_args_flag_param,
                        actor_weight_decay=actor_weight_decay,
                        critic_weight_decay=critic_weight_decay,
                        lstm_weight_decay=lstm_weight_decay,
                        lstm_lr=lstmLr)

    if load_weights:
        agent.actor.load_state_dict(torch.load(actor_weights_path))
        agent.critic.load_state_dict(torch.load(critic_weights_path))
        agent.lstm.load_state_dict(torch.load(lstm_weights_path))

    envWrapper = gym.wrappers.vector.RecordEpisodeStatistics(env, buffer_length=10000)
    os.makedirs("eval_logs", exist_ok=True)
    with open("eval_logs/episode_info.csv", "w") as f:
        f.write("update,return,target,first_action\n")
    with open("eval_logs/update_info.csv", "w") as f:
        f.write("update,critic_loss,actor_loss,entropy,critic_grad_norm,actor_grad_norm,lstm_grad_norm\n")

    criticLosses = []
    actorLosses = []
    entropies = []
    current_max_steps = max_steps
    env.set_attr("max_steps", current_max_steps)

    # Tracks, per env, the action taken on the first REAL step of that env's
    # current (in-progress) episode.
    #
    # SyncVectorEnv's auto-reset is "next-step": when env i terminates during
    # a .step() call, that env isn't actually reset until the FOLLOWING
    # .step() call -- and whatever action is passed in for env i on that
    # following call gets silently discarded so the wrapper can return the
    # fresh reset observation instead. Debug output confirmed this: current_step
    # for a just-terminated env stays flat (doesn't zero out) until one extra
    # .step() call has passed.
    #
    # So there are two stages after a termination, not one:
    #   awaitingReset[i]        -- next action for env i will be silently
    #                              discarded while the env internally resets.
    #   awaitingFirstAction[i]  -- env i has now actually reset; the NEXT
    #                              action chosen is the real first move.
    #
    # The very first episode for each env (from the synchronous envWrapper.reset()
    # call before the loop starts) doesn't go through this discard step, so
    # awaitingFirstAction starts True and awaitingReset starts False.
    firstActions = np.full(nEnvs, -1, dtype=np.int64)
    awaitingReset = np.zeros(nEnvs, dtype=bool)
    awaitingFirstAction = np.ones(nEnvs, dtype=bool)

    resetOptions = {
        "randomSpawn": False,
        "randomSize": False,
        "randomTargetCoords": False
    }
    with open("eval_logs/eval_results.txt", "w") as f:
        f.write("")

    _ = compute_gae(np.zeros((2, nEnvs), np.float32), np.ones((2, nEnvs), np.float32),
                    np.zeros((2, nEnvs), np.float32), gamma, lam, 2, nEnvs)

    # Early-stop tracker: the most recent early_stop_patience completed
    # (non-miss) episode outcomes, pooled across all envs. If they're all
    # the same target once we're past early_stop_min_updates, the policy
    # has clearly committed and training stops early.
    recent_hit_targets = deque(maxlen=early_stop_patience) if early_stop_patience > 0 else None
    early_stopped = False

    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) if useTorchProfiler else None

    if prof:
        prof.__enter__()

    critic_loss = actor_loss = train_entropy = None

    for samplePhase in tqdm(range(nUpdates)):
        entropyBonus = max(endEntropy, beginEntropy - (samplePhase * 2 / entropyDecayHorizon) * (beginEntropy - endEntropy))
        lr = criticLr * (1 - samplePhase / lrDecayHorizon)
        lr = max(lr, criticLrFloor)
        for param_group in agent.critic_optim.param_groups:
            param_group['lr'] = lr
        lstmLr_current = lstmLr * (1 - samplePhase / lrDecayHorizon)
        lstmLr_current = max(lstmLr_current, lstmLrFloor)
        for param_group in agent.lstm_optim.param_groups:
            param_group['lr'] = lstmLr_current
        actorLr_current = actorLr * (1 - samplePhase / lrDecayHorizon)
        actorLr_current = max(actorLr_current, actorLrFloor)
        for param_group in agent.actor_optim.param_groups:
            param_group['lr'] = actorLr_current
        if (samplePhase + 1) % 100 == 0 and current_max_steps > min_steps:
            current_max_steps -= min(step_decay, current_max_steps - min_steps)
            env.set_attr("max_steps", current_max_steps)
            print(f"Tightening the clock! New Max Steps: {current_max_steps}")

        epValuePreds = torch.zeros(nStepsPerUpdate, nEnvs, device=cpu_device)
        epRewards = torch.zeros(nStepsPerUpdate, nEnvs, device=cpu_device)
        epActionLogProbs = torch.zeros(nStepsPerUpdate, nEnvs, device=cpu_device)
        epActions = torch.zeros(nStepsPerUpdate, nEnvs, device=cpu_device, dtype=torch.long)
        masks = torch.zeros(nStepsPerUpdate, nEnvs, device=cpu_device)

        if samplePhase == 0:
            states, __ = envWrapper.reset(options=resetOptions)

        epEntropies = torch.zeros(nStepsPerUpdate, nEnvs, device=cpu_device)
        epStates = torch.zeros(nStepsPerUpdate, nEnvs, obsShape, device=cpu_device)
        initialLstmState = (lstmState[0].detach().clone(), lstmState[1].detach().clone()) if lstmState is not None else None

        agent.to(cpu_device)
        if lstmState is not None:
            lstmState = (lstmState[0].to(cpu_device), lstmState[1].to(cpu_device))

        epInfos = []
        epTargetHits = {}
        epFirstActions = {}
        for step in range(nStepsPerUpdate):
            actions, actionLogProbs, stateValuePreds, entropy, lstmState = agent.select_action(states, lstmState)
            epStates[step] = torch.as_tensor(states, dtype=torch.float32, device=cpu_device)
            epEntropies[step] = entropy
            epActions[step] = actions

            actions_np = actions.numpy()
            for i in range(nEnvs):
                if awaitingReset[i]:
                    # This action is about to be discarded internally by the
                    # vector env's auto-reset -- NOT a real first move.
                    awaitingReset[i] = False
                    awaitingFirstAction[i] = True
                elif awaitingFirstAction[i]:
                    firstActions[i] = actions_np[i]
                    awaitingFirstAction[i] = False

            states, rewards, terminated, truncated, infos = envWrapper.step(actions.numpy())
            for i, term in enumerate(terminated):
                if term:
                    epTargetHits[(step, i)] = env.get_attr("last_target_hit")[i]
                    epFirstActions[(step, i)] = int(firstActions[i])
                    awaitingReset[i] = True
            epInfos.append(infos)

            dones_list = [term or trunc for term, trunc in zip(terminated, truncated)]
            dones = torch.as_tensor(dones_list, dtype=torch.float32, device=cpu_device)

            if lstmState is not None:
                h, c = lstmState
                mask = (1 - dones).view(1, -1, 1)
                lstmState = (h * mask, c * mask)

            epValuePreds[step] = torch.squeeze(stateValuePreds)
            epRewards[step] = torch.as_tensor(rewards, dtype=torch.float32, device=cpu_device)
            epActionLogProbs[step] = actionLogProbs

            if lstmState is not None:
                lstmState = (lstmState[0].detach(), lstmState[1].detach())

            masks[step] = torch.as_tensor([not d for d in dones_list], dtype=torch.float32, device=cpu_device)

        agent.to(device)
        epStates = epStates.to(device)
        epRewards = epRewards.to(device)
        epActionLogProbs = epActionLogProbs.to(device)
        epValuePreds = epValuePreds.to(device)
        epEntropies = epEntropies.to(device)
        epActions = epActions.to(device)
        masks = masks.to(device)
        if lstmState is not None:
            lstmState = (lstmState[0].to(device), lstmState[1].to(device))
        if initialLstmState is not None:
            initialLstmState = (initialLstmState[0].to(device), initialLstmState[1].to(device))

        if check_for_NaN_errors:
            for _ in range(ppo_epochs):
                with torch.autograd.detect_anomaly():
                    critic_loss, actor_loss, train_entropy = agent.get_losses(
                        epStates, epRewards, epActionLogProbs, epValuePreds, epEntropies,
                        masks, gamma, lam, entropyBonus, device,
                        epActions, clip_eps,
                        initialLstmState
                    )
                    critic_grad_norm, actor_grad_norm, lstm_grad_norm = agent.update_parameters(critic_loss, actor_loss)
        else:
            for _ in range(ppo_epochs):
                if useTorchProfiler and samplePhase < 3:
                    with torch.mps.profiler.profile(mode="interval", wait_until_completed=True):
                        critic_loss, actor_loss, train_entropy = agent.get_losses(
                            epStates, epRewards, epActionLogProbs, epValuePreds, epEntropies,
                            masks, gamma, lam, entropyBonus, device,
                            epActions, clip_eps,
                            initialLstmState
                        )
                        critic_grad_norm, actor_grad_norm, lstm_grad_norm = agent.update_parameters(critic_loss, actor_loss)
                else:
                    critic_loss, actor_loss, train_entropy = agent.get_losses(
                        epStates, epRewards, epActionLogProbs, epValuePreds, epEntropies,
                        masks, gamma, lam, entropyBonus, device,
                        epActions, clip_eps,
                        initialLstmState
                    )
                    critic_grad_norm, actor_grad_norm, lstm_grad_norm = agent.update_parameters(critic_loss, actor_loss)

        with open("eval_logs/episode_info.csv", "a") as f:
            for step_idx, step_info in enumerate(epInfos):
                if step_info is not None and "_episode" in step_info:
                    episode_mask = step_info["_episode"]
                    episode_returns = step_info["episode"]["r"]
                    for i, finished in enumerate(episode_mask):
                        if finished:
                            target = epTargetHits.get((step_idx, i), -1)
                            first_action = epFirstActions.get((step_idx, i), -1)
                            first_action_name = ACTION_NAMES.get(first_action, "none")
                            f.write(f"{samplePhase},{episode_returns[i]},{target},{first_action_name}\n")
                            if recent_hit_targets is not None:
                                recent_hit_targets.append(target)

        criticLosses.append(critic_loss.detach().cpu().numpy())
        actorLosses.append(actor_loss.detach().cpu().numpy())
        entropies.append(train_entropy.detach().mean().cpu().numpy())

        with open("eval_logs/update_info.csv", "a") as f:
            f.write(f"{samplePhase},{float(criticLosses[-1])},{float(actorLosses[-1])},"
                    f"{float(entropies[-1])},{critic_grad_norm},{actor_grad_norm},{lstm_grad_norm}\n")

        if progress_callback is not None and (samplePhase % progress_callback_interval == 0):
            this_update_left = sum(1 for v in epTargetHits.values() if v == 0)
            this_update_right = sum(1 for v in epTargetHits.values() if v == 1)
            this_update_miss = sum(1 for v in epTargetHits.values() if v == -1)
            progress_callback(samplePhase, {
                "critic_loss": float(criticLosses[-1]),
                "actor_loss": float(actorLosses[-1]),
                "entropy": float(entropies[-1]),
                "critic_grad_norm": critic_grad_norm,
                "actor_grad_norm": actor_grad_norm,
                "lstm_grad_norm": lstm_grad_norm,
                "left_count": this_update_left,
                "right_count": this_update_right,
                "miss_count": this_update_miss,
            })

        if (recent_hit_targets is not None
                and samplePhase + 1 >= early_stop_min_updates
                and len(recent_hit_targets) == early_stop_patience
                and len(set(recent_hit_targets)) == 1
                and recent_hit_targets[0] in (0, 1)):
            committed_target = recent_hit_targets[0]
            print(f"Early stopping at update {samplePhase + 1}/{nUpdates}: "
                  f"last {early_stop_patience} hits all went to target "
                  f"{committed_target} ({'left' if committed_target == 0 else 'right'}) "
                  f"-- policy appears to have converged.")
            early_stopped = True
            break

        if samplePhase == 5 and (useCProfiler or useTorchProfiler):
            if prof:
                prof.__exit__(None, None, None)
                print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
                prof.export_chrome_trace("./visualize/profile_trace.json")
                print("Chrome trace saved to ./visualize/profile_trace.json")
            if useCProfiler:
                profiler.disable()
                stream = io.StringIO()
                stats = pstats.Stats(profiler, stream=stream)
                stats.sort_stats('tottime')
                stats.print_stats(20)
                print(stream.getvalue())
                profiler.dump_stats("./visualize/profile_output.prof")
            break

    """ Run 100 eval episodes after training """
    agent.actor.eval()
    agent.critic.eval()
    agent.lstm.eval()

    left_count, right_count, miss_count, total_episodes, stalled = run_eval_until_target_hits(
        agent, low, high, spawn, target_awards, target_coords, step_penalty,
        target_hits=target_hits,
    )

    if stalled:
        print(f"WARNING: eval stalled after {total_episodes} episodes "
              f"({left_count + right_count} hits, hit rate below the 10% "
              f"floor) -- stopped early rather than continuing indefinitely.")
    print(f"Left target: {left_count}, Right target: {right_count}, No reward: {miss_count} "
          f"(over {total_episodes} episodes)")

    if save_weights:
        if not os.path.exists("weights"):
            os.mkdir("weights")

        torch.save(agent.actor.state_dict(), actor_weights_path)
        torch.save(agent.critic.state_dict(), critic_weights_path)
        torch.save(agent.lstm.state_dict(), lstm_weights_path)

    env.close()

    return left_count, right_count, miss_count