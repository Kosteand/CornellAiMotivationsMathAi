import os
import io
import cProfile
import pstats

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from numba import njit

import matplotlib
matplotlib.use('Agg')  # must be before importing pyplot
import matplotlib.pyplot as plt

from Utilities.SpaceEnv import MazeEnv


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
    ) -> None:

        super().__init__()
        self.device = device
        self.n_envs = n_envs
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.validate_args_flag = validate_args_flag

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

        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=critic_lr, weight_decay=0)
        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=actor_lr, weight_decay=0)
        self.lstm_optim = optim.AdamW(self.lstm.parameters(), lr=critic_lr, weight_decay=0)

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

        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        nn.utils.clip_grad_norm_(self.lstm.parameters(), max_norm=0.5)

        self.critic_optim.step()
        self.actor_optim.step()
        self.lstm_optim.step()


def run_training(
    # core PPO / optimization hyperparameters
    criticLr: float = 0.0003,
    actorLr: float = 0.0001,
    criticLrFloor: float = 3e-5,
    actorLrFloor: float = 1e-5,
    nUpdates: int = 5000,
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
) -> tuple[int, int]:
    """
    Runs the full PPO training loop (identical in behavior to the original
    train.py script) with the given hyperparameters/settings, then runs 100
    post-training eval episodes.

    criticLrFloor and actorLrFloor are the linear-decay floors: criticLr
    decays linearly to criticLrFloor over nUpdates (and the LSTM optimizer
    shares this same schedule), while actorLr decays linearly to
    actorLrFloor.

    All of the original script's side effects are preserved
    (eval_logs/returns.csv, eval_logs/eval_results.txt, the saved weights,
    and the result1 plot). The only thing this function returns is:

        (left_count, right_count)

    i.e. how many of the 100 final eval episodes ended by reaching the left
    target vs. the right target (episodes that timed out without hitting
    either target are not counted in either number).

    load_weights, if True, loads the existing weight files (actor/critic/
    lstm) into the agent right after it's constructed, before training
    begins -- training then proceeds as normal on top of those loaded
    weights.

    save_weights, if True (the default), saves the trained actor/critic/
    lstm weights to disk at the end of the run, as the original script
    always did. If False, that save is skipped.
    """
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
                        validate_args_flag=validate_args_flag_param)

    if load_weights:
        agent.actor.load_state_dict(torch.load(actor_weights_path))
        agent.critic.load_state_dict(torch.load(critic_weights_path))
        agent.lstm.load_state_dict(torch.load(lstm_weights_path))

    envWrapper = gym.wrappers.vector.RecordEpisodeStatistics(env, buffer_length=10000)
    os.makedirs("eval_logs", exist_ok=True)
    with open("eval_logs/returns.csv", "w") as f:
        f.write("update,return,target\n")
    criticLosses = []
    actorLosses = []
    entropies = []
    current_max_steps = max_steps
    env.set_attr("max_steps", current_max_steps)
    resetOptions = {
        "randomSpawn": False,
        "randomSize": False,
        "randomTargetCoords": False
    }
    with open("eval_logs/eval_results.txt", "w") as f:
        f.write("")

    _ = compute_gae(np.zeros((2, nEnvs), np.float32), np.ones((2, nEnvs), np.float32),
                    np.zeros((2, nEnvs), np.float32), gamma, lam, 2, nEnvs)

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
        entropyBonus = max(endEntropy, beginEntropy - (samplePhase * 2 / nUpdates) * (beginEntropy - endEntropy))
        lr = criticLr * (1 - samplePhase / nUpdates)
        lr = max(lr, criticLrFloor)
        for param_group in agent.critic_optim.param_groups:
            param_group['lr'] = lr
        for param_group in agent.lstm_optim.param_groups:
            param_group['lr'] = lr
        actorLr_current = actorLr * (1 - samplePhase / nUpdates)
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
        for step in range(nStepsPerUpdate):
            actions, actionLogProbs, stateValuePreds, entropy, lstmState = agent.select_action(states, lstmState)
            epStates[step] = torch.as_tensor(states, dtype=torch.float32, device=cpu_device)
            epEntropies[step] = entropy
            epActions[step] = actions
            states, rewards, terminated, truncated, infos = envWrapper.step(actions.numpy())
            for i, term in enumerate(terminated):
                if term:
                    epTargetHits[(step, i)] = env.get_attr("last_target_hit")[i]
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
                    agent.update_parameters(critic_loss, actor_loss)
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
                        agent.update_parameters(critic_loss, actor_loss)
                else:
                    critic_loss, actor_loss, train_entropy = agent.get_losses(
                        epStates, epRewards, epActionLogProbs, epValuePreds, epEntropies,
                        masks, gamma, lam, entropyBonus, device,
                        epActions, clip_eps,
                        initialLstmState
                    )
                    agent.update_parameters(critic_loss, actor_loss)

        with open("eval_logs/returns.csv", "a") as f:
            for step_idx, step_info in enumerate(epInfos):
                if step_info is not None and "_episode" in step_info:
                    episode_mask = step_info["_episode"]
                    episode_returns = step_info["episode"]["r"]
                    for i, finished in enumerate(episode_mask):
                        if finished:
                            target = epTargetHits.get((step_idx, i), -1)
                            f.write(f"{samplePhase},{episode_returns[i]},{target}\n")

        criticLosses.append(critic_loss.detach().cpu().numpy())
        actorLosses.append(actor_loss.detach().cpu().numpy())
        entropies.append(train_entropy.detach().mean().cpu().numpy())

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

    left_count = 0
    right_count = 0
    no_reward_count = 0

    print("Running 100 post-training eval episodes...")
    for eval_run in range(100):
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
                _, actionLogits, lstmStateEval = agent.forward(obs[None, :], lstmStateEval)
                action = torch.argmax(actionLogits, dim=-1).item()
                obs, reward, terminated, truncated, info = evalEnv.step(action)
                done = terminated or truncated

        if terminated:
            if evalEnv.last_target_hit == 0:
                left_count += 1
            elif evalEnv.last_target_hit == 1:
                right_count += 1
            else:
                no_reward_count += 1

        evalEnv.close()

    print(f"Left target: {left_count}, Right target: {right_count}, No reward: {no_reward_count}")

    """ plot the results """
    rolling_length = 20
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
    fig.suptitle(
        f"Training plots for {agent.__class__.__name__} \n \
                (n_envs={nEnvs}, n_steps_per_update={nStepsPerUpdate})"
    )

    axs[0][0].set_title("Episode Returns")
    episode_returns_moving_average = (
        np.convolve(
            np.array(envWrapper.return_queue).flatten(),
            np.ones(rolling_length),
            mode="valid",
        )
        / rolling_length
    )
    axs[0][0].plot(
        np.arange(len(episode_returns_moving_average)) / nEnvs,
        episode_returns_moving_average,
    )
    axs[0][0].set_xlabel("Number of episodes")

    axs[1][0].set_title("Entropy")
    entropy_moving_average = (
        np.convolve(np.array(entropies), np.ones(rolling_length), mode="valid")
        / rolling_length
    )
    axs[1][0].plot(entropy_moving_average)
    axs[1][0].set_xlabel("Number of updates")

    axs[0][1].set_title("Critic Loss")
    critic_losses_moving_average = (
        np.convolve(np.array(criticLosses).flatten(), np.ones(rolling_length), mode="valid")
        / rolling_length
    )
    axs[0][1].plot(critic_losses_moving_average)
    axs[0][1].set_xlabel("Number of updates")

    axs[1][1].set_title("Actor Loss")
    actor_losses_moving_average = (
        np.convolve(np.array(actorLosses).flatten(), np.ones(rolling_length), mode="valid")
        / rolling_length
    )
    axs[1][1].plot(actor_losses_moving_average)
    axs[1][1].set_xlabel("Number of updates")

    plt.tight_layout()
    plt.savefig("result1")

    if save_weights:
        if not os.path.exists("weights"):
            os.mkdir("weights")

        torch.save(agent.actor.state_dict(), actor_weights_path)
        torch.save(agent.critic.state_dict(), critic_weights_path)
        torch.save(agent.lstm.state_dict(), lstm_weights_path)

    env.close()

    return left_count, right_count