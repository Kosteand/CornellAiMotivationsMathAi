import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import os  
import io

from Utilities.SpaceEnv import *
from Utilities.HeatMap import *
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from functools import partial

import cProfile
import pstats

import matplotlib
matplotlib.use('Agg')  # must be before importing pyplot
import matplotlib.pyplot as plt

useProfiler = True # !!!!! use to test performance
if(useProfiler):
    profiler = cProfile.Profile()
    profiler.enable()

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
    ) -> None:
        
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        self.lstm = nn.LSTM(n_features, self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True).to(device)


        critic_layers = [
            nn.Linear(self.lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # estimate V(s)
        ]

        actor_layers = [
            nn.Linear(self.lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(
                32, n_actions
            ),  # estimate action logits (will be fed into a softmax later)
        ]

        # define actor and critic networks
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        self.actor = nn.Sequential(*actor_layers).to(self.device)

        # define optimizers for actor and critic
        self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=actor_lr)
        self.lstm_optim = optim.RMSprop(self.lstm.parameters(), lr=critic_lr)
        
    def forward(self, x: np.ndarray,lstmState=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the networks.

        Args:
            x: A batched vector of states.

        Returns:
            state_values: A tensor with the state values, with shape [n_envs,].
            action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
        """
        x = torch.Tensor(x).to(next(self.parameters()).device)
        x, lstmState = self.lstm(x.unsqueeze(1), lstmState)
        x = x.squeeze(1)
        state_values = self.critic(x)  # shape: [n_envs,]
        action_logits_vec = self.actor(x)  # shape: [n_envs, n_actions]
        return (state_values, action_logits_vec, lstmState)

    def select_action(
        self, x: np.ndarray, lstmState=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of the chosen actions and the log-probs of those actions.

        Args:
            x: A batched vector of states.,

        Returns:
            actions: A tensor with the actions, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
            state_values: A tensor with the state values, with shape [n_steps_per_update, n_envs].
        """
        stateValues, action_logits, lstmState = self.forward(x, lstmState)
        if torch.isnan(action_logits).any():
            print(f"NaN in action_logits during rollout")
        action_pd = torch.distributions.Categorical(logits=action_logits, validate_args=False) # !!!!! change validate_args to True if expereincing NaN errors
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)  # compute while actions still on CPU
        entropy = action_pd.entropy()
        # !!!!! actions = actions.to(device)  # move to MPS after log_prob is computed
        return (actions, action_log_probs, stateValues, entropy, lstmState)

    def get_losses(self, states, rewards, action_log_probs_old, value_preds_old,
                  entropy_old, masks, gamma, lam, ent_coef, device, actions, clip_eps,
                  initial_lstm_state=None):
        T = len(rewards)
        n_envs_actual = states.shape[1]

        # Re-run LSTM forward pass on stored states
        states_seq = states.permute(1, 0, 2)
        x, _ = self.lstm(states_seq, initial_lstm_state)
        x = x.permute(1, 0, 2)
        if torch.isnan(x).any():
            print(f"NaN detected in LSTM output at update {samplePhase}, skipping update")
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        x_flat = x.reshape(T * n_envs_actual, self.lstm_hidden_size)



        value_preds = self.critic(x_flat).reshape(T, n_envs_actual)
        action_logits = self.actor(x_flat).reshape(T, n_envs_actual, -1)
        action_pd = torch.distributions.Categorical(logits=action_logits, validate_args=False) # !!!!! change validate_args to True if experiencing NaN errors
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()

        # GAE
        advantages = torch.zeros(T, n_envs_actual, device=device)
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = rewards[t] + gamma * masks[t] * value_preds[t+1].detach() - value_preds[t].detach()
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        returns = advantages + value_preds.detach()

        # Critic loss
        critic_loss = ((returns - value_preds).pow(2) * masks).sum() / masks.sum()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO clipped actor loss
        ratio = torch.exp(action_log_probs - action_log_probs_old.detach())
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        actor_loss = -(
            torch.min(ratio * advantages.detach(), clipped_ratio * advantages.detach()) * masks
        ).sum() / masks.sum() - ent_coef * (entropy * masks).sum() / masks.sum()

        return critic_loss, actor_loss

    def update_parameters(self, critic_loss, actor_loss):
        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.lstm_optim.zero_grad()

        critic_loss.backward(retain_graph=True)
        actor_loss.backward()

        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        nn.utils.clip_grad_norm_(self.lstm.parameters(), max_norm=0.5)

        self.critic_optim.step()
        self.actor_optim.step()
        self.lstm_optim.step()


#device = "cuda" if torch.cuda.is_available() else "cpu" !!!!!use for docker
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")
print(f"Running RL Training on: {device.upper()}")

saveWeights = True
load_weights = False

actor_weights_path = "weights/actor_weights.h5"
critic_weights_path = "weights/critic_weights.h5"
lstm_weights_path = "weights/lstm_weights.h5"

 # 3. Create Environment with Rendering
low = np.array([0, 0])
high = np.array([10, 10])      # inclusive bounds -> 11x11 map
num_cols = high[0] - low[0] + 1
num_rows = high[1] - low[1] + 1

spawn = np.array([7, 5])
target_coords = np.array([[0, 5], [10,5]])
target_awards = np.array([200, 10])
'''lookU = lambda lowLeft, topRight, targetCords: DirectionWrap(
    ManhattanDistanceTarget(lowLeft=lowLeft, topRight=topRight, targetCords=targetCords), 
    offset=np.array([0, 1]))

lookD = lambda lowLeft, topRight, targetCords: DirectionWrap(
    ManhattanDistanceTarget(lowLeft=lowLeft, topRight=topRight, targetCords=targetCords), 
    offset=np.array([0, -1]))

lookR = lambda lowLeft, topRight, targetCords: DirectionWrap(
    ManhattanDistanceTarget(lowLeft=lowLeft, topRight=topRight, targetCords=targetCords), 
    offset=np.array([1, 0]))

lookL = lambda lowLeft, topRight, targetCords: DirectionWrap(
    ManhattanDistanceTarget(lowLeft=lowLeft, topRight=topRight, targetCords=targetCords), 
    offset=np.array([-1, 0]))'''



def makeEnv():
    
    def _init():
        heatMapTypes = []
        # Create your specific env
        env = MazeEnv(low, high, spawn, target_awards, target_coords, heatMapTypes=heatMapTypes, vision_range=1, use_ray_scans=False)
        return env
    return _init

def runEval(agent, device, low, high, spawn, target_awards, target_coords, obsShape, samplePhase):
    evalHeatMapTypes = []
    evalEnv = MazeEnv(low, high, spawn, target_awards, target_coords, 
                      heatMapTypes=evalHeatMapTypes, vision_range=1, use_ray_scans=False)
    resetOptions = {
        "randomSpawn": False,
        "randomSize": False,
        "randomTargetCoords": False
    }
    obs, info = evalEnv.reset(options=resetOptions)
    done = False
    totalReward = 0
    lstmState = None
    steps = 0
    coordLog = []

    with torch.no_grad():
            while not done:
                _, actionLogits, lstmState = agent.forward(obs[None, :], lstmState)
                action = torch.argmax(actionLogits, dim=-1).item()
                obs, reward, terminated, truncated, info = evalEnv.step(action)
                totalReward += reward
                steps += 1
                done = terminated or truncated
                coordLog.append((int(evalEnv.coords[0]), int(evalEnv.coords[1])))
    print(f"[Eval @ {samplePhase}] Score: {totalReward:.2f}, Steps: {steps}, Success: {terminated}")
    evalEnv.close()

    os.makedirs("eval_logs", exist_ok=True)
    with open("eval_logs/eval_results.txt", "a") as f:
        f.write(f"Update {samplePhase}: score={totalReward:.2f}, steps={steps}, success={terminated}\n")
        for i, (x, y) in enumerate(coordLog):
            f.write(f"  step {i+1}: ({x}, {y})\n")
        f.write("\n")


nEnvs = 22

env = gym.vector.SyncVectorEnv([makeEnv() for _ in range(nEnvs)])

obsShape = env.single_observation_space.shape[0]
actionShape = env.single_action_space.n

#Defining core constants
criticLr = 0.0003
actorLr = 0.0001
nUpdates = 5000
nStepsPerUpdate = 512
ppo_epochs = 4

clip_eps = 0.2
gamma = 0.99
lam = 0.95
beginEntropy = 0.15
endEntropy = 0.05
entropyBonus = beginEntropy

lstmState = None 

if saveWeights:

    agent = ActoCritic(obsShape, actionShape, device, criticLr, actorLr, nEnvs, 256, 2)

    # Vector-specific wrapper
    envWrapper = gym.wrappers.vector.RecordEpisodeStatistics(env, buffer_length=10000)
    # Clear returns log at start of each run
    os.makedirs("eval_logs", exist_ok=True)
    with open("eval_logs/returns.csv", "w") as f:
        f.write("update,return\n")
    criticLosses = []
    actorLosses = []
    entropies = []
    current_max_steps = 500
    min_steps = 200
    step_decay = 20
    env.set_attr("max_steps", current_max_steps)
    resetOptions = {
    "randomSpawn": False,
    "randomSize": False, 
    "randomTargetCoords": False
    }
    os.makedirs("eval_logs", exist_ok=True)
    with open("eval_logs/eval_results.txt", "w") as f:
        f.write("")
           
           
    
    for samplePhase in tqdm(range(nUpdates)):
        entropyBonus = max(endEntropy,beginEntropy - (samplePhase*2 / nUpdates) * (beginEntropy - endEntropy))
        lr = criticLr * (1 - samplePhase / nUpdates)
        lr = max(lr, 3e-5)  # floor at 10% of starting rate
        for param_group in agent.critic_optim.param_groups:
            param_group['lr'] = lr
        for param_group in agent.lstm_optim.param_groups:
            param_group['lr'] = lr
        actorLr_current = actorLr * (1 - samplePhase / nUpdates)
        actorLr_current = max(actorLr_current, 1e-5)
        for param_group in agent.actor_optim.param_groups:
            param_group['lr'] = actorLr_current
        if samplePhase % 100 == 0 and samplePhase>0 and current_max_steps > min_steps:
            current_max_steps -= step_decay

            env.set_attr("max_steps", current_max_steps)
            print(f"Tightening the clock! New Max Steps: {current_max_steps}")
        # anneal gamma if needed:gamma = 0.95 + 0.04 * (current_max_steps - min_steps) / (1000 - min_steps)
            
        epValuePreds = torch.zeros(nStepsPerUpdate, nEnvs, device=cpu_device)
        epRewards = torch.zeros(nStepsPerUpdate, nEnvs, device=cpu_device)
        epActionLogProbs = torch.zeros(nStepsPerUpdate, nEnvs, device=cpu_device)
        epActions = torch.zeros(nStepsPerUpdate, nEnvs, device=cpu_device, dtype=torch.long)
        masks = torch.zeros(nStepsPerUpdate, nEnvs, device=cpu_device)

        if(samplePhase==0):
            states,__ = envWrapper.reset(options=resetOptions)
            
        epEntropies = torch.zeros(nStepsPerUpdate, nEnvs, device=cpu_device)
        epStates = torch.zeros(nStepsPerUpdate, nEnvs, obsShape, device=cpu_device)
        initialLstmState = (lstmState[0].detach().clone(), lstmState[1].detach().clone()) if lstmState is not None else None
        
        agent.to(cpu_device) # !!!!!
        if lstmState is not None:
            lstmState = (lstmState[0].to(cpu_device), lstmState[1].to(cpu_device))

        epInfos = []

        for step in range(nStepsPerUpdate):
            actions, actionLogProbs, stateValuePreds, entropy, lstmState = agent.select_action(
            states, lstmState)
            epStates[step] = torch.as_tensor(states, dtype=torch.float32, device=cpu_device)
            epEntropies[step] = entropy
            epActions[step] = actions
            states, rewards, terminated, truncated, infos = envWrapper.step(actions.numpy())
            epInfos.append(infos)
            
            dones_list = [term or trunc for term, trunc in zip(terminated, truncated)]
            dones = torch.as_tensor(dones_list, dtype=torch.float32, device=cpu_device)
        
            if lstmState is not None:
                h, c = lstmState
                # Zero out hidden state for finished envs
                mask = (1 - dones).view(1, -1, 1)
                lstmState = (h * mask, c * mask)
            
            
            epValuePreds[step] = torch.squeeze(stateValuePreds)
            epRewards[step] = torch.as_tensor(rewards, dtype=torch.float32, device=cpu_device)
            epActionLogProbs[step] = actionLogProbs
            
            if lstmState is not None:
                lstmState = (lstmState[0].detach(), lstmState[1].detach())
            
            masks[step] = torch.as_tensor([not d for d in dones_list], dtype=torch.float32, device=cpu_device)

            
            # MAYBE DELETE TODO
        #epRewards = (epRewards - epRewards.mean()) / (epRewards.std() + 1e-8)
        
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

        for _ in range(ppo_epochs):
            critic_loss, actor_loss = agent.get_losses(
                epStates, epRewards, epActionLogProbs, epValuePreds, epEntropies,
                masks, gamma, lam, entropyBonus, device,
                epActions, clip_eps,
                initialLstmState
            )
            agent.update_parameters(critic_loss, actor_loss)

        
        # Log episodes that completed during this update's rollout
        with open("eval_logs/returns.csv", "a") as f:
            for step_info in epInfos:
                if step_info is not None and "_episode" in step_info:
                    episode_mask = step_info["_episode"]
                    episode_returns = step_info["episode"]["r"]
                    for i, finished in enumerate(episode_mask):
                        if finished:
                            f.write(f"{samplePhase},{episode_returns[i]}\n")

        if (samplePhase+1) % 100 == 0:
            agent.actor.eval()
            agent.critic.eval()
            agent.lstm.eval()
            runEval(agent, device, low, high, spawn, target_awards, target_coords, obsShape, samplePhase+1)
            agent.actor.train()
            agent.critic.train()
            agent.lstm.train()

        # log the losses and entropy
        criticLosses.append(critic_loss.detach().cpu().numpy())
        actorLosses.append(actor_loss.detach().cpu().numpy())
        entropies.append(entropy.detach().mean().cpu().numpy())

        if samplePhase==50 and useProfiler==True:
            break



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
        f"Training plots for {agent.__class__.__name__} in the LunarLander-v3 environment \n \
                (n_envs={nEnvs}, n_steps_per_update={nStepsPerUpdate})"
    )

    # episode return
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

    # entropy
    axs[1][0].set_title("Entropy")
    entropy_moving_average = (
        np.convolve(np.array(entropies), np.ones(rolling_length), mode="valid")
        / rolling_length
    )
    axs[1][0].plot(entropy_moving_average)
    axs[1][0].set_xlabel("Number of updates")


    # critic loss
    axs[0][1].set_title("Critic Loss")
    critic_losses_moving_average = (
        np.convolve(
            np.array(criticLosses).flatten(), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[0][1].plot(critic_losses_moving_average)
    axs[0][1].set_xlabel("Number of updates")


    # actor loss
    axs[1][1].set_title("Actor Loss")
    actor_losses_moving_average = (
        np.convolve(np.array(actorLosses).flatten(), np.ones(rolling_length), mode="valid")
        / rolling_length
    )
    axs[1][1].plot(actor_losses_moving_average)
    axs[1][1].set_xlabel("Number of updates")

    plt.tight_layout()
    #plt.show() uncomment later, won't work in Docker !!!!!
    plt.savefig("result1")




    if not os.path.exists("weights"):
        os.mkdir("weights")

    """ save network weights """
    torch.save(agent.actor.state_dict(), actor_weights_path)
    torch.save(agent.critic.state_dict(), critic_weights_path)
    torch.save(agent.lstm.state_dict(), lstm_weights_path)


""" load network weights """
if load_weights:
    agent = ActoCritic(obsShape, actionShape, device, criticLr, actorLr, n_envs=1)

    agent.actor.load_state_dict(torch.load(actor_weights_path))
    agent.critic.load_state_dict(torch.load(critic_weights_path))
    agent.lstm.load_state_dict(torch.load("weights/lstm_weights.h5"))
    agent.actor.eval()
    agent.critic.eval()
    agent.lstm.eval()




env.close()
