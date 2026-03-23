import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import os  

from Utilities.SpaceEnv import *
from Utilities.HeatMap import *
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from functools import partial

import cProfile
import pstats
import io
useProfiler = False
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
    ) -> None:
        
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        

        critic_layers = [
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # estimate V(s)
        ]

        actor_layers = [
            nn.Linear(n_features, 128),
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
        
    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the networks.

        Args:
            x: A batched vector of states.

        Returns:
            state_values: A tensor with the state values, with shape [n_envs,].
            action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
        """
        x = torch.Tensor(x).to(self.device)
        state_values = self.critic(x)  # shape: [n_envs,]
        action_logits_vec = self.actor(x)  # shape: [n_envs, n_actions]
        return (state_values, action_logits_vec)

    def select_action(
        self, x: np.ndarray
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
        state_values, action_logits = self.forward(x)
        action_pd = torch.distributions.Categorical(
            logits=action_logits
        )  # implicitly uses softmax
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()
        return (actions, action_log_probs, state_values, entropy)

    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lam: float,
        ent_coef: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device=device)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        returns = advantages + value_preds
        critic_loss = (returns.detach() - value_preds).pow(2).mean()
        
        # calculate the loss of the minibatch for actor and critic
        #Prevents exploding gradient here
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # give a bonus for higher entropy to encourage exploration
        actor_loss = (
            -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
        )
        return (critic_loss, actor_loss)

    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        """
        Updates the parameters of the actor and critic networks.

        Args:
            critic_loss: The critic loss.
            actor_loss: The actor loss.
        """
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)

        self.actor_optim.step()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running RL Training on: {device.upper()}")

saveWeights = True
load_weights = False 

actor_weights_path = "weights/actor_weights.h5"
critic_weights_path = "weights/critic_weights.h5"

 # 3. Create Environment with Rendering
low = np.array([0, 0])
high = np.array([100, 100])
num_cols = high[0] - low[0] + 1
num_rows = high[1] - low[1] + 1
walls = np.zeros((num_cols, num_rows), dtype=bool)

spawn = np.array([5, 5])
target_coords = np.array([[35, 40], [70,20]])
target_awards = np.array([10, 5])

dummy_heatmap_ManhattanMid = ManhattanDistanceFromMiddle(lowLeft=low, topRight=high)

dummy_Heatmap_Target_Dist = DistanceTarget(lowLeft=low,topRight=high, targetCords=target_coords)

dummy_Heatmap_Target_Dist_Manhat = ManhattanDistanceTarget(lowLeft=low,topRight=high, targetCords=target_coords)
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
        heatMapTypes = [
            ManhattanDistanceTarget,
            DistanceTarget,
            partial(DirectionWrap, inner_map_type=ManhattanDistanceTarget, offset=np.array([0,  1])),
            partial(DirectionWrap, inner_map_type=ManhattanDistanceTarget, offset=np.array([0, -1])),
            partial(DirectionWrap, inner_map_type=DistanceTarget,          offset=np.array([1,  0])),
            partial(DirectionWrap, inner_map_type=DistanceTarget,          offset=np.array([-1, 0])),
]
        # Create your specific env
        env = MazeEnv(low, high, spawn, target_awards, target_coords, heatMapTypes=heatMapTypes, walls=walls)
        return env
    return _init

nEnvs = 22

env = gym.vector.SyncVectorEnv([makeEnv() for _ in range(nEnvs)])

obsShape = env.single_observation_space.shape[0]
actionShape = env.single_action_space.n

#Defining core constants
criticLr = 0.0001
actorLr = 0.00005
nUpdates = 4000
nStepsPerUpdate = 256

gamma = 0.99
lam = 0.95
beginEntropy = 0.15
endEntropy = 0.01
entropyBonus = beginEntropy

if saveWeights:

    agent = ActoCritic(obsShape, actionShape, device, criticLr, actorLr, nEnvs)

    # Vector-specific wrapper
    envWrapper = gym.wrappers.vector.RecordEpisodeStatistics(env, buffer_length=10000)
    criticLosses = []
    actorLosses = []
    entropies = []
    current_max_steps = 1000
    min_steps = 250
    step_decay = 15
    resetOptions = {
    "randomSpawn": True,
    "randomSize": True, 
    "randomTargetCoords": True,
    "max_steps": current_max_steps
}
           
           
    
    for samplePhase in tqdm(range(nUpdates)):
        entropyBonus = max(0.05,beginEntropy - (samplePhase*2 / nUpdates) * (beginEntropy - endEntropy))
        if samplePhase % 100 == 0 and current_max_steps > min_steps:
            current_max_steps -= step_decay
            # New step val in all 4 envs
            print(f"Tightening the clock! New Max Steps: {current_max_steps}")
        # anneal gamma if needed:gamma = 0.95 + 0.04 * (current_max_steps - min_steps) / (1000 - min_steps)
            
        epValuePreds = torch.zeros(nStepsPerUpdate, nEnvs, device=device)
        epRewards = torch.zeros(nStepsPerUpdate, nEnvs, device=device)
        epActionLogProbs = torch.zeros(nStepsPerUpdate, nEnvs, device=device)
        masks = torch.zeros(nStepsPerUpdate, nEnvs, device=device)

        if(samplePhase==0):
            states,__ = envWrapper.reset(options=resetOptions)
            
        epEntropies = torch.zeros(nStepsPerUpdate, nEnvs, device=device)
        
        
        for step in range(nStepsPerUpdate):
            actions, actionLogProbs, stateValuePreds, entropy = agent.select_action(
            states)
            epEntropies[step] = entropy
            states, rewards, terminated, truncated, infos = envWrapper.step(
            actions.cpu().numpy())
            
            
            epValuePreds[step] = torch.squeeze(stateValuePreds)
            epRewards[step] = torch.tensor(rewards, device=device)
            epActionLogProbs[step] = actionLogProbs
            
            masks[step] = torch.tensor([not term for term in terminated])
            
            # MAYBE DELETE TODO
        epRewards = (epRewards - epRewards.mean()) / (epRewards.std() + 1e-8)

            # calculate the losses for actor and critic
        critic_loss, actor_loss = agent.get_losses(
            epRewards,
            epActionLogProbs,
            epValuePreds,
            epEntropies,
            masks,
            gamma,
            lam,
            entropyBonus,
            device,
        )

        # update the actor and critic networks
        agent.update_parameters(critic_loss, actor_loss)

        # log the losses and entropy
        criticLosses.append(critic_loss.detach().cpu().numpy())
        actorLosses.append(actor_loss.detach().cpu().numpy())
        entropies.append(entropy.detach().mean().cpu().numpy())


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
    plt.show()
    plt.savefig("result1")




    if not os.path.exists("weights"):
        os.mkdir("weights")

    """ save network weights """
    torch.save(agent.actor.state_dict(), actor_weights_path)
    torch.save(agent.critic.state_dict(), critic_weights_path)


""" load network weights """
if load_weights:
    agent = ActoCritic(obsShape, actionShape, device, criticLr, actorLr, n_envs=1)

    agent.actor.load_state_dict(torch.load(actor_weights_path))
    agent.critic.load_state_dict(torch.load(critic_weights_path))
    agent.actor.eval()
    agent.critic.eval()
agent.critic.eval()
agent.actor.eval()        # Create your specific env
        
low = np.array([0, 0])
high = np.array([100, 100])

spawn = np.array([5, 5])
target_coords = np.array([[35, 40], [70,20]])
target_awards = np.array([10, 5])
evalHeatMapTypes = [
            ManhattanDistanceTarget,
            DistanceTarget,
            partial(DirectionWrap, inner_map_type=ManhattanDistanceTarget, offset=np.array([0,  1])),
            partial(DirectionWrap, inner_map_type=ManhattanDistanceTarget, offset=np.array([0, -1])),
            partial(DirectionWrap, inner_map_type=DistanceTarget,          offset=np.array([1,  0])),
            partial(DirectionWrap, inner_map_type=DistanceTarget,          offset=np.array([-1, 0])),
]
evalEnv = MazeEnv(low, high, spawn, target_awards, target_coords, 
                   heatMapTypes=evalHeatMapTypes)
resetOptions = {
    "randomSpawn": True,
    "randomSize": True, 
    "randomTargetCoords": True,
    "max_steps": current_max_steps
}
obs, info = evalEnv.reset(options=resetOptions)
done = False
totalReward = 0

with torch.no_grad(): # No training pytorch stuff
    while not done:
        # 1. Get the action (add a batch dimension with [None, :] for the network)
        _, actionLogits = agent.forward(obs[None, :])
        
        # 2. Pick the BEST action (Argmax)
        action = torch.argmax(actionLogits, dim=-1).item()
        
        # 3. Step the environment
        obs, reward, terminated, truncated, info = evalEnv.step(action)
        #evalEnv.unwrapped.visualize(0)
        print(evalEnv.unwrapped.coords)
        totalReward += reward
        done = terminated or truncated

print(f"Final Score: {totalReward}")
evalEnv.close()

env.close()
