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
from Utilities.MultiHeatMap import *
import cProfile
import pstats
import io
useProfiler = False
if(useProfiler):
    profiler = cProfile.Profile()
    profiler.enable()


def mapLabel(m) -> str:
    """Filesystem-safe label for a heatmap entry (class, partial, or instance).

    The entries in a mapsTypes list are uninstantiated (classes / functools.partial),
    so we can't call the instance-level toString(); derive a stable name instead.
    """
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
            parts.append(f"N:c{int(N)}")
        return "_".join(parts)
    if isinstance(m, type):
        ts = getattr(m, "toString", None)
        if ts is not None:
            try:
                return ts()
            except TypeError:
                pass
        return m.__name__
    ts = getattr(m, "toString", None)
    print("HEREHEREHERE::"+ts)
    if ts is not None:
        try:
            return ts()
        except Exception:
            pass
    return type(m).__name__

# Currently this file, was used to sanity check the model with PPO. It also ahs the skeleton
# For testing various noise levels. The ewnxt step is to integrate the different complexity heatmaps here
#Which should just be a few lines
# Noise std per sweep index, in normalized [0,1] heatmap units (because getObs now
# divides each channel by its range). The per-step gradient of a distance field is
# ~1/range ≈ 0.012 normalized, so NOISE_STEP=0.01 sweeps from clean (i=0) up to
# ~0.09 std at i=9 — roughly 7x the single-step signal: hard but the static global
# gradient still survives. Tune this one constant to make the task easier/harder.
NOISE_STEP = 0.01


def buildMapsTypes(i):
    """Full heatmap-observation list for noise experiment index i (noise = NOISE_STEP*i).

    Elements [0] and [1] are the two competing base proxies (L2 vs Manhattan,
    both noised); the rest are direction-wrapped "look" sensors fed to the agent.
    Noise is static (cached per cell in NoiseWrap) and held constant for the whole
    training run at this level — no warmup/curriculum.
    """
    noise = NOISE_STEP * i
    return [
        partial(NoiseWrap, inner_map_type=DistanceTarget, noiseLevel=noise),
        partial(NoiseWrap, inner_map_type=ManhattanDistanceTarget, noiseLevel=noise),
        partial(DirectionWrap, inner_map_type=partial(NoiseWrap, inner_map_type=DistanceTarget, noiseLevel=noise), offset=np.array([0,  1])),
        partial(DirectionWrap, inner_map_type=partial(NoiseWrap, inner_map_type=DistanceTarget, noiseLevel=noise), offset=np.array([0, -1])),
        partial(DirectionWrap, inner_map_type=partial(NoiseWrap, inner_map_type=ManhattanDistanceTarget, noiseLevel=noise), offset=np.array([0,  1])),
        partial(DirectionWrap, inner_map_type=partial(NoiseWrap, inner_map_type=ManhattanDistanceTarget, noiseLevel=noise), offset=np.array([0, -1])),
        partial(DirectionWrap, inner_map_type=partial(NoiseWrap, inner_map_type=DistanceTarget, noiseLevel=noise), offset=np.array([1,  0])),
        partial(DirectionWrap, inner_map_type=partial(NoiseWrap, inner_map_type=DistanceTarget, noiseLevel=noise), offset=np.array([-1, 0])),
        partial(DirectionWrap, inner_map_type=partial(NoiseWrap, inner_map_type=ManhattanDistanceTarget, noiseLevel=noise), offset=np.array([1,  0])),
        partial(DirectionWrap, inner_map_type=partial(NoiseWrap, inner_map_type=ManhattanDistanceTarget, noiseLevel=noise), offset=np.array([-1, 0])),
    ]

class ActoCritic(nn.Module):
    def __init__(
        self,
        n_features: np.int_,
        n_actions: np.int_,
        device: torch.device,
        critic_lr: np.float32,
        actor_lr: np.float32,
        n_envs: int
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

        #print("x going into network:", x.shape)          # currently (1, 22) — suspicious

        """
        Forward pass of the networks.

        Args:
            x: A batched vector of states.

        Returns:
            state_values: A tensor with the state values, with shape [n_envs,].
            action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
        """
        x = torch.Tensor(x).to(self.device)
        x = x.reshape(x.shape[0], -1)
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
        stateValues, action_logits = self.forward(x)
        action_pd = torch.distributions.Categorical(
            logits=action_logits
        )  # implicitly uses softmax
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()
        return (actions, action_log_probs, stateValues, entropy)

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
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)

        self.actor_optim.step()


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
target_awards = np.array([10, 5])


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
    if(len(mapsTypes)>1):
        label = mapLabel(mapsTypes[0]) + mapLabel(mapsTypes[1])
    else:
        label = mapLabel(mapsTypes[0])
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



    agent = ActoCritic(obsShape, actionShape, device, criticLr, actorLr, nEnvs)

    # Vector-specific wrapper
    envWrapper = gym.wrappers.vector.RecordEpisodeStatistics(env, buffer_length=10000)
    criticLosses = []
    actorLosses = []
    entropies = []
    current_max_steps = 250
    min_steps = 250
    step_decay = 15
    resetOptions = {
    "randomSpawn": True,
    "randomSize": True, 
    "randomTargetCoords": True,
    "max_steps": current_max_steps
}
        
        
    
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
            
        epEntropies = torch.zeros(nStepsPerUpdate, nEnvs, device=device)
        
        for step in range(nStepsPerUpdate):
            actions, actionLogProbs, stateValuePreds, entropy= agent.select_action(
            states)
            epEntropies[step] = entropy
            states, rewards, terminated, truncated, infos = envWrapper.step(
            actions.cpu().numpy())
            
            dones = torch.tensor(
            [term or trunc for term, trunc in zip(terminated, truncated)],
            dtype=torch.float32, device=device)
        
        
            
            epValuePreds[step] = torch.squeeze(stateValuePreds)
            epRewards[step] = torch.tensor(rewards, device=device)
            epActionLogProbs[step] = actionLogProbs
            
            
            
            masks[step] = torch.tensor([not (term or trunc) for term, trunc in zip(terminated, truncated)])

            
            # MAYBE DELETE TODO
        #epRewards = (epRewards - epRewards.mean()) / (epRewards.std() + 1e-8)

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
        f"Training plots for {agent.__class__.__name__} in the 2d space \n \
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
    plt.savefig("result1"+ mapLabel(mapsTypes[0]))##todo




    if not os.path.exists("weights"):
        os.mkdir("weights")
    actor_weights_path = "weights/actor_weights"+mapLabel(mapsTypes[0])+"UV"+".h5"
    critic_weights_path = "weights/critic_weights"+mapLabel(mapsTypes[0])+"UV"+".h5"
    """ save network weights """
    torch.save(agent.actor.state_dict(), actor_weights_path)
    torch.save(agent.critic.state_dict(), critic_weights_path)
    
    env.close()

def testValidity(mapsTypes):
    """ load network weights """
    
    nEnvs = 22

    env = gym.vector.SyncVectorEnv([makeEnv(mapsTypes) for _ in range(nEnvs)])

    obsShape = env.single_observation_space.shape[0]
    actionShape = env.single_action_space.n

    #Defining core constants
    criticLr = 0.0001
    actorLr = 0.00005


    
    agent = ActoCritic(obsShape, actionShape, device, criticLr, actorLr, n_envs=1)
    
    actor_weights_path = "weights/actor_weights"+mapLabel(mapsTypes[0])+"UV"+".h5"
    critic_weights_path = "weights/critic_weights"+mapLabel(mapsTypes[0])+"UV"+".h5"

    agent.actor.load_state_dict(torch.load(actor_weights_path))
    agent.critic.load_state_dict(torch.load(critic_weights_path))
    agent.actor.eval()
    agent.critic.eval()
    agent.critic.eval()
    agent.actor.eval()        # Create your specific env
            
    low = np.array([0, 0])
    high = np.array([100, 100])

    spawn = np.array([5, 5])
    target_coords = np.array([[55, 43], [80,20]])
    target_awards = np.array([10, 5])
    evalEnv = MazeEnv(low, high, spawn, target_awards, target_coords, 
                    mapsTypes)
    resetOptions = {
        "randomSpawn": True,
        "randomSize": True, 
        "randomTargetCoords": True,
        "max_steps": 1000
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
            print(evalEnv.unwrapped.coords)
            totalReward += reward
            done = terminated or truncated
            if(done):
                if(terminated):
                    print(mapLabel(mapsTypes[0])+"pass")
                    os.rename("weights/actor_weights"+mapLabel(mapsTypes[0])+"UV"+".h5","weights/actor_weights"+mapLabel(mapsTypes[0])+"pass"+".h5")
                    os.rename("weights/critic_weights"+mapLabel(mapsTypes[0])+"UV"+".h5","weights/critic_weights"+mapLabel(mapsTypes[0])+"pass"+".h5")
                else:
                    print(mapLabel(mapsTypes[0])+mapLabel(mapsTypes[0])+"fail")
                    print(f"Tareget was at{evalEnv.targetCords[0]}{evalEnv.targetCords[1]}")
                    os.rename("weights/critic_weights"+mapLabel(mapsTypes[0])+"UV"+".h5","weights/critic_weights"+mapLabel(mapsTypes[0])+"fail"+".h5")
                    os.rename("weights/actor_weights"+mapLabel(mapsTypes[0])+"UV"+".h5","weights/actor_weights"+mapLabel(mapsTypes[0])+"fail"+".h5")


    print(f"Final Score: {totalReward}")
    evalEnv.close()
    
    
    
    
def testBinaryMapStrength(mapsTypes):
    
    criticLr = 0.0001
    actorLr = 0.00005
    low = np.array([-50, 0])
    high = np.array([50,50])
    num_cols = high[0] - low[0] + 1
    num_rows = high[1] - low[1] + 1

    spawn = np.array([0, 25])
    target_coords = np.array([[10, 14]])
    target_awards = np.array([10])
    evalHeatMapTypes = [
                xDescending,
                xAscending,
                partial(DirectionWrap, inner_map_type= xAscending, offset=np.array([0,  1])),
                partial(DirectionWrap, inner_map_type= xAscending, offset=np.array([0, -1])),
                partial(DirectionWrap, inner_map_type=yAscending,          offset=np.array([1,  0])),
                partial(DirectionWrap, inner_map_type=yAscending,          offset=np.array([-1, 0])),
    ]
    evalEnv = MazeEnv(low, high, spawn, target_awards, target_coords, 
                    heatMapTypes=evalHeatMapTypes)

    obsShape = evalEnv.observation_space.shape[0]
    actionShape = evalEnv.action_space.n


    agent = ActoCritic(obsShape, actionShape, device, criticLr, actorLr, n_envs=1)
    actor_weights_path = "weights/actor_weights"+mapLabel(mapsTypes[0])+"UV"+".h5"
    critic_weights_path = "weights/critic_weights"+mapLabel(mapsTypes[0])+"UV"+".h5"

    if not(os.path.isfile(actor_weights_path)):
        print("EIther agent not failed or not validated")
    else:
        agent.actor.load_state_dict(torch.load(actor_weights_path))
        agent.critic.load_state_dict(torch.load(critic_weights_path))
        agent.actor.eval()
        agent.critic.eval()
        agent.critic.eval()
        agent.actor.eval()        



        resetOptions = {
            "randomSpawn":False,
            "randomSize": False, 
            "randomTargetCoords": False,
            "max_steps": 1000
        }
        obs, info = evalEnv.reset(options=resetOptions)
        done = False
        totalReward = 0

        coords = 0
        print("Running tester")
        with torch.no_grad(): # No training pytorch stuff
            for i in range(20):
                # 1. Get the action (add a batch dimension with [None, :] for the network)
                _, actionLogits= agent.forward(obs[None, :])

                # 2. Pick the BEST action (Argmax)
                action = torch.argmax(actionLogits, dim=-1).item()

                # 3. Step the environment
                obs, reward, terminated, truncated, info = evalEnv.step(action)
                evalEnv.unwrapped.visualize(0)
                evalEnv.unwrapped.visualize(1)
                coords = evalEnv.unwrapped.coords[0]
                totalReward += reward

        if(coords<0):
            print(f"{mapLabel(mapsTypes[0])} is stronger than {mapLabel(mapsTypes[1])} by {coords*-1}")
        else:
            print(f"{mapLabel(mapsTypes[0])} is stronger than {mapLabel(mapsTypes[1])} by {coords}")
        evalEnv.close()

# --- Sanity check: train ONCE on the new OptimalActionTarget heatmap, NO noise ---
# Channels [0]/[1] are the two single-target-locked variants (needed for weight
# naming); [2] is the nearest-target version. The agent is handed the optimal
# action directly, so it should learn fast — this just confirms the new map and
# TargetWrap integrate end-to-end.
'''sanityMaps = [
    partial(TargetWrap, inner_map_type=OptimalActionTarget, targetIndex=0),
    partial(TargetWrap, inner_map_type=OptimalActionTarget, targetIndex=1),
    OptimalActionTarget,
]'''
polyMap = [partial(MultiPolynomialInverse, map=OptimalActionTarget, N=1)]
polyMap2 = [partial(MultiPolynomialInverse, map=OptimalActionTarget, N=2)]

polyMap3 = [partial(MultiPolynomialInverse, map=OptimalActionTarget, N=3)]

polyMap4 = [partial(MultiPolynomialInverse, map=OptimalActionTarget, N=4)]


trainAgent(polyMap)
trainAgent(polyMap2)

trainAgent(polyMap3)

trainAgent(polyMap4)


print("trained")
testValidity(polyMap)
# --- Original noise sweep (re-enable when the sanity run looks good) ---
# for i in range(10):
#     trainAgent(buildMapsTypes(i))
# for i in range(10):
#     testValidity(buildMapsTypes(i))
# for i in range(10):
#     testBinaryMapStrength(buildMapsTypes(i))




'''[
            DistanceTarget,
            ManhattanDistanceTarget,
            partial(DirectionWrap, inner_map_type=ManhattanDistanceTarget, offset=np.array([0,  1])),
            partial(DirectionWrap, inner_map_type=ManhattanDistanceTarget, offset=np.array([0, -1])),
            partial(DirectionWrap, inner_map_type=DistanceTarget,          offset=np.array([1,  0])),
            partial(DirectionWrap, inner_map_type=DistanceTarget,          offset=np.array([-1, 0])),
]'''
