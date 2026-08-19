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

from train import ActoCritic



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running RL Training on: {device.upper()}")

saveWeights = True
load_weights = False

actor_weights_path = "weights/actor_weights.h5"
critic_weights_path = "weights/critic_weights.h5"

 # 3. Create Environment with Rendering
low = np.array([-50, 0])
high = np.array([50,15])
num_cols = high[0] - low[0] + 1
num_rows = high[1] - low[1] + 1
walls = np.zeros((num_cols, num_rows), dtype=bool)

spawn = np.array([0, 5])
target_coords = np.array([[10, 14]])
target_awards = np.array([10])

dummy_heatmap_ManhattanMid = ManhattanDistanceFromMiddle(lowLeft=low, topRight=high)

dummy_Heatmap_Target_Dist = DistanceTarget(lowLeft=low,topRight=high, targetCords=target_coords)

dummy_Heatmap_Target_Dist_Manhat = ManhattanDistanceTarget(lowLeft=low,topRight=high, targetCords=target_coords)

def makeEnv():

    def _init():
        heatMapTypes = [
            xAscending,
            xDescending,
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

criticLr = 0.0001
actorLr = 0.00005


low = np.array([-50, 0])
high = np.array([50,50])
num_cols = high[0] - low[0] + 1
num_rows = high[1] - low[1] + 1
walls = np.zeros((num_cols, num_rows), dtype=bool)

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


print("Running tester")
with torch.no_grad(): # No training pytorch stuff
    while not done:
        # 1. Get the action (add a batch dimension with [None, :] for the network)
        _, actionLogits= agent.forward(obs[None, :])

        # 2. Pick the BEST action (Argmax)
        action = torch.argmax(actionLogits, dim=-1).item()

        # 3. Step the environment
        obs, reward, terminated, truncated, info = evalEnv.step(action)
        evalEnv.unwrapped.visualize(0)
        evalEnv.unwrapped.visualize(1)
        print(evalEnv.unwrapped.coords)
        totalReward += reward
        done = terminated or truncated

print(f"Final Score: {totalReward}")
evalEnv.close()