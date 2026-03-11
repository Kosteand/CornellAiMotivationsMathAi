import gymnasium as gym
from pyvirtualdisplay import Display
from gymnasium.wrappers import RecordVideo
from Utilities.SpaceEnv import *
from Utilities.HeatMap import *
import torch

# 1. Start the Virtual Display (The "Fake Monitor")

# 2. Check Hardware
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running RL Training on: {device.upper()}")

# 3. Create Environment with Rendering
low = np.array([0, 0])
high = np.array([100, 100])

spawn = np.array([1, 1])
target_coords = np.array([[20, 8], [80, 30]])
target_awards = np.array([100, 50])

dummy_heatmap_ManhattanMid = ManhattanDistanceFromMiddle(lowLeft=low, topRight=high)

dummy_Heatmap_Target_Dist = DistanceTarget(lowLeft=low,topRight=high, targetCords=target_coords)

dummy_Heatmap_Target_Dist_Manhat = ManhattanDistanceTarget(lowLeft=low,topRight=high, targetCords=target_coords)


env = MazeEnv(low, high, spawn, target_awards, target_coords, dummy_heatmap_ManhattanMid, dummy_Heatmap_Target_Dist, dummy_Heatmap_Target_Dist_Manhat)


# 4. Wrap to record videos into a 'videos' folder
obs, info = env.reset()
for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
