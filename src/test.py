import numpy as np
import Utilities.SpaceEnv as SEnv
from Utilities.HeatMap import ManhattanDistanceFromMiddle
from Utilities.HeatMap import DistanceTarget
from Utilities.HeatMap import ManhattanDistanceTarget
low = np.array([0, 0])
high = np.array([100, 100])

spawn = np.array([1, 1])
target_coords = np.array([[20, 8], [80, 30]])
target_awards = np.array([100, 50])

dummy_heatmap_ManhattanMid = ManhattanDistanceFromMiddle(lowLeft=low, topRight=high)

dummy_Heatmap_Target_Dist = DistanceTarget(lowLeft=low,topRight=high, targetCords=target_coords)

dummy_Heatmap_Target_Dist_Manhat = ManhattanDistanceTarget(lowLeft=low,topRight=high, targetCords=target_coords)


env = SEnv.MazeEnv(low, high, spawn, target_awards, target_coords, dummy_heatmap_ManhattanMid, dummy_Heatmap_Target_Dist, dummy_Heatmap_Target_Dist_Manhat)


env.visualize(mapInt=0)
env.visualize(mapInt=1)
env.visualize(mapInt=2)

