import numpy as np
import random
from numpy.typing import NDArray
from typing import Protocol
import numpy.typing as npt

class HeatMappable(Protocol):
    def map(self, cordArr: NDArray[np.int_]) -> np.int_:
        ...
    def getRange(self) -> tuple[float, float]:
        ...
    def generate_heatmap(obj, noise=0): #current impl uses floats
      #current implementation is probably slow
      shape = [int(obj.topRight[i] - obj.lowLeft[i] + 1) for i in range(obj.dimension)]
      heatmap = np.zeros(shape, dtype=float)
      for idx in np.ndindex(*shape):
        coord = np.array([obj.lowLeft[i] + idx[i] for i in range(obj.dimension)], dtype=int)
        heatmap[idx] = obj.map(coord)
      
      #add noise
      heatmap_average = np.mean(heatmap)
      heatmap_std = np.std(heatmap)
      for idx in np.ndindex(*shape):
        heatmap[idx]=random.uniform(-noise*heatmap_std, noise*heatmap_std)
      
        

class DistanceTarget():
    def __init__(self, targetCords: NDArray[np.int_], lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        self.targetCords = targetCords
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.dimension = targetCords.shape[-1]

    def map(self, cordArr: NDArray[np.int_]) -> np.int_:
        return np.min(np.linalg.norm(self.targetCords - cordArr, axis=1))

    def getRange(self) -> tuple[float, float]:
        corners = np.array(np.meshgrid(*zip(self.lowLeft, self.topRight))).T.reshape(-1, self.dimension)
        max_dists = [np.min(np.linalg.norm(self.targetCords - c, axis=1)) for c in corners]
        return (0.0, float(np.max(max_dists)))

class ManhattanDistanceTarget():
    def __init__(self, targetCords: NDArray[np.int_], lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        self.targetCords = targetCords
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.dimension = targetCords.shape[-1]

    def map(self, cordArr: NDArray[np.int_]) -> np.int_:
        return np.min(np.sum(np.abs(self.targetCords - cordArr), axis=1))

    def getRange(self) -> tuple[float, float]:
        corners = np.array(np.meshgrid(*zip(self.lowLeft, self.topRight))).T.reshape(-1, self.dimension)
        max_dists = [np.min(np.sum(np.abs(self.targetCords - c), axis=1)) for c in corners]
        return (0.0, float(np.max(max_dists)))

class LInftyDistanceTarget():
    def __init__(self, targetCords: NDArray[np.int_], lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        self.targetCords = targetCords
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.dimension = targetCords.shape[-1]

    def map(self, cordArr: NDArray[np.int_]) -> np.int_:
        return np.min(np.amax(np.abs(self.targetCords - cordArr), axis=1))

    def getRange(self) -> tuple[float, float]:
        corners = np.array(np.meshgrid(*zip(self.lowLeft, self.topRight))).T.reshape(-1, self.dimension)
        max_dists = [np.min(np.amax(np.abs(self.targetCords - c), axis=1)) for c in corners]
        return (0.0, float(np.max(max_dists)))

class DistanceFromMiddle():
    def __init__(self, lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        if not (lowLeft.size == topRight.size):
            raise Exception("Wrong size inputs")
        if np.any(lowLeft >= topRight):
            raise Exception("Not proper corner points")
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.midPoint = lowLeft + (topRight - lowLeft) / 2
        self.dimension = lowLeft.size

    def map(self, cordArr: NDArray[np.int_]) -> np.int_:
        if not (cordArr.size == self.dimension):
            raise Exception("Wrong array size")
        return np.linalg.norm(cordArr - self.midPoint)

    def getRange(self) -> tuple[float, float]:
        return (0.0, float(np.linalg.norm(self.topRight - self.midPoint)))

class ManhattanDistanceFromMiddle():
    def __init__(self, lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        if not (lowLeft.size == topRight.size):
            raise Exception("Wrong size inputs")
        if np.any(lowLeft >= topRight):
            raise Exception("Not proper corner points")
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.midPoint = lowLeft + (topRight - lowLeft) / 2
        self.dimension = lowLeft.size

    def map(self, cordArr: NDArray[np.int_]) -> np.int_:
        if not (cordArr.size == self.dimension):
            raise Exception("Wrong array size")
        return np.sum(np.abs(cordArr - self.midPoint))

    def getRange(self) -> tuple[float, float]:
        return (0.0, float(np.sum(np.abs(self.topRight - self.midPoint))))

class LInftyDistanceFromMiddle():
    def __init__(self, lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        if not (lowLeft.size == topRight.size):
            raise Exception("Wrong size inputs")
        if np.any(lowLeft >= topRight):
            raise Exception("Not proper corner points")
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.midPoint = lowLeft + (topRight - lowLeft) / 2
        self.dimension = lowLeft.size

    def map(self, cordArr: NDArray[np.int_]) -> np.int_:
        if not (cordArr.size == self.dimension):
            raise Exception("Wrong array size")
        return np.amax(np.abs(cordArr - self.midPoint))

    def getRange(self) -> tuple[float, float]:
        return (0.0, float(np.amax(np.abs(self.topRight - self.midPoint))))
    
class DirectionWrap():
    def __init__(self, heatmap: HeatMappable, offset: npt.NDArray[np.int_]):
        """
        heatmap: The original HeatMappable object (e.g., DistanceTarget)
        offset: A 2D vector like [0, 1] for Up, [0, -1] for Down, etc.
        """
        self.heatmap = heatmap
        self.offset = np.array(offset)
        
    def map(self, current_coords: npt.NDArray[np.int_]) -> np.float32:
        """
        Returns the heatmap value at the agent's position + the offset.
        Essentially 'looking' one step in a specific direction.
        """
        look_ahead_point = current_coords + self.offset
        return self.heatmap.map(look_ahead_point)
        
    def getRange(self) -> tuple[float, float]:
        return self.heatmap.range
class NoiseWrap:
    def __init__(self, targetMap: HeatMappable, noiseLevel: float = 0.05):
        """
        target_map: The original HeatMap object.
        noise_level: The standard deviation of the Gaussian noise to add.
        """
        self.target_map = targetMap
        self.noise_level = noiseLevel
        
    def map(self, cordArr: NDArray[np.int_]) -> float:
        # 1. Get the "clean" value from the original map
        clean_value = self.target_map.map(cordArr)
        
        # 2. Calculate the range to scale noise appropriately
        low, high = self.getRange()
        scale = (high - low) * self.noise_level
        
        # 3. Add Gaussian (Normal) noise
        noise = np.random.normal(0, scale)
        
        return clean_value + noise
        
    def getRange(self) -> tuple[float, float]:
        # Return the range of the underlying map
        return self.target_map.getRange()
    
