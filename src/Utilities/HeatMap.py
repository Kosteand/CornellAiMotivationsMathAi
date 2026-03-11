import numpy as np
import random
from numpy.typing import NDArray
from typing import Protocol

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
    def __init__(self,map:HeatMappable,cordArr: NDArray[np.int_]):
        self.map = map
        self.cordArr = cordArr
        
        
    def map(self, cordArr: NDArray[np.int_]) -> np.int_:
        return self.map.map(cordArr+self.cordArr)
        
    def getRange(self)-> tuple[float, float]:
        return self.map.range
    
class NoiseWrap():
    def __init__(self,map:HeatMappable,cordArr: NDArray[np.int_]):
        self.map = map
        min, max = map.getRange()
        
        
    def map(self, cordArr: NDArray[np.int_]) -> np.int_:
        return self.map.map(cordArr+self.cordArr)
        
    def getRange(self)-> tuple[float, float]:
        return self.map.range
    
