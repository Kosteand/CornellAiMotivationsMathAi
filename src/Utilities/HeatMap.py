import inspect

import numpy as np
import random
from numpy.typing import NDArray
from typing import Protocol
import numpy.typing as npt




def generate_heatmap_impl(obj, noise=0, blockSize=1): #current impl uses floats #higher block size should mean less salient data
  #current implementation is probably slow
  shape = [int(obj.topRight[i] - obj.lowLeft[i] + 1) for i in range(obj.dimension)]
  width=shape[0]
  height=shape[1]


  heatmap = np.zeros(shape, dtype=float)
  for idx in np.ndindex(*shape):
    coord = np.array([obj.lowLeft[i] + idx[i] for i in range(obj.dimension)], dtype=int)
    heatmap[idx] = obj.map(coord)

  heatmap_std = np.std(heatmap)

  #blocking
  if blockSize > 1:
    #0=bottom-left, 1=bottom-right, 2=top-left, 3=top-right
    corner = random.randint(0, 3)
    if corner == 0:
        x_step = 1
        y_step = 1
        x_start = 0
        y_start = 0
    elif corner == 1:
        x_step = -1
        y_step = 1
        x_start = width - 1
        y_start = 0
    elif corner == 2:
        x_step = 1
        y_step = -1
        x_start = 0
        y_start = height - 1
    else:
        x_step = -1
        y_step = -1
        x_start = width - 1
        y_start = height - 1

    blocked = np.zeros_like(heatmap)

    y = y_start
    while 0 <= y < height:
        x = x_start
        while 0 <= x < width:
            if x_step == 1:
                x_end = min(x + blockSize, width)
            else:
                x_end = x + 1
                x_start_block = max(x - blockSize + 1, 0)
            if y_step == 1:
                y_end = min(y + blockSize, height)
            else:
                y_end = y + 1
                y_start_block = max(y - blockSize + 1, 0)

            if x_step == 1:
                x_slice = slice(x, x_end)
            else:
                x_slice = slice(x_start_block, x_end) # type: ignore
            if y_step == 1:
                y_slice = slice(y, y_end)
            else:
                y_slice = slice(y_start_block, y_end) # type: ignore

            block_mean = np.mean(heatmap[x_slice, y_slice])+random.uniform(-noise*heatmap_std, noise*heatmap_std) #adds noise
            blocked[x_slice, y_slice] = block_mean

            if x_step == 1:
                x += blockSize
            else:
                x -= blockSize

        if y_step == 1:
            y += blockSize
        else:
            y -= blockSize

    heatmap = blocked
  else:
      #add noise
      for idx in np.ndindex(*shape):
        heatmap[idx]+=random.uniform(-noise*heatmap_std, noise*heatmap_std)
  return heatmap


class HeatMappable(Protocol):
    def map(self, cordArr: NDArray[np.int_]) -> np.int_:
        ...
    def getRange(self) -> tuple[float, float]:
        ...
    def generate_heatmap(self, noise=0, blockSize=1) -> np.ndarray:
        ...  

class DistanceTarget():
    def __init__(self, targetCords: NDArray[np.int_], lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        self.targetCords = targetCords
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.dimension = targetCords.shape[-1]

    def map(self, cordArr: NDArray[np.int_]) -> np.int_:
        min_dist = float('inf')
        x, y = int(cordArr[0]), int(cordArr[1])
        for tx, ty in self.targetCords:
            dx, dy = int(tx) - x, int(ty) - y
            d = (dx*dx + dy*dy) ** 0.5
            if d < min_dist:
                min_dist = d
        return min_dist

    def getRange(self) -> tuple[float, float]:
        corners = np.array(np.meshgrid(*zip(self.lowLeft, self.topRight))).T.reshape(-1, self.dimension)
        max_dists = [np.min(np.linalg.norm(self.targetCords - c, axis=1)) for c in corners]
        return (0.0, float(np.max(max_dists)))
    
    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)

class ManhattanDistanceTarget():
    def __init__(self, targetCords: NDArray[np.int_], lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        self.targetCords = targetCords
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.dimension = targetCords.shape[-1]

    def map(self, cordArr: NDArray[np.int_]) -> np.int_:
        min_dist = float('inf')
        x, y = int(cordArr[0]), int(cordArr[1])
        for tx, ty in self.targetCords:
            d = abs(int(tx) - x) + abs(int(ty) - y)
            if d < min_dist:
                min_dist = d
        return min_dist

    def getRange(self) -> tuple[float, float]:
        corners = np.array(np.meshgrid(*zip(self.lowLeft, self.topRight))).T.reshape(-1, self.dimension)
        max_dists = [np.min(np.sum(np.abs(self.targetCords - c), axis=1)) for c in corners]
        return (0.0, float(np.max(max_dists)))
    
    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)

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
    
    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)

class DistanceFromMiddle():
    def __init__(self,targetCords:NDArray[np.int_], lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
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
        diffs = self.midPoint - cordArr
        return float(np.sqrt((diffs**2).sum()))

    def getRange(self) -> tuple[float, float]:
        return (0.0, float(np.linalg.norm(self.topRight - self.midPoint)))
    
    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)

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
        return int(np.abs(self.midPoint - cordArr).sum())



    def getRange(self) -> tuple[float, float]:
        return (0.0, float(np.sum(np.abs(self.topRight - self.midPoint))))
    
    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)

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
    
    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)
    
class DirectionWrap(HeatMappable):
    def __init__(self, inner_map_type, offset, **kwargs):
         # Filter kwargs to only what inner_map_type accepts
        inner_sig = inspect.signature(inner_map_type.__init__)
        if any(p.kind == p.VAR_KEYWORD for p in inner_sig.parameters.values()):
            filtered = kwargs
        else:
            filtered = {k: v for k, v in kwargs.items() if k in inner_sig.parameters}
        self.inner = inner_map_type(**filtered)
        self.offset = np.array(offset)
    def map(self, coords):
        return self.inner.map(coords + self.offset)
    def getRange(self):
        return self.inner.getRange()
class NoiseWrap:
    def __init__(self, targetMap: HeatMappable, noiseLevel: float = 0.05):
        """
        target_map: The original HeatMap object.
        noise_level: The amount of noise to add.
        """
        self.target_map = targetMap
        self.noise_level = noiseLevel
        self.hash_table = dict()
        
    def map(self, cordArr: NDArray[np.int_]) -> float:
        # 1. Get the "clean" value from the original map
        clean_value = self.target_map.map(cordArr)
        key = tuple(cordArr)
        
        #check if value has alread been computed
        if key not in self.hash_table: 
            low, high = self.getRange()
            scale = (high - low) * self.noise_level
            noise = np.random.normal(0, scale)
            print("temp")
            self.hash_table[key]=clean_value+noise

        return self.hash_table[key]
        
    def getRange(self) -> tuple[float, float]:
        # Return the range of the underlying map
        return self.target_map.getRange()
    
