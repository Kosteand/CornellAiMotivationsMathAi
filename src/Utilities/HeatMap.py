import numpy as np
from numpy.typing import NDArray
from typing import Protocol

class HeatMappable(Protocol):
    def map(self, cordArr: NDArray[np.int_]) -> np.int_:
        ...
    def getRange(self) -> tuple[float, float]:
        ...

class DistanceTarget():
    def __init__(self, targetCords: NDArray[np.int_], lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        self.targetCords = targetCords
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.dimension = targetCords.size

    def map(self, cordArr: NDArray[np.int_]) -> np.int_:
        if not (cordArr.size == self.dimension):
            raise Exception("Wrong array size")
        return np.linalg.norm(cordArr - self.targetCords)

    def getRange(self) -> tuple[float, float]:
        corners = np.array(np.meshgrid(*zip(self.lowLeft, self.topRight))).T.reshape(-1, self.dimension)
        distances = [np.linalg.norm(c - self.targetCords) for c in corners]
        return (0.0, float(np.max(distances)))

class ManhattanDistanceTarget():
    def __init__(self, targetCords: NDArray[np.int_], lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        self.targetCords = targetCords
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.dimension = targetCords.size

    def map(self, cordArr: NDArray[np.int_]) -> np.int_:
        if not (cordArr.size == self.dimension):
            raise Exception("Wrong array size")
        return np.sum(np.abs(cordArr - self.targetCords))

    def getRange(self) -> tuple[float, float]:
        max_dist = np.sum(np.maximum(np.abs(self.topRight - self.targetCords), np.abs(self.lowLeft - self.targetCords)))
        return (0.0, float(max_dist))

class LInftyDistanceTarget():
    def __init__(self, targetCords: NDArray[np.int_], lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        self.targetCords = targetCords
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.dimension = targetCords.size

    def map(self, cordArr: NDArray[np.int_]) -> np.int_:
        if not (cordArr.size == self.dimension):
            raise Exception("Wrong array size")
        return np.amax(np.abs(cordArr - self.targetCords))

    def getRange(self) -> tuple[float, float]:
        max_dist = np.amax(np.maximum(np.abs(self.topRight - self.targetCords), np.abs(self.lowLeft - self.targetCords)))
        return (0.0, float(max_dist))

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