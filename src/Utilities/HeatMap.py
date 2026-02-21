import gymnasium as gym
import numpy as np 
from gym import spaces
import pygame 
import MazeCreater
from openpyxl import Workbook 
from typing import Protocol 
class HeatMappable(Protocol):
    def map(cordArr: NDArray[np.int_])->np.int:
        ...
class DistanceTarget():
    def __init__(self, targetCords:NDArray[np.int_]):
        self.targetCords = targetCords
        self.dimension,_ = targetCords.shape
        
    def euclidianMap(self, cordArr: NDArray[np.int_])->np.int:
        d,_ = cordArr.size
        if(not (d ==self.dimension)):
            raise Exception("Wrong array size")
        return np.linalg.norm(cordArr - self.targetCords)
    
    def manhattanMap(self, cordArr: NDArray[np.int_])->np.int:
        d,_ = cordArr.size
        if(not (d ==self.dimension)):
            raise Exception("Wrong array size")
        return np.sum(np.abs(cordArr - self.targetCords))
    
    def lInftyMap(self, cordArr: NDArray[np.int_])->np.int:
        d,_ = cordArr.size
        if(not (d ==self.dimension)):
            raise Exception("Wrong array size")
        return np.amax(np.abs(cordArr - self.targetCords))
    
    

class DistanceFromMiddle():
    def __init__(self, lowLeft: NDArray[np.int_], topRight:NDArray[np.int_]):
        if not(lowLeft.size==topRight.size ):
            raise Exception("Wrong size inputs")
        if (np.mina(lowLeft-topRight) < 0):
            raise Exception("Not proper corner points")
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.midPoint = (topRight-lowLeft)/2
        self.dimension,_ = lowLeft.size
    def euclidianMap(self, cordArr: NDArray[np.int_])->np.int:
        d,_ = cordArr.size
        if(not (d ==self.dimension)):
            raise Exception("Wrong array size")
        return np.linalg.norm(cordArr - self.midPoint)
    def manhattanMap(self, cordArr: NDArray[np.int_])->np.int:
        d,_ = cordArr.size
        if(not (d ==self.dimension)):
            raise Exception("Wrong array size")
        return np.sum(np.abs(cordArr - self.midPoint))
    
    def lInftyMap(self, cordArr: NDArray[np.int_])->np.int:
        d,_ = cordArr.size
        if(not (d ==self.dimension)):
            raise Exception("Wrong array size")
        return np.amax(np.abs(cordArr - self.midPoint))