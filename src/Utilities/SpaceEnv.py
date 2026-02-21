import gymnasium as gym
import numpy as np 
from gym import spaces
import pygame 
import MazeCreater
from openpyxl import Workbook 
class MazeEnv(gym.Env):
    def __init__(self):
        self.maze = ""  #TODO 