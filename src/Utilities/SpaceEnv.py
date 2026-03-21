import inspect

import gymnasium as gym
import numpy as np 
from gymnasium import spaces
from openpyxl import Workbook 
from Utilities.HeatMap import DistanceTarget, HeatMappable, LInftyDistanceTarget, ManhattanDistanceTarget
import matplotlib.pyplot as plt
import numpy.typing as npt
from datetime import datetime


class MazeEnv(gym.Env):
    #lowerLeft: The corner of the world that has the lowest possible coords.
    #uppeRight: Corner with highest possible coords
    #spawn: Where the agent begins
    #targetCords: List of coords of each target
    #targetAwards: List of the awards of each 
    #*args: List of heatMaps (probably from HeatMap.py)
    def __init__(self,lowerLeft: npt.NDArray[np.int_], upperRight: npt.NDArray[np.int_], 
                 spawn: npt.NDArray[np.int_],targetAwards: npt.NDArray[np.int_],
                 targetCords : npt.NDArray[np.int_], heatMapTypes:list,
                 walls: np.ndarray | None = None):
        super(MazeEnv, self).__init__()
        self.heatMapTypes = heatMapTypes
        self.max_steps = 1000  # Default starting limit
        self.current_step = 0
        self.lowerLeft = np.array(lowerLeft)
        self.upperRight = np.array(upperRight)
        self.targetCords = np.array(targetCords)
        self.targetAwards = np.array(targetAwards)
        self.spawn = np.array(spawn)
        self.coords = self.spawn.copy()
        self.num_cols = self.upperRight[0] - self.lowerLeft[0] + 1
        self.num_rows = self.upperRight[1] - self.lowerLeft[1] + 1
        self.maps = self.buildMaps()

        if not (self.lowerLeft.shape == (2,) and 
                self.upperRight.shape == (2,) and 
                self.spawn.shape == (2,) and 
                self.targetCords.shape[1:] == (2,) and 
                len(self.targetCords) == len(self.targetAwards)):
                    raise Exception("Not proper array dimensions")
        if np.any(lowerLeft >= upperRight):
            raise ValueError("lowerLeft must be strictly less than upperRight in all dimensions.")
        # Check if spawn is within bounds
        if np.any(self.spawn <= self.lowerLeft) or np.any(self.spawn > self.upperRight):
            raise ValueError("Spawn point is outside the environment boundaries.")

        # Check if all target coordinates are within bounds
        for coord in self.targetCords:
            if np.any(coord < self.lowerLeft) or np.any(coord > self.upperRight):
             raise ValueError(f"Target coordinate {coord} is outside the environment boundaries.")
            

        if walls is not None:
          if walls.shape != (self.num_cols, self.num_rows):
              raise ValueError(f"Walls array must have shape ({self.num_cols}, {self.num_rows})")
          # Indexing convention: walls[x - lowerLeft[0], y - lowerLeft[1]]
          if walls[self.spawn[0] - self.lowerLeft[0], self.spawn[1] - self.lowerLeft[1]]:
              raise ValueError("Spawn point cannot be on a wall")
          for coord in self.targetCords:
              if walls[coord[0] - self.lowerLeft[0], coord[1] - self.lowerLeft[1]]:
                  raise ValueError(f"Target {coord} cannot be on a wall")
          self.walls = walls
        else:
          self.walls = np.zeros((self.num_cols, self.num_rows), dtype=bool)
          
          
          
        
    
    #Size of distance obervation space is now fixed because of issues with using same weights for diff map sizes
    #TODO fix to a maxDim-shouldl be easy
        self.observation_space = spaces.Dict({"distU":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), "distD":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                               "distR":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), "distL":spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                                               "typeU":spaces.Discrete(3), "typeD":spaces.Discrete(3),
                                               "typeR":spaces.Discrete(3), "typeL":spaces.Discrete(3),
                                               "extras":spaces.Box(low=-np.inf, high=np.inf,shape=(len(heatMapTypes),), dtype=np.float32)
                                               })
        
        


        # 0=up, 1=down, 3=left, 2=right 
        self.action_space = spaces.Discrete(4); 
        


    def buildMaps(self):
        env_context = {
            'lowLeft': self.lowerLeft,
            'topRight': self.upperRight,
            'targetCords': self.targetCords,
            'spawn': self.spawn,
            'targetAwards':self.targetAwards
        }
        

        
        
        maps = []
        for mapType in self.heatMapTypes:
            if isinstance(mapType, type):
                innitSignature = inspect.signature(mapType.__init__)
            else:
                innitSignature = inspect.signature(mapType)
            kwargs = {k: v for k, v in env_context.items() if k in innitSignature.parameters.keys()}
            if any(p.kind == p.VAR_KEYWORD for p in innitSignature.parameters.values()):
                maps.append(mapType(**env_context))
            else:
                maps.append(mapType(**kwargs))
        return maps 
    
    def reset(self, seed=None, options=None):
        # Gymnasium reset must handle seeds and return (obs, info)
        opts = options if options is not None else {}
        randomSize = opts.get("randomSize", False)
        randomSpawn = opts.get("randomSpawn", False)
        randomTargetCoords = opts.get("randomTargetCoords", False)
        super().reset(seed=seed)
        self.current_step = 0
        if(randomSize):
            random_x = self.np_random.integers(-20, 20)
            random_y = self.np_random.integers(-20, 20)
        
            random_x_size = self.np_random.integers(10, 120)
            random_y_size = self.np_random.integers(10,120)
            
            self.lowerLeft[0] = random_x
            self.lowerLeft[1] = random_y
            self.upperRight = self.lowerLeft+[random_x_size,random_y_size]
            self.num_cols = self.upperRight[0] - self.lowerLeft[0] + 1
            self.num_rows = self.upperRight[1] - self.lowerLeft[1] + 1
            self.walls = np.zeros((self.num_cols, self.num_rows), dtype=bool)
        
        if(randomSpawn or randomSize):
            random_x = self.np_random.integers(self.lowerLeft[0], self.upperRight[0])
            random_y = self.np_random.integers(self.lowerLeft[1], self.upperRight[1])
            self.coords = np.array([random_x, random_y])
        else:
            self.coords = self.spawn.copy()
            
        if (randomTargetCoords):
            # Create a new array for the targets to avoid modifying the original list
            new_targets = []
            for _ in range(len(self.targetCords)):
                # Generate random X and Y within the current world bounds
                tx = self.np_random.integers(self.lowerLeft[0], self.upperRight[0] + 1)
                ty = self.np_random.integers(self.lowerLeft[1], self.upperRight[1] + 1)
                new_targets.append([tx, ty])
            self.targetCords = np.array(new_targets)
        # Simple way to ensure spawn isn't a target
        while any(np.array_equal(self.coords, t) for t in self.targetCords):
            self.coords = self.np_random.integers(self.lowerLeft, self.upperRight + 1, size=2)
        self.maps = self.buildMaps()
        observation = self.getObs()
        info = {}
        print(self.num_cols)
        print(self.num_rows)
        #TODO


        return observation, info
    
    def _scan_direction(self, dx, dy):
        """Returns (distance, type) where type: 0=boundary/wall, 1=target"""
        x, y = self.coords
        dist = 0
        while True:
            x += dx
            y += dy
            dist += 1
            if np.any([x, y] < self.lowerLeft) or np.any([x, y] > self.upperRight):
                return dist - 1, 0  # hit boundary
            xi = x - self.lowerLeft[0]
            yi = y - self.lowerLeft[1]
            if self.walls[xi, yi]:
                return dist, 0      # hit wall
            if any(np.array_equal([x, y], t) for t in self.targetCords):
                return dist, 1      # hit target
    
    def getObs(self):
        distU = float(self.upperRight[1] - self.coords[1]) / self.num_rows
        distD = float(self.coords[1] - self.lowerLeft[1]) / self.num_rows
        distR = float(self.upperRight[0] - self.coords[0]) / self.num_cols
        distL = float(self.coords[0] - self.lowerLeft[0]) / self.num_cols
        
        distU, typeU = self._scan_direction(0, 1)
        distD, typeD = self._scan_direction(0, -1)
        distR, typeR = self._scan_direction(1, 0)
        distL, typeL = self._scan_direction(-1, 0)
        
        return {
            "distU": np.array([distU / self.num_rows], dtype=np.float32), "distD": np.array([distD / self.num_rows], dtype=np.float32),
            "distR": np.array([distR / self.num_cols], dtype=np.float32), "distL": np.array([distL / self.num_cols], dtype=np.float32),
            "typeU": typeU, "typeD": typeD, "typeR": typeR, "typeL": typeL,
            "extras": np.array([m.map(self.coords) for m in self.maps], dtype=np.float32)
        }
    
    
    def visualize(self, mapInt):
        # 1. Access the specific heatmap
        target_map = self.maps[mapInt]

        # 2. Define the discrete ranges for X (dim 0) and Y (dim 1)
        # We use lowerLeft and upperRight which we've enforced are shape (2,)
        x_range = np.arange(self.lowerLeft[0], self.upperRight[0] + 1)
        y_range = np.arange(self.lowerLeft[1], self.upperRight[1] + 1)

        # 3. Initialize the Z-value grid (Y rows, X columns)
        zVal = np.zeros((len(y_range), len(x_range)))

        # 4. Fill the grid
        # We loop through Y first because that represents the rows in the image/array
        for idx_y, val_y in enumerate(y_range):
            for idx_x, val_x in enumerate(x_range):
                # Create the 2D coordinate for this specific cell
                current_coord = np.array([val_x, val_y])
                
                # Call the mapping function for this single point
                zVal[idx_y, idx_x] = target_map.map(current_coord)

        # 5. Plotting
        fig, ax = plt.subplots(figsize=(8, 6))
    
        # extent: maps the array indices to the actual coordinate values
        # [xmin, xmax, ymin, ymax]
        # origin='lower': puts (0,0) at the bottom left
        im = ax.imshow(zVal,
            extent=[self.lowerLeft[0] , self.upperRight[0] ,
            self.lowerLeft[1], self.upperRight[1] ],
            origin='lower', aspect='equal', cmap='viridis')
    
        plt.colorbar(im, ax=ax, label='Heatmap Value')
        ax.set_xlabel('X ')
        ax.set_ylabel('Y ')
        ax.set_title(f'Heatmap Visualization: Map {mapInt}')
        ax.set_xlim(self.lowerLeft[0], self.upperRight[0])
        ax.set_ylim(self.lowerLeft[1], self.upperRight[1])
        ax.scatter(self.coords[0], self.coords[1], 
           color='red', marker='*', s=200, label='agent', edgecolors='white')
        
        # !!!!! Ekadh add visualization heres

        # Save and cleanup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"./visualize/heatmap__{mapInt}_{timestamp}.png")        
        plt.close(fig)

    def step(self, action):
        self.current_step += 1
        truncated = False
        terminated = False
        
        if self.current_step >= self.max_steps:
            print("Truncated")
            truncated = True
            reward = -10
            return self.getObs(), reward, terminated, truncated, {}
            
            
        direction = action
        stepSize = 1
        if direction%2 == 0:
            change = stepSize
        else:
            change = -stepSize
        oldLoc = self.coords.copy()
        if direction//2 ==0 :
            self.coords = self.coords+[0,change]
        else:
            self.coords = self.coords+[change,0]
        matches = (self.targetCords == self.coords)

        # 2. A target is hit only if BOTH x and y match (logical AND across axis 1)
        # hits will be a 1D boolean array of shape (N,)
        hits = np.all(matches, axis=1)

        if np.any(hits):
            print("Target hit")
            # Get the index of the first target hit
            target_idx = np.where(hits)[0][0]
            reward = self.targetAwards[target_idx]
            terminated = True
        else:
            # Check if we hit a wall (already calculated) or just a normal step
            if not self._is_valid_position(self.coords):
                self.coords = oldLoc
                reward = -0.1
            else:
                reward = -0.01
                
        truncated  = False
        return self.getObs(), reward, terminated, truncated, {}
            
            

    def _is_valid_position(self, pos):
        row, col = pos
   
        # If agent goes out of the grid
        if np.any(pos < self.lowerLeft) or np.any(pos > self.upperRight):
            return False
        #check if the agent is going into a wall
        x_idx = pos[0] - self.lowerLeft[0]
        y_idx = pos[1] - self.lowerLeft[1]
        if self.walls[x_idx, y_idx]:
          return False
        return True
