import inspect

import gymnasium as gym
import numpy as np 
from gymnasium import spaces
from openpyxl import Workbook 
from Utilities.HeatMap import DistanceTarget, HeatMappable, LInftyDistanceTarget, ManhattanDistanceTarget
import matplotlib.pyplot as plt
import numpy.typing as npt
from datetime import datetime
from functools import partial
from Utilities.generateWalls import generateWalls


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
        self.max_steps = 1000  # Default starting limit decays
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
        ##For when pure python is better than vectorized
        self.targetsTuple = set(map(tuple, self.targetCords.tolist()))
        self.walls = np.zeros((self.num_cols,self.num_rows))

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
        obs_size = 8 + len(heatMapTypes)  # 4 dists + 4 types + extras
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        


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
            if isinstance(mapType, partial):
                # Always pass everything to partials, DirectionWrap filters internally
                maps.append(mapType(**env_context))
            elif isinstance(mapType, type):
                sig = inspect.signature(mapType.__init__)
                has_var_keyword = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
                if has_var_keyword:
                    maps.append(mapType(**env_context))
                else:
                    kwargs = {k: v for k, v in env_context.items() if k in sig.parameters}
                    maps.append(mapType(**kwargs))
            else:
                sig = inspect.signature(mapType)
                kwargs = {k: v for k, v in env_context.items() if k in sig.parameters}
                maps.append(mapType(**kwargs))
        return maps
    
    def reset(self, seed=None, options=None):
        # Gymnasium reset must handle seeds and return (obs, info)
        opts = options if options is not None else {}
    
        # If new options provided, save them for future auto-resets
        if options is not None:
            self._reset_options = opts
        else:
            # Use saved options from last manual reset
            opts = getattr(self, '_reset_options', {})
        randomSize = opts.get("randomSize", False)
        randomSpawn = opts.get("randomSpawn", False)
        randomTargetCoords = opts.get("randomTargetCoords", False)
        self.max_steps = opts.get("max_steps", self.max_steps)
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
            #TO stop walls from gettign in the way until they are fully impl
            #self.walls = np.zeros((self.num_cols, self.num_rows), dtype=bool)

            self.walls = np.zeros((self.num_cols,self.num_rows))
            #self.walls = np.zeros(self.num_cols,self.num_rows)
        
        if(randomSpawn or randomSize):
            foundValidSpawn = False
            while not foundValidSpawn:
                random_x = self.np_random.integers(self.lowerLeft[0], self.upperRight[0])
                random_y = self.np_random.integers(self.lowerLeft[1], self.upperRight[1])
                if not self.walls[random_x - self.lowerLeft[0], random_y - self.lowerLeft[1]]:
                    foundValidSpawn=True
                    self.coords = np.array([random_x, random_y])
        else:
            self.coords = self.spawn.copy()
            
        if randomTargetCoords:
          occupied = set()
          
          # mark spawn as occupied so targets don't land on it
          occupied.add((int(self.coords[0]), int(self.coords[1])))
          
          new_targets = []
          for _ in range(len(self.targetCords)):
              while True:
                  tx = int(self.np_random.integers(self.lowerLeft[0], self.upperRight[0] + 1))
                  ty = int(self.np_random.integers(self.lowerLeft[1], self.upperRight[1] + 1))
                  xi = tx - self.lowerLeft[0]
                  yi = ty - self.lowerLeft[1]
                  if not self.walls[xi, yi] and (tx, ty) not in occupied:
                      occupied.add((tx, ty))
                      new_targets.append([tx, ty])
                      break
          self.targetCords = np.array(new_targets)
        self.targetsTuple = set(map(tuple, self.targetCords.tolist()))

        self.maps = self.buildMaps()
        observation = self.getObs()
        info = {}


        return observation, info
    
    def _scan_direction(self, dx, dy):
        """Returns (distance, type) where type: 0=boundary/wall, 1=target"""
        x, y = int(self.coords[0]), int(self.coords[1])
        ll0, ll1 = int(self.lowerLeft[0]), int(self.lowerLeft[1])
        ur0, ur1 = int(self.upperRight[0]), int(self.upperRight[1])
        dist = 0
                
        while True:
            x += dx
            y += dy
            dist += 1
            # Pure Python bounds check — no numpy
            if x < ll0 or x > ur0 or y < ll1 or y > ur1:
                return dist - 1, 0
            if self.walls[x - ll0, y - ll1]:
                return dist, 0
            # O(1) set lookup instead of O(n) array comparison
            if (x, y) in self.targetsTuple:
                return dist, 1
    
    def getObs(self):
        distU, typeU = self._scan_direction(0, 1)
        distD, typeD = self._scan_direction(0, -1)
        distR, typeR = self._scan_direction(1, 0)
        distL, typeL = self._scan_direction(-1, 0)
        extras = np.array([m.map(self.coords) for m in self.maps], dtype=np.float32)
        return np.array([
            distU/self.num_rows, distD/self.num_rows,
            distR/self.num_cols, distL/self.num_cols,
            typeU, typeD, typeR, typeL,
            *extras
        ], dtype=np.float32)
    
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
        
        # build a wall overlay array in (y, x) order to match imshow
        wall_overlay = np.zeros((len(y_range), len(x_range)))
        for idx_y, val_y in enumerate(y_range):
            for idx_x, val_x in enumerate(x_range):
                xi = val_x - self.lowerLeft[0]
                yi = val_y - self.lowerLeft[1]
                if self.walls[xi, yi]:
                    wall_overlay[idx_y, idx_x] = 1

        # mask so only wall cells are drawn, then overlay in solid grey
        wall_masked = np.ma.masked_where(wall_overlay == 0, wall_overlay)
        ax.imshow(wall_masked, extent=[self.lowerLeft[0], self.upperRight[0],
                  self.lowerLeft[1], self.upperRight[1]],
                  origin='lower', aspect='equal', cmap='gray', vmin=0, vmax=1, alpha=0.8)

        # plot targets
        for i, coord in enumerate(self.targetCords):
            ax.scatter(coord[0], coord[1], color='yellow', marker='X',
                      s=150, label=f'target {i}' if i == 0 else '', edgecolors='black')

        # Save and cleanup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ax.legend(loc='upper right')
        plt.savefig(f"./visualize/heatmap__{mapInt}_{timestamp}.png")        
        plt.close(fig)

    def step(self, action):
        self.current_step += 1
        truncated = False
        terminated = False
        
        if self.current_step >= self.max_steps:
            truncated = True
            reward = -0.01
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

        # 2. A target is hit only if BOTH x and y match (logical AND across axis 1)
        # hits will be a 1D boolean array of shape (N,)
        hits = np.where(
             np.all(self.targetCords == self.coords[None, :], axis=1)
        )[0]
        if len(hits) > 0:
            reward = self.targetAwards[hits[0]]
            terminated = True
        else:
            if not self._is_valid_position(self.coords):
                self.coords = oldLoc
                reward = -0.1
            else:
                reward = -0.1
              
        truncated  = False
        return self.getObs(), reward, terminated, truncated, {}
            
            

    def _is_valid_position(self, pos):
        x, y = int(pos[0]), int(pos[1])
        ll0, ll1 = int(self.lowerLeft[0]), int(self.lowerLeft[1])
        ur0, ur1 = int(self.upperRight[0]), int(self.upperRight[1])
        if x < ll0 or x > ur0 or y < ll1 or y > ur1:
            return False
        if self.walls[x - ll0, y - ll1]:
            return False
        return True