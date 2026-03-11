import gymnasium as gym
import numpy as np 
from gymnasium import spaces
from openpyxl import Workbook 
from Utilities.HeatMap import HeatMappable
import matplotlib.pyplot as plt
import numpy.typing as npt


class MazeEnv(gym.Env):
    #lowerLeft: The corner of the world that has the lowest possible coords.
    #uppeRight: Corner with highest possible coords
    #spawn: Where the agent begins
    #targetCords: List of coords of each target
    #targetAwards: List of the awards of each 
    #*args: List of heatMaps (probably from HeatMap.py)
    def __init__(self,lowerLeft: npt.NDArray[np.int_], upperRight: npt.NDArray[np.int_], 
                 spawn: npt.NDArray[np.int_],targetAwards: npt.NDArray[np.int_],
                 targetCords : npt.NDArray[np.int_], *args: HeatMappable):
        super(MazeEnv, self).__init__()
        self.lowerLeft = lowerLeft
        self.upperRight = upperRight
        self.targetCords = targetCords
        self.targetAwards = targetAwards
        self.maps = np.array(args)
        self.spawn = spawn
        self.coords = spawn
        if not (self.lowerLeft.shape == (2,) and 
                self.upperRight.shape == (2,) and 
                self.spawn.shape == (2,) and 
                self.targetCords.shape[1:] == (2,) and 
                len(self.targetCords) == len(self.targetAwards)):
                    raise Exception("Not proper array dimensions")
        self.num_cols = self.upperRight[0] - self.lowerLeft[0] + 1
        self.num_rows = self.upperRight[1] - self.lowerLeft[1] + 1
        if np.any(lowerLeft >= upperRight):
            raise ValueError("lowerLeft must be strictly less than upperRight in all dimensions.")
        # Check if spawn is within bounds
        if np.any(self.spawn < self.lowerLeft) or np.any(self.spawn > self.upperRight):
            raise ValueError("Spawn point is outside the environment boundaries.")

        # Check if all target coordinates are within bounds
        for coord in self.targetCords:
            if np.any(coord < self.lowerLeft) or np.any(coord > self.upperRight):
             raise ValueError(f"Target coordinate {coord} is outside the environment boundaries.")
    
        self.observation_space = spaces.Dict({"distU":spaces.Discrete(self.num_rows), "distD":spaces.Discrete(self.num_rows),
                                               "distR":spaces.Discrete(self.num_cols), "distL":spaces.Discrete(self.num_rows),
                                               "typeU":spaces.Discrete(3), "typeD":spaces.Discrete(3),
                                               "typeR":spaces.Discrete(3), "typeL":spaces.Discrete(3),
                                               "extras":spaces.Box(low=-np.inf, high=np.inf,shape=(len(args),), dtype=np.int_)
                                               })
        
        


        # 0=up, 1=down, 3=left, 2=right 
        self.action_space = spaces.Dict({"direction":spaces.Discrete(4), "stepSize":spaces.Box(low = 0, high=100, shape=(1,),dtype = np.byte)} ); 
        


    
    def reset(self):
        self.coords = self.spawn
        return self.coords
    
    
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
                    extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], 
                    origin='lower', 
                    aspect='equal', 
                    cmap='viridis')
    
        plt.colorbar(im, ax=ax, label='Heatmap Value')
        ax.set_xlabel('X ')
        ax.set_ylabel('Y ')
        ax.set_title(f'Heatmap Visualization: Map {mapInt}')
        ax.scatter(self.coords[0], self.coords[1], 
           color='red', marker='*', s=200, label='agent', edgecolors='white')
        
        

        # Save and cleanup
        plt.savefig(f"heatmap_visual_{mapInt}.png")
        plt.close(fig)

    def step(self, action):
        if action%2 == 0:
            change = action.stepSize
        else:
            change = -action.stepSize
        oldLoc = self.coords
        if action/2 ==0 :
            self.coords = self.coords+[0,change]
        else:
            self.coords = self.coords+[change,0]
        if not self._is_valid_position(self, pos):
            self.coords = oldLoc
            return -5
        if (np.any(self.coords == self.targetCords)):
            self.reset
            return 100
        return -1
            
            

    def _is_valid_position(self, pos):
        row, col = pos
   
        # If agent goes out of the grid
        if np.any(pos < self.lowerLeft) or np.any(pos > self.upperRight)
            return False

        return True
