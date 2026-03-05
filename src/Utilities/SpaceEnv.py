import gymnasium as gym
import numpy as np 
from gym import spaces
import pygame 
import MazeCreater
from openpyxl import Workbook 
from Utilities.HeatMap import HeatMappable
maze_127 = MazeCreater.generate_sparse_maze(127, 127, density=0.2)
np.savetxt("output.csv", maze_127, delimiter=",", fmt='%d')


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
        super(MazeGameEnv, self).__init__()
        if not(lowerLeft.shape == upperRight.shape and upperRight.shape == spawn.shape 
               and spawn.shape == targetCoords.shape[1:] and targetCoords.shape[0] ==targetAwards[0]):
                        raise Exception("Not proper array dimensions")

        if np.any(lowerLeft >= upperRight):
            raise ValueError("lowerLeft must be strictly less than upperRight in all dimensions.")
        # Check if spawn is within bounds
        if np.any(self.spawn < self.lowerLeft) or np.any(self.spawn > self.upperRight):
            raise ValueError("Spawn point is outside the environment boundaries.")

        # Check if all target coordinates are within bounds
        for coord in self.targetCords:
            if np.any(coord < self.lowerLeft) or np.any(coord > self.upperRight):
             raise ValueError(f"Target coordinate {coord} is outside the environment boundaries.")
        self.lowerLeft = lowerLeft
        self.upperRight = upperRight
        self.targetCords = targetCords
        self.targetAwards = targetAwards
        self.maps = np.array(args)
        self.spawn = spawn
        self.cords = spawn
        self.observation_space = spaces.Dict({"positionX": spaces.Discrete(self.num_rows), "positionY": spaces.Discrete(self.num_cols),
                                               "distU":spaces.Discrete(self.num_rows), "distD":spaces.Discrete(self.num_rows),
                                               "distR":spaces.Discrete(self.num_cols), "distL":spaces.Discrete(self.num_rows),
                                               "typeU":spaces.Discrete(2), "typeD":spaces.Discrete(2),
                                               "typeR":spaces.Discrete(2), "typeL":spaces.Discrete(2),
                                               "extras":spaces.Box(low=-np.inf, high=np.inf,shape=(len(args),), dtype=np.float32)
                                               })


        # 0=up, 1=down, 2=left, 3=right for 2-d
        self.action_space = spaces.Discrete(2*len())  
        


    
    def reset(self):
        self.current_pos = self.spawn
        return self.current_pos
    
    
    def visualize(self, mapInt, coord1Int, coord2Int, fixedInts):
        target_map = self.maps[mapInt]

        ax1Range = np.arange(self.lowerLeft[coord1Int], self.upperRight[coord1Int] + 1)
        ax2Range = np.arange(self.lowerLeft[coord2Int], self.upperRight[coord2Int] + 1)

        zVal = np.zeros((len(axis2_range), len(axis1_range)))
    

        for idx2, val2 in enumerate(ax2Range):
            for idx1, val1 in enumerate(ax1Range):
                # Start with a base coordinate the spawn or lowerLeft and voerwrite from there
                current_coord = self.lowerLeft.copy()
            
                # Use the fixedInts provides in parameters
                for axis, val in fixedInts.items():
                    current_coord[axis] = val
            
                # These are the two dimensions to be displayed 
                current_coord[coord1Int] = val1
                current_coord[coord2Int] = val2
            
                # Call the actual mappable method to get the value
                zVal[idx2, idx1] = target_map.map(current_coord)

        #Plotting
        fig, ax = plt.subplots(figsize=(8, 6))
    
        # Gemini told me to put this here ngl, does some sort of aligning
        im = ax.imshow(zVal, 
                    extent=[ax1Range[0], ax1Range[-1], ax2Range[0], ax2Range[-1]], 
                    origin='lower', 
                    aspect='auto',
                    cmap='viridis')
    
        plt.colorbar(im, ax=ax, label='Value')
        ax.set_xlabel(f'Dimension {coord1Int}')
        ax.set_ylabel(f'Dimension {coord2Int}')
        ax.set_title(f'Heatmap {mapInt} Slice (Fixed dims: {fixedInts})')

        plt.savefig("a_figure.png")
        plt.close(fig) 


    def step(self, action):
        new_pos = self.current_pos.copy()
    
        # 1. Determine which dimension to move in
        # (e.g., if action is 0 or 1, dim is 0; if 2 or 3, dim is 1...)
        dim = action // 2
    
        # 2. Determine direction: Even actions decrease, Odd actions increase
        # (or vice-versa, depending on your preference)
        if action % 2 == 0:
            new_pos[dim] -= 1
        else:
            new_pos[dim] += 1
        
        # 3. Add Boundary Checking (Crucial for N-dims)
        if np.all(new_pos >= self.lowerLeft) and np.all(new_pos <= self.upperRight):
            self.current_pos = new_pos
            # Check if the new position is valid TODO
            if self._is_valid_position(new_pos):
                self.current_pos = new_pos

            # Reward function TODO
            if np.array_equal(self.current_pos, self.goal_pos):
                reward = 1.0
                done = True
            else:
                reward = 0.0
                done = False

            return self.current_pos, reward, done, {}

    def _is_valid_position(self, pos):
        row, col = pos
   
        # If agent goes out of the grid
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False

        return True
#TODO 
    def render(self):
        # Clear the screen
        self.screen.fill((255, 255, 255))  

        # Draw env elements one cell at a time
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size
            
                try:
                    print(np.array(self.current_pos)==np.array([row,col]).reshape(-1,1))
                except Exception as e:
                    print('Initial state')

                if self.maze[row, col] == '#':  # Obstacle
                    pygame.draw.rect(self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'S':  # Starting position
                    pygame.draw.rect(self.screen, (0, 255, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'G':  # Goal position
                    pygame.draw.rect(self.screen, (255, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))

                if np.array_equal(np.array(self.current_pos), np.array([row, col]).reshape(-1,1)):  # Agent position
                    pygame.draw.rect(self.screen, (0, 0, 255), (cell_left, cell_top, self.cell_size, self.cell_size))

        pygame.display.update()  # Update the display