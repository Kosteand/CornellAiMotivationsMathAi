import gymnasium as gym
import numpy as np 
from gym import spaces
import pygame 
import MazeCreater
from openpyxl import Workbook 
import matplotlib.pyplot as plt
import Utilities.HeatMap
maze_127 = MazeCreater.generate_sparse_maze(127, 127, density=0.2)
np.savetxt("output.csv", maze_127, delimiter=",", fmt='%d')


class MazeEnv(gym.Env):
    def __init__(self, spawn: npt.NDArray[np.int_], targets: npt.NDArray[np.int_], *args: Utilities.HeatMap.HeatMappable):
        super(MazeGameEnv, self).__init__()
        self.maze = np.array(maze)  # Maze is a 2D numpy array
        self.spawn_pos = np.where(self.maze == -2)  # Spawn position
        self.target_pos = np.where(self.maze == -1)  # Target position
        self.current_pos = self.spawn_pos #spawn position is current posiiton of agent
        self.num_rows, self.num_cols = self.maze.shape
        
        self.heatmaps = args
        
         
        self.observation_space = spaces.Dict({"positionX": spaces.Discrete(self.num_rows), "positionY": spaces.Discrete(self.num_cols),
                                               "targetX":spaces.Discrete(self.num_rows), "targetY":spaces.Discrete(self.num_cols),
                                               "distU":spaces.Discrete(self.num_rows), "distD":spaces.Discrete(self.num_rows),
                                               "distR":spaces.Discrete(self.num_cols), "distL":spaces.Discrete(self.num_rows),
                                               "extras":spaces.Box(low=-np.inf, high=np.inf,shape=(len(args),), dtype=np.float32)
                                               })
        
        
        # 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)  


        self.cell_size = 125
        
        


    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos
    
    def visualize(self, mapInt, ):
        fig, ax = plt.subplots()
        heatMapPlot = ax.imshow(data, cmap='viridis', interpolation='nearest')
        
        plt.colorbar(heatmap_plot, ax=ax, label='Value')

        ax.set_xlabel('X-axis Label')
        ax.set_ylabel('Y-axis Label')
        ax.set_title('My Matplotlib Heatmap')

        plt.savefig("a_figure_png")

    def step(self, action):
        # Move the agent based on the selected action
        new_pos = np.array(self.current_pos)
        if action == 0:  # Up
            new_pos[0] -= 1
        elif action == 1:  # Down
            new_pos[0] += 1
        elif action == 2:  # Left
            new_pos[1] -= 1
        elif action == 3:  # Right
            new_pos[1] += 1

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

        # If the agent hits an obstacle
        if self.maze[row, col] == 1:
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
    def getColorGrad(val, minVal, maxVal):
        adjustedVal = (val - minVal)/(maxVal - minVal)
        r = int(255*adjustedVal)
        g = 0
        b = int(255*(1-adjustedVal))
        return(r, g, b)
    