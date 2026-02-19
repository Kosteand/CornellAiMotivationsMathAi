import numpy as np
import random
from openpyxl import Workbook
import pandas as pd
def generate_maze(width=127, height=127, iterations=5, diagonal_density=0):
    maze = np.full((height, width), 1, dtype=object)
    rooms = []

    def partition(y, x, w, h, depth):
        if depth == 0 or w < 20 or h < 20:
            max_rw, max_rh = w - 4, h - 4
            if max_rw >= 5 and max_rh >= 5:
                rw, rh = random.randint(5, max_rw), random.randint(5, max_rh)
                rx, ry = x + random.randint(2, w - rw - 2), y + random.randint(2, h - rh - 2)
                maze[ry:ry+rh, rx:rx+rw] = 0
                rooms.append({'center': (rx + rw // 2, ry + rh // 2), 'id': len(rooms)})
            return

        if w > h:
            split = random.randint(int(w * 0.4), int(w * 0.6))
            partition(y, x, split, h, depth - 1)
            partition(y, x + split, w - split, h, depth - 1)
        else:
            split = random.randint(int(h * 0.4), int(h * 0.6))
            partition(y, x, w, split, depth - 1)
            partition(y + split, x, w, h - split, depth - 1)

    partition(1, 1, width - 2, height - 2, iterations)

    def draw_fat_line(p1, p2):
        x1, y1 = p1; x2, y2 = p2
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        sx, sy = (1 if x1 < x2 else -1), (1 if y1 < y2 else -1)
        err = dx - dy
        while True:
            maze[max(1, y1-1):min(height-1, y1+3), max(1, x1-1):min(width-1, x1+3)] = 0
            if x1 == x2 and y1 == y2: break
            e2 = 2 * err
            if e2 > -dy: err -= dy; x1 += sx
            if e2 < dx: err += dx; y1 += sy

    # --- THE FIX: SHUFFLE FOR NON-LINEAR FLOW ---
    # Keep track of start/end rooms before shuffling
    start_room = min(rooms, key=lambda r: sum(r['center']))
    goal_room = max(rooms, key=lambda r: sum(r['center']))
    
    connection_order = rooms.copy()
    random.shuffle(connection_order) # This removes the "direct diagonal spine"

    for i in range(len(connection_order)):
        r1 = connection_order[i]['center']
        r2 = connection_order[(i + 1) % len(connection_order)]['center']
        
        # Mix of diagonal and Manhattan for local flow
        if random.random() < diagonal_density:
            draw_fat_line(r1, r2)
        else:
            maze[r1[1]-1:r1[1]+3, min(r1[0], r2[0]):max(r1[0], r2[0])+1] = 0
            maze[min(r1[1], r2[1]):max(r1[1], r2[1])+1, r2[0]-1:r2[0]+3] = 0

    # Mark Start/Goal using the original extreme points
    maze[start_room['center'][1]-1:start_room['center'][1]+2, 
         start_room['center'][0]-1:start_room['center'][0]+2] = -2
    maze[goal_room['center'][1]-1:goal_room['center'][1]+2, 
         goal_room['center'][0]-1:goal_room['center'][0]+2] = -1
    
    return maze

# Usage
maze_127 = generate_maze()
np.savetxt("output22.csv", maze_127, delimiter=",", fmt='%d')