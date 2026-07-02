import random
import numpy as np

def generateWalls(num_cols, num_rows, wall_thickness=2, corridor_thickness=2, loop_factor=0.4):
    scale = wall_thickness + corridor_thickness

    walls = np.ones((num_cols, num_rows), dtype=bool)

    logical_cols = (num_cols - 1) // scale
    logical_rows = (num_rows - 1) // scale

    if logical_cols < 1 or logical_rows < 1:
        return walls

    visited = np.zeros((logical_cols, logical_rows), dtype=bool)

    def cell_center(lx, ly):
        return 1 + lx * scale, 1 + ly * scale

    def carve_cell(lx, ly):
        cx, cy = cell_center(lx, ly)
        walls[cx:min(cx + corridor_thickness, num_cols),
              cy:min(cy + corridor_thickness, num_rows)] = False

    def carve_passage(lx1, ly1, lx2, ly2):
        cx1, cy1 = cell_center(lx1, ly1)
        cx2, cy2 = cell_center(lx2, ly2)

        if cx1 == cx2:  # vertical
            walls[cx1:min(cx1 + corridor_thickness, num_cols),
                  min(cy1, cy2):max(cy1, cy2) + 1] = False
        else:           # horizontal
            walls[min(cx1, cx2):max(cx1, cx2) + 1,
                  cy1:min(cy1 + corridor_thickness, num_rows)] = False

    # build spanning tree
    stack = [(0, 0)]
    visited[0, 0] = True
    carve_cell(0, 0)

    while stack:
        lx, ly = stack[-1]
        directions = [(1,0),(-1,0),(0,1),(0,-1)]
        random.shuffle(directions)
        moved = False

        for dx, dy in directions:
            nlx, nly = lx + dx, ly + dy
            if 0 <= nlx < logical_cols and 0 <= nly < logical_rows and not visited[nlx, nly]:
                visited[nlx, nly] = True
                carve_passage(lx, ly, nlx, nly)
                carve_cell(nlx, nly)
                stack.append((nlx, nly))
                moved = True
                break

        if not moved:
            stack.pop()

    # add loops
    for lx in range(logical_cols):
        for ly in range(logical_rows):
            for dx, dy in [(1, 0), (0, 1)]:
                nlx, nly = lx + dx, ly + dy
                if 0 <= nlx < logical_cols and 0 <= nly < logical_rows:
                    if random.random() < loop_factor:
                        carve_passage(lx, ly, nlx, nly)

    return walls

def generateWallsFixed1(num_cols=11, num_rows=11):
    walls = np.zeros((num_cols, num_rows), dtype=bool)

    wall_cols = [1, 3, 5, 9]

    for col in wall_cols:
        if col >= num_cols:
            continue
        # fill entire column with walls
        walls[col, :] = True
        # punch one random hole
        hole = random.randint(0, num_rows - 1)
        walls[col, hole] = False

    return walls