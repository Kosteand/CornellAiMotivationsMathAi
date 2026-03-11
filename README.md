## Command Usage

| Command | Description |
|---|---|
| `init <r : int> <c : int>` | Initializes Maze with dimension r x c. |
| `train` | Begins training. |
| `speed <float>` | Sets speed of training. |
| `pause` | Pause training. |
| `stop` | Stop training. |
| `reset` | Resets the position of the agent. |
| `visualize <int>` | Creates visualization of Map / HeatMap and writes plot to file named `heatmap_visual_<int>.png`. |
| `step <action : int>` | Steps based on action provided. |
| `valid <r : int> <c : int>` | Returns whether `maze[r][c]` is a valid position. |
| `render` | Renders maze state. |