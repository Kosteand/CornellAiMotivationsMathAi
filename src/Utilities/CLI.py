""" TODO: 
Ight David, can you make a command line for MAzeEnv init, visualize, and step
Look at the HeatMappable protocol if needed
also, make sure that the matplotlib code in visualize is right
"""
# utils: matplotlib and numpy
import matplotlib as plt
import numpy as np
import asyncio
import sys
# usage: CLI cli = CLI()
#        cli.run() -- async read for input started.
class CLI:
        task     = None
        running  = False
        maze     = None
        speed    = .3
        def __init__(self):
                return;
        """
        Sets up async for STDIN.
        """
        def run (self):
                asyncio.run (self.main ())
                
        # see https://stackoverflow.com/a/71627494 for implementation.
        async def get_stream_reader(self, pipe) -> asyncio.StreamReader:
                loop = asyncio.get_event_loop()
                reader = asyncio.StreamReader(loop=loop)
                protocol = asyncio.StreamReaderProtocol(reader)
                await loop.connect_read_pipe(lambda: protocol, pipe)
                return reader

                
        # before exiting remove all tasks from being listened on.
        async def deconstruct_task (self) -> None:
                if self.task is not None:
                        self.task.cancel()
        # actual runner.
        async def main(self):
                reader = await self.get_stream_reader(sys.stdin)
                bl = True
                while bl:
                        print("> ", end="", flush=True)
                        data = await reader.readline()
                        if not data:
                                break
                        data = data.decode().strip()
                        bl = self.parse (data)
                        
                await self.deconstruct_task ()

        # simply parses the user input.
        def parse (self, line : str) -> bool:
                tokens = line.split()
                if not tokens:
                        return True


                match tokens[0]:
                        case "init":
                                self.create(tokens[1:])
                        case "train":
                                if check(tokens[0]):
                                        self.train()
                        case "pause":
                                self.pause()
                        case "reset":
                                self.reset()
                        case "visualize":
                                if len(tokens) <= 1:
                                        print ("incorrect usage. visualize <int> required", file = sys.stderr)
                                else:
                                        self.visualize(int(tokens[1]))
                        case "step":
                                if len(tokens) <= 1:
                                        print ("incorrect usage. step <action> required", file = sys.stderr)
                                else:
                                        self.step (int(tokens[1]))
                        case "valid":
                                if len(tokens) < 3:
                                        print ("incorrect usage. valid <r : int> <c : int> required", file = sys.stderr)
                                else:
                                        self.valid (int(tokens[1]), int(tokens[2]))
                        case "render":
                                self.render()
                        case "help":
                                self.info()
                        case "speed":
                                self.speed = int(token[1])
                        case "quit":
                                # reset stuff
                                return False
                        case _:
                                print ("not a valid command", file = sys.stderr)
                return True
                                
        # returns True if we are good to go, else we need to err. sys.stderr.write("Error message goes here\n") for more natural java like syntax.
        def check(self, word) -> None:
                if not self.not_null():
                        print (f"instantialize maze before running {word}", file=sys.stderr)
                        return False
                else:
                        return True



        def not_null(self):
                return self.maze is not None

        # performs step.
        def step (self, dx : int) -> None:
                if self.running:
                        print ("cannot step while running.", file = sys.stderr)
                        return
                current_pos, reward, done, lst = maze.step (dx)
                print(f"current_pos: {current_pos}\nreward: {reward}\ndone: {done}\n empty:{lst}")
                return

        
        def pause (self) -> None:
                if self.running:
                        self.running = False
        
        def valid (self, r : int, c : int) -> None:
                if self.not_null():
                        if maze.valid (r,c):
                                print ("valid")
                        else:
                                print (f"({r},{c}) is not valid", file = sys.stderr)
                else:
                        print ("maze has not been created.", file = sys.stderr)
                
        def train (self) -> None:
                if self.not_null():
                        if self.running:
                                print ("already running.", file = sys.stderr)
                                return
                        else:
                                self.running = True
                                self.task    = asyncio.create_task(self.run_loop())
                else:
                        print ("maze has not been created.", file = sys.stderr)

        async def run_loop(self) -> None:
                while self.running:
                        self.maze.step()
                        await asyncio.sleep(self.speed)  # control speed

        # for help.
        def info(self) -> None:
            print("""
command usage:
    init <r : int> <c : int>
        initializes Maze with dimension r x c.

    train
        begins training.

    speed <float>
        sets speed of training

    pause
        pause training.

    reset
        resets the position of the agent.

    visualize <int>
        creates visualization of Map / HeatMap and writes plot to
        heatmap_visual_<int>.png.

    step <action : int>
        steps based on action provided.

    valid <r : int> <c : int>
        returns whether maze[r][c] is a valid position.
        Equivalent to: cat valid <r> <c>

    render
        renders maze state.
        Equivalent to: cat render
""")
                
