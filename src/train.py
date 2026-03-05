import gymnasium as gym
from pyvirtualdisplay import Display
from gymnasium.wrappers import RecordVideo
import torch

# 1. Start the Virtual Display (The "Fake Monitor")
with Display(visible=0, size=(1400, 900)) as disp:
    
    # 2. Check Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running RL Training on: {device.upper()}")

    # 3. Create Environment with Rendering
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    # 4. Wrap to record videos into a 'videos' folder
    env = RecordVideo(env, video_folder="./videos", name_prefix="training")

    obs, info = env.reset()
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
