"""PPO training on MazeEnv using the heatmap observation channels.

Why PPO: the hand-rolled A2C in conductExperiment.py is correct but very
sample-inefficient and prone to single-action collapse under sparse reward.
PPO (clipped objective + minibatch reuse) is far more robust on exactly this
kind of task. Env, reward, and heatmaps are reused unchanged.

Run:  python3 trainPPO.py
"""
import os
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from functools import partial

from Utilities.SpaceEnv import MazeEnv
from Utilities.HeatMap import *

# --- environment config (mirrors conductExperiment.py) ---
low = np.array([0, 0])
high = np.array([100, 100])
num_cols = high[0] - low[0] + 1
num_rows = high[1] - low[1] + 1
walls = np.zeros((num_cols, num_rows), dtype=bool)
spawn = np.array([5, 5])
target_coords = np.array([[35, 40], [70, 20]])
target_awards = np.array([10, 5])

# Reset behaviour the env should use on EVERY reset (incl. SB3 auto-resets,
# which call reset() with no options — hence the wrapper below).
RESET_OPTIONS = {
    "randomSpawn": True,
    "randomSize": True,
    "randomTargetCoords": True,
    "max_steps": 300,
}

# --- sanity map set: the new optimal-action heatmap, NO noise ---
# [0]/[1] = single-target-locked variants, [2] = nearest-target.
sanityMaps = [
    partial(TargetWrap, inner_map_type=OptimalActionTarget, targetIndex=0),
    partial(TargetWrap, inner_map_type=OptimalActionTarget, targetIndex=1),
    OptimalActionTarget,
]

# Optional dense progress reward. OFF by default to keep the sparse-reward
# design faithful (dense reward injects a clean true-distance signal that would
# confound the noise study). Flip on only if PPO needs help bootstrapping.
SHAPE_REWARD = False
SHAPE_COEF = 0.3


class ResetOptionsWrapper(gym.Wrapper):
    """Inject fixed reset options on every reset (SB3 vec envs don't pass them)."""
    def __init__(self, env, options):
        super().__init__(env)
        self._opts = options

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options if options is not None else self._opts)


class ProgressRewardWrapper(gym.Wrapper):
    """Add coef * (decrease in L2 distance to nearest target) to the reward."""
    def __init__(self, env, coef=0.3):
        super().__init__(env)
        self.coef = coef
        self._prev = None

    def _dist(self):
        c = self.env.unwrapped.coords
        t = self.env.unwrapped.targetCords
        return float(np.min(np.linalg.norm(np.array(t) - np.array(c), axis=1)))

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev = self._dist()
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        d = self._dist()
        r = r + self.coef * (self._prev - d)
        self._prev = d
        return obs, r, term, trunc, info


def make_env(maps):
    def _init():
        env = MazeEnv(low, high, spawn, target_awards, target_coords,
                      heatMapTypes=maps, walls=walls)
        env = ResetOptionsWrapper(env, RESET_OPTIONS)
        if SHAPE_REWARD:
            env = ProgressRewardWrapper(env, coef=SHAPE_COEF)
        return env
    return _init


def evaluate(model, maps, episodes=200):
    """Greedy reach-rate over `episodes` (oracle≈100%, random≈10%)."""
    env = make_env(maps)()
    reached = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(int(action))
            done = term or trunc
        reached += term
    return reached, episodes


def train(maps, total_timesteps=400_000, n_envs=8, label="optimalAction"):
    # MLP PPO runs fastest on CPU (tiny net → GPU overhead dominates), and it
    # sidesteps the container's flaky CUDA state.
    vec_env = DummyVecEnv([make_env(maps) for _ in range(n_envs)])
    model = PPO(
        "MlpPolicy",
        vec_env,
        device="cpu",
        verbose=1,
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=dict(pi=[128, 64, 32], vf=[128, 64, 32])),
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    os.makedirs("weights", exist_ok=True)
    path = f"weights/ppo_{label}"
    model.save(path)
    print(f"saved -> {path}.zip")

    reached, n = evaluate(model, maps)
    print(f"PPO greedy eval: reached {reached}/{n}  (oracle≈{n}, random≈{n//10})")
    return model


if __name__ == "__main__":
    train(sanityMaps, label="optimalAction_sanity")
