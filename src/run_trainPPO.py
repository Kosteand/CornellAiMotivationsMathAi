"""Runnable entry point for PPO training on BanditEnv.

trainPPO.py defines the `train()` function and its supporting classes but
no longer calls it - every input `train()` accepts is spelled out below as
its own variable, so changing the run config means editing a value here,
not the training code itself.

Run:  python3 run_trainPPO.py
"""
from Utilities.bandit_env import MarginGroup
from trainPPO import train

# --- environment config ---

# List of ComplexityGroup(g, k, value) instances passed to BanditEnv.
# g = number of options in the group, k = complexity dial (higher = harder
# to learn, see ComplexityGroup's docstring), value = reward for matching
# that group's label. Two groups here, same g, different k: group 0 is
# low-frequency/easy (k=1), group 1 is high-frequency/hard (k=25). Nothing
# in the observation marks which secret belongs to which group - the agent
# only sees the concatenated secrets.
GROUPS = [
    MarginGroup(4, 1, 1)
]

# Reward given when the chosen action doesn't match any group's label.
INCORRECT_REWARD = 0.0

# Number of BanditEnv copies run in parallel via DummyVecEnv.
N_ENVS = 8

# --- training loop config ---

# Total env steps across all N_ENVS. Since each BanditEnv episode is
# exactly one step, this is also the total number of independent bandit
# episodes sampled over the whole run.
TOTAL_TIMESTEPS = 200_000

# Used to name the saved model (weights_dir/ppo_{label}.zip) and the CSV
# logs (eval_logs_dir/ppo_{label}_train_log.csv and ppo_{label}_eval.csv).
LABEL = "complexity_g4"

# Passed to model.learn(). Requires tqdm and rich
# (pip install "stable-baselines3[extra]") - set False to avoid that
# dependency.
PROGRESS_BAR = True

# If True, log every training transition's x/label/action to
# eval_logs_dir/ppo_{label}_train_log.csv. Measured to add ~0% overhead
# even at tens of thousands of timesteps (buffers in memory, writes once
# at the end).
LOG_TRAINING_DATA = True

# How often (in training iterations, i.e. every N_STEPS * N_ENVS
# timesteps) SB3 prints its stats box during training. 1 = every
# iteration, 5 = every 5th, etc. Set to None to suppress periodic
# printing entirely - combine with PRINT_FINAL_SUMMARY for a single box
# at the very end instead of one per iteration. Has no effect if
# VERBOSE=0 (which disables that box, and all other SB3 console output,
# regardless of this setting).
LOG_INTERVAL = 1

# If True, force one extra print of the stats box after training
# finishes, showing the final iteration's numbers even if LOG_INTERVAL
# suppressed it during the loop. Also gated by VERBOSE >= 1.
PRINT_FINAL_SUMMARY = False

# --- PPO hyperparameters (passed straight through to stable_baselines3.PPO) ---

DEVICE = "cpu"
VERBOSE = 1
SEED = None
LEARNING_RATE = 3e-4
N_STEPS = 512
BATCH_SIZE = 512
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Hidden layer sizes for the actor (pi) and critic (vf) MLPs.
NET_ARCH_PI = (64, 32)
NET_ARCH_VF = (64, 32)

# L2 penalty (Adam's weight_decay) applied to both actor and critic
# parameters, UNLESS one or both of ACTOR_WEIGHT_DECAY/CRITIC_WEIGHT_DECAY
# below is set, in which case each unset one falls back to this value.
WEIGHT_DECAY = 0.0

# Set either of these (not None) to give the actor and critic their own
# weight_decay instead of sharing WEIGHT_DECAY. Leaving both None keeps
# SB3's normal single optimizer with WEIGHT_DECAY applied uniformly.
ACTOR_WEIGHT_DECAY = None
CRITIC_WEIGHT_DECAY = None

# Optional dict of any additional stable_baselines3.PPO keyword arguments
# not already covered above (e.g. target_kl, use_sde, tensorboard_log).
PPO_KWARGS = None

# --- evaluation config ---

# Number of fresh episodes used for the post-training greedy accuracy
# evaluation.
EVAL_EPISODES = 500

# --- output config ---

# Directory the trained model checkpoint is saved to.
WEIGHTS_DIR = "weights"

# Directory the training-data and evaluation CSV logs are written to.
EVAL_LOGS_DIR = "eval_logs"


if __name__ == "__main__":
    train(
        GROUPS,
        incorrect_reward=INCORRECT_REWARD,
        n_envs=N_ENVS,
        total_timesteps=TOTAL_TIMESTEPS,
        label=LABEL,
        progress_bar=PROGRESS_BAR,
        log_training_data=LOG_TRAINING_DATA,
        log_interval=LOG_INTERVAL,
        print_final_summary=PRINT_FINAL_SUMMARY,
        device=DEVICE,
        verbose=VERBOSE,
        seed=SEED,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        net_arch_pi=NET_ARCH_PI,
        net_arch_vf=NET_ARCH_VF,
        weight_decay=WEIGHT_DECAY,
        actor_weight_decay=ACTOR_WEIGHT_DECAY,
        critic_weight_decay=CRITIC_WEIGHT_DECAY,
        ppo_kwargs=PPO_KWARGS,
        eval_episodes=EVAL_EPISODES,
        weights_dir=WEIGHTS_DIR,
        eval_logs_dir=EVAL_LOGS_DIR,
    )
