"""Fixed-k-list sweep for plotting the training-data SHAPE - not an
early-stopping sweep like run_trainPPO_sweep.py. This runs PPO training at
a specific, hand-picked set of k values (fine resolution where k is small,
coarser where it's large), and keeps LOG_TRAINING_DATA on so the
per-episode training curves are actually saved for every one of them, not
just the final hit rate.

K values tested (three ranges, deduplicated at the k=10 / k=100 overlaps):
  - k =   1,   2, ...,  10   (every k)
  - k =  10,  20, ..., 100   (every 10th k)
  - k = 100, 110, ..., 500   (every 10th k)
That's 59 runs total: K_VALUES below is built from exactly those three
ranges, so edit the range() calls there if you want a different resolution.

LOG_TRAINING_DATA is ON here (run_trainPPO_sweep.py leaves it off by
default) because capturing the training curves is the whole point of this
sweep. That means MASTER_TRAIN_LOG_PATH WILL grow large across 59 runs -
each 200_000-timestep run adds roughly 7-9 MB, so expect the master file to
reach several hundred MB by the end. A previous unbounded sweep with this
flag on filled an entire disk, so this script checks free disk space
before every run and stops early - printing how far it got - rather than
crashing mid-write if space runs low. Keep an eye on
eval_logs/shape_sweep_train_log_all_runs.csv's size while this runs, and
raise MIN_FREE_BYTES if you want more headroom.

Run:  python3 run_trainPPO_shape_sweep.py
"""
import csv
import os
import shutil

from Utilities.bandit_env import MarginGroup
from trainPPO import train

# --- sweep-specific config ---
G = 4
VALUE = 1.0

K_VALUES = []
_seen = set()
for _r in (range(1, 11), range(10, 101, 10), range(100, 501, 10)):
    for _k in _r:
        if _k not in _seen:
            _seen.add(_k)
            K_VALUES.append(_k)

# Stop early (rather than crash mid-write) if free disk space on the drive
# holding EVAL_LOGS_DIR drops below this.
MIN_FREE_BYTES = 2 * 1024 ** 3  # 2 GB

# Output paths.
SUMMARY_CSV_PATH = "eval_logs/shape_sweep_summary.csv"
MASTER_TRAIN_LOG_PATH = "eval_logs/shape_sweep_train_log_all_runs.csv"
MASTER_EVAL_LOG_PATH = "eval_logs/shape_sweep_eval_log_all_runs.csv"

# Reused every iteration so per-run temp files get overwritten, not
# accumulated - same pattern as run_trainPPO_sweep.py.
TEMP_LABEL = "shape_sweep_temp"

# --- environment config (groups is NOT here - built by the loop) ---
INCORRECT_REWARD = 0.0
N_ENVS = 8

# --- training loop config ---
TOTAL_TIMESTEPS = 200_000
PROGRESS_BAR = False
LOG_TRAINING_DATA = True  # on: this sweep exists to capture training curves
LOG_INTERVAL = None
PRINT_FINAL_SUMMARY = False

# --- PPO hyperparameters ---
DEVICE = "cpu"
VERBOSE = 0
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
NET_ARCH_PI = (64, 32)
NET_ARCH_VF = (64, 32)
WEIGHT_DECAY = 0.0
ACTOR_WEIGHT_DECAY = None
CRITIC_WEIGHT_DECAY = None
PPO_KWARGS = None

# --- evaluation config ---
EVAL_EPISODES = 500

# --- output config ---
WEIGHTS_DIR = "weights"
EVAL_LOGS_DIR = "eval_logs"


def _fold_into_master(source_csv, dest_csv, k):
    """Append source_csv's data rows into dest_csv, tagging each row with
    k. Writes dest_csv's header (source's header + a leading 'k' column)
    only the first time dest_csv is created."""
    if not os.path.exists(source_csv):
        return

    with open(source_csv, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header, data_rows = rows[0], rows[1:]

    write_header = not os.path.exists(dest_csv)
    with open(dest_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["k"] + header)
        for row in data_rows:
            writer.writerow([k] + row)


def _free_space_ok():
    check_dir = EVAL_LOGS_DIR if os.path.exists(EVAL_LOGS_DIR) else "."
    return shutil.disk_usage(check_dir).free >= MIN_FREE_BYTES


def run_sweep():
    os.makedirs(EVAL_LOGS_DIR, exist_ok=True)

    # Fresh output files at the start of every sweep run.
    with open(SUMMARY_CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["k", "hit_rate"])
    for path in (MASTER_TRAIN_LOG_PATH, MASTER_EVAL_LOG_PATH):
        if os.path.exists(path):
            os.remove(path)

    last_k_run = None
    for k in K_VALUES:
        if not _free_space_ok():
            print(
                f"Stopping early: free disk space dropped below "
                f"{MIN_FREE_BYTES / 1024**3:.1f} GB before k={k}. "
                f"Last completed k was {last_k_run}."
            )
            break

        groups = [MarginGroup(g=G, delta=1.0 / k, value=VALUE)]

        result = train(
            groups,
            incorrect_reward=INCORRECT_REWARD,
            n_envs=N_ENVS,
            total_timesteps=TOTAL_TIMESTEPS,
            label=TEMP_LABEL,
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

        hit_rate = result.hit_rate

        if LOG_TRAINING_DATA:
            _fold_into_master(
                f"{EVAL_LOGS_DIR}/ppo_{TEMP_LABEL}_train_log.csv",
                MASTER_TRAIN_LOG_PATH,
                k,
            )
        _fold_into_master(
            f"{EVAL_LOGS_DIR}/ppo_{TEMP_LABEL}_eval.csv",
            MASTER_EVAL_LOG_PATH,
            k,
        )

        with open(SUMMARY_CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([k, hit_rate])

        print(f"k={k}: hit_rate={hit_rate:.1%}")
        last_k_run = k

    print(f"Summary   -> {SUMMARY_CSV_PATH}")
    if LOG_TRAINING_DATA:
        print(f"Train log -> {MASTER_TRAIN_LOG_PATH}")
    print(f"Eval log  -> {MASTER_EVAL_LOG_PATH}")


if __name__ == "__main__":
    run_sweep()
