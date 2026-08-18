"""Sweep k for a single MarginGroup(g=4, value=1.0, delta=1/k), stopping
once STOP_AFTER_CONSECUTIVE_LOW straight k values all land under
LOW_HIT_RATE_THRESHOLD hit rate - or once MAX_RUNS runs have happened,
whichever comes first. The MAX_RUNS cap exists because MarginGroup doesn't
necessarily degrade to a low hit rate the way the old frequency-based
ComplexityGroup did; without a cap, a sweep whose hit rate stays high could
run k up indefinitely.

Every input train() accepts is exposed below as an editable variable,
EXCEPT `groups` - the sweep loop builds that itself each iteration
(g=4, value=1.0, k starting at 1 and incrementing by 1).

Per-run CSV outputs: train() normally writes one train_log.csv and one
eval.csv per run (named after `label`). Sweeping k across dozens of runs
that way would mean dozens of separate files. Instead, every iteration
here reuses the SAME on-disk label (TEMP_LABEL), so train() overwrites the
same pair of files each time rather than creating a new one - and right
after each run, this script folds that run's rows into two master CSVs
(tagged with the k that produced them) and then the temp files get
overwritten again next iteration. So only one run's worth of per-run CSV
data ever exists at a time; nothing accumulates across the whole sweep
except the small master files and the new sweep-summary CSV.

The sweep-summary CSV (SUMMARY_CSV_PATH) is the one this was actually
built for: one row per run with k, hit_rate, and (as of the weight-decay
change below) each run's final weight norm alongside everything else
train()'s TrainResult carries (correct/episodes/mean_reward) plus the
weight_decay value used. Named sweep_summary_weight_decay.csv, distinct
from plain sweep_summary.csv, so this doesn't clobber output from the
no-decay sweep or the exponential-fit sweep.

WEIGHT_DECAY is no longer 0.0: AdamW's weight_decay directly shrinks the
policy's weight norm over training (see weight_norm() below - it's
reported per run precisely so you can see this happening across k). The
value here (1.0) was picked empirically, not guessed: at 20_000-40_000
timesteps, hit_rate was unaffected (within noise) all the way up to
weight_decay=2.0 on both an easy (k=8) and a harder (k=20) MarginGroup,
while weight_decay=1.0 already visibly shrinks the final weight norm
(e.g. ~16.1 -> ~13.7 on the k=20 comparison) - AdamW's decay compounds
over every one of the ~3900 optimizer steps in a full 200_000-timestep
run (n_epochs * total_timesteps / batch_size), so the effect at the full
TOTAL_TIMESTEPS below will be considerably larger than in those shorter
smoke tests, without the shorter-run evidence suggesting hit_rate is at
any real risk. If you want even smaller weights, increasing this further
is reasonable given how flat hit_rate stayed even at weight_decay=2.0 -
just re-check hit_rate isn't dropping once you go meaningfully higher.

LOG_TRAINING_DATA defaults to False: with it on, MASTER_TRAIN_LOG_PATH
accumulates every training episode from every run in the sweep, with no
size cap - this previously filled an entire disk (8.7GB+) during a long
sweep. Leave it off unless you specifically need the raw per-episode
training curves and are watching that file's size.

Note: since TEMP_LABEL is fixed, the model checkpoint
(weights_dir/ppo_{TEMP_LABEL}.zip) also gets overwritten every iteration -
only the LAST k's model is kept on disk. If you want every k's model kept,
change TEMP_LABEL to something like f"g4_k{k}" inside the loop below, at
the cost of one checkpoint file per k.

Run:  python3 run_trainPPO_sweep.py
"""
import csv
import os

import torch

from Utilities.bandit_env import MarginGroup
from trainPPO import train

# --- sweep-specific config ---

# Fixed group shape being swept: g=4, value=1.0, k=1,2,3,... .
G = 4
VALUE = 1.0
START_K = 1.0

# Stop once this many CONSECUTIVE k values all land under
# LOW_HIT_RATE_THRESHOLD.
STOP_AFTER_CONSECUTIVE_LOW = 10
LOW_HIT_RATE_THRESHOLD = 0.50  # 30%

# Safety net: stop after this many runs total even if hit_rate never drops
# under LOW_HIT_RATE_THRESHOLD for STOP_AFTER_CONSECUTIVE_LOW straight runs
# (e.g. MarginGroup stays learnable well past k=100 - without this the
# sweep would just keep incrementing k forever).
MAX_RUNS = 100

# Output paths. SUMMARY_CSV_PATH is intentionally NOT sweep_summary.csv -
# that name is used by the no-weight-decay sweep and by
# run_exponential_fit_sweep.py, and overwriting it would clobber either
# one's output.
SUMMARY_CSV_PATH = "eval_logs/sweep_summary_weight_decay.csv"
MASTER_TRAIN_LOG_PATH = "eval_logs/sweep_train_log_all_runs.csv"
MASTER_EVAL_LOG_PATH = "eval_logs/sweep_eval_log_all_runs.csv"

# Reused every iteration - see module docstring above.
TEMP_LABEL = "sweep_temp"

# --- environment config (groups is NOT here - built by the loop) ---
INCORRECT_REWARD = 0.0
N_ENVS = 8

# --- training loop config ---
TOTAL_TIMESTEPS = 200_000
PROGRESS_BAR = False
# Off by default for sweeps: this logs every training episode of every run
# into MASTER_TRAIN_LOG_PATH, which has no size cap and grows with
# TOTAL_TIMESTEPS * (number of runs) - it filled an entire disk during a
# long sweep. The (k, hit_rate) summary and per-run eval logs are
# unaffected either way. Only turn this back on for a short, deliberately
# bounded sweep where you actually want the raw per-episode training
# curves, and keep an eye on MASTER_TRAIN_LOG_PATH's size while it runs.
LOG_TRAINING_DATA = False
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
# See module docstring: chosen empirically to keep the final weight norm
# meaningfully smaller without visibly hurting hit_rate.
WEIGHT_DECAY = 1.0
ACTOR_WEIGHT_DECAY = None
CRITIC_WEIGHT_DECAY = None
PPO_KWARGS = None

# --- evaluation config ---
EVAL_EPISODES = 500

# --- output config ---
WEIGHTS_DIR = "weights"
EVAL_LOGS_DIR = "eval_logs"

SUMMARY_FIELDNAMES = [
    "k", "hit_rate", "correct", "episodes", "mean_reward",
    "weight_norm_actor", "weight_norm_critic", "weight_norm_total",
    "weight_decay",
]


def weight_norm(parameters):
    """L2 norm across every element of every tensor in `parameters`
    (flattening each into one giant vector conceptually, without ever
    materializing it) - the standard way to summarize "how big are this
    network's weights" as a single number. Works the same whether
    `parameters` is the full policy, or just its actor/critic sub-nets."""
    total = 0.0
    for p in parameters:
        total += float(torch.sum(p.detach() ** 2))
    return total ** 0.5


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


def run_sweep():
    os.makedirs(EVAL_LOGS_DIR, exist_ok=True)

    # Fresh output files at the start of every sweep run.
    with open(SUMMARY_CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES).writeheader()
    for path in (MASTER_TRAIN_LOG_PATH, MASTER_EVAL_LOG_PATH):
        if os.path.exists(path):
            os.remove(path)

    consecutive_low = 0
    run_count = 0
    k = START_K

    while consecutive_low < STOP_AFTER_CONSECUTIVE_LOW and run_count < MAX_RUNS:
        # Recomputed every iteration - delta must shrink as k grows, not
        # stay fixed at 1/START_K for the whole sweep.
        DELTA = 1 / k
        groups = [MarginGroup(g=G, delta=DELTA, value=VALUE)]

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

        policy = result.model.policy
        actor_params = list(policy.mlp_extractor.policy_net.parameters()) + list(
            policy.action_net.parameters()
        )
        critic_params = list(policy.mlp_extractor.value_net.parameters()) + list(
            policy.value_net.parameters()
        )
        norm_actor = weight_norm(actor_params)
        norm_critic = weight_norm(critic_params)
        norm_total = weight_norm(policy.parameters())

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
            csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES).writerow({
                "k": k,
                "hit_rate": hit_rate,
                "correct": result.correct,
                "episodes": result.episodes,
                "mean_reward": result.mean_reward,
                "weight_norm_actor": norm_actor,
                "weight_norm_critic": norm_critic,
                "weight_norm_total": norm_total,
                "weight_decay": WEIGHT_DECAY,
            })

        if hit_rate < LOW_HIT_RATE_THRESHOLD:
            consecutive_low += 1
        else:
            consecutive_low = 0

        print(
            f"k={k}: hit_rate={hit_rate:.1%}  weight_norm={norm_total:.2f}  "
            f"(consecutive runs under {LOW_HIT_RATE_THRESHOLD:.0%}: "
            f"{consecutive_low}/{STOP_AFTER_CONSECUTIVE_LOW})"
        )

        run_count += 1
        k += 1

    if consecutive_low >= STOP_AFTER_CONSECUTIVE_LOW:
        print(
            f"Stopped: {STOP_AFTER_CONSECUTIVE_LOW} consecutive runs under "
            f"{LOW_HIT_RATE_THRESHOLD:.0%} hit rate (last k={k - 1})."
        )
    else:
        print(
            f"Stopped: reached MAX_RUNS={MAX_RUNS} without "
            f"{STOP_AFTER_CONSECUTIVE_LOW} consecutive runs under "
            f"{LOW_HIT_RATE_THRESHOLD:.0%} hit rate (last k={k - 1})."
        )
    print(f"Summary  -> {SUMMARY_CSV_PATH}")
    if LOG_TRAINING_DATA:
        print(f"Train log -> {MASTER_TRAIN_LOG_PATH}")
    print(f"Eval log  -> {MASTER_EVAL_LOG_PATH}")


if __name__ == "__main__":
    run_sweep()
