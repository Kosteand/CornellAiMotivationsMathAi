import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import defaultdict

updates = []
returns = []
targets = []
first_actions = []

with open("eval_logs/episode_info.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        updates.append(int(row[0]))
        returns.append(float(row[1]))
        targets.append(int(row[2]))
        # older episode_info.csv files may not have a first_action column
        first_actions.append(row[3] if len(row) > 3 else "none")

updates = np.array(updates)
returns = np.array(returns)
targets = np.array(targets)
first_actions = np.array(first_actions)

# ---------------------------------------------------------
# Plot 1: raw returns + rolling average (existing behavior)
# ---------------------------------------------------------
window = 50
rolling = np.convolve(returns, np.ones(window)/window, mode='valid')
rolling_updates = updates[window-1:]

plt.figure(figsize=(12, 5))
plt.plot(updates, returns, alpha=0.2, color='blue', label='Raw returns')
plt.plot(rolling_updates, rolling, color='blue', linewidth=2, label=f'{window}-episode rolling avg')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
plt.xlim(left=0)
plt.xlabel("Update")
plt.ylabel("Episode Return")
plt.title("Training Returns")
plt.legend()
plt.tight_layout()
plt.savefig("eval_logs/returns_plot.png")
print("Saved to eval_logs/returns_plot.png")

# ---------------------------------------------------------
# Plot 2: rolling target-hit percentages, grouped by update
# ---------------------------------------------------------
# Group episode outcomes by the update they occurred on. Multiple episodes
# (up to 22, one per env) can finish on the same update, so we aggregate
# per-update before computing rolling percentages -- a "10 update" window
# below means 10 distinct update values, not a fixed row count.
by_update = defaultdict(list)
for u, t in zip(updates, targets):
    by_update[u].append(t)

unique_updates = np.array(sorted(by_update.keys()))

left_pct = np.zeros(len(unique_updates))
right_pct = np.zeros(len(unique_updates))
either_pct = np.zeros(len(unique_updates))

for i, u in enumerate(unique_updates):
    outcomes = np.array(by_update[u])
    total = len(outcomes)
    left_pct[i] = 100.0 * np.sum(outcomes == 0) / total
    right_pct[i] = 100.0 * np.sum(outcomes == 1) / total
    either_pct[i] = left_pct[i] + right_pct[i]

target_window = 10  # rolling average over the last 10 updates (not rows)


def rolling_avg(arr, window):
    if len(arr) < window:
        return np.array([]), np.array([])
    avg = np.convolve(arr, np.ones(window) / window, mode='valid')
    return unique_updates[window - 1:], avg


left_x, left_avg = rolling_avg(left_pct, target_window)
right_x, right_avg = rolling_avg(right_pct, target_window)
either_x, either_avg = rolling_avg(either_pct, target_window)

plt.figure(figsize=(12, 5))
plt.plot(left_x, left_avg, color='tab:blue', linewidth=2, label='Left target %')
plt.plot(right_x, right_avg, color='tab:orange', linewidth=2, label='Right target %')
plt.plot(either_x, either_avg, color='tab:green', linewidth=2, label='Either target %')
plt.ylim(0, 100)
plt.xlim(left=0)
plt.xlabel("Update")
plt.ylabel(f"% of episodes ({target_window}-update rolling avg)")
plt.title("Target Hit Rate Over Training")
plt.legend()
plt.tight_layout()
plt.savefig("eval_logs/target_percentage_plot.png")
print("Saved to eval_logs/target_percentage_plot.png")

# ---------------------------------------------------------
# Plot 3: rolling % of left-reward (out of left+right, misses excluded),
# split out by the agent's first move that episode
# ---------------------------------------------------------
# Only episodes that actually reached a target (target 0 or 1) count here --
# misses (target == -1) are excluded entirely, both from the numerator and
# the denominator, per your request to ignore them.
action_names = ["up", "down", "left", "right"]

# left_by_update_action[action][update] = count of left hits with that first move on that update
# total_by_update_action[action][update] = count of left+right hits with that first move on that update
left_by_update_action = {a: defaultdict(int) for a in action_names}
total_by_update_action = {a: defaultdict(int) for a in action_names}

for u, t, a in zip(updates, targets, first_actions):
    if t not in (0, 1):
        continue  # ignore misses
    if a not in left_by_update_action:
        continue  # ignore unrecognized/"none" first_action values
    total_by_update_action[a][u] += 1
    if t == 0:
        left_by_update_action[a][u] += 1

# Rolling window is over 10 *updates* (not rows), consistent with plot 2.
# To handle updates where a given first move has few/no reward episodes,
# we sum counts across the window before taking the percentage, rather
# than averaging percentages -- this avoids blowing up on small samples
# and naturally skips windows with zero data (left as a gap in the line).
first_move_window = 10

# Baseline: overall left-rate (out of left+right, misses excluded) pooled
# across ALL first moves -- this is the same quantity as plot 2's left %
# line, restricted to the same rolling-sum-of-counts method used below.
baseline_left_arr = np.zeros(len(unique_updates))
baseline_total_arr = np.zeros(len(unique_updates))
for a in action_names:
    baseline_left_arr += np.array([left_by_update_action[a].get(u, 0) for u in unique_updates], dtype=np.float64)
    baseline_total_arr += np.array([total_by_update_action[a].get(u, 0) for u in unique_updates], dtype=np.float64)

baseline_pct = None
if len(unique_updates) >= first_move_window:
    baseline_left_sum = np.convolve(baseline_left_arr, np.ones(first_move_window), mode='valid')
    baseline_total_sum = np.convolve(baseline_total_arr, np.ones(first_move_window), mode='valid')
    with np.errstate(invalid='ignore', divide='ignore'):
        baseline_pct = 100.0 * baseline_left_sum / baseline_total_sum
    baseline_pct[baseline_total_sum == 0] = np.nan

plt.figure(figsize=(12, 5))
colors = {"up": "tab:purple", "down": "tab:brown", "left": "tab:blue", "right": "tab:orange"}

for a in action_names:
    left_arr = np.array([left_by_update_action[a].get(u, 0) for u in unique_updates], dtype=np.float64)
    total_arr = np.array([total_by_update_action[a].get(u, 0) for u in unique_updates], dtype=np.float64)

    if len(unique_updates) < first_move_window:
        continue

    left_sum = np.convolve(left_arr, np.ones(first_move_window), mode='valid')
    total_sum = np.convolve(total_arr, np.ones(first_move_window), mode='valid')

    with np.errstate(invalid='ignore', divide='ignore'):
        pct = 100.0 * left_sum / total_sum
    pct[total_sum == 0] = np.nan  # leave gaps where this move had no reward episodes in the window

    deviation = pct - baseline_pct
    x = unique_updates[first_move_window - 1:]
    plt.plot(x, deviation, color=colors[a], linewidth=2, label=f'First move: {a}')

plt.axhline(y=0, color='black', linestyle='--', alpha=0.4, label='Overall left-rate (baseline)')
plt.xlim(left=0)
plt.xlabel("Update")
plt.ylabel(f"Left-rate deviation from baseline, pp ({first_move_window}-update rolling)")
plt.title("First Move vs. Which Target Is Reached, Relative to Baseline (misses excluded)")
plt.legend()
plt.tight_layout()
plt.savefig("eval_logs/first_move_correlation_plot.png")
print("Saved to eval_logs/first_move_correlation_plot.png")

# ---------------------------------------------------------
# Plot 4: training_plots.png -- episode returns, entropy, critic loss,
# actor loss. This used to be produced automatically at the end of every
# run_training.py call (as result1.png); it's now only produced here,
# when plot_returns.py is actually called, reading the per-update losses
# that run_training.py persists to eval_logs/update_info.csv.
# ---------------------------------------------------------
import os

loss_updates = []
critic_losses = []
actor_losses = []
entropies_arr = []

with open("eval_logs/update_info.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        loss_updates.append(int(row[0]))
        critic_losses.append(float(row[1]))
        actor_losses.append(float(row[2]))
        entropies_arr.append(float(row[3]))

critic_losses = np.array(critic_losses)
actor_losses = np.array(actor_losses)
entropies_arr = np.array(entropies_arr)

rolling_length = 20

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
fig.suptitle("Training plots")

axs[0][0].set_title("Episode Returns")
# NOTE: uses the same (update, return) pairs as Plot 1 above, plotted
# against update rather than raw episode count -- run_training.py no
# longer exposes envWrapper.return_queue's episode-ordered sequence to
# any persisted file, so update-indexed rolling average (same data Plot 1
# uses) is the closest equivalent available after the fact.
if len(returns) >= rolling_length:
    returns_rolling = np.convolve(returns, np.ones(rolling_length) / rolling_length, mode="valid")
    axs[0][0].plot(updates[rolling_length - 1:], returns_rolling)
axs[0][0].set_xlabel("Update")

axs[1][0].set_title("Entropy")
if len(entropies_arr) >= rolling_length:
    entropy_rolling = np.convolve(entropies_arr, np.ones(rolling_length) / rolling_length, mode="valid")
    axs[1][0].plot(loss_updates[rolling_length - 1:], entropy_rolling)
axs[1][0].set_xlabel("Update")

axs[0][1].set_title("Critic Loss")
if len(critic_losses) >= rolling_length:
    critic_rolling = np.convolve(critic_losses, np.ones(rolling_length) / rolling_length, mode="valid")
    axs[0][1].plot(loss_updates[rolling_length - 1:], critic_rolling)
axs[0][1].set_xlabel("Update")

axs[1][1].set_title("Actor Loss")
if len(actor_losses) >= rolling_length:
    actor_rolling = np.convolve(actor_losses, np.ones(rolling_length) / rolling_length, mode="valid")
    axs[1][1].plot(loss_updates[rolling_length - 1:], actor_rolling)
axs[1][1].set_xlabel("Update")

plt.tight_layout()
os.makedirs("eval_logs", exist_ok=True)
plt.savefig("eval_logs/training_plots.png")
print("Saved to eval_logs/training_plots.png")

# ---------------------------------------------------------
# Plot 6: left_reward over training (curriculum_experiment.py only --
# distance_reward_nudges.csv is written by DistanceRewardCurriculum.
# on_update(), every single update, regardless of whether a hit happened
# that update -- see curriculum_experiment.py). Skipped gracefully if that
# file doesn't exist, e.g. when just running run_training.py directly
# without the curriculum wrapper.
# ---------------------------------------------------------
nudge_log_path = "eval_logs/distance_reward_nudges.csv"
if os.path.exists(nudge_log_path):
    nudge_updates = []
    nudge_distances = []
    nudge_left_values = []

    with open(nudge_log_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            nudge_updates.append(int(row[0]))
            nudge_distances.append(int(row[1]))
            nudge_left_values.append(float(row[2]))

    nudge_updates = np.array(nudge_updates)
    nudge_distances = np.array(nudge_distances)
    nudge_left_values = np.array(nudge_left_values)

    plt.figure(figsize=(12, 5))
    plt.plot(nudge_updates, nudge_left_values, color="tab:red", linewidth=1.5, label="left_reward")

    # Mark each distance change with a vertical line, so you can see how
    # left_reward's trajectory lines up with the curriculum advancing --
    # e.g. whether it's still oscillating/settling right before a change,
    # or was already stable well before advancing.
    distance_change_updates = nudge_updates[1:][np.diff(nudge_distances) != 0]
    for i, u in enumerate(distance_change_updates):
        plt.axvline(x=u, color="gray", linestyle="--", alpha=0.4,
                    label="distance change" if i == 0 else None)

    plt.xlim(left=0)
    plt.xlabel("Update")
    plt.ylabel("left_reward")
    plt.title("left_reward Over Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eval_logs/left_reward_plot.png")
    print("Saved to eval_logs/left_reward_plot.png")
else:
    print(f"Skipped left_reward plot -- {nudge_log_path} not found "
          f"(only written when curriculum_experiment.py is the one running training).")
