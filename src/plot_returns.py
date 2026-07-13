import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import defaultdict

updates = []
returns = []
targets = []

with open("eval_logs/returns.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        updates.append(int(row[0]))
        returns.append(float(row[1]))
        targets.append(int(row[2]))

updates = np.array(updates)
returns = np.array(returns)
targets = np.array(targets)

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