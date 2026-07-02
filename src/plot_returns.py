import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv

updates = []
returns = []

with open("eval_logs/returns.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        updates.append(int(row[0]))
        returns.append(float(row[1]))

updates = np.array(updates)
returns = np.array(returns)

# Rolling average
window = 50
rolling = np.convolve(returns, np.ones(window)/window, mode='valid')
rolling_updates = updates[window-1:]

plt.figure(figsize=(12, 5))
plt.plot(updates, returns, alpha=0.2, color='blue', label='Raw returns')
plt.plot(rolling_updates, rolling, color='blue', linewidth=2, label=f'{window}-episode rolling avg')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
plt.xlabel("Update")
plt.ylabel("Episode Return")
plt.title("Training Returns")
plt.legend()
plt.tight_layout()
plt.savefig("eval_logs/returns_plot.png")
print("Saved to eval_logs/returns_plot.png")