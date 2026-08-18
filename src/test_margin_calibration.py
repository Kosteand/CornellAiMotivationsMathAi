"""Supervised calibration sweep for MarginGroup.

This is deliberately NOT an RL script. Before wiring MarginGroup into PPO
training, we want a cheap sanity check on how a plain feedforward
classifier's accuracy depends on the margin delta = 1/c, at a fixed g=4
and s=1.0, trained directly on (x, label) pairs via cross-entropy loss (no
bandit reward, no environment stepping).

Sweep rule (as specified): g=4, s=1.0, delta(c) = 1/c. Start at c=1,
increment by 1, stop once 10 CONSECUTIVE c values all land under 50%
held-out accuracy, or once c=100 is reached - whichever comes first.

Each c gets its own freshly-initialized small MLP, trained from scratch on
freshly sampled data (train/test are independently drawn from
MarginGroup(g=4, delta=1/c, value=1.0, s=1.0).sample(), so there's no
leakage between them and no leakage across c values).

Output: eval_logs/margin_calibration_summary.csv with columns
(c, delta, train_accuracy, test_accuracy).

Run:  python3 test_margin_calibration.py
"""
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Utilities.bandit_env import MarginGroup

# --- sweep config ---
G = 4
S = 1.0
START_C = 1
MAX_C = 100
STOP_AFTER_CONSECUTIVE_LOW = 10
LOW_ACCURACY_THRESHOLD = 0.50  # 50%

# --- dataset config ---
TRAIN_EXAMPLES = 20_000
TEST_EXAMPLES = 5_000
SEED = 0

# --- model / training config ---
HIDDEN_SIZES = (64, 32)
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
DEVICE = "cpu"

# --- output config ---
EVAL_LOGS_DIR = "eval_logs"
SUMMARY_CSV_PATH = os.path.join(EVAL_LOGS_DIR, "margin_calibration_summary.csv")


def make_mlp(input_size: int, output_size: int, hidden_sizes) -> nn.Module:
    layers = []
    prev = input_size
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_size))
    return nn.Sequential(*layers)


def make_dataset(group: MarginGroup, num_examples: int, rng: np.random.Generator):
    xs = np.empty((num_examples, group.observation_size), dtype=np.float32)
    ys = np.empty((num_examples,), dtype=np.int64)
    for i in range(num_examples):
        x, label = group.sample(rng)
        xs[i] = x
        ys[i] = label
    return torch.from_numpy(xs), torch.from_numpy(ys)


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
        return float((preds == y).float().mean().item())


def train_one_c(c: int, rng: np.random.Generator) -> tuple[float, float, float]:
    delta = 1.0 / c
    group = MarginGroup(g=G, delta=delta, value=1.0, s=S)

    train_x, train_y = make_dataset(group, TRAIN_EXAMPLES, rng)
    test_x, test_y = make_dataset(group, TEST_EXAMPLES, rng)

    train_x, train_y = train_x.to(DEVICE), train_y.to(DEVICE)
    test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)

    model = make_mlp(group.observation_size, G, HIDDEN_SIZES).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.CrossEntropyLoss()

    num_train = train_x.shape[0]
    for _epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(num_train)
        for start in range(0, num_train, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            batch_x, batch_y = train_x[idx], train_y[idx]

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

    train_acc = accuracy(model, train_x, train_y)
    test_acc = accuracy(model, test_x, test_y)
    return delta, train_acc, test_acc


def run_calibration():
    os.makedirs(EVAL_LOGS_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    with open(SUMMARY_CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["c", "delta", "train_accuracy", "test_accuracy"])

    consecutive_low = 0
    c = START_C

    while consecutive_low < STOP_AFTER_CONSECUTIVE_LOW and c <= MAX_C:
        delta, train_acc, test_acc = train_one_c(c, rng)

        with open(SUMMARY_CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([c, delta, train_acc, test_acc])

        if test_acc < LOW_ACCURACY_THRESHOLD:
            consecutive_low += 1
        else:
            consecutive_low = 0

        print(
            f"c={c:3d}  delta={delta:.4f}  "
            f"train_acc={train_acc:.1%}  test_acc={test_acc:.1%}  "
            f"(consecutive under {LOW_ACCURACY_THRESHOLD:.0%}: "
            f"{consecutive_low}/{STOP_AFTER_CONSECUTIVE_LOW})"
        )

        c += 1

    if consecutive_low >= STOP_AFTER_CONSECUTIVE_LOW:
        print(
            f"Stopped: {STOP_AFTER_CONSECUTIVE_LOW} consecutive c values under "
            f"{LOW_ACCURACY_THRESHOLD:.0%} test accuracy (last c={c - 1})."
        )
    else:
        print(f"Stopped: reached MAX_C={MAX_C}.")

    print(f"Summary -> {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    run_calibration()
