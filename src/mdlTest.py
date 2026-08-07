"""Supervised sanity check, sibling to decoderTest.py: instead of asking "can the actor's
network decode this heatmap encoding", ask "how many bits of information does this encoding
actually carry, and how efficiently, per channel spent". Same oracle setup as decoderTest.py
(no environment -- a random number gets encoded into N polynomial channels, we try to recover
it), reused directly via decoderTest's own encode()/draw_params()/pad_channels() so this stays
in sync with the real MultiPolynomialInverse math instead of re-deriving it by hand.

Two numbers, per N:

  1. "bits conveyed" -- treat the decode residual as if it were additive Gaussian noise on
     top of the true value, and compute the Shannon capacity of that implied channel:
         bits = 0.5 * log2(1 + var(y) / mse)
     This is the standard way to turn an MSE into a single comparable "bits of information
     that got through" number. It's an APPROXIMATION (the residual isn't necessarily
     Gaussian), not a rigorous entropy estimate -- treat it as a consistent yardstick for
     comparing configs against each other, not an absolute bit count.

  2. "encoding cost" -- each channel is a float64 value, so N channels cost N*64 raw bits to
     store/transmit, regardless of how much of that is actually informative. bits conveyed /
     encoding cost is the efficiency ratio: how much of what's being spent on channels is
     actually getting useful information through, vs redundant/wasted capacity. THIS is the
     MDL-flavored comparison -- a heatmap that needs many channels to convey the same bits as
     a leaner one is a more wasteful (longer-description) encoding of the same information.

Also reports the classic two-part MDL code length (bits to describe the decoder's residual
error, `L_residual`, plus a rough `L_model` = param_count * 32 bits) for completeness, but
since every N in the sweep uses the SAME decoder architecture (in_dim padded to MAX_N, same
hidden width), L_model is constant across the sweep and doesn't differentiate -- L_residual
(equivalently, bits conveyed) is the term that actually varies with N here.

Esp. the polynomial heatmaps (MultiPolynomialInverse) for now -- extending to other
HeatMap/MultiHeatMap types later just means swapping what builds X in evaluate_N().
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import decoderTest as dt
from PPO import MLP


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_VALUES = list(range(1, 11))    # polynomial degrees to sweep -- every integer 1-10 (was a
                                  # hand-picked subset [1,2,3,5,7,10] that skipped 4,6,8,9,
                                  # which left gaps like N=6 with no exact matching bits value
                                  # for the experiment2.py (a,b) decode-difficulty pairings
                                  # that actually use it). Complete range so any (a,b) pair
                                  # tested on the RL side has real MDL data, not interpolation.
MAX_N = max(N_VALUES)            # every config pads to this in_dim (decoderTest's ZeroMap-
                                  # equivalent technique) so decoder capacity/L_model stays
                                  # fixed across the sweep -- isolates "does raising N change
                                  # how much information the ENCODING carries" from "does the
                                  # decoder just have more/fewer weights to work with".
NOISE_SCALE = 2.0                # matches MultiPolynomialInverse.noiseScale (hardcoded 2 in
                                  # Utilities/MultiHeatMap.py) -- this is the REAL production
                                  # noise level, not an arbitrary choice.
LO, HI = 0.0, 20.0                # continuous oracle range -- general encoding-capacity
                                  # characterization, not tied to any one RL run's narrow
                                  # output range (e.g. OptimalActionTarget's {0,1,2,3}).
N_SAMPLES = 20000
REGEN_EVERY = 1                  # fresh polynomial per sample -- matches the RL env's
                                  # per-episode pregen() (see decoderTest.py's docstring for
                                  # why this, not a fixed polynomial, is the honest setting).

HIDDEN_DIM = 128                 # matches the real PPO actor's MLPh128x3
LR = 1e-3
WEIGHT_DECAY = 9e-3               # same tuned value as decoderTest.py
EPOCHS = 1200                    # was 500 -- decoderTest.py's own N=7 run needed ~1200 to
                                  # fully converge; the N>=5 decline seen at 500 epochs was
                                  # partly still-converging, not a hard ceiling. Matching
                                  # decoderTest's proven budget instead of guessing higher.
BATCH_SIZE = 256
VAL_FRAC = 0.2
SEEDS = [0, 1, 2, 3, 4]           # 5 independent trials -- each reseeds numpy AND torch (see
                                  # main()), so dataset draws, polynomial draws, MLP init, and
                                  # minibatch order all vary together per trial. Reports
                                  # mean+-std per (N, probe) instead of trusting a single run.
BITS_PER_CHANNEL = 64             # channels are float64 (see decoderTest.DTYPE) -- raw
                                  # storage/transmission cost per channel, used for the
                                  # encoding-cost / efficiency comparison.
BITS_PER_PARAM = 32               # rough L_model proxy: float32-equivalent cost per decoder
                                  # weight. Constant across N here (see module docstring).


def _bits_conveyed(y_true: np.ndarray, y_pred: np.ndarray):
    """Gaussian-channel-capacity proxy -- see module docstring. Returns (bits, mse)."""
    mse = float(np.mean((y_true - y_pred) ** 2))
    var = float(np.var(y_true))
    snr = var / max(mse, 1e-12)
    bits = 0.5 * np.log2(1.0 + snr)
    return bits, mse


def _mdl_residual_bits_per_sample(mse: float) -> float:
    """Two-part MDL residual term: differential entropy (in bits) of N(0, mse) -- the cost
    of describing the leftover error under a Gaussian noise model."""
    return 0.5 * np.log2(2 * np.pi * np.e * max(mse, 1e-12))


def evaluate_N(N, rng):
    """Build the dataset for this N (via decoderTest's encode/draw_params, padded to MAX_N),
    fit a linear probe and the real-actor-architecture MLP, and return both probes' MDL
    numbers. Mirrors decoderTest.run_for_N's structure, minus the epoch-by-epoch printing."""
    y = rng.uniform(LO, HI, size=N_SAMPLES).astype(np.float32)

    n_blocks = int(np.ceil(N_SAMPLES / REGEN_EVERY))
    if n_blocks <= 1:
        w, nz = dt.draw_params(N, NOISE_SCALE, rng)
    else:
        wb, nzb = dt.draw_params(N, NOISE_SCALE, rng, size=n_blocks)
        block_id = np.arange(N_SAMPLES) // REGEN_EVERY
        w, nz = wb[block_id], nzb[block_id]
    X, clip_frac = dt.encode(y, w, nz, N, NOISE_SCALE)
    X = dt.pad_channels(X, MAX_N)

    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - x_mean) / x_std

    perm = rng.permutation(len(Xn))
    n_val = int(len(Xn) * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    print(f"\n=== N={N} (padded to {MAX_N}, {clip_frac:.2%} clipped) ===")

    # Linear probe: cheapest possible decoder, establishes the affine-shortcut floor.
    A = np.concatenate([Xn[train_idx], np.ones((len(train_idx), 1), np.float32)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y[train_idx], rcond=None)
    B = np.concatenate([Xn[val_idx], np.ones((len(val_idx), 1), np.float32)], axis=1)
    lin_pred = B @ coef
    lin_bits, lin_mse = _bits_conveyed(y[val_idx], lin_pred)
    lin_params = A.shape[1]

    # Real-actor-architecture MLP: the actual decode capability that matters for training.
    Xtr = torch.as_tensor(Xn[train_idx], device=device, dtype=torch.float64)
    ytr = torch.as_tensor(y[train_idx], device=device, dtype=torch.float64)
    Xval = torch.as_tensor(Xn[val_idx], device=device, dtype=torch.float64)
    yval_t = torch.as_tensor(y[val_idx], device=device, dtype=torch.float64)

    model = MLP(in_dim=MAX_N, hidden=HIDDEN_DIM, out=1).to(device=device, dtype=torch.float64)
    mlp_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    n_train = Xtr.shape[0]
    for epoch in range(EPOCHS):
        model.train()
        idx = torch.randperm(n_train, device=device)
        for b in range(0, n_train, BATCH_SIZE):
            batch = idx[b:b + BATCH_SIZE]
            pred = model(Xtr[batch]).squeeze(-1)
            loss = loss_fn(pred, ytr[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        mlp_pred = model(Xval).squeeze(-1).cpu().numpy()
    mlp_bits, mlp_mse = _bits_conveyed(y[val_idx], mlp_pred)

    def _row(name, bits, mse, params):
        cost = N * BITS_PER_CHANNEL
        l_residual = N_SAMPLES * _mdl_residual_bits_per_sample(mse)
        l_model = params * BITS_PER_PARAM
        print(f"  {name:>6}: bits/sample={bits:.3f}  bits/channel={bits/N:.3f}  "
              f"encoding_cost={cost}b  efficiency={bits/cost:.4f}  "
              f"L_model={l_model:.0f}b  L_residual={l_residual:.0f}b  mse={mse:.4f}")
        return dict(N=N, name=name, bits=bits, bits_per_channel=bits / N,
                    encoding_cost=cost, efficiency=bits / cost,
                    l_model=l_model, l_residual=l_residual, mse=mse)

    lin_row = _row("linear", lin_bits, lin_mse, lin_params)
    mlp_row = _row("MLP", mlp_bits, mlp_mse, mlp_params)
    return lin_row, mlp_row


def main():
    # (N, probe_name) -> list of that probe's row dict from each seed's evaluate_N call.
    by_config = {(N, name): [] for N in N_VALUES for name in ("linear", "MLP")}

    for seed in SEEDS:
        print(f"\n{'#'*24} SEED {seed} {'#'*24}")
        # Reseed BOTH numpy and torch per trial (not just once globally) -- otherwise only
        # the dataset draws would vary across "trials" while MLP init/minibatch order stayed
        # identical every time, understating the real run-to-run spread.
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        for N in N_VALUES:
            lin_row, mlp_row = evaluate_N(N, rng)
            by_config[(N, "linear")].append(lin_row)
            by_config[(N, "MLP")].append(mlp_row)

    print(f"\n=== Summary across {len(SEEDS)} seeds: MDL-style comparison across N={N_VALUES} "
          f"(oracle range [{LO},{HI}], noise_scale={NOISE_SCALE}, epochs={EPOCHS}) ===")
    print(f"{'N':>3}  {'probe':>6}  {'bits/sample':>17}  {'bits/channel':>15}  "
          f"{'cost(b)':>8}  {'efficiency':>17}")
    for N in N_VALUES:
        for name in ("linear", "MLP"):
            rows = by_config[(N, name)]
            bits = np.array([r["bits"] for r in rows])
            bpc = np.array([r["bits_per_channel"] for r in rows])
            eff = np.array([r["efficiency"] for r in rows])
            cost = rows[0]["encoding_cost"]
            print(f"{N:>3}  {name:>6}  {bits.mean():>8.3f} +- {bits.std():>5.3f}  "
                  f"{bpc.mean():>7.4f} +- {bpc.std():>5.4f}  {cost:>8}  "
                  f"{eff.mean():>8.4f} +- {eff.std():>6.4f}")


if __name__ == "__main__":
    main()
