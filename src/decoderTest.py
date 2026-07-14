"""Supervised sanity check: can the PPO actor's network DECODE the polynomial encoding?

There is NO environment here — no grid, no coordinates, no SpaceEnv. An oracle emits a
random number, MultiPolynomialInverse encodes it into N polynomial channels, and we
train the same MLP that PPO uses for its actor (plus strong weight decay) to guess the
original number back. Plain regression, square loss.

    oracle:   v ~ random number in [LO, HI]        (continuous -> must GENERALIZE)
    encode:   x = (v * w + noise + 1) ** [1..N]     (the MultiHeatMap poly encoding)
    target:   y = v
    network:  y_hat = MLP(x)                        # same arch as the PPO actor

Goal: make the net GENUINELY invert the polynomial, not find a linear shortcut. The key
fact: MultiPolynomialInverse builds weights with mean 1 and noise with mean 0 ACROSS the N
channels (see its __init__/pregen). So the exact decode is nonlinear but memoryless:

    root each channel:  f_i = channel_i ** (1/power_i) = v*w_i + noise_i + 1
    average them:       mean_i f_i = v*mean(w) + mean(noise) + 1 = v + 1   (nuisance cancels!)
    -> v = mean_i(channel_i ** (1/power_i)) - 1

This holds for ANY draw of w/noise, so it works even when the polynomial is REGENERATED
every sample/episode -- no memory needed. That's why regenerating weights is the right
move, not a problem:

  * Fixed polynomial  -> a linear read already decodes v (~0.99 linear-probe R^2). Shortcut.
  * Regenerated weights (REGEN_EVERY small, as the env does via pregen() each reset) -> a
    single global linear map can't fit all the polynomials, so the linear probe drops (and
    drops further with the max power N). The net is forced onto the genuine nonlinear
    root-average decode, which still recovers v exactly. THIS is genuine decoding.

Levers:
  * REGEN_EVERY -> small regenerates the polynomial often (kills the linear shortcut).
  * N (max power) -> the real difficulty knob. ALL channels [1..N] stay exposed (the decode
    needs them all to cancel noise); raising the MAX power makes the top channels sharper /
    less linearly approximable, so the linear probe falls (N=2: 0.96, 3: 0.87, 5: 0.75,
    12: 0.67) while the exact root-average decode still recovers v.
  * noiseScale is NOT a useful lever. Bigger noise makes channel 0 ALONE a worse read (its
    solo linear R^2 falls, 0.46 -> 0.16), but the full linear probe uses ALL channels and a
    linear combination cancels the mean-0 noise just like the decode does -- so the overall
    linear shortcut doesn't weaken (it even edges up). And with the fixed +1 offset, big
    noise only forces clipping (v*w+noise+1 < 1e-3), which breaks the exact decode. Max
    power N forces the root nonlinearity that linear can't do -- that's the shortcut-killer.

The MLP's val R^2 must clearly BEAT the printed linear-probe R^2 to count as genuine decode.

Sweep N=1..MAX_N using the SAME dummy-map padding technique as experiment2.py's ZeroMap:
each run's real N-channel encoding is padded with always-0 columns out to MAX_N, so the
MLP's in_dim is IDENTICAL across every N in the sweep. That isolates "is this N's encoding
decodable" from "does the network just have more/fewer input weights to work with" --
exactly like padding every SpaceEnv config to the same observation size with ZeroMap.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from PPO import MLP
from Utilities.MultiHeatMap import MultiPolynomialInverse


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_N = 10                   # fixed input width every run pads up to (the ZeroMap-equivalent
                            # dummy-map technique from experiment2.py): the MLP always sees
                            # MAX_N channels, real ones for N and always-0 padding for the rest,
                            # so in_dim never changes across the sweep -- isolates decodability
                            # from capacity, exactly like padding SpaceEnv configs with ZeroMap.
N_VALUES = [1,2,3,4,5,6,7,8,9,10]    # sweep N (max polynomial power/degree) -- restricted to the top
                            # end of the 1..MAX_N range, where the earlier full sweep showed
                            # val R^2 starting to slip (0.9868 -> 0.9683 for N=7..10). THIS is
                            # the difficulty knob: higher max power -> higher-degree roots to
                            # invert -> less linearly approximable, so the linear probe drops
                            # (N=2: 0.96, 3: 0.87, 5: 0.75, 12: 0.67) while the exact decode holds.
MIN_POWER = 1               # keep at 1. ALL of a run's real channels [1..N] must be exposed
                            # for the noise-cancellation decode (mean w=1, mean noise=0 over
                            # N channels); difficulty comes from the MAX power N, not from
                            # hiding low channels.
DTYPE = np.float64          # np.float32 or np.float64 for channels + net. Raw f^N needs
                            # precision: float64's ~16 digits move the numerical wall from
                            # ~N=8 to ~N=16-18 (graceful decay after), so N stays genuinely
                            # harder as it grows -- the honest way to push N, no shortcuts.
NOISE_SCALE = 20.0          # per-channel noise magnitude. Bigger noise degrades a shortcut that
                            # only reads a FEW channels (mean(noise)=0 only cancels across the
                            # FULL N-channel set, so a subset sees an uncanceled residual that
                            # grows with noise) -- while the FULL linear probe and the exact
                            # decode are UNAFFECTED, since both use every channel. The offset
                            # (see encode()) scales as 1+NOISE_SCALE, giving enough headroom
                            # that v*w+noise+offset never clips: decode stays exact (checked to
                            # ~1e-14 max err) at ANY noise level, all the way to 100+.
PARTIAL_PROBE_CHANNELS = 3  # how many of the N real channels the "partial shortcut" probe gets
                            # to see (min(this, N) if N is smaller). This is the metric that
                            # should get WORSE as NOISE_SCALE grows, unlike the full probe.
TORCH_DTYPE = torch.float64 if DTYPE is np.float64 else torch.float32
LO, HI = 0.0, 20.0          # range the oracle draws numbers from
INTEGER = False             # False: continuous oracle -> real generalization to unseen
                            # values. True: integer oracle (also prints exact-int acc).
N_SAMPLES = 20000           # how many numbers the oracle emits
REGEN_EVERY = 1             # regenerate the polynomial every this many inputs. 1 = a fresh
                            # polynomial per number, which KILLS the linear shortcut and
                            # forces the genuine nonlinear (root-average) decode. The env
                            # regenerates per episode (pregen() each reset) -> training
                            # across many episodes poses the same cross-polynomial problem.
                            # Set >= N_SAMPLES for one fixed polynomial (linear-decodable).

# Training
HIDDEN_DIM = 512   # same width as the PPO actor (see experiment2.py)
LR = 1e-3                   # constant, no annealing: an LR schedule was tried and reliably
                            # shrank |w| growth late in training without moving val R^2 at all
                            # (the oscillation it was meant to fix was only ~0.004 R^2 worth of
                            # noise) -- not worth trading away the weight-growth signal for that.
WEIGHT_DECAY = 9e-3        # meaningful is relative to STEP COUNT: cumulative pull-back over a
                            # run = (1-lr*wd)^n_steps. At the short runs used to rule out decay
                            # as an accuracy limiter (~600-1200 epochs), 3e-3 only gave 7-20%
                            # cumulative pull-back -- looked safe but was too weak to matter. At
                            # the current EPOCHS (~4000-5000, ~250-315k steps), the SAME 3e-3
                            # compounds to 47-61% pull-back -- strong enough to actually counter
                            # the |w| drift (loss plateaus but |w| keeps climbing at wd=1e-4,
                            # which only gives ~3% pull-back at this length) without re-fighting
                            # real learning, since 3e-3 was already the value that didn't move
                            # R^2 in the earlier decay-vs-capacity test.
EPOCHS = 5000
BATCH_SIZE = 256
VAL_FRAC = 0.2
SEED = 0
PRINT_EVERY = 100           # epoch-progress print interval within each N's run (kept coarse
                            # since this now runs once per N in the sweep)


class _DummyInner:
    """MultiPolynomialInverse requires an inner map, but we feed it scalars directly
    (not coordinates), so the inner map is never used — this just satisfies the
    constructor and lets us reuse the exact same weight/noise generation."""
    def __init__(self, **kwargs):
        pass

    def map(self, cordArr):
        return 0.0

    def getRange(self):
        return (0.0, 0.0)

    def toString(self):
        return "oracle"


def draw_params(N, noise_scale, rng, size=None):
    """Draw (weights, noise) exactly as MultiPolynomialInverse does in __init__/pregen.
    size=None -> one polynomial (N,); size=M -> M independent polynomials (M, N)."""
    shape = (N,) if size is None else (size, N)
    w = np.abs(rng.random(shape)).astype(DTYPE)
    w = w / w.mean(axis=-1, keepdims=True)
    nz = np.abs(rng.random(shape)).astype(DTYPE)
    nz = (nz - nz.mean(axis=-1, keepdims=True)) * noise_scale
    return w, nz


def encode(values, w, nz, N, noise_scale):
    """x = (v * w + noise + offset) ** [MIN_POWER .. MIN_POWER+N-1]  — mirrors
    MultiPolynomialInverse.mapAll but on scalars, with the exponents shifted up by
    MIN_POWER so the linear (degree-1) shortcut channel isn't handed to the net.
    w/nz broadcast as either one shared polynomial or one per sample.

    offset = 1 + noise_scale (NOT a fixed 1, like the real MultiPolynomialInverse): this
    gives enough headroom that v*w+noise+offset never goes negative regardless of how big
    noise_scale is, so the root-average decode stays exactly exact (no clipping) at ANY
    noise level -- checked to ~1e-14 max error up to noise_scale=100. A fixed offset of 1
    would clip once noise got large, which is genuinely destructive, not just decryptable.

    Returns (channels, clip_frac) -- clip_frac is the fraction of (v*w+noise+offset) values
    that hit the 1e-3 floor BEFORE clipping. With the scaling offset this should stay 0."""
    v = np.asarray(values, dtype=DTYPE)[:, None]               # (M, 1)
    offset = 1.0 + noise_scale
    f = v * w + nz + offset                                     # (M, N)  -- DTYPE all the way
    clip_frac = float((f < 1e-3).mean())
    f = np.clip(f, 1e-3, None)
    powers = np.arange(N) + MIN_POWER                          # [MIN_POWER .. +N-1]
    x = (f ** powers).astype(DTYPE)                            # the f^N blow-up happens HERE,
                                                               # so DTYPE must be set before it
    return x, clip_frac


def pad_channels(X, target_width):
    """ZeroMap-equivalent: pad the (M, N) channel matrix with always-0 columns out to
    target_width. Those columns carry zero information (same as ZeroMap.map always
    returning 0.0 in experiment2.py) -- they exist only to keep the MLP's in_dim fixed
    across every N in the sweep."""
    M, n = X.shape
    if n > target_width:
        raise ValueError(f"N={n} exceeds MAX_N={target_width}")
    if n == target_width:
        return X
    pad = np.zeros((M, target_width - n), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)


def run_for_N(N, rng):
    """Build the dataset and train the fixed-in_dim MLP for one N in the sweep. Returns a
    small dict of final metrics for the cross-N summary table."""
    # Build a poly instance only to grab its toString (authoritative). noiseScale is NOT read
    # from it -- we use NOISE_SCALE (much larger, with a matching offset so it stays exact).
    poly = MultiPolynomialInverse(N=N, map=_DummyInner)
    noise_scale = NOISE_SCALE

    # Oracle: draw random numbers; target is the number itself.
    if INTEGER:
        y = rng.integers(int(LO), int(HI) + 1, size=N_SAMPLES).astype(np.float32)
    else:
        y = rng.uniform(LO, HI, size=N_SAMPLES).astype(np.float32)

    # Encode. Each block of REGEN_EVERY consecutive inputs shares one polynomial (an
    # "episode"); blocks get fresh weights — mirroring the env's per-reset pregen().
    n_blocks = int(np.ceil(N_SAMPLES / REGEN_EVERY))
    if n_blocks <= 1:
        w, nz = draw_params(N, noise_scale, rng)                   # (N,) one fixed poly
    else:
        wb, nzb = draw_params(N, noise_scale, rng, size=n_blocks)  # (n_blocks, N)
        block_id = np.arange(N_SAMPLES) // REGEN_EVERY
        w, nz = wb[block_id], nzb[block_id]                        # (N_SAMPLES, N)
    X, clip_frac = encode(y, w, nz, N, noise_scale)
    # ZeroMap-equivalent padding: extra always-0 columns out to MAX_N so the MLP's in_dim
    # stays fixed across every N in the sweep (isolates decodability from capacity).
    X = pad_channels(X, MAX_N)

    baseline = float(y.var())   # MSE if the net just predicts the mean of y
    powers = list(np.arange(N) + MIN_POWER)
    print(f"\n=== N={N} (real channels={N}, padded to {MAX_N} with {MAX_N - N} zero cols, "
          f"weight_decay={WEIGHT_DECAY:g}, noise_scale={noise_scale:g}) ===")
    print(f"dataset: X={X.shape}  y={y.shape}  target range [{y.min():.3f}, {y.max():.3f}]  "
          f"distinct={len(np.unique(y))}  exponents={powers}  "
          f"blocks={n_blocks} (regen every {REGEN_EVERY})  "
          f"baseline var(y)={baseline:.4f}  encoding={poly.toString()}")
    if clip_frac > 0:
        print(f"WARNING: clip_frac={clip_frac:.4f} -- noise_scale is large enough that "
              f"{clip_frac:.2%} of values hit the 1e-3 floor; the decode is NO LONGER exact "
              f"for those samples (noise has become destructive, not just decryptable).")
    else:
        print(f"clip_frac=0.0000 -- noise stays fully decryptable (root-average decode exact)")

    # Standardize the channels — powers up to N make raw magnitudes wildly different,
    # which wrecks a plain MLP. (The real env does its own getRange normalization.) The
    # padded zero columns have std=0, so x_std falls back to 1e-8 and they standardize to
    # exactly 0 -- same always-0 behavior as ZeroMap after the env's own normalization.
    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - x_mean) / x_std

    # Train / val split — with a continuous oracle, val holds numbers never trained on.
    perm = rng.permutation(len(Xn))
    n_val = int(len(Xn) * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Linear probe: how much of the decode is a trivial AFFINE read of the channels? Fit
    # a least-squares linear map on train, score on val. If this R^2 is already ~1 (MAE
    # already low), the "polynomial" is a shortcut and the MLP below isn't doing genuine
    # nonlinear decoding -- the MLP must clearly BEAT both to count as genuine decoding.
    def _lin_probe(Xa, ya, Xb, yb):
        A = np.concatenate([Xa, np.ones((len(Xa), 1), np.float32)], axis=1)
        coef, *_ = np.linalg.lstsq(A, ya, rcond=None)
        B = np.concatenate([Xb, np.ones((len(Xb), 1), np.float32)], axis=1)
        pred = B @ coef
        r2 = 1.0 - ((yb - pred) ** 2).mean() / yb.var()
        mae = float(np.abs(yb - pred).mean())
        return r2, mae
    lin_r2, lin_mae = _lin_probe(Xn[train_idx], y[train_idx], Xn[val_idx], y[val_idx])
    print(f"linear-probe val R^2 = {lin_r2:.4f}  val MAE = {lin_mae:.4f}  "
          f"(~1 R^2 / low MAE => affine shortcut; the MLP must beat both to be genuine decode)")

    # Partial-channel linear probe: a "shortcut" that only reads a FEW channels instead of
    # all N. mean(noise)=0 only cancels across the FULL channel set, so a subset carries an
    # uncanceled noise residual that grows with noise_scale -- this is the metric that shows
    # bigger noise hurting a partial shortcut while the FULL probe above stays fine.
    k = min(PARTIAL_PROBE_CHANNELS, N)
    part_r2, part_mae = _lin_probe(Xn[train_idx, :k], y[train_idx], Xn[val_idx, :k], y[val_idx])
    print(f"partial-probe (first {k} of {N} real channels) val R^2 = {part_r2:.4f}  "
          f"val MAE = {part_mae:.4f}  (bigger noise_scale should make THIS worse, not the full probe)")

    Xtr = torch.as_tensor(Xn[train_idx], device=device, dtype=TORCH_DTYPE)
    ytr = torch.as_tensor(y[train_idx], device=device, dtype=TORCH_DTYPE)
    Xval = torch.as_tensor(Xn[val_idx], device=device, dtype=TORCH_DTYPE)
    yval = torch.as_tensor(y[val_idx], device=device, dtype=TORCH_DTYPE)

    # Same network as the PPO actor: MLP(in_dim, hidden, out). in_dim is ALWAYS MAX_N.
    model = MLP(in_dim=MAX_N, hidden=HIDDEN_DIM, out=1).to(device=device, dtype=TORCH_DTYPE)
    # AdamW = decoupled weight decay -> the "strong weight reg" knob.
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()   # square loss on how off the guess was

    n_train = Xtr.shape[0]
    final = {}
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

        if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                tr_mse = loss_fn(model(Xtr).squeeze(-1), ytr).item()
                val_pred = model(Xval).squeeze(-1)
                val_mse = loss_fn(val_pred, yval).item()
                val_mae = (val_pred - yval).abs().mean().item()
                # R^2 vs the predict-the-mean baseline: 1.0 = perfect, 0 = no better.
                r2 = 1.0 - val_mse / baseline
                # Weight norm ticker: global L2 norm of all params. Under strong weight
                # decay this settles low; if it keeps climbing, the reg isn't binding.
                weight_norm = torch.sqrt(
                    sum(p.pow(2).sum() for p in model.parameters())
                ).item()
                # Reward-framed loss breakdown (reward = -loss, matching the RL side of
                # this project): how much is attributable to being WRONG about the number
                # vs to the WEIGHT PENALTY. AdamW applies decay directly to the weights
                # (decoupled, not added into the backprop graph), so reg_penalty is the
                # equivalent L2 term (0.5*WEIGHT_DECAY*|w|^2) purely for reporting -- it's
                # not what's literally being differentiated, just the same quantity in units.
                reg_penalty = 0.5 * WEIGHT_DECAY * (weight_norm ** 2)
                reward_from_number = -val_mse
                reward_from_reg = -reg_penalty
                msg = (f"epoch {epoch:4d}  "
                       f"train MSE {tr_mse:.4f}  val MSE {val_mse:.4f}  "
                       f"val MAE {val_mae:.4f}  R^2 {r2:.4f}  |w| {weight_norm:.3f}  "
                       f"reward[number] {reward_from_number:.4f}  reward[reg] {reward_from_reg:.4f}")
                if INTEGER:
                    acc = (val_pred.round().clamp(y.min(), y.max()) == yval).float().mean().item()
                    msg += f"  exact-int acc {acc:.3f}"
                print(msg)
                final = dict(N=N, lin_r2=lin_r2, lin_mae=lin_mae, val_r2=r2,
                             val_mae=val_mae, weight_norm=weight_norm)

    return final


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    results = [run_for_N(N, rng) for N in N_VALUES]

    print(f"\n=== Summary: MLP(in_dim={MAX_N}, weight_decay={WEIGHT_DECAY:g}) "
          f"across N={N_VALUES} ===")
    print(f"{'N':>3}  {'linear R^2':>10}  {'lin MAE':>8}  {'MLP val R^2':>11}  "
          f"{'val MAE':>8}  {'|w|':>8}  {'genuine?':>8}")
    for r in results:
        genuine = "yes" if r["val_r2"] > r["lin_r2"] + 0.05 else "no"
        print(f"{r['N']:>3}  {r['lin_r2']:>10.4f}  {r['lin_mae']:>8.4f}  {r['val_r2']:>11.4f}  "
              f"{r['val_mae']:>8.4f}  {r['weight_norm']:>8.3f}  {genuine:>8}")


if __name__ == "__main__":
    main()
