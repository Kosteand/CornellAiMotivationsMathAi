"""Plotting helpers for run_trainPPO_sweep.py's output CSVs.

plot_sweep_summary(...)
    The (k, hit_rate) curve - final eval accuracy vs k, one point per run.
    This is "where each run landed."

plot_sweep_train_log(...)
    Per-k rolling-average training accuracy curves - one line per k,
    showing HOW each run got there over the course of its own training
    (fast/slow rise, noisy, early plateau, etc.), not just the final
    number. Colored continuously by k rather than a legend, since a sweep
    can span dozens of k values.

fit_learning_curves(...)
    Fits a few candidate parametric learning-curve shapes (exponential
    approach to an asymptote, logistic, power-law approach) to a SINGLE
    k's rolling-average curve - ignoring the first `skip_first` episodes,
    which are too noisy to fit meaningfully - overlays the fits on that
    k's plot_sweep_train_log plot, and prints each fit's R^2, RMSE, and
    per-parameter standard errors.

All three functions take a csv_path (defaulting to the paths
run_trainPPO_sweep.py writes to), an optional output_path (saves a PNG
there if given), and show (whether to also pop up an interactive window -
default True). All three return the created matplotlib Figure so it can
be inspected/saved/composed further if wanted.

Run directly (`python3 plot_sweep_results.py`) to plot both sweep graphs,
using the default eval_logs/ paths. Curve fitting is only exposed as a
function for now, not a CLI flag - call fit_learning_curves(...) directly.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.optimize import curve_fit


def _as_bool_int(series: pd.Series) -> pd.Series:
    """Coerce a 'correct' column to 0/1 ints, regardless of whether it was
    read back as real bools, or as "True"/"False" strings (both are valid
    depending on how a given CSV was written)."""
    if series.dtype == object:
        series = series.map(lambda v: str(v).strip().lower() == "true")
    return series.astype(int)


def plot_sweep_summary(
    csv_path: str = "eval_logs/sweep_summary.csv",
    output_path: str | None = None,
    show: bool = True,
):
    """Plot hit_rate vs k from a run_trainPPO_sweep.py summary CSV.

    Styled to match plot_sweep_train_log: same figure size, grid, and a
    viridis colormap over k via a colorbar rather than a legend. No
    markers - the line itself is colored continuously along its length
    (each segment colored by its k), via a LineCollection, so there's a
    color gradient without discrete dots at each point.
    """
    df = pd.read_csv(csv_path).sort_values("k")
    ks = df["k"].to_numpy()
    hit_rates = df["hit_rate"].to_numpy()

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=ks.min(), vmax=ks.max())

    fig, ax = plt.subplots(figsize=(9, 6))

    points = np.array([ks, hit_rates]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2)
    lc.set_array(ks[:-1])  # color each segment by its left endpoint's k
    ax.add_collection(lc)
    ax.set_xlim(ks.min(), ks.max())
    ax.set_ylim(-0.02, 1.02)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="k")

    ax.set_xlabel("k")
    ax.set_ylabel("hit rate (eval accuracy)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Sweep summary: hit rate vs k\n({csv_path})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_sweep_train_log(
    csv_path: str = "eval_logs/sweep_train_log_all_runs.csv",
    output_path: str | None = None,
    window: int = 200,
    show: bool = True,
    k: float | list[float] | None = None,
    max_points_per_line: int = 300,
):
    """
    Plot a rolling-average training-accuracy curve for every k in a
    run_trainPPO_sweep.py master training-log CSV.

    The x axis is each row's position WITHIN its own k's block of rows
    (0, 1, 2, ...), i.e. training episode index since THAT k's run
    started - not an absolute row number in the concatenated file - so
    every k's curve starts at x=0 regardless of where its rows happen to
    land in the file.

    k: restrict the plot to one k value (a single number) or several (a
    list). None (the default) plots every k in the file. Matched with a
    small floating-point tolerance since k is stored as a float. Raises
    ValueError if none of the requested k values are present.

    max_points_per_line: after computing the rolling mean, downsample
    each line to at most this many evenly-spaced points before plotting.
    This is purely a rendering optimization (the rolling mean already
    smooths the curve, so plotting every single one of e.g. 200,000 raw
    points per line - x59 lines for a big sweep - is redundant and slow
    for matplotlib to draw). Set to None to disable and plot every point.
    """
    # Only k and correct are ever used here - the CSV also has x_*, label_*,
    # action, matched_group columns that would otherwise be parsed and
    # loaded into memory for nothing.
    df = pd.read_csv(csv_path, usecols=["k", "correct"])
    df["correct"] = _as_bool_int(df["correct"])

    all_ks = sorted(df["k"].unique())

    if k is None:
        ks = all_ks
    else:
        wanted = [k] if isinstance(k, (int, float)) else list(k)
        ks = [kv for kv in all_ks if any(abs(kv - w) < 1e-9 for w in wanted)]
        if not ks:
            raise ValueError(
                f"No matching k value(s) for k={wanted} in {csv_path}. "
                f"Available k values: {all_ks}"
            )

    cmap = plt.get_cmap("viridis")
    # A degenerate (single-value) color range would otherwise divide by
    # zero in Normalize - pad it out when there's only one k to plot.
    norm_min, norm_max = min(ks), max(ks)
    if norm_min == norm_max:
        norm_min, norm_max = norm_min - 0.5, norm_max + 0.5
    norm = plt.Normalize(vmin=norm_min, vmax=norm_max)

    # Group once up front instead of re-scanning the whole dataframe with
    # df["k"] == kv inside the loop - that was O(rows * number of k's),
    # which is the actual source of the slowdown on a large sweep (e.g.
    # 59 k's x 200k rows = tens of millions of redundant comparisons).
    grouped = {kv: group for kv, group in df.groupby("k")["correct"]}

    fig, ax = plt.subplots(figsize=(9, 6))
    for kv in ks:
        sub = grouped[kv].reset_index(drop=True)
        rolling = sub.rolling(window=window, min_periods=1).mean()

        x_vals = rolling.index.to_numpy()
        y_vals = rolling.to_numpy()
        if max_points_per_line is not None and len(x_vals) > max_points_per_line:
            idx = np.linspace(0, len(x_vals) - 1, max_points_per_line).astype(int)
            x_vals = x_vals[idx]
            y_vals = y_vals[idx]

        ax.plot(
            x_vals,
            y_vals,
            color=cmap(norm(kv)),
            linewidth=1.0 if len(ks) > 1 else 2.0,
            alpha=0.85 if len(ks) > 1 else 1.0,
            label=f"k={kv}" if len(ks) == 1 else None,
        )

    if len(ks) == 1:
        ax.legend()
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="k")

    ax.set_xlabel(f"training episode within run (rolling mean, window={window})")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.02, 1.02)
    title_suffix = f" (k={ks[0]})" if len(ks) == 1 else ""
    ax.set_title(f"Per-k training accuracy curves{title_suffix}\n({csv_path})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    return fig


def _build_fit_specs(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Build the candidate (function, initial-guess, param-names) triples for
    every learning-curve shape fit_learning_curves knows about, seeded
    from the actual (x, y) data being fit. Rebuilt fresh per call since
    the power-law fit needs an x-offset baked in from THIS data's x.min()
    (learning curves start well after x=0 once the noisy early episodes
    are skipped, and a power law needs a positive base).
    """
    def exp_func(x, L, A, tau):
        # Exponential approach to an asymptote L, starting `A` below it,
        # with time constant tau. Rising curves have A > 0 (accuracy
        # starts low and climbs toward L); tau controls how fast.
        return L - A * np.exp(-x / tau)

    exp_p0 = [float(y[-1]), float(y[-1] - y[0]), max((x[-1] - x[0]) / 3.0, 1.0)]

    def logistic_func(x, L, x0, steepness):
        # Sigmoid rising to asymptote L, centered at x0.
        return L / (1.0 + np.exp(-steepness * (x - x0)))

    logistic_p0 = [
        max(float(y[-1]), 1e-3),
        float(x[len(x) // 2]),
        4.0 / max(float(x[-1] - x[0]), 1.0),
    ]

    x_min = float(x.min())

    def power_func(x, L, A, p):
        # Power-law approach to asymptote L: L - A*(x - x_min + 1)^-p.
        # The + 1 offset (fixed at fit time, not a free parameter) keeps
        # the base positive since x starts at x_min, not 0.
        return L - A * np.power(np.asarray(x, dtype=float) - x_min + 1.0, -p)

    power_p0 = [float(y[-1]), float(y[-1] - y[0]) or 0.1, 0.5]

    return {
        "exponential": (
            exp_func, exp_p0,
            ["L (asymptote)", "A (amplitude)", "tau (time constant)"],
        ),
        "logistic": (
            logistic_func, logistic_p0,
            ["L (asymptote)", "x0 (midpoint)", "steepness"],
        ),
        "power": (
            power_func, power_p0,
            ["L (asymptote)", "A (amplitude)", "p (exponent)"],
        ),
    }


def fit_learning_curves(
    csv_path: str = "eval_logs/sweep_train_log_all_runs.csv",
    k: float | None = None,
    window: int = 200,
    skip_first: int = 1000,
    curves: list[str] | None = None,
    output_path: str | None = None,
    show: bool = True,
):
    """
    Fit a few candidate parametric learning-curve shapes to a SINGLE k's
    rolling-average training-accuracy curve, overlay the fits on top of
    that k's plot_sweep_train_log plot, and print each fit's R^2, RMSE,
    and per-parameter standard errors.

    k is required - fitting one parametric curve to several different
    k's overlaid lines at once wouldn't mean anything; call this once per
    k you want to fit.

    skip_first: drop this many of the EARLIEST episodes before fitting
    (default 1000). Training is too noisy in that opening stretch - the
    policy hasn't settled into any consistent trend yet - and including
    it skews every fit toward explaining transient noise instead of the
    actual approach-to-asymptote shape. The full, un-skipped data is
    still plotted underneath for context; only the FIT itself ignores
    that stretch.

    curves: which candidate shapes to try, a subset of ["exponential",
    "logistic", "power"]. None (default) tries all three.

    Fitting uses the FULL-resolution rolling series (not the
    max_points_per_line-downsampled render plot_sweep_train_log draws) -
    the downsampling in that function is a rendering-speed optimization
    only and would throw away real signal if reused for fitting.
    """
    if k is None:
        raise ValueError("k is required - specify exactly which k's curve to fit.")

    curves = curves or ["exponential", "logistic", "power"]

    # Build (and, for now, keep off-screen) the base single-k plot via the
    # existing function - same styling/legend/title as normal, just with
    # show=False so we can add the fitted curves before anything pops up.
    fig = plot_sweep_train_log(
        csv_path=csv_path, output_path=None, window=window, show=False, k=k,
    )
    ax = fig.axes[0]

    df = pd.read_csv(csv_path, usecols=["k", "correct"])
    df["correct"] = _as_bool_int(df["correct"])
    matches = [kv for kv in df["k"].unique() if abs(kv - k) < 1e-9]
    if not matches:
        raise ValueError(f"k={k} not found in {csv_path}.")
    kv = matches[0]

    rolling = (
        df.loc[df["k"] == kv, "correct"]
        .reset_index(drop=True)
        .rolling(window=window, min_periods=1)
        .mean()
    )
    x_full = rolling.index.to_numpy(dtype=float)
    y_full = rolling.to_numpy(dtype=float)

    if skip_first >= len(x_full):
        raise ValueError(
            f"skip_first={skip_first} skips the entire series "
            f"({len(x_full)} points total) for k={kv} - lower skip_first "
            f"or use a run with more training data."
        )
    x, y = x_full[skip_first:], y_full[skip_first:]

    fit_specs = _build_fit_specs(x, y)
    colors = {"exponential": "tab:red", "logistic": "tab:orange", "power": "tab:purple"}
    x_dense = np.linspace(x.min(), x.max(), 300)

    print(
        f"Fitting learning curves for k={kv} "
        f"(skipped first {skip_first} episodes, {len(x)} points used):"
    )

    for name in curves:
        if name not in fit_specs:
            print(f"  {name}: unknown curve name - choose from {list(fit_specs)}")
            continue

        func, p0, param_names = fit_specs[name]
        try:
            popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=20000)
        except RuntimeError as exc:
            print(f"  {name}: fit FAILED to converge ({exc})")
            continue

        residuals = y - func(x, *popt)
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        param_errs = np.sqrt(np.diag(pcov))

        print(f"  {name}:  R^2 = {r_squared:.4f}   RMSE = {rmse:.4f}")
        for pname, val, err in zip(param_names, popt, param_errs):
            print(f"      {pname} = {val:.5g} +/- {err:.3g}")

        ax.plot(
            x_dense,
            func(x_dense, *popt),
            color=colors.get(name, "black"),
            linewidth=2.0,
            linestyle="--",
            label=f"{name} fit (R^2={r_squared:.3f})",
        )

    ax.legend(fontsize=8)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_fit_vs_k(
    csv_path: str = "eval_logs/sweep_summary.csv",
    output_path: str | None = None,
    show: bool = True,
):
    """
    Plot the exponential learning-curve fit's two meaningful parameters -
    L (asymptote reached, i.e. how much got learned) and tau (time
    constant, i.e. how fast it got there) - against k, from a summary CSV
    like the one run_exponential_fit_sweep.py writes (columns: k,
    fit_status, fit_L, fit_tau, fit_L_err, fit_tau_err, ...).

    Two stacked subplots sharing the k axis:
      - top:    L vs k, with vertical error bars sized by fit_L_err
                (curve_fit's parameter standard error - how tightly the
                asymptote was pinned down by that k's data).
      - bottom: tau vs k, same idea with fit_tau_err. tau is plotted on a
                log y-axis, since learning speed typically varies over
                orders of magnitude across k (a good tanh/margin task
                might learn in a few hundred episodes; a hard one might
                take tens of thousands) - a linear axis would flatten the
                small-k detail into invisibility. Error bars are clipped
                so the lower whisker never crosses into log-scale-illegal
                zero/negative territory (this only clips the DRAWN
                whisker, not the underlying fit_tau_err value).

    Rows where fit_status != "ok" (fit didn't converge, or there wasn't
    enough data to attempt one) are dropped before plotting - there's no
    L/tau to show for those k's. Raises ValueError if that leaves nothing
    to plot.
    """
    df = pd.read_csv(csv_path).sort_values("k")
    df = df[df["fit_status"] == "ok"].dropna(subset=["fit_L", "fit_tau"])
    if df.empty:
        raise ValueError(
            f"No rows with fit_status == 'ok' in {csv_path} - nothing to plot."
        )

    k = df["k"].to_numpy(dtype=float)
    L = df["fit_L"].to_numpy(dtype=float)
    tau = df["fit_tau"].to_numpy(dtype=float)
    # Missing/NaN error columns (e.g. an older summary CSV without them)
    # degrade gracefully to zero-length error bars rather than crashing.
    L_err = df["fit_L_err"].to_numpy(dtype=float) if "fit_L_err" in df else np.zeros_like(L)
    tau_err = df["fit_tau_err"].to_numpy(dtype=float) if "fit_tau_err" in df else np.zeros_like(tau)
    L_err = np.nan_to_num(L_err, nan=0.0)
    tau_err = np.nan_to_num(tau_err, nan=0.0)

    fig, (ax_L, ax_tau) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    # --- L (asymptote / "how much got learned") vs k ---
    ax_L.errorbar(
        k, L, yerr=L_err,
        fmt="o-", color="tab:blue", ecolor="tab:blue", elinewidth=1,
        capsize=3, markersize=4, alpha=0.9,
    )
    ax_L.set_ylabel("L  (fitted asymptote accuracy)")
    ax_L.set_title("Learning-curve fit vs k")
    ax_L.grid(True, alpha=0.3)

    # --- tau (time constant / "how fast it got there") vs k ---
    # Log y-axis: an asymmetric error bar whose lower end would go <= 0
    # is undrawable on a log scale, so the lower whisker length is capped
    # just short of reaching tau itself (a tiny positive floor) - this
    # only affects how far DOWN the whisker is drawn, not tau or
    # fit_tau_err themselves.
    lower_err = np.minimum(tau_err, tau * (1 - 1e-6))
    ax_tau.errorbar(
        k, tau, yerr=[lower_err, tau_err],
        fmt="o-", color="tab:red", ecolor="tab:red", elinewidth=1,
        capsize=3, markersize=4, alpha=0.9,
    )
    ax_tau.set_yscale("log")
    ax_tau.set_xlabel("k")
    ax_tau.set_ylabel("tau  (episodes; log scale)")
    ax_tau.grid(True, alpha=0.3, which="both")

    fig.suptitle(f"({csv_path})", fontsize=9, y=0.995)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    return fig


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot run_trainPPO_sweep.py's output CSVs."
    )
    parser.add_argument(
        "--summary-csv", default="eval_logs/sweep_summary.csv",
        help="Path to the (k, hit_rate) summary CSV.",
    )
    parser.add_argument(
        "--train-log-csv", default="eval_logs/sweep_train_log_all_runs.csv",
        help="Path to the master per-episode training-log CSV.",
    )
    parser.add_argument(
        "--window", type=int, default=200,
        help="Rolling-average window (in episodes) for the train-log plot.",
    )
    parser.add_argument(
        "--k", type=float, nargs="+", default=None,
        help=(
            "Only plot these k value(s) in the train-log plot "
            "(e.g. --k 5 or --k 1 5 10). Default: plot every k."
        ),
    )
    parser.add_argument(
        "--max-points", type=int, default=300,
        help=(
            "Downsample each train-log line to at most this many points "
            "before plotting (rendering speed only - the curve shape is "
            "unaffected since it's already rolling-averaged). Pass 0 to "
            "disable and plot every raw point."
        ),
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Don't open interactive windows, just save the PNGs.",
    )
    parser.add_argument(
        "--plot",
        choices=["summary", "train-log", "fit-vs-k", "both"],
        default="both",
        help=(
            "Which graph(s) to generate: 'summary' for just the (k, "
            "hit_rate) plot, 'train-log' for just the per-episode "
            "training-curve plot, 'fit-vs-k' for the L/tau-vs-k plot "
            "(requires a summary CSV with fit_L/fit_tau columns, e.g. "
            "from run_exponential_fit_sweep.py), or 'both' (default - "
            "summary + train-log only; pass --plot fit-vs-k explicitly)."
        ),
    )
    parser.add_argument(
        "--fit-k", type=float, default=None,
        help=(
            "Fit learning-curve shapes (exponential/logistic/power) to "
            "this single k's rolling-average training curve from "
            "--train-log-csv, overlay them, and print R^2/RMSE/parameter "
            "errors for each. Default: don't fit anything. Requires a "
            "single k, e.g. --fit-k 5."
        ),
    )
    parser.add_argument(
        "--fit-skip-first", type=int, default=1000,
        help=(
            "Drop this many of the earliest episodes before fitting (too "
            "noisy to fit meaningfully). Only used with --fit-k. "
            "Default: 1000."
        ),
    )
    parser.add_argument(
        "--fit-curves", choices=["exponential", "logistic", "power"],
        nargs="+", default=None,
        help=(
            "Which candidate curve shape(s) to fit. Only used with "
            "--fit-k. Default: try all three."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    show = not args.no_show

    # Build every requested figure first with show=False (each function's
    # own plt.show() would otherwise block until THAT window is closed
    # before the next figure even gets created) - then, once everything's
    # built, pop all of them up together with a single plt.show() call
    # below. That way both windows appear at once and neither plot's
    # creation waits on the other being closed.
    if args.plot in ("summary", "both"):
        plot_sweep_summary(
            csv_path=args.summary_csv,
            show=False,
            output_path="eval_logs/sweep_summary_plot.png",
        )
        print("Saved eval_logs/sweep_summary_plot.png")

    if args.plot in ("train-log", "both"):
        if args.fit_k is not None:
            # fit_learning_curves already builds the same base train-log
            # plot internally and overlays the fitted curves on top of it -
            # call that instead of plot_sweep_train_log so we don't draw
            # the same k's line twice in two separate figures.
            fit_learning_curves(
                csv_path=args.train_log_csv,
                k=args.fit_k,
                window=args.window,
                skip_first=args.fit_skip_first,
                curves=args.fit_curves,
                show=False,
                output_path="eval_logs/sweep_train_log_fit_plot.png",
            )
            print("Saved eval_logs/sweep_train_log_fit_plot.png")
        else:
            plot_sweep_train_log(
                csv_path=args.train_log_csv,
                show=False,
                window=args.window,
                max_points_per_line=(args.max_points or None),
                k=args.k,
                output_path="eval_logs/sweep_train_log_plot.png",
            )
            print("Saved eval_logs/sweep_train_log_plot.png")

    if args.plot == "fit-vs-k":
        plot_fit_vs_k(
            csv_path=args.summary_csv,
            show=False,
            output_path="eval_logs/fit_vs_k_plot.png",
        )
        print("Saved eval_logs/fit_vs_k_plot.png")

    if show:
        plt.show()
