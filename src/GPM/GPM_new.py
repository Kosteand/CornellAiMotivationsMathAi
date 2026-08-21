import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os
import pytensor.tensor as pt

# ----------------------------------------------------------------------
# USER-SETTABLE OPTIONS
# ----------------------------------------------------------------------
# Everything in this block is meant to be edited directly, rather than
# buried in the __main__ block below or in fit_beta_binomial_monotonic_gp's
# own default arguments - per direct request, ALL user-settable options
# live up here from now on (and should keep landing here for anything
# added to this file in the future), not scattered further down.

# --- which comparison(s) to run ---
# If RUN_ALL_COMPARISONS is True, __main__ runs this fit on EVERY CSV
# already sitting in p_curve_data/ (i.e. every comparison
# sample_p_curve_adaptive.py/run_p_curve_experiments.py has already
# sampled and plot_p_curve_results.py has already graphed - currently 7)
# instead of just DEFAULT_TEST_NAME. Each one gets its own
# outputs/<test_name>/ directory, same convention as the single-test case.
RUN_ALL_COMPARISONS = False
DEFAULT_TEST_NAME = "margin_vs_margin_harder_variable"  # used when RUN_ALL_COMPARISONS is False

# --- shape-model overlay ---
# When True, also fits plot_p_curve_results.py's SHAPE_MODELS (logistic,
# probit, laplace, gennorm, plus whichever else is uncommented there) to
# the SAME data and draws each successfully-fit curve on top of this
# script's own GP plot - a direct visual "does the flexible GP agree
# with any of the simple parametric shapes" check. Optional (default
# off) since it pulls in plot_p_curve_results.py's own imports
# (sklearn, statsmodels, scipy.optimize) and re-fits every shape model,
# neither of which this script needs unless asked for.
OVERLAY_SHAPE_FITS = False

# --- data transformation ---
LOG_TRANSFORM_X = True

# --- model structure ---
N_INDUCING = 30           # number of inducing points = monotone grid

# --- MCMC settings ---
TUNE = 1000
DRAWS = 1000
CHAINS = 4
CORES = 1
TARGET_ACCEPT = 0.99
PREDICTION_SAMPLES = 200

# --- priors for GP hyperparameters ---
MEAN_C_SIGMA = 2.0
LS_BETA = 1.0
ETA_SIGMA = 2.0

# --- priors for overdispersion ---
LOG_PHI_MU = 1.0
LOG_PHI_SIGMA = 1.0

# --- prior for monotone increments and starting value ---
INCREMENT_SIGMA = 1.0
G0_SIGMA = 2.0

# --- jitter / reproducibility ---
JITTER = 1e-6
RANDOM_SEED = 42


# ----------------------------------------------------------------------
# Kernel functions
# ----------------------------------------------------------------------
def matern32_cov_pt(X1, X2, ls, eta):
    """
    PyTensor Matérn 3/2 kernel.
    X1: (n1, 1) tensor, X2: (n2, 1) tensor
    Returns covariance matrix (n1, n2)
    """
    r = pt.abs(X1 - X2.T) / ls
    sqrt3 = pt.sqrt(3.0)
    K = eta**2 * (1.0 + sqrt3 * r) * pt.exp(-sqrt3 * r)
    return K

def matern32_cov_np(X1, X2, ls, eta):
    """
    Numpy Matérn 3/2 kernel.
    X1: (n1, 1) array, X2: (n2, 1) array
    Returns covariance matrix (n1, n2)
    """
    r = np.abs(X1 - X2.T) / ls
    sqrt3 = np.sqrt(3.0)
    K = eta**2 * (1.0 + sqrt3 * r) * np.exp(-sqrt3 * r)
    return K

# ----------------------------------------------------------------------
# Main fitting function
# ----------------------------------------------------------------------
def fit_beta_binomial_monotonic_gp(
    # Data and output
    csv_path: str,
    output_dir: str,
    # Data transformation
    log_transform_x: bool = LOG_TRANSFORM_X,
    # Model structure
    n_inducing: int = N_INDUCING,           # number of inducing points = monotone grid
    # MCMC settings
    tune: int = TUNE,
    draws: int = DRAWS,
    chains: int = CHAINS,
    cores: int = CORES,
    target_accept: float = TARGET_ACCEPT,
    prediction_samples: int = PREDICTION_SAMPLES,
    # Priors for GP hyperparameters
    mean_c_sigma: float = MEAN_C_SIGMA,
    ls_beta: float = LS_BETA,
    eta_sigma: float = ETA_SIGMA,
    # Priors for overdispersion
    log_phi_mu: float = LOG_PHI_MU,
    log_phi_sigma: float = LOG_PHI_SIGMA,
    # Prior for monotone increments and starting value
    increment_sigma: float = INCREMENT_SIGMA,
    g0_sigma: float = G0_SIGMA,
    # Jitter
    jitter: float = JITTER,
    random_seed: int = RANDOM_SEED,
    # Optional shape-model overlay (see OVERLAY_SHAPE_FITS at the top of
    # this file)
    overlay_shape_fits: bool = OVERLAY_SHAPE_FITS,
):
    """
    Fit a monotonic Beta-Binomial GP using a sparse GP with inducing points.
    """

    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # 1. Read and prepare data
    # =========================================================================
    df = pd.read_csv(csv_path)
    required = ['x', 'chose_variable', 'hits_per_run', 'y_hat']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV.")

    x_raw = df['x'].values.astype(float)
    k = df['chose_variable'].values.astype(int)
    N = df['hits_per_run'].values.astype(int)
    y_hat = df['y_hat'].values.astype(float)

    if not np.allclose(k / N, y_hat, rtol=1e-6):
        print("Warning: chose_variable / hits_per_run does not exactly match y_hat.")

    if log_transform_x:
        z = np.log(x_raw)
    else:
        z = x_raw

    mask = np.isfinite(z) & (k >= 0) & (N > 0)
    z = z[mask]
    k = k[mask]
    N = N[mask]
    y_hat = y_hat[mask]
    x_raw = x_raw[mask]

    n_obs = len(z)
    print(f"Number of valid observations: {n_obs}")

    # =========================================================================
    # 2. Build inducing points (monotone grid)
    # =========================================================================
    if n_inducing < 3:
        n_inducing = 3
    if n_inducing >= n_obs:
        n_inducing = max(3, n_obs // 2)

    probs = np.linspace(0.0, 1.0, n_inducing)
    z_inducing = np.quantile(z, probs)
    z_inducing = np.unique(z_inducing)
    n_inducing = len(z_inducing)

    print(f"Using {n_inducing} inducing points.")
    print(f"Inducing range: [{z_inducing.min():.3f}, {z_inducing.max():.3f}]")

    # =========================================================================
    # 3. Define PyMC model
    # =========================================================================
    with pm.Model() as model:
        # ----- GP hyperparameters -----
        mean_c = pm.Normal('mean_c', mu=0.0, sigma=mean_c_sigma)
        ls = pm.HalfCauchy('ls', beta=ls_beta)          # lengthscale
        eta = pm.HalfNormal('eta', sigma=eta_sigma)     # amplitude

        # ----- Monotone inducing values -----
        g0 = pm.Normal('g0', mu=0.0, sigma=g0_sigma)
        delta = pm.HalfNormal('delta', sigma=increment_sigma, shape=n_inducing - 1)

        g_inducing = pm.Deterministic(
            'g_inducing',
            g0 + pm.math.concatenate([
                pm.math.zeros(1),
                pm.math.cumsum(delta)
            ])
        )

        # ----- GP prior on inducing points -----
        X_u = z_inducing[:, None]
        K_uu = matern32_cov_pt(X_u, X_u, ls, eta)
        K_uu = K_uu + pt.eye(n_inducing) * jitter

        pm.Potential(
            'gp_prior_inducing',
            pm.logp(
                pm.MvNormal.dist(mu=pm.math.zeros(n_inducing), cov=K_uu),
                g_inducing
            )
        )

        # ----- Deterministic mapping to observed points (sparse GP mean) -----
        X_obs = z[:, None]
        K_ou = matern32_cov_pt(X_u, X_obs, ls, eta)   # (n_inducing, n_obs)

        # Solve K_uu^{-1} g_inducing
        A = pt.slinalg.solve(K_uu, g_inducing)

        # g_obs = K_ou^T @ K_uu^{-1} @ g_inducing
        g_obs = K_ou.T @ A

        # ----- Link to probability -----
        f_obs = pm.Deterministic('f_obs', mean_c + g_obs)
        mu_obs = pm.Deterministic('mu_obs', pm.math.invlogit(f_obs))

        # ----- Beta-Binomial overdispersion -----
        log_phi = pm.Normal('log_phi', mu=log_phi_mu, sigma=log_phi_sigma)
        phi = pm.Deterministic('phi', pm.math.exp(log_phi))

        alpha = mu_obs * phi
        beta = (1.0 - mu_obs) * phi

        y_obs = pm.BetaBinomial(
            'y_obs',
            alpha=alpha,
            beta=beta,
            n=N,
            observed=k,
        )

    # =========================================================================
    # 4. Sample
    # =========================================================================
    print("Sampling...")
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            return_inferencedata=True,
            random_seed=random_seed,
        )

    # =========================================================================
    # 5. Predictions
    # =========================================================================
    x_pred_raw = np.linspace(x_raw.min(), x_raw.max(), 200)
    z_pred = np.log(x_pred_raw) if log_transform_x else x_pred_raw
    n_pred = len(z_pred)

    posterior = trace.posterior
    total_samples = posterior.dims['chain'] * posterior.dims['draw']
    idx_samples = np.linspace(
        0, total_samples - 1,
        min(prediction_samples, total_samples),
        dtype=int
    )

    def get_flat(var_name):
        vals = posterior[var_name].values
        return vals.reshape(total_samples, *vals.shape[2:])[idx_samples]

    ls_s = get_flat('ls')
    eta_s = get_flat('eta')
    mean_c_s = get_flat('mean_c')
    g_inducing_s = get_flat('g_inducing')

    f_pred_samples = np.zeros((len(idx_samples), n_pred))

    for i in range(len(idx_samples)):
        ls_i = ls_s[i]
        eta_i = eta_s[i]
        mean_i = mean_c_s[i]
        g_u_i = g_inducing_s[i]

        K_uu_i = matern32_cov_np(z_inducing[:, None], z_inducing[:, None], ls_i, eta_i)
        K_uu_i += np.eye(n_inducing) * jitter
        K_pu_i = matern32_cov_np(z_pred[:, None], z_inducing[:, None], ls_i, eta_i)

        # Solve K_uu @ alpha = g_inducing
        alpha_vec = np.linalg.solve(K_uu_i, g_u_i)
        f_pred_samples[i, :] = mean_i + K_pu_i @ alpha_vec

    f_pred_mean = f_pred_samples.mean(axis=0)
    f_pred_lower = np.percentile(f_pred_samples, 2.5, axis=0)
    f_pred_upper = np.percentile(f_pred_samples, 97.5, axis=0)

    # Transform to probability scale
    mu_pred_mean = 1.0 / (1.0 + np.exp(-f_pred_mean))
    mu_pred_lower = 1.0 / (1.0 + np.exp(-f_pred_lower))
    mu_pred_upper = 1.0 / (1.0 + np.exp(-f_pred_upper))

    # =========================================================================
    # 6. Save plot
    # =========================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(x_raw, y_hat, 'o', alpha=0.2, markersize=3, label='Observed y_hat')
    plt.plot(x_pred_raw, mu_pred_mean, 'r-', label='Posterior mean bag bias μ(x)')
    plt.fill_between(
        x_pred_raw,
        mu_pred_lower,
        mu_pred_upper,
        color='red',
        alpha=0.25,
        label='95% credible interval',
    )

    if overlay_shape_fits:
        # Lazy import (only pulled in when this option is actually on -
        # see OVERLAY_SHAPE_FITS at the top of this file) of
        # plot_p_curve_results.py's own shape-fitting machinery, from its
        # sibling location in this same GPM/ folder.
        import sys
        sys.path.insert(0, _HERE)
        from plot_p_curve_results import per_point_summary, fit_shape_diagnostic, SHAPE_MODELS

        points_df = per_point_summary(df)
        shape = fit_shape_diagnostic(points_df)
        shape_models = shape.get("models") or {}
        shape_colors = {
            "logistic": "tab:blue", "probit": "tab:green",
            "laplace": "tab:orange", "gennorm": "tab:purple",
        }
        for spec in SHAPE_MODELS:
            m = shape_models.get(spec["name"])
            if not m or "error" in m:
                continue
            curve = spec["fn"](z_pred, *[m["params"][p] for p in spec["param_names"]])
            is_best = spec["name"] == shape.get("best_model")
            label = spec["name"] + (" (best AIC)" if is_best else "")
            plt.plot(x_pred_raw, curve, color=shape_colors.get(spec["name"]),
                      linewidth=2.0 if is_best else 1.2,
                      linestyle="-" if is_best else "--",
                      label=label, zorder=3)

    plt.xlabel('x (original scale)')
    plt.ylabel('P(choose variable)')
    plt.title('Monotonic sparse GP with Beta-Binomial likelihood')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/beta_binomial_gp_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 7. Save summaries
    # =========================================================================
    summary = az.summary(
        trace,
        var_names=['mean_c', 'ls', 'eta', 'log_phi', 'g0'],
    )
    summary.to_csv(f"{output_dir}/parameter_summary.csv")

    with open(f"{output_dir}/model_info.txt", 'w') as f:
        f.write("Monotonic Beta-Binomial sparse GP with Matérn 3/2 kernel\n")
        f.write("=========================================================\n\n")
        f.write(f"Number of observations: {n_obs}\n")
        f.write(f"Number of inducing points: {n_inducing}\n")
        f.write(f"Inducing points (z = log(x)): {z_inducing}\n")
        f.write(f"Log-transform of x: {log_transform_x}\n\n")
        f.write("Parameter summary (mean ± sd):\n")
        f.write(summary.to_string())

    az.to_netcdf(trace, f"{output_dir}/trace.nc")

    print(f"Done. Outputs saved to {output_dir}")

    return trace, mu_pred_mean, mu_pred_lower, mu_pred_upper


# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
# 2026-08-19: this file moved from src/ into src/GPM/ (its own folder,
# alongside its own outputs/ dir) as part of a repo cleanup. p_curve_data/
# (the shared p-curve pipeline's data, still used by run_p_curve_
# experiments.py/sample_p_curve_adaptive.py/plot_p_curve_results.py too)
# ALSO moved, into src/GPM/p_curve_data/ (alongside this file), since
# everything p_curve-related now lives under GPM/. Paths below are
# resolved relative to THIS FILE's own location (not the current working
# directory), so `python3 GPM_new.py` and `python3 GPM/GPM_new.py` (run
# from src/, or from anywhere else) all find the same files.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_ROOT = os.path.dirname(_HERE)  # src/ - the parent of GPM/

if __name__ == "__main__":
    # Every value plugged into `params` below (other than csv_path/
    # output_dir, which are derived per test name) comes straight from
    # the USER-SETTABLE OPTIONS block at the top of this file - edit
    # there, not here.
    if RUN_ALL_COMPARISONS:
        import glob
        test_names = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(_HERE, "p_curve_data", "*.csv"))
        )
        if not test_names:
            raise SystemExit(f"RUN_ALL_COMPARISONS is True but no CSVs were found in "
                              f"{os.path.join(_HERE, 'p_curve_data')!r}.")
    else:
        test_names = [DEFAULT_TEST_NAME]

    print(f"Running on {len(test_names)} comparison(s): {test_names}")

    for test_name in test_names:
        params = {
            "csv_path": os.path.join(_HERE, "p_curve_data", f"{test_name}.csv"),
            "output_dir": os.path.join(_HERE, "outputs", test_name),
            "log_transform_x": LOG_TRANSFORM_X,
            "n_inducing": N_INDUCING,
            "tune": TUNE,
            "draws": DRAWS,
            "chains": CHAINS,
            "cores": CORES,
            "target_accept": TARGET_ACCEPT,
            "prediction_samples": PREDICTION_SAMPLES,
            "mean_c_sigma": MEAN_C_SIGMA,
            "ls_beta": LS_BETA,
            "eta_sigma": ETA_SIGMA,
            "log_phi_mu": LOG_PHI_MU,
            "log_phi_sigma": LOG_PHI_SIGMA,
            "increment_sigma": INCREMENT_SIGMA,
            "g0_sigma": G0_SIGMA,
            "jitter": JITTER,
            "random_seed": RANDOM_SEED,
            "overlay_shape_fits": OVERLAY_SHAPE_FITS,
        }

        print(f"\n=== {test_name} ===")
        trace, mu_mean, mu_lower, mu_upper = fit_beta_binomial_monotonic_gp(**params)

    print("Fitting complete.")