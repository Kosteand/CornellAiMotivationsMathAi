import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os
import pytensor.tensor as pt

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
    log_transform_x: bool = True,
    # Model structure
    n_inducing: int = 30,           # number of inducing points = monotone grid
    # MCMC settings
    tune: int = 1000,
    draws: int = 1000,
    chains: int = 4,
    cores: int = 4,
    target_accept: float = 0.95,
    prediction_samples: int = 200,
    # Priors for GP hyperparameters
    mean_c_sigma: float = 2.0,
    ls_beta: float = 1.0,
    eta_sigma: float = 2.0,
    # Priors for overdispersion
    log_phi_mu: float = 1.0,
    log_phi_sigma: float = 1.0,
    # Prior for monotone increments and starting value
    increment_sigma: float = 1.0,
    g0_sigma: float = 2.0,
    # Jitter
    jitter: float = 1e-6,
    random_seed: int = 42,
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
    params = {
        "csv_path": os.path.join(_HERE, "p_curve_data", "margin_vs_margin_harder_variable.csv"),
        "output_dir": os.path.join(_HERE, "outputs", "margin_vs_margin_harder_variable"),
        "log_transform_x": True,
        "n_inducing": 30,
        "tune": 1000,
        "draws": 1000,
        "chains": 4,
        "cores": 1,
        "target_accept": 0.99,
        "prediction_samples": 200,
        "mean_c_sigma": 2.0,
        "ls_beta": 1.0,
        "eta_sigma": 2.0,
        "log_phi_mu": 1.0,
        "log_phi_sigma": 1.0,
        "increment_sigma": 1.0,
        "g0_sigma": 2.0,
        "jitter": 1e-6,
        "random_seed": 42,
    }

    trace, mu_mean, mu_lower, mu_upper = fit_beta_binomial_monotonic_gp(**params)
    print("Fitting complete.")