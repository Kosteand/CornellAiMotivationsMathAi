import numpy as np
from scipy.optimize import differential_evolution

# ==========  Fixed parameters  ==========
g = 4
n = 3
s = 2.0
powers = np.arange(1, n + 1)   # [1, 2, 3]

# ==========  Vectorized success probability  ==========
def success_probability(a, n_samples=100000000):
    """
    Vectorized Monte Carlo estimate of P(linear classifier picks correct row).
    """
    # --- Signal row (gradient G = 1) ---
    # Generate V for signal: V_k = 1 + U_k - mean(U)
    U_sig = np.random.rand(n_samples, n)
    V_sig = 1 + U_sig - U_sig.mean(axis=1, keepdims=True)
    
    # Generate W for signal: W = n * Dirichlet(1,...,1)
    E_sig = np.random.exponential(1, size=(n_samples, n))
    W = n * E_sig / E_sig.sum(axis=1, keepdims=True)
    
    # Base and features for signal
    B = s * V_sig + W
    x_sig = B ** powers          # shape: (n_samples, n)
    score_sig = np.dot(x_sig, a) # shape: (n_samples,)
    
    # --- Noise rows (gradient G = 0, three rows per sample) ---
    # Generate V for all 3 noise rows at once: shape (n_samples, g-1, n)
    U_noise = np.random.rand(n_samples, g - 1, n)
    V_noise = 1 + U_noise - U_noise.mean(axis=2, keepdims=True)
    
    # Features for noise: C = s * V_noise (no W term)
    C = s * V_noise
    x_noise = C ** powers        # shape: (n_samples, g-1, n)
    scores_noise = np.dot(x_noise, a)  # shape: (n_samples, g-1)
    
    # For each sample, find the highest score among the 3 noise rows
    max_noise_score = scores_noise.max(axis=1)  # shape: (n_samples,)
    
    # Success if signal score beats all noise scores
    correct = (score_sig > max_noise_score)
    return np.mean(correct)

# ==========  Optimisation  ==========
def neg_success(a):
    # Differential evolution minimises; we negate the probability
    return -success_probability(a, n_samples=80000)  # 80k samples per eval

# Bounds for coefficients (loose, safe bounds)
bounds = [(-5.0, 5.0)] * n

# Run global optimisation
print("Optimising linear weights... (this takes 1–2 minutes)")
result = differential_evolution(
    neg_success, 
    bounds, 
    maxiter=50,      # number of generations
    popsize=15,      # population size
    seed=42,
    disp=True        # shows progress
)

best_a = result.x
best_p = -result.fun

print(f"\nOptimal weight vector a: {best_a}")
print(f"Maximum success probability p* = {best_p:.4f}")