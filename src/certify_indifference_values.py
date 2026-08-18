"""Certify whether specific (k_fixed, k_variable, M) triples are true
reward indifference points - WITHOUT running find_indifference_reward's
search phase at all. This is certify_reward() only: given a CANDIDATE M
you already have in mind, it runs the same rigorous anytime-valid
sequential test find_indifference_reward()'s own verify step uses,
checking whether P(goes for fixed) at that EXACT M lands inside
(lo, hi) - default (0.40, 0.60) - with (1-alpha) confidence.

Why this exists: testing "is reward compensation for complexity
multiplicative?" - i.e. does M(a, c) == M(a, b) * M(b, c) for some
intermediate b - requires certifying a SPECIFIC candidate M (the product
of two other, already-measured M's), not searching for whatever M
happens to be right for that (a, c) pair on its own. find_indifference_
reward()'s search phase would just find its OWN best M for (a, c), which
tells you nothing about whether that particular multiplicative
prediction holds. certify_reward() already does exactly the right thing
here (a rigorous, sequential, from-scratch test AT a fixed x) - this
file just makes it easy to queue up a whole list of (k_fixed, k_variable,
M) checks and run them unattended, each with its own detail CSV, the same
overnight-batch shape as run_indifference_batch.py.

CHECKS below is a tuple of (k_fixed, k_variable, M) triples - e.g. to
test whether M(1,3) is well-approximated by M(1,2) * M(2,3), using two
values you already measured elsewhere:
    CHECKS = (
        (1, 3, 1.0299717639322616 * 1.0498731572965168),
    )
Add as many checks as you want - there's no fixed count, and each check
is fully independent (one rejected/inconclusive/errored check has no
effect on any other).

BONFERRONI_CORRECT (default True): if you run several checks in one call
and want the FAMILY-WISE false-accept probability across the whole batch
bounded by `alpha` (not just each individual check's own alpha - see
find_indifference_reward.py's module docstring, "alpha-splitting across
attempts", for the identical reasoning applied there to verify()'s
retries), this divides `alpha` by the number of checks before running
any of them. Turn it off if you'd rather each check just use the flat
alpha on its own (e.g. because you consider each check its own
standalone question, not part of one combined claim).

Outputs - same folder-per-run convention as run_indifference_batch.py:
every call to run_checks() creates ONE NEW SUBFOLDER inside
EVAL_LOGS_DIR (named from the wall-clock start time by default, or
CHECKS_DIR_NAME below for a fixed name), containing:
    checks_summary.csv                        - one row per check
    k{K_FIXED}_k{K_VARIABLE}_M{M}_certify_log.csv  - that check's full
                                 per-run detail (same CSV columns
                                 find_indifference_reward()'s own certify
                                 phase logs - see that module's docstring)
so nothing from one check's run collides with another's, and a run
killed partway through still leaves every already-finished check's
summary row and detail CSV intact.

Every certify_reward() hyperparameter (lo/hi/alpha/target_n/max_runs/...,
every trainPPO.train() knob) is set ONCE in COMMON_KWARGS below and reused
for every check, same as run_indifference_batch.py's COMMON_KWARGS.
save_model defaults to False here too (throwaway runs, never reloaded -
see find_indifference_reward.py's docstring for why).

Run:  python3 certify_indifference_values.py
"""
import csv
import os
import traceback
from datetime import datetime

from find_indifference_reward import certify_reward, _RunCsvLogger

# Fixed subfolder name inside EVAL_LOGS_DIR for this run's outputs - leave
# None to auto-name it from the time this script starts (see module
# docstring); set a fixed string here instead for a predictable path.
CHECKS_DIR_NAME = None

# See module docstring for what this does and why it defaults to True.
BONFERRONI_CORRECT = True

# --- the (k_fixed, k_variable, M) triples to certify, in order ---
CHECKS = (
    (1, 3, 1.0299717639322616 * 1.0498731572965168),
)

# --- every certify_reward() keyword argument (its own knobs, plus every
# trainPPO.train() knob forwarded via train_kwargs), set once and reused
# for every check above - same names/defaults as certify_reward's own
# signature (see find_indifference_reward.py's docstring for what each
# does). ---
COMMON_KWARGS = dict(
    # --- certify_reward's own config ---
    lo=0.40,
    hi=0.60,
    alpha=0.05,
    target_n=200,
    max_runs=5000,
    stall_patience=5,
    hits_per_run=500,
    g=4,
    value_fixed=1.0,
    incorrect_reward=0.0,
    base_seed=10_000_000,
    # --- normal training parameters (mirrors trainPPO.train()'s full
    # signature, minus `groups`/`seed`/`label` - set internally per run) ---
    n_envs=8,
    total_timesteps=200_000,
    progress_bar=False,
    log_training_data=False,
    log_interval=None,
    print_final_summary=False,
    device="cpu",
    verbose=0,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=512,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    net_arch_pi=(64, 32),
    net_arch_vf=(64, 32),
    weight_decay=0.0,
    actor_weight_decay=None,
    critic_weight_decay=None,
    ppo_kwargs=None,
    eval_episodes=1,
    periodic_eval_freq=None,
    periodic_eval_episodes=100,
    weights_dir="weights",
    eval_logs_dir="eval_logs",
    # --- output config ---
    save_model=False,  # throwaway runs, never reloaded - see find_indifference_reward.py's docstring
)

# certify_reward's OWN keyword arguments - everything else in
# COMMON_KWARGS is a trainPPO.train() knob forwarded via train_kwargs.
_CERTIFY_ARG_NAMES = {
    "lo", "hi", "alpha", "target_n", "max_runs", "stall_patience",
    "hits_per_run", "g", "value_fixed", "incorrect_reward", "base_seed",
}

SUMMARY_FIELDNAMES = [
    "k_fixed", "k_variable", "M_tested", "alpha_used",
    "certify_status", "certified",
    "n_runs", "stalled_runs", "mean", "ci_lo", "ci_hi",
    "n_fixed_total", "n_variable_total", "csv_path", "error",
]


def _check_label(k_fixed, k_variable, M):
    return f"k{k_fixed}_k{k_variable}_M{M:.6f}"


def _write_summary(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def run_checks(checks=CHECKS, common_kwargs=None, checks_dir_name=None,
               bonferroni_correct=None):
    """Certify each (k_fixed, k_variable, M) triple in `checks`, in
    order, with every output written inside one new subfolder of
    EVAL_LOGS_DIR created for this call - see module docstring. Returns
    the list of summary row dicts (same rows written to the summary CSV)
    - one per check, in the same order as `checks`, whether that check
    accepted/rejected/errored.

    A single check raising (e.g. a genuinely pathological M) does NOT
    stop the batch - the traceback is printed, that check's row records
    certify_status="error"/the exception message, and the loop moves on
    to the next check. Same reasoning as run_indifference_batch.py's
    run_batch().
    """
    common_kwargs = dict(common_kwargs if common_kwargs is not None else COMMON_KWARGS)
    if bonferroni_correct is None:
        bonferroni_correct = BONFERRONI_CORRECT

    eval_logs_dir = common_kwargs.pop("eval_logs_dir", "eval_logs")
    certify_kwargs = {k: v for k, v in common_kwargs.items() if k in _CERTIFY_ARG_NAMES}
    train_kwargs = {k: v for k, v in common_kwargs.items() if k not in _CERTIFY_ARG_NAMES}

    base_alpha = certify_kwargs.pop("alpha", 0.05)
    if bonferroni_correct and len(checks) > 1:
        alpha_used = base_alpha / len(checks)
    else:
        alpha_used = base_alpha

    if checks_dir_name is None:
        checks_dir_name = CHECKS_DIR_NAME
    if checks_dir_name is None:
        checks_dir_name = f"certify_checks_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checks_dir = os.path.join(eval_logs_dir, checks_dir_name)
    os.makedirs(checks_dir, exist_ok=True)
    train_kwargs["eval_logs_dir"] = checks_dir
    print(
        f"=== check outputs -> {checks_dir}/  "
        f"(alpha_used={alpha_used:.5f} per check"
        + (f", Bonferroni-corrected across {len(checks)} checks" if bonferroni_correct and len(checks) > 1 else "")
        + ") ==="
    )

    summary_path = f"{checks_dir}/checks_summary.csv"
    summary_rows = []

    total = len(checks)
    for i, (k_fixed, k_variable, M) in enumerate(checks, start=1):
        label = _check_label(k_fixed, k_variable, M)
        csv_path = f"{checks_dir}/{label}_certify_log.csv"

        print(
            f"=== [{i}/{total}] certifying M={M:.6f} for "
            f"(k_fixed={k_fixed}, k_variable={k_variable}) -> {csv_path} ==="
        )

        error = ""
        result = None
        csv_logger = _RunCsvLogger(csv_path)
        check_train_kwargs = dict(train_kwargs)
        check_train_kwargs["label"] = label
        check_train_kwargs["print_eval_summary"] = False
        try:
            result = certify_reward(
                M, k_fixed, k_variable,
                train_kwargs=check_train_kwargs,
                csv_logger=csv_logger, iteration=1,
                alpha=alpha_used,
                **certify_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            error = repr(exc)
            traceback.print_exc()
        finally:
            csv_logger.close()

        if result is not None:
            certify_status = result["status"]
            certified = certify_status == "accepted"
            n_runs = result["n_runs"]
            stalled_runs = result["stalled_runs"]
            mean = result["mean"]
            ci_lo = result["history"][-1]["ci_lo"] if result["history"] else None
            ci_hi = result["history"][-1]["ci_hi"] if result["history"] else None
            n_fixed_total = result["n_fixed_total"]
            n_variable_total = result["n_variable_total"]
        else:
            certify_status = "error"
            certified = False
            n_runs = stalled_runs = mean = ci_lo = ci_hi = None
            n_fixed_total = n_variable_total = None

        print(
            f"=== [{i}/{total}] finished: status={certify_status}  "
            f"certified={certified}  mean={mean}  CI=({ci_lo}, {ci_hi}) ==="
        )

        summary_rows.append({
            "k_fixed": k_fixed,
            "k_variable": k_variable,
            "M_tested": M,
            "alpha_used": alpha_used,
            "certify_status": certify_status,
            "certified": certified,
            "n_runs": n_runs,
            "stalled_runs": stalled_runs,
            "mean": mean,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_fixed_total": n_fixed_total,
            "n_variable_total": n_variable_total,
            "csv_path": csv_path,
            "error": error,
        })
        _write_summary(summary_path, summary_rows)

    print(f"Checks summary -> {summary_path}")
    return summary_rows


if __name__ == "__main__":
    run_checks()
