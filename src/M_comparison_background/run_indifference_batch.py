"""Run find_indifference_reward() overnight across a whole batch of
(k_fixed, k_variable) pairs - one run per pair, each writing its own
per-run detail CSV (everything find_indifference_reward() itself logs -
see that module's docstring) plus a single running summary CSV (M, beta,
certified, status - one row per pair, rewritten after every pair
finishes) all into ONE NEW SUBFOLDER this script creates inside
EVAL_LOGS_DIR for this run - e.g.
    eval_logs/indifference_batch_20260810_061200/
        batch_summary.csv
        k_fixed1_k_variable3_run_log.csv
        k_fixed3_k_variable13_run_log.csv
        ...
so every overnight run's outputs are self-contained in their own folder
and never collide with (or get mixed in with) another run's - including
the incidental per-run ppo_{label}_eval.csv files trainPPO.train() itself
writes, which also land inside that same subfolder (this script points
find_indifference_reward's own eval_logs_dir at it too). The subfolder is
named from the wall-clock time this script started, by default - pass
batch_dir_name=... to run_batch() (or set BATCH_DIR_NAME below) for a
fixed, predictable name instead. If this gets killed partway through an
overnight run, every pair that already finished is still there in both
the summary and its own detail CSV inside that run's folder; nothing
already-completed is lost.

PAIRS below is a tuple of (k_fixed, k_variable) tuples - e.g.
    PAIRS = ((1, 3), (3, 13), (13, 20))
runs three calls to find_indifference_reward(): the first with the FIXED
target's difficulty k=1 and the VARIABLE target's difficulty k=3, the
second k_fixed=3/k_variable=13, the third k_fixed=13/k_variable=20 - same
(k_fixed, k_variable) argument order find_indifference_reward() itself
takes. Add or remove pairs freely; there's no fixed count.

Not MarginGroup-only: GROUP_FACTORY below controls what "k_fixed"/
"k_variable" actually mean and what Group type gets built from them - see
find_indifference_reward.py's module docstring and its own
`group_factory` parameter for the full explanation of why this is a safe,
drop-in swap (nothing in the search/certify statistics looks at what kind
of Group produced them). GROUP_FACTORY=None (the default) reproduces this
script's original behavior EXACTLY: a MarginGroup with delta=1/k for each
side, and PAIRS entries are plain k integers. To use a different Group
type, set GROUP_FACTORY to a `(g, difficulty, value) -> Group` callable
and make PAIRS entries whatever shape that factory expects - e.g.:

    from groups import AlternatingGroup
    from Utilities.bandit_env import HeatmapGroup

    # AlternatingGroup: difficulty is k, same as MarginGroup's default,
    # just a different Group subclass.
    GROUP_FACTORY = lambda g, k, value: AlternatingGroup(g=g, k=k, value=value)
    PAIRS = ((1, 3), (3, 13))

    # HeatmapGroup: difficulty is a (noise_scale, n) tuple, since that
    # group needs two numbers instead of one - PAIRS entries become
    # ((noise_scale_fixed, n_fixed), (noise_scale_variable, n_variable)).
    GROUP_FACTORY = lambda g, spec, value: HeatmapGroup(
        g=g, noise_scale=spec[0], n=spec[1], value=value,
    )
    PAIRS = ((((0.5, 3), (1.5, 3)), ((1.0, 4), (2.0, 4))))

    # Mixing group TYPES within one pair (one side MarginGroup, the other
    # HeatmapGroup) needs a factory that can tell the two apart - tag each
    # side's own spec with which Group it wants (see mixed_group_factory
    # and the actual GROUP_FACTORY/PAIRS below, which use exactly this
    # pattern): a spec is `("margin", k)`, `("margin", k, s)`, or
    # `("heatmap", noise_scale, n)`, dispatched on its first element. This
    # is what's actually wired up below by default, since it's the only
    # shape flexible enough to cover every one of this script's current
    # PAIRS (some pure-MarginGroup, some mixed, some pure-HeatmapGroup)
    # with one factory. (As of 2026-08-13, "margin_scaled" was renamed to
    # plain "margin" - it was always the same MarginGroup spec, just an
    # unnecessary second name for s != 1.0; see mixed_group_factory below.)

    # Scaled MarginGroup: `("margin", k, s)` upscales EVERYTHING about a
    # MarginGroup by s - the random backdrop is Uniform[0, s) instead of
    # Uniform[0, 1), and the margin is s/k instead of 1/k, so the RELATIVE
    # difficulty (margin as a fraction of the backdrop) stays exactly the
    # same for a given k while the absolute numbers involved scale up or
    # down with s. Use this to test whether upscaling changes anything
    # (e.g. weight_norm, M) that a purely-relative quantity like k
    # shouldn't affect if the network is truly scale-invariant:
    #
    #     PAIRS = ((("margin", 9, 1.0), ("margin", 9, 10.0)),)
    #
    # runs k=9 at s=1 against k=9 at s=10 - same relative margin (1/9) both
    # times, only the absolute scale of the observation differs. s=1.0
    # IS the plain, unscaled MarginGroup case (delta=1/k, s=1) - `("margin",
    # k)` (2-tuple, no s) is accepted as shorthand for exactly this.

Every hyperparameter find_indifference_reward() accepts (search config,
certify config, shared env config, every trainPPO.train() knob) is set
ONCE below in COMMON_KWARGS and reused identically for every pair in this
run - edit COMMON_KWARGS to change them, the same way you'd pass them
directly to find_indifference_reward(). save_model and fit_exponential
both default to False here too (see find_indifference_reward's own
docstring for why) - set fit_exponential=True in COMMON_KWARGS if you
want the exponential-curve comparison fit logged for every pair in this
batch.

Each pair's own CSV is named from the pair itself
({batch_dir}/k_fixed{K_FIXED}_k_variable{K_VARIABLE}_run_log.csv, with
each side's spec sanitized into a filesystem-safe string - see
_safe_spec_str - so plain int/float k's format exactly as before, e.g.
"3", and non-numeric specs like HeatmapGroup's (noise_scale, n) tuples
format as e.g. "0.5-3" instead of a raw, filesystem-unsafe tuple repr) so
pairs never collide with each other. `label` (used for the transient
weights/eval_logs filenames trainPPO.train() itself writes per run) is
set the same way, so those don't collide across pairs either.

Run:  python3 run_indifference_batch.py
"""
import csv
import os
import traceback
from datetime import datetime

import torch

from find_indifference_reward import find_indifference_reward
from Utilities.bandit_env import MarginGroup, HeatmapGroup

# Fixed subfolder name for this batch's outputs, inside EVAL_LOGS_DIR -
# leave None to auto-name it from the time this script starts (see module
# docstring); set a fixed string here instead if you want a predictable
# path to point other tooling at.
BATCH_DIR_NAME = None


def mixed_group_factory(g, spec, value):
    """`(g, difficulty, value) -> Group` factory that dispatches on a tag
    in `spec[0]` rather than assuming one Group type - lets a single
    factory (and a single PAIRS list) mix MarginGroup and HeatmapGroup
    pairs freely, side by side, run to run, or even within the SAME pair
    (one side MarginGroup, the other HeatmapGroup) - see module docstring.
    `spec` is `("margin", k)`, `("margin", k, s)`, `("margin", k, s, err)`,
    or `("heatmap", noise_scale, n)`.

    `("margin", k, s)` is a MarginGroup with everything upscaled by `s`:
    the per-option random backdrop is Uniform[0, s) instead of the default
    Uniform[0, 1), and the margin itself is `s / k` instead of `1 / k` -
    i.e. delta stays proportional to the backdrop's own scale (same k =>
    same RELATIVE margin, s/k over a s-wide backdrop, as 1/k over a
    1-wide one), so k keeps meaning the same thing it always has while s
    controls the absolute scale everything sits at. This needs no changes
    to MarginGroup itself - `s` has been a constructor parameter there all
    along (default 1.0); the 2-tuple `("margin", k)` is just shorthand for
    the s=1.0 special case of this, with delta=1/k. Use
    `("margin", k, 1.0)` to spell that out explicitly. (As of 2026-08-13,
    "margin_scaled" was renamed to plain "margin" - it was always this
    exact same spec shape, just an unnecessary second name for s != 1.0.)

    `("margin", k, s, err)` additionally sets MarginGroup's `err` - the
    per-episode probability that the REWARDED correct option is NOT the
    one the proxy (argmax of the observation) points to (see
    Utilities/bandit_env.py's MarginGroup docstring). Omitting it (the
    2- and 3-tuple forms above) defaults to err=0.0, i.e. the original,
    error-free MarginGroup behavior - every margin spec anywhere in this
    project before err existed is implicitly err=0.0.
    """
    kind, params = spec[0], spec[1:]
    if kind == "margin":
        if len(params) == 1:
            (k,) = params
            s, err = 1.0, 0.0
        elif len(params) == 2:
            k, s = params
            err = 0.0
        else:
            k, s, err = params
        return MarginGroup(g=g, k=k, value=value, s=s, err=err)
    if kind == "heatmap":
        noise_scale, n = params
        return HeatmapGroup(g=g, noise_scale=noise_scale, n=n, value=value)
    raise ValueError(f"unknown group spec kind {kind!r} in {spec!r}")


# `(g, difficulty, value) -> Group` callable controlling what Group type
# gets built for each side of every pair below - None would reproduce
# this script's original MarginGroup(delta=1/k)-only behavior exactly
# (plain int k's, no tag needed); mixed_group_factory (the current
# default) instead dispatches per-side on a `("margin", k)`/
# `("heatmap", noise_scale, n)` tag, since PAIRS below mixes both Group
# types - see mixed_group_factory's own docstring and the module
# docstring's "Not MarginGroup-only" section.
GROUP_FACTORY = mixed_group_factory

# --- the pairs to run, in order - (k_fixed, k_variable) ---
#
# Three groups of pairs, each testing a different question (see the
# conversation this batch came from for the full rationale):
#
# 1. k_fixed LARGER than k_variable (pure MarginGroup) - these are the
#    exact REVERSES of three already-certified pairs ((2,9)->M=1.4040,
#    (5,25)->M=1.3386, (16,18)->M=1.0189), directly testing the symmetry
#    prediction M(b,a) ~= 1/M(a,b).
#
# 2. One side MarginGroup, the other HeatmapGroup(n=3) - chosen from each
#    side's actual measured weight_norm_actor (MarginGroup at
#    weight_decay=0; HeatmapGroup at n=3 from the noise_scale sweep) to
#    span both directions plus a near-balanced pair in each ordering, so
#    M's relationship to weight-norm ratio can be checked ACROSS Group
#    types, not just within MarginGroup's own k's.
#
# 3. Both sides HeatmapGroup(n=3) at various noise_scale - a spread of
#    gap sizes and absolute positions across the already-swept
#    noise_scale range, to fit weight-norm-ratio vs M purely within the
#    HeatmapGroup family (the analogous test to MarginGroup's k-pairs).
PAIRS = (
    # --- 2026-08-13 batch #5: pure HeatmapGroup(n=3) <-> HeatmapGroup(n=3)
    # tests, spanning noise_scale in [0.05, 2.5] (all values already have
    # weight_norm/tau data in weight_norm_data.csv at 0.05 resolution - no
    # new sweep needed, just new M measurements). Goal: the question that
    # motivated this batch is "does noise_scale - heatmap's analogue of
    # margin's s, since it's the magnitude/scale knob rather than
    # the structural-difficulty knob n - explain M as well as (or better
    # than) the trained weight_norm ratio?" The only prior data (7 pairs
    # from batch_summary_6) couldn't really answer this: 3 of the 7 pairs
    # have noise_scale=0.0 on one side (undefined ratio, since noise_scale
    # ratio requires dividing by it), and every pair sits on the same flat
    # weight_norm/M plateau (M within 0.02 of 1.0 for 6/7 pairs), so there
    # was almost no variance in the outcome to explain either way.
    #
    # This batch deliberately keeps BOTH sides nonzero (so noise_scale
    # ratio is always well-defined) and separates RATIO from GAP
    # (variable - fixed), since margin's earlier same-ratio-
    # different-base tests showed ratio-alone models can miss an
    # absolute-scale effect - the same design applies here.
    #
    # 1-3: SAME ratio (2x), three different absolute bases (0.3->0.6,
    # 1.1->2.2, 1.25->2.5 - the largest base achievable at ratio 2 within
    # the noise_scale<=2.5 ceiling). If noise_scale_ratio alone predicts
    # M, these three should land close together; if M drifts with base,
    # ratio alone isn't sufficient (same logic as the margin
    # same-ratio-different-base batch).
    (("heatmap", 0.3, 3), ("heatmap", 0.6, 3)),
    (("heatmap", 1.1, 3), ("heatmap", 2.2, 3)),
    (("heatmap", 1.25, 3), ("heatmap", 2.5, 3)),

    # 4-5: SAME gap (1.0), two different bases/ratios (0.2->1.2 [ratio 6],
    # 1.5->2.5 [ratio 1.67]). If GAP (not ratio) is what actually predicts
    # M, these two should land close together instead of #1-3's triplet -
    # lets us tell gap-driven from ratio-driven effects apart directly,
    # the same way the k=13 vs k=21 "matched weight_norm" tests did for
    # margin.
    (("heatmap", 0.2, 3), ("heatmap", 1.2, 3)),
    (("heatmap", 1.5, 3), ("heatmap", 2.5, 3)),

    # 6-7: SAME gap (2.0) as the one existing large-gap anchor
    # (heatmap0.0,3 -> heatmap2.0,3, M=0.821, the biggest M-deviation seen
    # in any heatmap<->heatmap pair so far) but shifted OFF zero, at two
    # different ratios (0.4->2.4 [ratio 6], 0.05->2.05 [ratio 41]). Lets
    # us tell whether that big deviation was really about the gap=2.0
    # itself, or specifically about one side being noise_scale=0 (a
    # qualitatively different "no noise at all" regime, not just a very
    # small scale) - noise_scale_ratio couldn't even be computed for the
    # original 0.0 anchor, so this is the only way to get a ratio-based
    # read on that same gap.
    (("heatmap", 0.4, 3), ("heatmap", 2.4, 3)),
    (("heatmap", 0.05, 3), ("heatmap", 2.05, 3)),

    # 8: extreme ratio (25x, 0.1->2.5) and large gap (2.4) at once - the
    # widest span achievable within the noise_scale<=2.5 ceiling with
    # neither side at exactly 0. Gives one point far outside the cluster
    # of small-ratio/small-gap pairs above, useful for constraining the
    # slope of any noise_scale_ratio-based fit rather than just its
    # intercept.
    (("heatmap", 0.1, 3), ("heatmap", 2.5, 3)),

    # 9-10: small ratio/small gap, but at bases NOT already covered by the
    # 7 existing pairs (which only ever used bases at or below 1.0 on the
    # low side) - (2.0->2.5 [ratio 1.25, gap 0.5]) and (0.05->0.5 [ratio
    # 10, gap 0.45]). Extends coverage toward the high end of the allowed
    # noise_scale range and toward a smaller absolute base than any
    # existing nonzero pair, without repeating any of the 7 already-
    # measured combinations.
    (("heatmap", 2.0, 3), ("heatmap", 2.5, 3)),
    (("heatmap", 0.05, 3), ("heatmap", 0.5, 3)),
)

# --- MODEL HYPERPARAMETERS (added 2026-08-14, per direct instruction) -
# everything about the policy/value NETWORK itself, as opposed to the
# training LOOP around it (learning_rate, n_steps, etc., set further down
# in COMMON_KWARGS). Change these here to run this whole batch with a
# different model instead of editing find_indifference_reward.py or
# trainPPO.py directly - every knob below is forwarded straight through
# find_indifference_reward() to trainPPO.train() (see that module's
# docstring for the full explanation of each).
#
# MODEL_NET_ARCH_PI / MODEL_NET_ARCH_VF - model SIZE: hidden layer widths
# for the actor ("pi") and critic ("vf") MLPs. (64, 32) is this project's
# long-standing default (two hidden layers, 64 then 32 units); e.g.
# (128, 64) or (32,) would make the model bigger/smaller.
MODEL_NET_ARCH_PI = (64, 32)
MODEL_NET_ARCH_VF = (64, 32)

# MODEL_ACTIVATION_FN - the nonlinearity between hidden layers, as a
# torch.nn.Module CLASS (not an instance). None keeps SB3's own MlpPolicy
# default (torch.nn.Tanh) - what every run before this option existed
# used. Other options: torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.GELU,
# ... (any torch.nn activation module works).
MODEL_ACTIVATION_FN = None

# MODEL_OPTIMIZER_CLASS - the torch.optim optimizer CLASS (not an
# instance). None keeps this project's own AdamW default (see
# trainPPO.train's docstring for why AdamW over SB3's own Adam default).
# Other options: torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop,
# ... (any torch.optim optimizer works, though ones with a different
# constructor signature than Adam/AdamW may not accept `weight_decay`/
# `lr` the same way - check before using one here alongside a nonzero
# MODEL_WEIGHT_DECAY below).
MODEL_OPTIMIZER_CLASS = None

# MODEL_WEIGHT_DECAY / MODEL_ACTOR_WEIGHT_DECAY / MODEL_CRITIC_WEIGHT_DECAY
# - L2 penalty applied by the optimizer above. Leave the actor/critic
# variants as None (the default) to apply MODEL_WEIGHT_DECAY uniformly to
# every parameter; set either one to give the actor and critic their own
# separate decay instead (see trainPPO.train's docstring for the
# actor/critic-split mechanics).
MODEL_WEIGHT_DECAY = 0.0
MODEL_ACTOR_WEIGHT_DECAY = None
MODEL_CRITIC_WEIGHT_DECAY = None

# MODEL_POLICY_KWARGS_EXTRA - catch-all dict for any OTHER
# ActorCriticPolicy keyword not already covered above (e.g. `ortho_init`,
# `log_std_init`) - merged into the model's policy_kwargs, overriding the
# options above on conflict. None (the default) adds nothing extra.
MODEL_POLICY_KWARGS_EXTRA = None

# --- every find_indifference_reward() keyword argument, set once and
# reused for every pair above. See find_indifference_reward's own
# docstring (and find_candidate_reward's/certify_reward's) for what each
# of these does - these are exactly the same names/defaults, just
# collected here instead of typed out at every call site. ---
COMMON_KWARGS = dict(
    verify=True,
    # --- search (phase 1) config ---
    search_range=None,
    hits_per_point=500,
    search_alpha=0.05,
    target_n_per_point=40,
    min_seeds_per_point=3,
    max_seeds_per_point=40,
    bracket_tol=None,
    m_stable_tol=None,
    m_stable_window=3,
    max_points=200,
    # --- certify (phase 2) config - only used since verify=True above ---
    lo=0.40,
    hi=0.60,
    alpha=0.05,
    target_n=200,
    max_runs=5000,
    stall_patience=5,
    hits_per_certify_run=500,
    max_search_iterations=10,
    # --- shared env config ---
    g=4,
    value_fixed=1.0,
    incorrect_reward=0.0,
    base_seed=0,
    # --- normal training parameters (mirrors trainPPO.train()'s full
    # signature, minus `groups`/`seed`/`label` - label is set per-pair
    # below, seed is set internally per run) ---
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
    net_arch_pi=MODEL_NET_ARCH_PI,
    net_arch_vf=MODEL_NET_ARCH_VF,
    activation_fn=MODEL_ACTIVATION_FN,
    optimizer_class=MODEL_OPTIMIZER_CLASS,
    weight_decay=MODEL_WEIGHT_DECAY,
    actor_weight_decay=MODEL_ACTOR_WEIGHT_DECAY,
    critic_weight_decay=MODEL_CRITIC_WEIGHT_DECAY,
    policy_kwargs_extra=MODEL_POLICY_KWARGS_EXTRA,
    ppo_kwargs=None,
    eval_episodes=1,
    periodic_eval_freq=None,
    periodic_eval_episodes=100,
    weights_dir="weights",
    eval_logs_dir="eval_logs",
    # --- output config ---
    save_model=False,  # see find_indifference_reward's docstring - throwaway runs, never reloaded
    fit_exponential=False,  # comparison diagnostic only, off by default - see module docstring
)

# One row per pair, rewritten (not appended) after every pair finishes -
# always reflects every pair completed so far, even if the batch is
# killed or crashes partway through.
SUMMARY_FIELDNAMES = [
    "k_fixed", "k_variable", "label", "csv_path",
    "M", "beta", "certified", "status", "error",
]


def _safe_spec_str(spec):
    """Filesystem-safe string for one side's difficulty spec. Plain
    numbers stringify exactly as before (e.g. 3 -> "3", matching this
    script's original int-only labels byte-for-byte), so existing
    MarginGroup-based folder/file names are unaffected. A tuple/list
    (e.g. HeatmapGroup's (noise_scale, n)) joins its elements with "-"
    (e.g. (0.5, 3) -> "0.5-3"). Anything else falls back to str() with
    every non-alphanumeric character (other than ".", "_", "-") replaced
    by "_", so a custom group_factory can use whatever spec shape it
    wants without ever producing a path-breaking filename."""
    if isinstance(spec, (int, float)):
        return str(spec)
    if isinstance(spec, (tuple, list)):
        return "-".join(_safe_spec_str(s) for s in spec)
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(spec))


def _pair_label(k_fixed, k_variable):
    return f"k_fixed{_safe_spec_str(k_fixed)}_k_variable{_safe_spec_str(k_variable)}"


def _write_summary(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def run_batch(pairs=PAIRS, common_kwargs=None, batch_dir_name=None, group_factory=None):
    """Run find_indifference_reward() once per (k_fixed, k_variable) pair
    in `pairs`, in order, with every output (each pair's own detail CSV,
    the running batch summary CSV, and trainPPO.train()'s own incidental
    per-run eval CSVs) written inside ONE NEW SUBFOLDER of EVAL_LOGS_DIR
    created for this call - see module docstring. Returns the list of
    summary row dicts (same rows written to the summary CSV) - one per
    pair, in the same order as `pairs`, whether that pair succeeded or
    raised.

    A single pair raising (e.g. a genuinely pathological x_star, or an
    environment error) does NOT stop the batch - the traceback is printed
    and that pair's summary row records status="error"/the exception
    message, and the loop moves on to the next pair. That's deliberate
    for an unattended overnight run: one bad pair losing you every pair
    after it would defeat the entire point of queuing them up together.

    group_factory: defaults to GROUP_FACTORY (module-level, itself
    defaulting to None = plain MarginGroup(delta=1/k)) if not given here
    directly - see module docstring for how to point this at a different
    Group type.
    """
    common_kwargs = dict(common_kwargs if common_kwargs is not None else COMMON_KWARGS)
    eval_logs_dir = common_kwargs.get("eval_logs_dir", "eval_logs")
    if group_factory is None:
        group_factory = GROUP_FACTORY

    if batch_dir_name is None:
        batch_dir_name = BATCH_DIR_NAME
    if batch_dir_name is None:
        batch_dir_name = f"indifference_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir = os.path.join(eval_logs_dir, batch_dir_name)
    os.makedirs(batch_dir, exist_ok=True)
    common_kwargs["eval_logs_dir"] = batch_dir
    print(f"=== batch outputs -> {batch_dir}/ ===")

    summary_path = f"{batch_dir}/batch_summary.csv"
    summary_rows = []

    total = len(pairs)
    for i, (k_fixed, k_variable) in enumerate(pairs, start=1):
        label = _pair_label(k_fixed, k_variable)
        csv_path = f"{batch_dir}/{label}_run_log.csv"

        print(
            f"=== [{i}/{total}] starting (k_fixed={k_fixed}, k_variable={k_variable}) "
            f"-> {csv_path} ==="
        )

        kwargs = dict(common_kwargs)
        kwargs["label"] = label
        kwargs["csv_path"] = csv_path
        kwargs["group_factory"] = group_factory

        status = "ok"
        error = ""
        M = beta = certified = None
        try:
            result = find_indifference_reward(k_fixed, k_variable, **kwargs)
            M = result["M"]
            beta = result["beta"]
            certified = result["certified"]
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            status = "error"
            error = repr(exc)
            traceback.print_exc()

        print(
            f"=== [{i}/{total}] finished (k_fixed={k_fixed}, k_variable={k_variable}): "
            f"M={M}  beta={beta}  certified={certified}  status={status} ==="
        )

        summary_rows.append({
            "k_fixed": k_fixed,
            "k_variable": k_variable,
            "label": label,
            "csv_path": csv_path,
            "M": M,
            "beta": beta,
            "certified": certified,
            "status": status,
            "error": error,
        })
        _write_summary(summary_path, summary_rows)

    print(f"Batch summary -> {summary_path}")
    return summary_rows


if __name__ == "__main__":
    run_batch()
