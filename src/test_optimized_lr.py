"""
Runs 10 seeds at trial 15's winning LR settings from the hit-rate search
(actorLr=4.7134e-3, criticLr=6.1245e-4), with left_reward overridden to
200 -- testing whether this LR setting, tuned/confirmed at left_reward=
1000, generalizes to a harder, less lopsided reward ratio.

All other settings are pulled directly from run_optuna_search.py's
FIXED_KWARGS (imported, not retyped), so this exactly reproduces trial
15's actual training conditions except for the one deliberate change.

Each seed's result is printed the same way run_training() always prints
it (its own internal "Left target: X, Right target: Y, No reward: Z"
line) -- this script adds only a header line per seed for readability,
the same way run_optuna_confirm.py labels its own per-candidate runs.
"""

from run_training import run_training
from run_optuna_search import FIXED_KWARGS

N_SEEDS = 10

# Trial 15's exact winning params (see the confirmation run's results).
ACTOR_LR = 0.0047134491951152544
CRITIC_LR = 0.000612447061450724

# Reuse FIXED_KWARGS exactly as the search used it, overriding only
# left_reward. Weights aren't saved by default (FIXED_KWARGS already has
# save_weights=False) -- flip that here if you want to keep any of these
# 10 runs' weights; note run_training() always writes to the same fixed
# path, so saving more than one seed's weights would just overwrite the
# previous one unless you also change the save paths per seed.
RUN_KWARGS = dict(FIXED_KWARGS)
RUN_KWARGS["left_reward"] = 200

if __name__ == "__main__":
    for seed_idx in range(N_SEEDS):
        print(f"\n=== Seed {seed_idx + 1}/{N_SEEDS} "
              f"(actorLr={ACTOR_LR:.4e}, criticLr={CRITIC_LR:.4e}, left_reward={RUN_KWARGS['left_reward']}) ===")

        run_training(
            criticLr=CRITIC_LR,
            actorLr=ACTOR_LR,
            lstmLr=CRITIC_LR,  # matches the convention used throughout the search/confirmation
            lstmLrFloor=RUN_KWARGS["criticLrFloor"],
            **RUN_KWARGS,
        )

    RUN_KWARGS = dict(FIXED_KWARGS)
    RUN_KWARGS["left_reward"] = 40
    
    for seed_idx in range(N_SEEDS):
            print(f"\n=== Seed {seed_idx + 1}/{N_SEEDS} "
                  f"(actorLr={ACTOR_LR:.4e}, criticLr={CRITIC_LR:.4e}, left_reward={RUN_KWARGS['left_reward']}) ===")
    
            run_training(
                criticLr=CRITIC_LR,
                actorLr=ACTOR_LR,
                lstmLr=CRITIC_LR,  # matches the convention used throughout the search/confirmation
                lstmLrFloor=RUN_KWARGS["criticLrFloor"],
                **RUN_KWARGS,
            )

    RUN_KWARGS = dict(FIXED_KWARGS)
    RUN_KWARGS["left_reward"] = 122.5

    for seed_idx in range(N_SEEDS):
            print(f"\n=== Seed {seed_idx + 1}/{N_SEEDS} "
                  f"(actorLr={ACTOR_LR:.4e}, criticLr={CRITIC_LR:.4e}, left_reward={RUN_KWARGS['left_reward']}) ===")
    
            run_training(
                criticLr=CRITIC_LR,
                actorLr=ACTOR_LR,
                lstmLr=CRITIC_LR,  # matches the convention used throughout the search/confirmation
                lstmLrFloor=RUN_KWARGS["criticLrFloor"],
                **RUN_KWARGS,
            )