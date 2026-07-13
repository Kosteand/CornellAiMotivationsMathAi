from run_training import run_training


if __name__ == "__main__":
    left_count, right_count = run_training(
        # core PPO / optimization hyperparameters
        criticLr=0.0003,  # make this 5 e-6 or so
        actorLr=0.0001,
        criticLrFloor=3e-5,
        actorLrFloor=1e-5,
        nUpdates=5000,
        nStepsPerUpdate=512,
        ppo_epochs=4,
        clip_eps=0.2,
        gamma=0.99,
        lam=0.95,
        beginEntropy=0.15,
        endEntropy=0.05,
        # environment / reward shaping
        step_penalty=0.1,
        left_reward=1000,
        right_reward=10,
        max_steps=500,
        min_steps=500,
        step_decay=20,
        # run settings / debug flags
        useCProfiler=False,
        useTorchProfiler=False, # use to test performance, don't use with check_for_NaN_errors
        validate_args_flag_param=True,
        check_for_NaN_errors=False,
        load_weights=False,
        save_weights=True,
    )

    print(f"Left target hits: {left_count},\nRight target hits: {right_count},\nMisses: {100-left_count-right_count}")