from midpoint_search import certify_candidate

if __name__ == "__main__":
    result = certify_candidate(
        x_star=122.5,
        target_hits=100,
        lo=0.40,
        hi=0.60,
        alpha=0.05,
        max_runs=300,
        fixed_kwargs={
            "right_reward": 10,
            "nUpdates": 5000,
            "nStepsPerUpdate": 512,
            "max_steps": 500,
            "min_steps": 500,
            "step_penalty": 0.0,
            "early_stop_patience": 200,
            "early_stop_min_updates": 500,
        },
    )

    print("\n=== CERTIFICATION RESULT ===")
    print(result)