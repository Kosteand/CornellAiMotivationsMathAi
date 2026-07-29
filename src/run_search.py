from midpoint_search import find_candidate_x_star

if __name__ == "__main__":
    result = find_candidate_x_star(
        x_bounds=(1200/11, 200),
        n_search_points=6,
        hits_per_point=100,
        n_seeds_per_point=3,
        fixed_kwargs={
            "right_reward": 10,
            "nUpdates": 5000,
            "nStepsPerUpdate": 512,
            "max_steps": 500,
            "min_steps": 500,   # == max_steps, so step_decay never fires
            "step_penalty": 0.0,
            "early_stop_patience": 200,
            "early_stop_min_updates": 1000,
        },
    )

    print("\n=== SEARCH RESULT ===")
    print(f"Candidate x* (M): {result['x_star']:.4g}")
    print(f"Fitted beta:      {result['beta']:.4g}")
    print(f"Fit SSE:          {result['sse']:.4g}")
    print(f"xs:               {list(result['xs'])}")
    print(f"y_hats:           {list(result['y_hats'])}")