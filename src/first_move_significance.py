import csv
import math
from collections import defaultdict

ACTIONS = ["up", "down", "left", "right"]


def read_episode_info_csv(path="eval_logs/episode_info.csv"):
    """
    Reads episode_info.csv and returns, per first_action, the (left_count,
    right_count) tally -- misses (target == -1) are excluded entirely,
    same convention used everywhere else in this project.
    """
    left_by_action = defaultdict(int)
    right_by_action = defaultdict(int)

    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            target = int(row[2])
            first_action = row[3] if len(row) > 3 else "none"

            if target not in (0, 1):
                continue  # ignore misses
            if first_action not in ACTIONS:
                continue  # ignore "none"/unrecognized values (e.g. older csv rows)

            if target == 0:
                left_by_action[first_action] += 1
            else:
                right_by_action[first_action] += 1

    return left_by_action, right_by_action


def _norm_cdf(z):
    """Standard normal CDF via math.erf -- no scipy dependency."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_proportion_z_test(x1, n1, x2, n2):
    """
    Two-proportion z-test: is the left-rate for group 1 (x1 lefts out of
    n1 hits) different from group 2 (x2 lefts out of n2 hits)?

    Returns (z, p_value_two_tailed). Returns (None, None) if either group
    has zero hits (test undefined).
    """
    if n1 == 0 or n2 == 0:
        return None, None

    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        # only possible if p_pool is 0 or 1 (every hit in both groups
        # went the same way) -- no variance, so no meaningful test
        return None, None

    z = (p1 - p2) / se
    p_value = 2 * (1 - _norm_cdf(abs(z)))
    return z, p_value


def analyze_first_move_correlation(path="eval_logs/episode_info.csv", alpha=0.05):
    left_by_action, right_by_action = read_episode_info_csv(path)

    total_left = sum(left_by_action.values())
    total_right = sum(right_by_action.values())
    total_hits = total_left + total_right

    if total_hits == 0:
        print("No non-miss rows found in episode_info.csv -- nothing to analyze.")
        return []

    overall_p_left = total_left / total_hits
    bonferroni_alpha = alpha / len(ACTIONS)

    print(f"Overall P(left) across all first moves (misses excluded): "
          f"{overall_p_left:.4f}  ({total_left} left / {total_right} right, "
          f"{total_hits} total hits)\n")

    results = []

    for action in ACTIONS:
        n_left = left_by_action.get(action, 0)
        n_right = right_by_action.get(action, 0)
        n_hits = n_left + n_right

        if n_hits == 0:
            print(f"{action:>5}: no hits recorded -- skipping")
            continue

        p_action = n_left / n_hits

        # Compare this action's group to the REST of the data (all other
        # actions combined) via a two-proportion z-test, rather than to
        # the overall average -- comparing to the overall average would
        # include this group's own data in the reference, which biases
        # the test toward "not significant" and violates the independence
        # assumption the z-test relies on. When an action is a small
        # share of the total data, "rest" and "overall average" are very
        # close anyway.
        rest_left = total_left - n_left
        rest_right = total_right - n_right
        rest_hits = rest_left + rest_right
        rest_p = rest_left / rest_hits if rest_hits > 0 else float("nan")

        z, p_value = two_proportion_z_test(n_left, n_hits, rest_left, rest_hits)

        if p_value is None:
            sig_note = "test undefined (need hits in both this group and the rest)"
        else:
            sig_05 = "YES" if p_value < alpha else "no"
            sig_bonf = "YES" if p_value < bonferroni_alpha else "no"
            sig_note = (f"p={p_value:.4f}  significant at alpha=0.05: {sig_05}  "
                        f"| Bonferroni-corrected (alpha={bonferroni_alpha:.4f}): {sig_bonf}")

        direction = "more" if p_action > rest_p else "less" if p_action < rest_p else "equally"
        print(f"{action:>5}: P(left|{action})={p_action:.4f}  "
              f"(n={n_hits}, {n_left} left / {n_right} right)  "
              f"vs. rest P(left)={rest_p:.4f} (n={rest_hits})  "
              f"-- {direction} likely to go left  |  {sig_note}")

        results.append({
            "action": action,
            "p_action": p_action,
            "n_hits": n_hits,
            "n_left": n_left,
            "n_right": n_right,
            "rest_p": rest_p,
            "rest_n": rest_hits,
            "z": z,
            "p_value": p_value,
            "significant_at_0.05": (p_value is not None and p_value < alpha),
            "significant_bonferroni": (p_value is not None and p_value < bonferroni_alpha),
        })

    return results


if __name__ == "__main__":
    analyze_first_move_correlation()