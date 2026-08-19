from __future__ import annotations

import csv
import os
from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Allows the module to import with legacy Gym installations.
    import gym
    from gym import spaces

# The group classes (ComplexityGroupBase and its concrete subclasses) live
# in the TOP-LEVEL groups.py (not under Utilities/ - Utilities/ is reserved
# for the environment itself; see groups.py's module docstring). Re-exported
# here (explicit names) so every existing `from Utilities.bandit_env import
# MarginGroup` (etc.) elsewhere in this codebase keeps working unchanged;
# groups.py is the source of truth. NOTE: the old ComplexityGroup class was
# renamed to AlternatingGroup - `from Utilities.bandit_env import
# ComplexityGroup` no longer works; update call sites to `AlternatingGroup`.
from groups import (
    DEFAULT_MC_SAMPLES,
    DEFAULT_MC_SEED,
    ComplexityGroupBase,
    AlternatingGroup,
    MarginGroup,
    HeatmapGroup,
    clear_metric_cache,
)


# ---------------------------------------------------------------------------
# Legacy encoding-based proxies (kept for reference, not used by BanditEnv).
#
# These model "index -> continuous encoding": the environment first chooses
# a discrete bandit index, then a Proxy encodes it into an observation the
# agent must invert. PolynomialProxy's difficulty is a closed-form algebra
# puzzle (parameterized by random weights/biases + increasing powers), which
# is a different kind of hardness than the frequency-based complexity dial
# below. Nothing in this file currently instantiates these; they're left
# here in case that encoding-based approach is useful again later.
# ---------------------------------------------------------------------------


class Proxy(ABC):
    """
    Converts a bandit index into an observation.

    Parameters
    ----------
    complexity:
        Number of values produced by the proxy.
    value:
        Reward given when the agent selects the bandit represented by
        this proxy.
    """

    def __init__(self, complexity: int, value: float):
        if complexity < 1:
            raise ValueError("complexity must be at least 1")

        self.complexity = int(complexity)
        self.value = float(value)

    @property
    def observation_size(self) -> int:
        return self.complexity

    def reset(self, rng: np.random.Generator) -> None:
        """Randomize the proxy for a new episode, if applicable."""

    @abstractmethod
    def __call__(self, bandit: int) -> npt.NDArray[np.float32]:
        """Encode a bandit index as an observation."""
        raise NotImplementedError


class BasicProxy(Proxy):
    """
    Identity proxy.

    The proxy directly exposes the suggested bandit index. Complexity and
    dimensionality are unnecessary because it always returns one value.
    """

    def __init__(self, value: float):
        super().__init__(complexity=1, value=value)

    def __call__(self, bandit: int) -> npt.NDArray[np.float32]:
        return np.asarray([bandit], dtype=np.float32)


class PolynomialProxy(Proxy):
    """
    Produces a nonlinear, multi-value encoding of a bandit index.

    For channel i:

        observation[i] = (weight[i] * bandit + bias[i]) ** (1 / power[i])

    where powers are 1, 2, ..., complexity.
    """

    def __init__(self, complexity: int, value: float):
        super().__init__(complexity=complexity, value=value)

        self.powers = np.arange(
            1,
            self.complexity + 1,
            dtype=np.float32,
        )

        power_sum = float(self.powers.sum())
        self.complexity_weights = self.powers / power_sum

        self.weights = np.empty(self.complexity, dtype=np.float32)
        self.biases = np.empty(self.complexity, dtype=np.float32)

    def reset(self, rng: np.random.Generator) -> None:
        raw_weights = (
            rng.random(self.complexity).astype(np.float32)
            * self.complexity_weights
        )

        # The random draw is almost surely nonzero, but handling zero makes
        # the method numerically safe.
        weight_sum = float(raw_weights.sum())
        if weight_sum == 0.0:
            raw_weights = self.complexity_weights.copy()
            weight_sum = float(raw_weights.sum())

        self.weights = (
            raw_weights * self.complexity / weight_sum
        ).astype(np.float32)

        raw_biases = rng.random(self.complexity).astype(np.float32)
        bias_sum = float(raw_biases.sum())

        if bias_sum == 0.0:
            raw_biases.fill(1.0)
            bias_sum = float(raw_biases.sum())

        self.biases = (
            raw_biases * self.complexity / bias_sum
        ).astype(np.float32)

    def __call__(self, bandit: int) -> npt.NDArray[np.float32]:
        encoded = self.weights * float(bandit) + self.biases
        return np.power(encoded, 1.0 / self.powers).astype(np.float32)

    def inverse(
        self,
        observation: npt.ArrayLike,
    ) -> float:
        """
        Estimate the original bandit index from an encoded observation.

        Each channel independently implies:

            bandit = (observation ** power - bias) / weight

        The estimates are averaged to reduce floating-point error.
        """
        observation_array = np.asarray(
            observation,
            dtype=np.float32,
        )

        if observation_array.shape != (self.complexity,):
            raise ValueError(
                "observation must have shape "
                f"({self.complexity},), got {observation_array.shape}"
            )

        estimates = (
            np.power(observation_array, self.powers) - self.biases
        ) / self.weights

        return float(estimates.mean())


class BanditEnv(gym.Env):
    """
    One-step bandit environment composed of one or more complexity groups.

    Episode construction
    --------------------
    1. Each group independently samples its own secret ``x_i`` and correct
       option ``label_i`` (see ``AlternatingGroup.sample``), a value in
       ``0 .. group.g - 1``, LOCAL to that group.
    2. The agent observes the concatenation of all groups' secrets, in the
       order the groups were supplied.
    3. The agent selects a single action from
       ``Discrete(sum(group.g for group in groups))`` - EACH GROUP OWNS ITS
       OWN DISJOINT BLOCK of that space, in the order the groups were
       supplied. Group ``i``'s block starts at
       ``action_offsets[i] = sum(g.g for g in groups[:i])`` and covers the
       next ``groups[i].g`` actions. An action can only ever fall inside
       ONE group's block - there is no shared/overlapping range, so it is
       never possible for a single action to simultaneously be "correct
       for" two different groups. Choosing an action anywhere in a group's
       block is how the agent "goes for" that group's target, whether or
       not it's the specific option that group's target happens to be that
       episode.

    Reward semantics
    ----------------
    An action matches AT MOST one group: whichever group's block it falls
    in, if the action equals ``action_offsets[i] + label_i`` for that
    group ``i``. If it matches, the agent earns that group's ``value``. If
    the action falls in some group's block but isn't that block's correct
    option this episode (or, degenerately, the action space of some other
    group is larger and this action falls past every group's block - not
    possible given the space is sized to exactly fit every block),
    ``incorrect_reward`` is returned instead.

    Nothing in the observation marks which secret belongs to which group,
    what that group's ``g``/``k``/``value`` is, or which group (if any) the
    action ended up matching - the agent has to pick up any structure that
    distinguishes the groups (e.g. different value, different k, different
    positions in the observation vector, different blocks of the action
    space) purely from repeated exposure to the observation/reward pattern.

    With two groups sharing the same ``g`` but different ``k``, this gives
    you exactly the "two groups of g options apiece (8 total, 4 per group),
    independently tunable difficulty" setup: pass
    ``AlternatingGroup(g, k_1, value_1)`` and
    ``AlternatingGroup(g, k_2, value_2)``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        groups: list[AlternatingGroup],
        incorrect_reward: float | Sequence[float] = 0.0,
    ):
        super().__init__()

        if len(groups) < 1:
            raise ValueError("must supply at least one group")

        self.groups = list(groups)

        if isinstance(incorrect_reward, (int, float)):
            # Backward-compatible scalar case (everything that constructed
            # BanditEnv before this changed still behaves identically):
            # every group's "landed in this group's block but didn't hit
            # its label" reward is the same constant.
            self.incorrect_reward = float(incorrect_reward)
            self._incorrect_rewards: Sequence[float] = [float(incorrect_reward)] * len(self.groups)
        else:
            # Per-group case: incorrect_reward[i] is the reward for an
            # action that falls in group i's block without matching its
            # label. Deliberately NOT copied - kept as the exact list/
            # tuple object the caller passed in, so if that same object is
            # shared across multiple BanditEnv instances (e.g. one per
            # DummyVecEnv sub-env, the same way `groups`' individual group
            # objects are already shared - see make_env in trainPPO.py),
            # mutating incorrect_reward[i] from outside after construction
            # is immediately visible to every instance sharing it. This is
            # what lets a caller flip a group's incorrect-answer reward
            # mid-training the same way it can already flip a group's
            # `.value` (e.g. run_shared_machinery_experiment.py's
            # gradient-starvation seam, where an "inactive" group should
            # get the same penalized reward whether the agent happens to
            # pick that group's correct label or not).
            if len(incorrect_reward) != len(self.groups):
                raise ValueError(
                    f"incorrect_reward has {len(incorrect_reward)} entries, "
                    f"expected one per group ({len(self.groups)})"
                )
            self.incorrect_reward = None
            self._incorrect_rewards = incorrect_reward

        # Each group gets its own disjoint block of actions - group i's
        # options occupy [offset_i, offset_i + group_i.g) of the action
        # space, so an action can only ever belong to (and possibly match)
        # ONE group, never accidentally satisfy a different group's target
        # the way a single shared Discrete(max_g) space would allow.
        self._action_offsets: tuple[int, ...] = tuple(
            sum(g.g for g in self.groups[:i]) for i in range(len(self.groups))
        )
        total_actions = sum(g.g for g in self.groups)
        self.action_space = spaces.Discrete(total_actions)

        observation_size = sum(g.observation_size for g in self.groups)

        # Bounds are concatenated per-group, since not every group's
        # secrets live in [0, 1) (e.g. MarginGroup's scores live in
        # [0, s + delta]).
        low = np.concatenate([g.observation_low for g in self.groups])
        high = np.concatenate([g.observation_high for g in self.groups])
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self._labels: tuple[int, ...] | None = None
        self._observation: npt.NDArray[np.float32] | None = None
        self._episode_finished = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)

        secret_chunks = []
        labels = []
        for group in self.groups:
            x, label = group.sample(self.np_random)
            secret_chunks.append(np.atleast_1d(np.asarray(x, dtype=np.float32)))
            labels.append(label)

        self._labels = tuple(labels)
        self._observation = np.concatenate(secret_chunks).astype(np.float32)
        self._episode_finished = False

        info = {
            # Useful for debugging/evaluation. Remove this entry if exposing
            # the hidden answer through info is undesirable.
            "labels": self._labels,
            "group_values": tuple(group.value for group in self.groups),
            # Where each group's block starts in the action space (see the
            # class docstring), and each group's correct label expressed as
            # an absolute action - i.e. what the agent would actually need
            # to submit to hit THIS group specifically. "labels" above
            # stays LOCAL (0..group.g-1) for backwards compatibility;
            # "global_labels" is "labels" already shifted by
            # "action_offsets", for callers that want to compare directly
            # against the action without doing that arithmetic themselves.
            "action_offsets": self._action_offsets,
            "global_labels": tuple(
                offset + label
                for offset, label in zip(self._action_offsets, self._labels)
            ),
        }

        return self._observation.copy(), info

    def step(
        self,
        action: int,
    ) -> tuple[
        npt.NDArray[np.float32],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        if self._episode_finished:
            raise RuntimeError(
                "The episode has finished. Call reset() before step()."
            )

        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action!r}; expected an integer from "
                f"0 to {self.action_space.n - 1}"
            )

        assert self._labels is not None
        assert self._observation is not None

        action = int(action)

        matched_group = None
        chosen_group = None
        # Only used if, in principle, no group's block contains the action
        # (shouldn't happen - see the loop's comment below) - falls back to
        # the first group's incorrect reward, matching this attribute's old
        # behavior of always being a single scalar in that (unreachable)
        # case.
        reward = self._incorrect_rewards[0]

        for index, (group, label, offset) in enumerate(
            zip(self.groups, self._labels, self._action_offsets)
        ):
            if offset <= action < offset + group.g:
                # The action falls in THIS group's block - this is the
                # group the agent "went for" this episode, regardless of
                # whether it picked the right option within that block.
                # Blocks are disjoint, so exactly one group satisfies this
                # (or, in principle, none - but the space is sized to
                # exactly cover every block, so every valid action always
                # belongs to exactly one).
                chosen_group = index
                if action == offset + label:
                    reward = group.value
                    matched_group = index
                else:
                    # Landed in this group's block but not on its label -
                    # this group's own incorrect-answer reward (equal to
                    # the shared scalar `incorrect_reward` in the common
                    # case, but independently overridable per-group - see
                    # `_incorrect_rewards` above).
                    reward = self._incorrect_rewards[index]
                break

        # Every episode is exactly one step long.
        terminated = True
        truncated = False
        self._episode_finished = True

        info = {
            "selected_action": action,
            "correct": matched_group is not None,
            "matched_group": matched_group,
            # Which group's action BLOCK this action fell in - i.e. which
            # target the agent went for, whether or not it hit it. Unlike
            # matched_group, this is never None for a valid action.
            "chosen_group": chosen_group,
            "labels": self._labels,
            "action_offsets": self._action_offsets,
        }

        return (
            self._observation.copy(),
            float(reward),
            terminated,
            truncated,
            info,
        )


class CsvLoggingWrapper(gym.Wrapper):
    """
    Records every episode's secret(s) and the agent's chosen action to a
    CSV file, one row per episode.

    Columns: ``episode``, ``x_0..x_{n-1}`` (each group's secret, i.e. what
    the agent observed), ``label_0..label_{n-1}`` (each group's correct
    option), ``action`` (what the agent actually picked), ``reward``,
    ``correct``, ``matched_group``.

    This wraps any ``BanditEnv`` (or a vector/gym-wrapped version of one) -
    it doesn't assume anything about the training loop driving it, since
    nothing in this repo currently trains on ``BanditEnv``. Just wrap the
    env once and step it as usual; rows are appended and flushed on every
    step so partial runs (e.g. a crashed training script) aren't lost.
    """

    def __init__(self, env: gym.Env, csv_path: str, append: bool = False):
        super().__init__(env)

        groups = self.env.unwrapped.groups
        num_groups = len(groups)

        # Each group may contribute more than one observation column (e.g.
        # MarginGroup contributes g columns), so name columns
        # x_{group}_{index_within_group} rather than assuming one x per
        # group. A group with observation_size == 1 still gets a single
        # x_{i}_0 column for consistency.
        self._x_columns = [
            f"x_{i}_{j}"
            for i, group in enumerate(groups)
            for j in range(group.observation_size)
        ]
        self._label_columns = [f"label_{i}" for i in range(num_groups)]
        self._fieldnames = (
            ["episode"]
            + self._x_columns
            + self._label_columns
            + ["action", "reward", "correct", "matched_group"]
        )

        file_exists = append and os.path.exists(csv_path)
        self._file = open(csv_path, "a" if append else "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)

        if not file_exists:
            self._writer.writeheader()
            self._file.flush()

        self._episode = 0
        self._pending_secrets: npt.NDArray[np.float32] | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._pending_secrets = np.asarray(obs, dtype=np.float32)
        self._episode += 1
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        row = {"episode": self._episode}
        row.update(
            zip(self._x_columns, self._pending_secrets.tolist())
        )
        row.update(zip(self._label_columns, info["labels"]))
        row["action"] = int(action)
        row["reward"] = float(reward)
        row["correct"] = bool(info["correct"])
        row["matched_group"] = info["matched_group"]

        self._writer.writerow(row)
        self._file.flush()

        return obs, reward, terminated, truncated, info

    def close(self):
        self._file.close()
        super().close()


if __name__ == "__main__":
    # Smoke test / usage example: two groups of the same size g, with
    # different complexity k and different reward values.
    env = BanditEnv(
        groups=[
            AlternatingGroup(g=4, k=1, value=1.0),   # easy, low-frequency
            AlternatingGroup(g=4, k=25, value=1.0),  # hard, high-frequency
        ],
        incorrect_reward=0.0,
    )

    obs, info = env.reset(seed=0)
    print("observation (secrets):", obs)
    print("true labels per group (local):", info["labels"])
    print("action offsets per group:", info["action_offsets"])
    print("true labels per group (global/absolute action):", info["global_labels"])

    action = info["global_labels"][0]  # deliberately match the easy group -
    # NOTE: must use the offset-adjusted global label, not the local one,
    # since the action space is now partitioned into per-group blocks.
    obs, reward, terminated, truncated, info = env.step(action)
    print("reward:", reward, "matched_group:", info["matched_group"], "chosen_group:", info["chosen_group"])

    # --- CSV logging demo: wrap the env and run a few random episodes ---
    logged_env = CsvLoggingWrapper(
        BanditEnv(
            groups=[
                AlternatingGroup(g=4, k=1, value=1.0),
                AlternatingGroup(g=4, k=25, value=1.0),
            ],
            incorrect_reward=0.0,
        ),
        csv_path="bandit_log.csv",
    )
    rng = np.random.default_rng(0)
    for _ in range(5):
        obs, info = logged_env.reset()
        action = int(rng.integers(logged_env.action_space.n))
        logged_env.step(action)
    logged_env.close()
    print("wrote bandit_log.csv")
