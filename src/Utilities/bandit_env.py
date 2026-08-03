from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Allows the module to import with legacy Gym installations.
    import gym
    from gym import spaces


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
    One-step contextual bandit environment.

    Episode construction
    --------------------
    1. Two distinct bandit indices are sampled.
    2. Each bandit is encoded by one proxy.
    3. The agent observes the concatenated proxy encodings.
    4. The agent selects one bandit index.

    Reward semantics
    ----------------
    Each proxy's ``value`` is the reward for correctly selecting the bandit
    represented by that proxy.

    Because the two represented bandits are distinct, an action can normally
    earn at most one proxy reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_bandits: int,
        proxy_1: Proxy,
        proxy_2: Proxy,
        incorrect_reward: float = 0.0,
    ):
        super().__init__()

        if num_bandits < 2:
            raise ValueError(
                "num_bandits must be at least 2 so that two distinct "
                "bandits can be sampled"
            )

        self.num_bandits = int(num_bandits)
        self.proxies = (proxy_1, proxy_2)
        self.incorrect_reward = float(incorrect_reward)

        # The action is the agent's guess of the represented bandit index.
        self.action_space = spaces.Discrete(self.num_bandits)

        observation_size = sum(
            proxy.observation_size for proxy in self.proxies
        )

        # The supplied proxies produce nonnegative values.
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self.suggested_bandits: tuple[int, int] | None = None
        self._observation: npt.NDArray[np.float32] | None = None
        self._episode_finished = True

    def _sample_distinct_bandits(self) -> tuple[int, int]:
        """
        Sample two different bandit indices without rejection sampling.
        """
        first = int(self.np_random.integers(self.num_bandits))
        second = int(self.np_random.integers(self.num_bandits - 1))

        if second >= first:
            second += 1

        return first, second

    def _build_observation(
        self,
        suggested_bandits: tuple[int, int],
    ) -> npt.NDArray[np.float32]:
        parts = [
            proxy(bandit).reshape(-1)
            for proxy, bandit in zip(self.proxies, suggested_bandits)
        ]

        observation = np.concatenate(parts).astype(
            np.float32,
            copy=False,
        )

        if not self.observation_space.contains(observation):
            raise ValueError(
                "A proxy produced an observation outside the declared "
                f"observation space: {observation}"
            )

        return observation

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)

        for proxy in self.proxies:
            proxy.reset(self.np_random)

        self.suggested_bandits = self._sample_distinct_bandits()
        self._observation = self._build_observation(
            self.suggested_bandits
        )
        self._episode_finished = False

        info = {
            # Useful for debugging/evaluation. Remove this entry if exposing
            # the hidden answer through info is undesirable.
            "suggested_bandits": self.suggested_bandits,
            "proxy_values": tuple(
                proxy.value for proxy in self.proxies
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
                f"0 to {self.num_bandits - 1}"
            )

        assert self.suggested_bandits is not None
        assert self._observation is not None

        action = int(action)

        matched_proxy = None
        reward = self.incorrect_reward

        for index, (proxy, bandit) in enumerate(
            zip(self.proxies, self.suggested_bandits)
        ):
            if action == bandit:
                reward = proxy.value
                matched_proxy = index
                break

        # Every episode is exactly one step long.
        terminated = True
        truncated = False
        self._episode_finished = True

        info = {
            "selected_bandit": action,
            "correct": matched_proxy is not None,
            "matched_proxy": matched_proxy,
            "suggested_bandits": self.suggested_bandits,
        }

        return (
            self._observation.copy(),
            float(reward),
            terminated,
            truncated,
            info,
        )