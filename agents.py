"""Small pure-NumPy learning agents for WindCartPole."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from controllers import LQRController, OBSERVATION_SCALE
from custom_cartpole import WindCartPoleEnv


@dataclass(frozen=True)
class LinearPolicy:
    """Threshold policy with learned weights over normalized CartPole state."""

    weights: np.ndarray

    def act(self, observation: np.ndarray) -> int:
        score = float(self.weights @ (np.asarray(observation) / OBSERVATION_SCALE))
        return int(score >= 0.0)


def _episode_return(
    env: WindCartPoleEnv, policy: LinearPolicy, seed: int, max_steps: int
) -> int:
    observation, _ = env.reset(seed=seed)
    for steps in range(1, max_steps + 1):
        observation, _, terminated, truncated, _ = env.step(policy.act(observation))
        if terminated or truncated:
            return steps
    return max_steps


def train_cem(
    seed: int = 2026,
    generations: int = 5,
    population_size: int = 12,
    max_steps: int = 500,
    scenarios: Iterable[tuple[str, float]] | None = None,
) -> LinearPolicy:
    """Fit a linear threshold policy with a deterministic CEM training loop.

    The search starts from the LQR controller and optimizes mean episode return
    over fixed seeded no-wind and disturbed scenarios.
    """
    if generations < 1 or population_size < 4 or max_steps < 1:
        raise ValueError("generations >= 1, population_size >= 4, and max_steps >= 1 are required")

    scenario_list = (
        list(scenarios)
        if scenarios is not None
        else [
            ("none", 0.0),
            ("random", 1.0),
            ("random", 10.0),
            ("random", 15.0),
            ("sinusoidal", 1.0),
            ("sinusoidal", 5.0),
            ("sinusoidal", 10.0),
        ]
    )
    if not scenario_list:
        raise ValueError("at least one training scenario is required")

    reference_env = WindCartPoleEnv(wind_mode="none", wind_intensity=0.0)
    initial_weights = LQRController(reference_env).normalized_weights
    reference_env.close()

    environments = [
        WindCartPoleEnv(wind_mode=mode, wind_intensity=intensity)
        for mode, intensity in scenario_list
    ]
    scenario_seeds = [seed + 100 + index for index in range(len(scenario_list))]

    def score(weights: np.ndarray) -> float:
        policy = LinearPolicy(np.asarray(weights, dtype=np.float64))
        returns = [
            _episode_return(env, policy, scenario_seed, max_steps)
            for env, scenario_seed in zip(environments, scenario_seeds)
        ]
        return float(np.mean(returns))

    rng = np.random.default_rng(seed)
    mean = initial_weights.copy()
    sigma = np.maximum(np.abs(mean) * 0.35, 0.25)
    best_weights = mean.copy()
    elite_count = max(2, population_size // 4)

    try:
        best_score = score(best_weights)
        for _ in range(generations):
            candidates = np.vstack(
                [mean, rng.normal(mean, sigma, size=(population_size - 1, mean.size))]
            )
            scores = np.array([score(candidate) for candidate in candidates])
            if float(scores.max()) > best_score:
                best_index = int(scores.argmax())
                best_score = float(scores[best_index])
                best_weights = candidates[best_index].copy()

            elites = candidates[np.argsort(scores)[-elite_count:]]
            mean = 0.25 * mean + 0.75 * elites.mean(axis=0)
            sigma = np.maximum(
                0.25 * sigma + 0.75 * elites.std(axis=0), 0.08
            )
    finally:
        for env in environments:
            env.close()

    return LinearPolicy(best_weights)
