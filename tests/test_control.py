import numpy as np

from agents import train_cem
from controllers import LQRController
from custom_cartpole import WindCartPoleEnv


def test_none_wind_does_not_apply_disturbance() -> None:
    no_wind = WindCartPoleEnv(wind_mode="none", wind_intensity=10.0)
    zero_random = WindCartPoleEnv(wind_mode="random", wind_intensity=0.0)
    try:
        obs_none, _ = no_wind.reset(seed=21)
        obs_zero, _ = zero_random.reset(seed=21)
        np.testing.assert_array_equal(obs_none, obs_zero)
        for action in (1, 0, 1, 1, 0):
            obs_none, *_ = no_wind.step(action)
            obs_zero, *_ = zero_random.step(action)
            np.testing.assert_array_equal(obs_none, obs_zero)
    finally:
        no_wind.close()
        zero_random.close()


def test_lqr_balances_at_least_400_steps_without_wind() -> None:
    env = WindCartPoleEnv(wind_mode="none", wind_intensity=0.0)
    controller = LQRController(env)
    try:
        observation, _ = env.reset(seed=7)
        steps = 0
        for steps in range(1, 501):
            observation, _, terminated, truncated, _ = env.step(
                controller.act(observation)
            )
            if terminated or truncated:
                break
        assert steps >= 400
    finally:
        env.close()


def test_cem_policy_is_seeded_and_returns_valid_actions() -> None:
    first = train_cem(seed=17, generations=1, population_size=4, max_steps=120)
    second = train_cem(seed=17, generations=1, population_size=4, max_steps=120)
    np.testing.assert_array_equal(first.weights, second.weights)
    action = first.act(np.zeros(4, dtype=np.float32))
    assert action in (0, 1)
