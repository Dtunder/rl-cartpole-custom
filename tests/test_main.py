import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from custom_cartpole import WindCartPoleEnv
from main import run_simulation, main


def test_env_initialization() -> None:
    env = WindCartPoleEnv(wind_mode="random", wind_intensity=2.0)
    assert env.wind_mode == "random"
    assert env.wind_intensity == 2.0
    assert env.current_step == 0


def test_env_reset() -> None:
    env = WindCartPoleEnv()
    env.current_step = 10
    obs, info = env.reset()
    assert env.current_step == 0
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)


def test_env_step_invalid_action() -> None:
    env = WindCartPoleEnv()
    env.reset()
    with pytest.raises(ValueError, match="invalid"):
        env.step(5)


def test_env_step_uninitialized() -> None:
    env = WindCartPoleEnv()
    with pytest.raises(RuntimeError, match="Call reset before using step method."):
        env.step(0)


def test_env_step_random_wind() -> None:
    env = WindCartPoleEnv(wind_mode="random", wind_intensity=5.0)
    obs, info = env.reset()

    # We can't perfectly predict the exact state due to randomness,
    # but we can check if it returns correct types and advances the step.
    obs, reward, terminated, truncated, info = env.step(1)
    assert env.current_step == 1
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_env_step_sinusoidal_wind() -> None:
    env = WindCartPoleEnv(wind_mode="sinusoidal", wind_intensity=2.0)
    obs, info = env.reset()

    obs, reward, terminated, truncated, info = env.step(0)
    assert env.current_step == 1

    obs, reward, terminated, truncated, info = env.step(1)
    assert env.current_step == 2


def test_env_euler_integrator() -> None:
    # Test with euler integrator
    env = WindCartPoleEnv()
    env.kinematics_integrator = "euler"
    env.reset()
    obs, reward, terminated, truncated, info = env.step(1)
    assert isinstance(obs, np.ndarray)


def test_env_termination() -> None:
    env = WindCartPoleEnv()
    env.reset()

    # Force the pole to fall to test termination logic
    env.state = np.array((0.0, 0.0, env.theta_threshold_radians + 0.1, 0.0))

    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated is True
    assert reward == 1.0  # First time it falls, reward is 1.0 based on env code
    assert env.steps_beyond_terminated == 0

    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 0.0  # Steps beyond terminated increases, reward becomes 0.0
    assert env.steps_beyond_terminated == 1


def test_render_mode_human() -> None:
    with patch("custom_cartpole.WindCartPoleEnv.render") as mock_render:
        env = WindCartPoleEnv(render_mode="human")
        env.reset()
        env.step(0)
        mock_render.assert_called()


@patch("main.gym.make")
def test_run_simulation(mock_gym_make: MagicMock) -> None:
    # Setup mock env
    mock_env = MagicMock()
    mock_env.reset.return_value = (np.zeros(4), {})
    # Simulate a 2-step episode
    mock_env.step.side_effect = [
        (np.zeros(4), 1.0, False, False, {}),
        (np.zeros(4), 1.0, True, False, {}),
    ]
    mock_env.action_space.sample.return_value = 0
    mock_gym_make.return_value = mock_env

    # Call function
    run_simulation(env_id="WindCartPole-v0", episodes=1)

    # Assertions
    mock_gym_make.assert_called_once_with("WindCartPole-v0", render_mode=None)
    mock_env.reset.assert_called_once()
    assert mock_env.step.call_count == 2
    mock_env.close.assert_called_once()


@patch("main.run_simulation")
def test_main(mock_run_simulation: MagicMock) -> None:
    main()
    assert mock_run_simulation.call_count == 2
    mock_run_simulation.assert_any_call(wind_mode="random", wind_intensity=5.0)
    mock_run_simulation.assert_any_call(wind_mode="sinusoidal", wind_intensity=10.0)


def test_custom_cartpole_init_validation() -> None:
    with pytest.raises(TypeError, match="wind_mode must be a string"):
        WindCartPoleEnv(wind_mode=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="wind_mode must be 'random' or 'sinusoidal'"):
        WindCartPoleEnv(wind_mode="unknown")
    with pytest.raises(TypeError, match="wind_intensity must be a number"):
        WindCartPoleEnv(wind_intensity="strong")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="wind_intensity must be non-negative"):
        WindCartPoleEnv(wind_intensity=-5.0)


def test_run_simulation_validation() -> None:
    with pytest.raises(TypeError, match="episodes must be an integer"):
        run_simulation(episodes="five")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="episodes must be positive"):
        run_simulation(episodes=0)


@patch("main.gym.make")
def test_run_simulation_make_error(
    mock_gym_make: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    mock_gym_make.side_effect = Exception("Gym make error")
    run_simulation(max_retries=0)
    assert "Failed to create environment after retries: Gym make error" in caplog.text


@patch("main.gym.make")
def test_run_simulation_reset_error(
    mock_gym_make: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    mock_env = MagicMock()
    mock_env.reset.side_effect = Exception("Gym reset error")
    mock_gym_make.return_value = mock_env
    run_simulation(episodes=1, max_retries=0)
    assert "Failed to reset environment after retries: Gym reset error" in caplog.text


@patch("main.gym.make")
def test_run_simulation_step_error(
    mock_gym_make: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    mock_env = MagicMock()
    mock_env.reset.return_value = (np.zeros(4), {})
    mock_env.step.side_effect = Exception("Gym step error")
    mock_gym_make.return_value = mock_env
    run_simulation(episodes=1, max_retries=0)
    assert "Error during step after retries: Gym step error" in caplog.text


@patch("main.gym.make")
def test_run_simulation_fallback(
    mock_gym_make: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    mock_gym_make.side_effect = [Exception("Gym make error"), MagicMock()]
    run_simulation(max_retries=0, fallback_kwargs={"wind_mode": "random"})
    assert "Using fallback configuration for environment creation." in caplog.text


@patch("main.gym.make")
def test_run_simulation_close_error(
    mock_gym_make: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    mock_env = MagicMock()
    mock_env.reset.return_value = (np.zeros(4), {})
    mock_env.step.return_value = (np.zeros(4), 1.0, True, False, {})
    mock_env.close.side_effect = Exception("Gym close error")
    mock_gym_make.return_value = mock_env
    run_simulation(episodes=1)
    assert "Error while closing environment: Gym close error" in caplog.text
