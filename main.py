"""
Main entry point for running the WindCartPole-v0 simulation.

This module provides functions to initialize the custom CartPole environment
and run episodic simulations using a random agent.
"""

import logging
import gymnasium as gym
from typing import Any, Optional, Dict
from resilience import execute_with_resilience
from config import CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Register the environment
gym.register(
    id="WindCartPole-v0",
    entry_point="custom_cartpole:WindCartPoleEnv",
    max_episode_steps=CONFIG["max_episode_steps"],
)


def run_simulation(
    env_id: str = "WindCartPole-v0",
    episodes: Optional[int] = None,
    render_mode: Optional[str] = None,
    max_retries: Optional[int] = None,
    fallback_kwargs: Optional[Dict[str, Any]] = None,
    **env_kwargs: Any,
) -> None:
    """
    Run a simulation of the specified environment using a random agent.

    Args:
        env_id (str): The ID of the registered Gymnasium environment to run. Defaults to 'WindCartPole-v0'.
        episodes (int): The number of episodes to simulate.
        render_mode (str, optional): The render mode for the environment (e.g., 'human', 'rgb_array'). Defaults to None.
        max_retries (int): Maximum number of retries for resilient operations.
        fallback_kwargs (Optional[Dict[str, Any]]): Fallback keyword arguments if environment creation fails.
        **env_kwargs: Additional keyword arguments passed to the environment upon creation (e.g., wind_mode, wind_intensity).

    Raises:
        TypeError: If `episodes` is not an integer.
        ValueError: If `episodes` is less than or equal to 0.
    """
    episodes = episodes if episodes is not None else CONFIG["episodes"]
    max_retries = max_retries if max_retries is not None else CONFIG["max_retries"]

    if not isinstance(episodes, int):
        logger.error(f"episodes must be an integer, got {type(episodes).__name__}")
        raise TypeError(f"episodes must be an integer, got {type(episodes).__name__}")
    if episodes <= 0:
        logger.error(f"episodes must be positive, got {episodes}")
        raise ValueError(f"episodes must be positive, got {episodes}")

    logger.info(f"Creating environment: {env_id} with kwargs: {env_kwargs}")

    def fallback_make(*args: Any, **kwargs: Any) -> gym.Env:
        logger.warning("Using fallback configuration for environment creation.")
        # Override original kwargs with fallback_kwargs
        new_kwargs = kwargs.copy()
        if fallback_kwargs is not None:
            new_kwargs.update(fallback_kwargs)
        return gym.make(*args, **new_kwargs)

    try:
        env = execute_with_resilience(
            gym.make,
            env_id,
            max_retries=max_retries,
            delay=CONFIG["main_make_delay"],
            fallback=fallback_make if fallback_kwargs else None,
            render_mode=render_mode,
            **env_kwargs,
        )
    except Exception as e:
        logger.error(f"Failed to create environment after retries: {e}")
        return

    try:
        for episode in range(episodes):
            success = _run_single_episode(env, episode, max_retries=max_retries)
            if not success:
                break
    finally:
        try:
            env.close()
            logger.info("Environment closed successfully.")
        except Exception as e:
            logger.error(f"Error while closing environment: {e}")


def _run_single_episode(env: gym.Env, episode_index: int, max_retries: Optional[int] = None) -> bool:
    """
    Run a single episode of the environment using a random action policy.

    Args:
        env (gym.Env): The initialized Gymnasium environment.
        episode_index (int): The current episode number (used for logging).
        max_retries (int): Maximum number of retries for resilient operations.

    Returns:
        bool: True if the episode completed successfully, False if an error occurred.
    """
    max_retries = max_retries if max_retries is not None else CONFIG["max_retries"]
    try:
        obs, info = execute_with_resilience(
            env.reset, max_retries=max_retries, delay=CONFIG["main_reset_delay"]
        )
        logger.info(f"Starting Episode {episode_index + 1}")
    except Exception as e:
        logger.error(f"Failed to reset environment after retries: {e}")
        return False

    done = False
    truncated = False
    total_reward = 0.0
    step_count = 0

    while not done and not truncated:
        # Simple random agent
        action = env.action_space.sample()

        try:
            obs, reward, done, truncated, info = execute_with_resilience(
                env.step, action, max_retries=max_retries, delay=CONFIG["main_step_delay"]
            )
        except Exception as e:
            logger.error(f"Error during step after retries: {e}")
            break

        total_reward += float(reward)
        step_count += 1

    logger.info(
        f"Episode {episode_index + 1}: Total Reward = {total_reward}, Steps = {step_count}"
    )
    return True


def main() -> None:
    """
    Main execution block.

    Runs two sets of simulations to demonstrate the WindCartPole environment:
    1. A simulation with 'random' wind mode.
    2. A simulation with 'sinusoidal' wind mode.
    """
    logger.info("Running with random wind...")
    run_simulation(wind_mode="random", wind_intensity=CONFIG["main_random_wind_intensity"])

    logger.info("Running with sinusoidal wind...")
    run_simulation(wind_mode="sinusoidal", wind_intensity=CONFIG["main_sinusoidal_wind_intensity"])


if __name__ == "__main__":
    main()
