import math
import logging
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

logger = logging.getLogger(__name__)


class WindCartPoleEnv(CartPoleEnv):
    """
    A custom Gymnasium environment based on the classic CartPole, introducing simulated external wind disturbances.

    The wind applies an external force to the cart, making the balancing task more challenging.
    This environment is registered as 'WindCartPole-v0'.

    Attributes:
        wind_mode (str): The mode of the wind disturbance ('random' or 'sinusoidal').
        wind_intensity (float): The maximum magnitude or scale of the wind force.
        current_step (int): Tracks the number of steps elapsed in the current episode.
    """

    def __init__(
        self,
        wind_mode: str = "random",
        wind_intensity: Union[int, float] = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the WindCartPole environment.

        Args:
            wind_mode (str): The type of wind disturbance. Options are 'random' (uniform noise)
                or 'sinusoidal' (periodic force based on step count). Defaults to "random".
            wind_intensity (float): The intensity/scale of the wind. Must be non-negative. Defaults to 1.0.
            **kwargs: Additional keyword arguments passed to the base `CartPoleEnv`.

        Raises:
            TypeError: If `wind_mode` is not a string or `wind_intensity` is not a number.
            ValueError: If `wind_mode` is invalid or `wind_intensity` is negative.
        """
        if not isinstance(wind_mode, str):
            logger.error(f"wind_mode must be a string, got {type(wind_mode).__name__}")
            raise TypeError(
                f"wind_mode must be a string, got {type(wind_mode).__name__}"
            )
        if wind_mode not in ("random", "sinusoidal"):
            logger.error(
                f"wind_mode must be 'random' or 'sinusoidal', got {wind_mode!r}"
            )
            raise ValueError(
                f"wind_mode must be 'random' or 'sinusoidal', got {wind_mode!r}"
            )
        if not isinstance(wind_intensity, (int, float)):
            logger.error(
                f"wind_intensity must be a number, got {type(wind_intensity).__name__}"
            )
            raise TypeError(
                f"wind_intensity must be a number, got {type(wind_intensity).__name__}"
            )
        if wind_intensity < 0:
            logger.error(f"wind_intensity must be non-negative, got {wind_intensity}")
            raise ValueError(
                f"wind_intensity must be non-negative, got {wind_intensity}"
            )

        super().__init__(**kwargs)
        self.wind_mode = wind_mode
        self.wind_intensity = float(wind_intensity)
        self.current_step = 0
        logger.info(
            f"Initialized WindCartPoleEnv with wind_mode='{self.wind_mode}' and wind_intensity={self.wind_intensity}"
        )

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Run one timestep of the environment's dynamics.

        Args:
            action (int): The action to take. Should be 0 (push cart to the left) or 1 (push cart to the right).

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: A tuple containing:
                - observation (np.ndarray): The current state of the environment `(x, x_dot, theta, theta_dot)`.
                - reward (float): The reward obtained from the action.
                - terminated (bool): Whether the episode has reached a terminal state (pole fell or out of bounds).
                - truncated (bool): Whether the episode was truncated (always False for this environment natively, but handled by wrappers).
                - info (dict): Additional information (empty in this environment).

        Raises:
            ValueError: If the action provided is not valid in the action space.
            RuntimeError: If the environment has not been reset before calling `step()`.
        """
        err_msg = f"{action!r} ({type(action)}) invalid"
        if not self.action_space.contains(action):
            logger.error(err_msg)
            raise ValueError(err_msg)
        if self.state is None:
            logger.error("Call reset before using step method.")
            raise RuntimeError("Call reset before using step method.")

        self.current_step += 1

        # Calculate wind force - optimized conditional check
        if self.wind_mode == "random":
            wind_force = self.np_random.uniform(
                -self.wind_intensity, self.wind_intensity
            )
        else:
            wind_force = self.wind_intensity * math.sin(0.1 * self.current_step)

        x, x_dot, theta, theta_dot = self.state

        # Action force + wind force
        force = self.force_mag if action == 1 else -self.force_mag
        total_force = force + wind_force

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        # Optimized mathematical operations (avoid **2)
        theta_dot_sq = theta_dot * theta_dot
        costheta_sq = costheta * costheta

        temp = (
            total_force + self.polemass_length * theta_dot_sq * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta_sq / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array((x, x_dot, theta, theta_dot))
        logger.debug(
            f"Step {self.current_step}: action={action}, wind_force={wind_force:.4f}, state={self.state}"
        )

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            logger.info(f"Pole fell at step {self.current_step}. Episode terminated.")
            self.steps_beyond_terminated = 0  # type: ignore[assignment]
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warning("Step called after episode was terminated.")
            self.steps_beyond_terminated += 1  # type: ignore[operator]
            reward = 0.0

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to an initial state and returns the initial observation.

        Args:
            seed (Optional[int]): The seed that is used to initialize the environment's PRNG. Defaults to None.
            options (Optional[Dict[str, Any]]): Additional information to specify how the environment is reset. Defaults to None.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: A tuple containing:
                - observation (np.ndarray): The initial state of the environment.
                - info (dict): Additional information dictionary.
        """
        self.current_step = 0
        logger.info("Environment reset.")
        return super().reset(seed=seed, options=options)
