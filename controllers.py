"""Classical controllers for the WindCartPole environment."""

import numpy as np


OBSERVATION_SCALE = np.array([2.4, 3.0, 0.21, 3.5], dtype=np.float64)


def _discrete_lqr_gain(env: object) -> np.ndarray:
    """Solve the discrete Riccati equation for the locally linearized dynamics."""
    dt = float(env.tau)
    total_mass = float(env.total_mass)
    polemass_length = float(env.polemass_length)
    pole_mass = float(env.masspole)
    pole_half_length = float(env.length)
    gravity = float(env.gravity)

    denominator = 4.0 / 3.0 - pole_mass / total_mass
    theta_from_theta = gravity / (pole_half_length * denominator)
    theta_from_force = -1.0 / (total_mass * pole_half_length * denominator)
    x_from_theta = -polemass_length * theta_from_theta / total_mass
    x_from_force = 1.0 / total_mass - polemass_length * theta_from_force / total_mass

    # Match CartPoleEnv's semi-implicit Euler update around the upright state.
    transition = np.eye(4, dtype=np.float64)
    transition[0, 1] = dt
    transition[0, 2] = dt * dt * x_from_theta
    transition[1, 2] = dt * x_from_theta
    transition[2, 2] = 1.0 + dt * dt * theta_from_theta
    transition[2, 3] = dt
    transition[3, 2] = dt * theta_from_theta
    control = np.array(
        [dt * dt * x_from_force, dt * x_from_force, dt * dt * theta_from_force, dt * theta_from_force],
        dtype=np.float64,
    ).reshape(4, 1)

    state_cost = np.diag([1.0, 1.0, 10.0, 1.0])
    action_cost = np.array([[0.1]], dtype=np.float64)
    value = state_cost.copy()
    for _ in range(10_000):
        gain = np.linalg.solve(
            action_cost + control.T @ value @ control,
            control.T @ value @ transition,
        )
        next_value = (
            state_cost
            + transition.T @ value @ transition
            - transition.T @ value @ control @ gain
        )
        if np.max(np.abs(next_value - value)) < 1e-11:
            value = next_value
            break
        value = next_value
    else:
        raise RuntimeError("Discrete LQR Riccati iteration did not converge")

    return np.linalg.solve(
        action_cost + control.T @ value @ control,
        control.T @ value @ transition,
    ).ravel()


class LQRController:
    """Bang-bang controller using an LQR feedback gain from the linearized model."""

    def __init__(self, env: object) -> None:
        self.gain = _discrete_lqr_gain(env)

    @property
    def normalized_weights(self) -> np.ndarray:
        """Weights for a threshold policy on normalized observations."""
        return -self.gain * OBSERVATION_SCALE

    def act(self, observation: np.ndarray) -> int:
        control = -float(self.gain @ np.asarray(observation, dtype=np.float64))
        return int(control >= 0.0)
