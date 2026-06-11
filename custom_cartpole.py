import math
import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

class WindCartPoleEnv(CartPoleEnv):
    """
    Custom CartPole environment that includes a simulated external wind disturbance.
    The wind applies an external force to the cart.
    """
    
    def __init__(self, wind_mode="random", wind_intensity=1.0, **kwargs):
        """
        wind_mode: 'random' or 'sinusoidal'
        wind_intensity: scale of the wind force
        """
        if not isinstance(wind_mode, str):
            raise TypeError(f"wind_mode must be a string, got {type(wind_mode).__name__}")
        if wind_mode not in ("random", "sinusoidal"):
            raise ValueError(f"wind_mode must be 'random' or 'sinusoidal', got {wind_mode!r}")
        if not isinstance(wind_intensity, (int, float)):
            raise TypeError(f"wind_intensity must be a number, got {type(wind_intensity).__name__}")
        if wind_intensity < 0:
            raise ValueError(f"wind_intensity must be non-negative, got {wind_intensity}")
            
        super().__init__(**kwargs)
        self.wind_mode = wind_mode
        self.wind_intensity = float(wind_intensity)
        self.current_step = 0

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        if not self.action_space.contains(action):
            raise ValueError(err_msg)
        if self.state is None:
            raise RuntimeError("Call reset before using step method.")
        
        self.current_step += 1
        
        # Calculate wind force - optimized conditional check
        if self.wind_mode == "random":
            wind_force = self.np_random.uniform(-self.wind_intensity, self.wind_intensity)
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

        self.state = (x, x_dot, theta, theta_dot)

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
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                pass # Already warned in CartPoleEnv
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
            
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        return super().reset(seed=seed, options=options)
