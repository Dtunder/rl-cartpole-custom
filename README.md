# rl-cartpole-custom

![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)
![Coverage](https://img.shields.io/badge/Coverage-100%25-brightgreen)

Gymnasium environment for CartPole with external wind disturbances.

This project implements a custom `Gymnasium` environment called `WindCartPoleEnv` that inherits from the classic `CartPoleEnv`. It introduces external wind disturbances that act as additional forces on the cart. 

## Features

- **Custom Environment**: `WindCartPole-v0`
- **Configurable Wind**:
  - `wind_mode`: Can be set to `'random'` (uniform random noise) or `'sinusoidal'` (periodic oscillating force).
  - `wind_intensity`: Adjust the scale of the wind force.

## Prerequisites

To install the necessary dependencies, run:

```bash
pip install gymnasium numpy
```

## Usage

### CLI Instructions

You can run a simple random agent to see how the environment behaves with the wind applied.

To run the simulation:

```bash
python main.py
```

The script registers the environment and runs a few episodes showing the step counts and total rewards obtained by the random agent.

## Configuration

When initializing the environment, you can configure the wind behavior using the following parameters:

- `wind_mode` (str): The type of wind disturbance.
  - `'random'`: Applies a uniform random force at each step.
  - `'sinusoidal'`: Applies a periodic oscillating force based on the current step count.
- `wind_intensity` (float): The maximum magnitude of the wind force. Must be a non-negative number.

Example configuration setup in Python:
```python
import gymnasium as gym
import custom_cartpole  # registers the environment

env = gym.make('WindCartPole-v0', wind_mode='sinusoidal', wind_intensity=2.5)
```

## API Reference

### `custom_cartpole.py`

#### `class WindCartPoleEnv(CartPoleEnv)`
A custom Gymnasium environment based on the classic CartPole, introducing simulated external wind disturbances.
- `__init__(self, wind_mode="random", wind_intensity=1.0, **kwargs)`: Initializes the environment with specific wind parameters.
- `step(self, action)`: Runs one timestep of the environment's dynamics, calculating the wind force and applying it to the cart. Returns `(observation, reward, terminated, truncated, info)`.
- `reset(self, *, seed=None, options=None)`: Resets the environment state and the internal step counter. Returns `(observation, info)`.

### `main.py`

#### Functions
- `run_simulation(env_id='WindCartPole-v0', episodes=5, render_mode=None, **env_kwargs)`: Runs a simulation of the specified environment using a random agent for a given number of episodes.
- `_run_single_episode(env, episode_index)`: Helper function to execute a single episode of the environment using a random action policy.
- `main()`: Main entry point that runs example simulations with both 'random' and 'sinusoidal' wind modes.
