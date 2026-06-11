# rl-cartpole-custom

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

You can run a simple random agent to see how the environment behaves with the wind applied.

To run the simulation:

```bash
python main.py
```

The script registers the environment and runs a few episodes showing the step counts and total rewards obtained by the random agent.
