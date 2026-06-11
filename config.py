import os
import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Default fallback configurations
DEFAULT_CONFIG = {
    "wind_mode": "random",
    "wind_intensity": 1.0,
    "sinusoidal_frequency": 0.1,
    "max_episode_steps": 500,
    "episodes": 5,
    "max_retries": 3,
    "resilience_delay": 1.0,
    "main_make_delay": 0.5,
    "main_reset_delay": 0.1,
    "main_step_delay": 0.1,
    "main_random_wind_intensity": 5.0,
    "main_sinusoidal_wind_intensity": 10.0,
}

CONFIG: Dict[str, Any] = DEFAULT_CONFIG.copy()

config_file = os.environ.get("CARTPOLE_CONFIG_FILE", "config.json")
if os.path.exists(config_file):
    try:
        with open(config_file, "r") as f:
            file_config = json.load(f)
            CONFIG.update(file_config)
            logger.info(f"Loaded configuration from {config_file}")
    except Exception as e:
        logger.warning(f"Failed to load config file {config_file}: {e}")
else:
    logger.info(f"Config file {config_file} not found. Using defaults.")

# Override with environment variables if present
for key in CONFIG.keys():
    env_key = f"CARTPOLE_{key.upper()}"
    if env_key in os.environ:
        val = os.environ[env_key]
        # Cast to correct type based on default
        default_val = DEFAULT_CONFIG[key]
        try:
            if isinstance(default_val, int):
                CONFIG[key] = int(val)
            elif isinstance(default_val, float):
                CONFIG[key] = float(val)
            else:
                CONFIG[key] = val
        except ValueError:
            logger.warning(
                f"Failed to cast env var {env_key}={val} to {type(default_val).__name__}. "
                f"Keeping current value: {CONFIG[key]}"
            )
