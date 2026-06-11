import os
from unittest.mock import patch, mock_open
import importlib
import config

def test_config_load_file_success():
    mock_json = '{"wind_intensity": 2.5}'
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=mock_json)):
            # Reload module to trigger the file reading block
            importlib.reload(config)
            assert config.CONFIG["wind_intensity"] == 2.5

def test_config_load_file_exception():
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="{invalid_json")):
            importlib.reload(config)
            # Should fall back to default
            assert config.CONFIG["wind_intensity"] == 1.0

def test_config_load_file_not_found():
    with patch("os.path.exists", return_value=False):
        importlib.reload(config)
        assert config.CONFIG["wind_intensity"] == 1.0

def test_config_env_override():
    with patch.dict(os.environ, {"CARTPOLE_WIND_INTENSITY": "5.0", "CARTPOLE_EPISODES": "10"}):
        importlib.reload(config)
        assert config.CONFIG["wind_intensity"] == 5.0
        assert config.CONFIG["episodes"] == 10

def test_config_env_override_invalid_type():
    with patch.dict(os.environ, {"CARTPOLE_WIND_INTENSITY": "not_a_float"}):
        importlib.reload(config)
        assert config.CONFIG["wind_intensity"] == 1.0

def test_config_env_override_string():
    with patch.dict(os.environ, {"CARTPOLE_WIND_MODE": "sinusoidal"}):
        importlib.reload(config)
        assert config.CONFIG["wind_mode"] == "sinusoidal"
