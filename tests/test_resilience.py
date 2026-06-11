import pytest
from unittest.mock import MagicMock
from resilience import execute_with_resilience


def test_execute_success() -> None:
    func = MagicMock(return_value="success")
    result = execute_with_resilience(func, 1, 2, kwarg1="a")
    assert result == "success"
    func.assert_called_once_with(1, 2, kwarg1="a")


def test_execute_retry_success() -> None:
    func = MagicMock(side_effect=[ValueError("error"), "success"])
    result = execute_with_resilience(
        func, max_retries=1, delay=0.01, exceptions=(ValueError,)
    )
    assert result == "success"
    assert func.call_count == 2


def test_execute_failure_after_retries() -> None:
    func = MagicMock(side_effect=ValueError("persistent error"))
    with pytest.raises(ValueError, match="persistent error"):
        execute_with_resilience(
            func, max_retries=2, delay=0.01, exceptions=(ValueError,)
        )
    assert func.call_count == 3


def test_execute_fallback() -> None:
    func = MagicMock(side_effect=ValueError("persistent error"))
    fallback = MagicMock(return_value="fallback_success")
    result = execute_with_resilience(
        func, max_retries=1, delay=0.01, exceptions=(ValueError,), fallback=fallback
    )
    assert result == "fallback_success"
    assert func.call_count == 2
    fallback.assert_called_once()


def test_execute_invalid_args() -> None:
    func = MagicMock()
    with pytest.raises(TypeError, match="max_retries must be an integer"):
        execute_with_resilience(func, max_retries="abc")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_retries must be non-negative"):
        execute_with_resilience(func, max_retries=-1)
    with pytest.raises(TypeError, match="delay must be a number"):
        execute_with_resilience(func, delay="slow")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="delay must be non-negative"):
        execute_with_resilience(func, delay=-1.0)
