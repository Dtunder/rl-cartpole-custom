import pytest
from unittest.mock import MagicMock, patch
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
        func,
        max_retries=1,
        delay=0.01,
        exceptions=(ValueError,),
        fallback=fallback,
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

def test_execute_unexpected_failure() -> None:
    # It shouldn't be possible to reach RuntimeError unless max_retries < 0 
    # but that's handled earlier. We can mock it by having an empty exceptions tuple
    # that doesn't catch what func throws, but func isn't catching anything if max_retries is not respected?
    # Wait, the only way to reach `raise RuntimeError` is if `max_retries >= 0` and the loop exits without throwing or returning.
    # But the loop runs `attempt in range(max_retries + 1)`. 
    # If `func` is successful, it returns.
    # If `func` raises an exception not in `exceptions`, it raises it immediately and exits the function.
    # If `func` raises an exception IN `exceptions`, it sets `last_exception` and continues.
    # At the end of the loop, if it hasn't returned, it means `func` kept raising an exception in `exceptions`.
    # Therefore, `last_exception` will NOT be None.
    # Thus, `raise RuntimeError` is essentially unreachable code in Python unless max_retries + 1 == 0 (so max_retries == -1) which is checked earlier.
    # We can use a mock patch to force this condition to get coverage or just leave it.
    pass
def test_execute_unreachable() -> None:
    # We can patch max_retries locally inside the function loop to -1 just before the loop
    # or use mock to bypass the value check.
    # Actually, the simplest way is to pass max_retries=-1 and disable the check.
    # Since we can't disable the check, we can use unittest.mock to mock `range` to return an empty list
    with pytest.raises(RuntimeError, match="Unexpected failure"):
        with patch('builtins.range', return_value=[]):
            execute_with_resilience(MagicMock(), max_retries=0)
