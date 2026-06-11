import time
import logging
from typing import Any, Callable, Optional, Tuple, Type, TypeVar
from config import CONFIG

T = TypeVar("T")

logger = logging.getLogger(__name__)


def execute_with_resilience(
    func: Callable[..., T],
    *args: Any,
    max_retries: Optional[int] = None,
    delay: Optional[float] = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    fallback: Optional[Callable[..., T]] = None,
    **kwargs: Any,
) -> T:
    """
    Executes a given function with retry logic and an optional fallback mechanism.

    This resilience layer helps verify system robustness by gracefully handling
    temporary failures (e.g., timeouts, connection issues, or bad configurations).

    Args:
        func (Callable[..., T]): The target function to execute.
        *args (Any): Positional arguments to pass to the target function.
        max_retries (int): The maximum number of retry attempts before giving up or falling back.
            Must be non-negative. Defaults to 3.
        delay (float): The delay in seconds between consecutive retry attempts.
            Must be non-negative. Defaults to 1.0.
        exceptions (Tuple[Type[Exception], ...]): A tuple of exception classes that should trigger a retry.
            Defaults to (Exception,).
        fallback (Optional[Callable[..., T]]): An optional fallback function to execute if all retries fail.
            It must accept the same arguments as `func`. Defaults to None.
        **kwargs (Any): Keyword arguments to pass to the target function.

    Returns:
        T: The result of the target function if successful, or the result of the fallback function if provided.

    Raises:
        TypeError: If `max_retries` is not an integer or `delay` is not a float/int.
        ValueError: If `max_retries` or `delay` is negative.
        Exception: The last exception raised by the target function if all retries fail and no fallback is provided.
    """
    max_retries = (
        max_retries if max_retries is not None else CONFIG["max_retries"]
    )
    delay = delay if delay is not None else CONFIG["resilience_delay"]

    if not isinstance(max_retries, int):
        raise TypeError(
            f"max_retries must be an integer, got {type(max_retries).__name__}"
        )
    if max_retries < 0:
        raise ValueError(
            f"max_retries must be non-negative, got {max_retries}"
        )
    if not isinstance(delay, (int, float)):
        raise TypeError(f"delay must be a number, got {type(delay).__name__}")
    if delay < 0:
        raise ValueError(f"delay must be non-negative, got {delay}")

    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            func_name = getattr(func, "__name__", str(func))
            if attempt < max_retries:
                logger.warning(
                    f"Execution of {func_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay} seconds..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"All {max_retries + 1} attempts failed for {func_name}. Last error: {e}"
                )

    if fallback is not None:
        func_name = getattr(func, "__name__", str(func))
        logger.info(f"Executing fallback for {func_name}...")
        return fallback(*args, **kwargs)

    if last_exception is not None:
        raise last_exception

    raise RuntimeError("Unexpected failure in execute_with_resilience")
