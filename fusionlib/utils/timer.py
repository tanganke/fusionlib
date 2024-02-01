import functools
import logging
import sys
import time
from typing import Callable, Optional, Union

logger = logging.getLogger(__name__)


class timer:
    """
    Usage:

        ```python
        # Usage as a decorator
        @timer
        def my_func():
            time.sleep(1)

        my_func()

        # Usage as a context manager
        with timer():
            time.sleep(1)
        ```
    """

    def __init__(
        self,
        func_or_msg: Optional[Union[Callable, str]] = None,
        msg: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        loglevel=logging.INFO,
        human_readable: bool = True,
    ) -> None:
        if isinstance(func_or_msg, str):
            self.func = None
            self.msg = func_or_msg
        else:
            self.func = func_or_msg
            self.msg = msg
        self.loglevel = loglevel
        self.human_readable = human_readable

        if logger is None:
            self._logger_fn = print
        else:
            self._logger_fn = functools.partial(
                logger.log, level=loglevel, stacklevel=3
            )

    @property
    def elapsed_time(self):
        return time.time() - self.start_time

    def reset(self):
        self.start_time = time.time()

    def __enter__(self) -> None:
        """
        Sets the start time and logs an optional message indicating the start of the code block execution.

        Args:
            msg: str, optional message to log
        """
        self.start_time = time.time()
        if self.msg is None:
            if self.func is not None:
                self.msg = self.func.__name__
            else:
                self.msg = ""
        self._logger_fn(f"[START] {self.msg}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Calculates the elapsed time and logs it, along with an optional message indicating the end of the code block execution.
        """
        elapsed_time = self.elapsed_time
        self._logger_fn(f"[END] {self.msg} {elapsed_time:.2f}s")

        # If an exception occurred, log it
        if exc_type is not None:
            logger.error(f"An error occurred: {exc_val}")
            # Return True to suppress the exception, False otherwise
            return False

    def __call__(self, *args, **kwargs):
        self.__enter__()
        try:
            assert self.func is not None, "func must be set if using as a decorator"
            return self.func(*args, **kwargs)
        except Exception as e:
            # Pass exception info to __exit__
            exc_type, exc_val, exc_tb = sys.exc_info()
            self.__exit__(exc_type, exc_val, exc_tb)
            raise  # re-raise the exception
        finally:
            # No exception, so pass None as all three arguments
            self.__exit__(None, None, None)
