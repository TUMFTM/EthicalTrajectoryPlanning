"""Module for timeoutig code."""
import signal

from commonroad_helper_functions.exceptions import ExecutionTimeoutError


class Timeout:
    """Context manager that raises exception after sec seconds."""

    def __init__(self, sec: int, section_name: str):
        """Initialize the Timeout class.

        Args:
            sec (float): Seconds until something is timed out.
        """
        self.sec = sec
        self.section_name = section_name

    def __enter__(self):
        """Enter the Timout class."""
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        """Exit the Timout class."""
        signal.alarm(0)  # disable alarm

    def raise_timeout(self, *args):
        """Raise a timeout."""
        raise ExecutionTimeoutError(self.section_name + " timed out")
