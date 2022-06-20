"""Class for easy timing different code snippets and functions."""
import time


class ExecTimer:
    """Class for easy timing of multiple code sections and function calls.

    This class provides a start and stop method, as well as
    a timing context manager
    and a timing decorator.
    """

    def __init__(self, timing_enabled=True):
        """Create a timer.

        Use this object to time all parts of a run.
        If subparts of functions have to be timed -> hand it over to
        subfunction.
        Args:
            timing_enabled (bool, optional):    If timing_enabled is set to
                                                false timing is deactivated
                                                completely.
                                                Defaults to True.
        """
        self.running = {}
        self.finished = {}
        self._timing_enabled = timing_enabled

    def reset(self):
        """Reset the timer object."""
        self.running = {}
        self.finished = {}

    def get_timing_dict(self):
        """Return the assembled timing dict.

        Returns:
            dict: timing dict.
        """
        return self.finished

    def start_timer(self, label):
        """Start a timer.

        Args:
            label (str): label of the timer.
                        label can be given as path.
                        e.g label1/label2
                        The timing dictionary then is assembled in this
                        structure.
        """
        if self._timing_enabled:

            self.running[label] = time.time()

    def stop_timer(self, label):
        """Start a timer.

        Args:
            label (str): label of the timer.
        """
        if self._timing_enabled:
            timer_time = time.time() - self.running.pop(label)
            if label in self.finished:
                self.finished[label].append(timer_time)
            else:
                self.finished[label] = [timer_time]

    def time_with_cm(self, label):
        """Time a code region using a context manager.

        Usage: See example in the timers.py script at the bottom.
        Args:
            label (str): label of the timer.
        """
        return self.TimerCM(self, label)

    def time_with_dec(self, label):
        """Time a code region using a decorator.

        Usage: See example in the timers.py script at the bottom.
        Args:
            label (str): label of the timer.
        """

        def decorator_function(function_to_decorate):
            def wrapper_function(*args, **kwargs):
                self.start_timer(label)
                result = function_to_decorate(*args, **kwargs)
                self.stop_timer(label)
                return result

            return wrapper_function

        return decorator_function

    class TimerCM:
        """Create a context manager for timing."""

        def __init__(self, timer, label):
            """Create context manager object.

            Args:
                timer (obj): superordinate timer class object
                label (str): label of the timer.
            """
            self.label = label
            self.timer = timer

        def __enter__(self):
            """Start the timer."""
            self.timer.start_timer(self.label)

        def __exit__(self, exc_type, exc_value, exc_tb):
            """Exit the timer."""
            self.timer.stop_timer(self.label)


if __name__ == "__main__":
    # Always create a instance of the timer to hold all timing values.
    timer = ExecTimer()

    # Example time with start() and stop() function - USE THE INSTANCE
    timer.start_timer("timed_with_start_stop")
    time.sleep(1)
    timer.stop_timer("timed_with_start_stop")

    # Example: time with context manager - USE THE INSTANCE
    with timer.time_with_cm("timed_with_context_manager"):
        time.sleep(2)

    # Example decorate a function - USE THE INSTANCE
    @timer.time_with_dec("timed_with_decorator")
    def test_function():
        """Arbitrary help function."""
        time.sleep(3)

    # Call the timed function. Timing is done with at function call
    # If function is called multiple times it stacks up a list
    test_function()
    test_function()

    with timer.time_with_cm("assemble/example1"):
        time.sleep(1)
    with timer.time_with_cm("assemble/example2"):
        time.sleep(2)

    print(timer.get_timing_dict())
