import shutil

TERM_WIDTH, TERM_HEIGHT = shutil.get_terminal_size(fallback=(156, 50))
class Console:
    """Fallback console object used as in case `rich` isn't installed."""
    # pylint:disable=too-few-public-methods,redefined-outer-name
    # pylint:disable=missing-function-docstring,missing-class-docstring
    @staticmethod
    def log(s, *args, **kwargs):  # noqa:E999
        print(s, *args, **kwargs)


class Logger:
    """Logger class for pretty printing metrics during training/testing."""
    def __init__(self):
        try:
            # pylint:disable=import-outside-toplevel
            from rich.console import Console as RichConsole
            console = RichConsole(log_path=False, width=TERM_WIDTH)
        except (ImportError, ModuleNotFoundError):
            console = Console()

        self.console = console

    def log(self, s, *args, **kwargs):
        """Print `s` using `self.console` object."""
        self.console.log(s, *args, **kwargs)
