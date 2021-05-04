import os
import shutil
import datetime
#  import torch
#  from torchvision import datasets, transforms

TERM_WIDTH, TERM_HEIGHT = shutil.get_terminal_size(fallback=(156, 50))

# noqa: E999
# pylint:disable=too-few-public-methods,redefined-outer-name
# pylint:disable=missing-function-docstring,missing-class-docstring
class Console:
    """Fallback console object used as in case `rich` isn't installed."""
    @staticmethod
    def log(s, *args, **kwargs):
        now = get_timestamp('%X')
        print(f'[{now}]  {s}', *args, **kwargs)


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


def print_metrics(metrics: dict, pre: list = None, logger: Logger = None):
    if logger is None:
        logger = Logger()

    mstr = ' '.join([
        f'{str(k):<5}: {v:<7.4g}' if isinstance(v, (float))
        else f'{str(k):<5}: {v:<7g}' for k, v in metrics.items()
    ])

    if pre is not None:
        logger.log(' '.join([*pre, mstr]))
    else:
        logger.log(mstr)


def get_timestamp(fstr=None):
    """Get formatted timestamp."""
    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime('%Y-%m-%d-%H%M%S')
    return now.strftime(fstr)

