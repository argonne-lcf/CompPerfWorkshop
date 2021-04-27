import os
import shutil
import datetime
import torch
from torchvision import datasets, transforms

TERM_WIDTH, TERM_HEIGHT = shutil.get_terminal_size(fallback=(156, 50))


class DistributedDataObject:
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            batch_size: int,
            rank: int,
            num_workers: int,
            **kwargs: dict
    ):
        self.dataset = dataset
        self.sampler = torch.utils.data.distributed.DistributedSampler(
            self.dataset, num_replicas=num_workers, rank=rank
        )
        self.loader = torch.utils.data.DataLoader(
            self.dataset, batch_size, sampler=self.sampler, **kwargs
        )



def prepare_datasets(
        args: dict,
        rank: int,
        num_workers: int,
        data: str = 'CIFAR10',
) -> (dict):
    """Build `train_data`, `test_data` as `DataObject`'s for easy access."""

    kwargs = {'rank': rank, 'num_workers': num_workers, 'pin_memory': False}
    #  if args.device.find('gpu') != -1:
    if not torch.cuda.is_available():
        kwargs = {'rank': 0, 'num_workers': 1, 'pin_memory': True}

    if str(data).lower() not in ['cifar10', 'mnist']:
        raise ValueError('Expected `data` to be one of "cifar10", "mnist"')

    if str(data).lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Build datasets
        datadir = os.path.abspath('datasets/CIFAR10')
        train_dataset = datasets.CIFAR10(
            datadir, train=True, download=True, transform=transform,
        )
        test_dataset = datasets.CIFAR10(
            datadir, train=False, transform=transform
        )
    elif str(data).lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        datadir = os.path.abspath('datasets/MNIST')
        train_dataset = datasets.MNIST(
            datadir, train=True, download=True, transform=transform,
        )
        test_dataset = datasets.MNIST(
            datadir, train=False, download=True, transform=transform,
        )

    print(f'rank: {rank}, num_workers: {num_workers}')
    train_data = DistributedDataObject(dataset=train_dataset,
                                       batch_size=args.batch_size, **kwargs)
    test_data = DistributedDataObject(test_dataset,
                                      args.test_batch_size, **kwargs)

    return {'training': train_data, 'testing': test_data}



def get_timestamp(fstr=None):
    """Get formatted timestamp."""
    now = datetime.datetime.now()
    if fstr is None:
        return now.strftime('%Y-%m-%d-%H%M%S')
    return now.strftime(fstr)


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
