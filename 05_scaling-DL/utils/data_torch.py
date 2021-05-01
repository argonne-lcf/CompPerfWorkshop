import sys
import os
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

here = os.path.abspath(os.path.dirname(__file__))
modulepath = os.path.dirname(here)
if modulepath not in sys.path:
    sys.path.append(modulepath)

DATA_PATH = os.path.join(modulepath, 'datasets')

class DistributedDataObject:
    """Object for grouping commonly used data structures."""
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
        data: str = 'MNIST',
) -> (dict):
    """Build `train_data`, `test_data` as `DataObject`'s for easy access."""

    #  kwargs = {'rank': rank, 'num_workers': num_workers, 'pin_memory': False}
    kwargs = {'rank': rank, 'num_workers': num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}
    #  if args.device.find('gpu') != -1:
    #  if not torch.cuda.is_available():
        #  kwargs = {'rank': 0, 'num_workers': 1, 'pin_memory': True}

    if str(data).lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Build datasets
        #  datadir = os.path.abspath('datasets/CIFAR10')
        datadir = os.path.join(DATA_PATH, 'CIFAR10')
        train_dataset = datasets.CIFAR10(
            datadir, train=True, download=True, transform=transform,
        )
        test_dataset = datasets.CIFAR10(
            datadir, train=False, transform=transform, download=True,
        )
    elif str(data).lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        #  datadir = os.path.abspath('datasets/MNIST')
        datadir = os.path.join(DATA_PATH, 'MNIST')
        train_dataset = datasets.MNIST(
            datadir, train=True, download=True, transform=transform,
        )
        test_dataset = datasets.MNIST(
            datadir, train=False, download=True, transform=transform,
        )

    else:
        raise ValueError('Expected `data` to be one of "cifar10", "mnist"')


    print(f'rank: {rank}, num_workers: {num_workers}')
    train_data = DistributedDataObject(train_dataset, batch_size=args.batch_size, **kwargs)
    test_data = DistributedDataObject(test_dataset, batch_size=args.test_batch_size, **kwargs)

    return {'training': train_data, 'testing': test_data}
