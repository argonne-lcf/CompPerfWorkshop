"""
pytorch10_cifar10.py

Contains end-to-end training example (CIFAR10) on GPU or CPU with optional DDP.
"""
from __future__ import print_function

import argparse
import os
import shutil
import socket
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms, models

# Set global variables for rank, local_rank and world_size
#  TERM_WIDTH, TERM_HEIGHT = os.get_terminal_size()
TERM_WIDTH, TERM_HEIGHT = shutil.get_terminal_size(fallback=(156, 50))
try:
    from mpi4py import MPI

    WITH_DDP = True
    LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', None)
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()

    # Pytorch will look for these:
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(SIZE)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(LOCAL_RANK)

    # It will want the master address, too, which we'll broadcast
    if RANK == 0:
        MASTER_ADDR = socket.gethostname()
    else:
        MASTER_ADDR = None

    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = str(2345)  # any open port

except (ImportError, ModuleNotFoundError) as err:
    WITH_DDP = False
    LOCAL_RANK = 0
    SIZE = 1
    RANK = 0
    print(f'WARNING: MPI Initialization Failed!\n Exception: {err}')


# --------------------------------------------------------------------
# Helper objects for pretty-printing metrics during training/testing
# --------------------------------------------------------------------
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


# pylint:disable=invalid-name
logger = Logger()


# -----------------------------------------------------
# Helper object for aggregating relevant data objects
# -----------------------------------------------------
@dataclass
class DataObject:
    dataset: torch.utils.data.Dataset
    sampler: torch.utils.data.Sampler
    loader: torch.utils.data.DataLoader



# --------------------------------
# Model constructor / definition
# --------------------------------
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # -----
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # -----
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # -----
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # -----
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.tensor) -> (torch.tensor):
        """Call the model on input data `x`."""
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


def setup_ddp(args: dict):
    """Specify device to use and setup backend for MPI communication."""
    if args.device not in ['gpu', 'cpu']:
        raise ValueError('Expected `args.device` to be one of "gpu", "cpu"')

    backend = 'nccl' if args.device == 'gpu' else 'gloo'
    if WITH_DDP:
        dist.init_process_group(backend=backend, init_method='env://')
    if args.device == 'gpu':
        # DDP: pin GPU to local rank
        # toch.cuda.set_device(int(LOCAL_RANK))
        torch.cuda.manual_seed(args.seed)

    if args.num_threads != 0:
        torch.set_num_threads(args.num_threads)

    if RANK == 0:
        logger.log(f'(setup torch threads) number of threads: {torch.get_num_threads()}')


def prepare_datasets(args: dict) -> (dict):
    """Build `train_data`, `test_data` as `DataObject`'s for easy access."""
    kwargs = {}
    if args.device.find('gpu') != -1:
        kwargs = {'num_workers': 1, 'pin_memory': True}

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = datasets.CIFAR10(
        'datasets/', train=True, download=True, transform=transform,
    )
    # Horovod: use DistributedSampler to partition the training data
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=SIZE, rank=RANK
    )
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               sampler=train_sampler, **kwargs)
    test_dataset = datasets.CIFAR10('datasets', train=False,
                                    transform=transform)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=SIZE, rank=RANK
    )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.test_batch_size,
                                              sampler=test_sampler, **kwargs)

    train_data = DataObject(train_dataset, train_sampler, train_loader)
    test_data = DataObject(test_dataset, test_sampler, test_loader)

    return {'training': train_data, 'testing': test_data}


def build_model(args: dict) -> (nn.Module):
    """Helper method for building model using hyperparams from `args`."""
    model = AlexNet(num_classes=10)
    #  model =models.resnet18(pretrained=False)

    if args.device == 'gpu' or args.device.find('gpu') != -1:
        model.cuda()  # move model to GPU
    if WITH_DDP:
        model = DDP(model)

    return model


#  def metric_average(val, name):
def metric_average(x: torch.tensor) -> (torch.tensor):
    """Compute global averages across all workers if using DDP. """
    if WITH_DDP:
        # Sum everything and divide by total size
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= SIZE
    else:
        pass

    return x



def train(
        epoch: int,
        data: DataObject,
        model: nn.Module,
        criterion: Callable[[torch.tensor], torch.tensor],
        optimizer: optim.Optimizer,
        args: dict
):
    logger.log(f'Training on {args.device}, cuda: {args.cuda}')

    model.train()
    # Horovod: set epoch to sampler for shuffling
    data.sampler.set_epoch(epoch)
    running_loss = torch.tensor(0.0)
    running_acc = torch.tensor(0.0)
    if args.device == 'gpu':
        running_loss = running_loss.cuda()
        running_acc = running_acc.cuda()
    for batch_idx, (batch, target) in enumerate(data.loader):
        if args.cuda:
            batch, target = batch.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        acc = pred.eq(target.data.view_as(pred)).cpu().float().sum()
        #  running_acc += pred.eq(target.data.view_as(pred)).cpu().float().sum()
        running_acc += acc
        running_loss += loss.item()

        #  if batch_idx % args.log_interval == 0 and RANK == 0:
        #      # Horovod: use train_sampler to determine
        #      # the number of examples in this worker's partition
        #      tstrs = {
        #          'epoch': epoch,
        #          'loss': loss.item() / args.batch_size,
        #          'batch': batch_idx * len(batch),
        #          'percent_complete': 100. * batch_idx / len(data.loader),
        #      }
        #      logger.log(' '.join([f'{k}: {v:.4g}' for k, v in tstrs.items()]))

        running_loss = running_loss / len(data.sampler)
        running_acc = running_acc / len(data.sampler)
        loss_avg = metric_average(running_loss)
        acc_avg = metric_average(running_acc)
        if RANK == 0 and batch_idx % args.log_interval == 0:
            batch_metrics = {
                'epoch': epoch,
                'batch_loss': loss.item() / args.batch_size,
                'global_loss': loss_avg,
                'batch_acc': acc / args.batch_size,
                #  'batch': batch_idx * len(batch),
                #  '% done': 100. * batch_idx / len(data.loader),
                #  'running_loss': running_loss,
                #  'running_accuracy': running_acc,
                'global_acc': acc_avg,
            }
            jdx = batch_idx * len(batch)
            frac = 100. * batch_idx / len(data.loader)
            str0 = f'[{jdx:5<}/{len(data.sampler):5<} ({frac:>3.3g}%)]'
            str1 = ' '.join([
                f'{str(k):>5}: {v:<7.4g}' for k, v in batch_metrics.items()
            ])
            logger.log(' '.join([str0, str1]))

            #mstr = ' '.join([
            #    f'{k}: {v:>7.5f}' for k, v in metrics.items()
            #])

            #logger.log(
            #    f'[{jdx:5<}/{len(data.sampler):5<} ({frac:>3.3g}%)]'
            #    f'  {mstr}'
            #)


def test(
        data: DataObject,
        model: nn.Module,
        criterion: Callable[[torch.tensor], torch.tensor],
        args: dict
):
    model.eval()
    test_loss = torch.tensor(0.0)
    test_accuracy = torch.tensor(0.0)
    if args.device == 'gpu':
        test_loss = test_loss.cuda()
        test_accuracy = test_accuracy.cuda()

    for batch, target in data.loader:
        if args.cuda:
            batch, target = batch.cuda(), target.cuda()

        output = model(batch)
        # Sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).float().sum()

    # Horovod: use test_sampler to determine the number of examples in this
    # workers partition
    test_loss /= len(data.sampler)
    test_accuracy /= len(data.sampler)

    # Horovod: average metric values across workers
    loss_avg = metric_average(test_loss)
    acc_avg = metric_average(test_accuracy)

    # Horovod: print output only on chief rank
    if RANK == 0:
        avg_metrics = {
            'loss_avg': loss_avg,
            'accuracy_avg': acc_avg,
        }
        logger.log('    ' + ' '.join(
            [f'{k}: {v:.3g}' for k, v in avg_metrics.items()]
        ))


def main(args):
    start = time.time()
    setup_ddp(args)
    data = prepare_datasets(args)

    model = build_model(args)
    criterion = nn.CrossEntropyLoss()
    # Horovod: scale learning rate by the number of GPUs
    #optimizer = optim.SGD(model.parameters(),
    #                      lr=args.lr * SIZE,
    #                      momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr * SIZE)
    epoch_times = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train(epoch, data['training'], model, criterion, optimizer, args)
        test(data['testing'], model, criterion, args)
        epoch_times.append(time.time() - t0)

    end = time.time()
    avg_dt = np.mean(epoch_times[-5:])
    if RANK == 0:
        logger.log(
            f'Total training time: {end - start:.5g}\n'
            f'Average time per epoch in the last 5: {avg_dt:.5g}'
        )


def parse_args(*args):
    """Parse command line arguments containing settings for training."""
    description = 'PyTorch CIFAR10 Example using DDP'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '--batch_size', type=int, default=64, required=False,
        help='input `batch_size` for training (default: 64)',
    )
    parser.add_argument(
        '--test_batch_size', type=int, default=64, required=False,
        help='input `batch_size` for testing (default: 64)',
    )
    parser.add_argument(
        '--epochs', type=int, default=10, required=False,
        help='training epochs (default: 10)',
    )
    parser.add_argument(
        '--lr', type=float, default=0.01, required=False,
        help='learning rate (default: 0.01)',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.5, required=False,
        help='SGD momentum (default: 0.5)',
    )
    parser.add_argument(
        '--seed', type=int, default=42, required=False,
        help='random seed (default: 42)',
    )
    parser.add_argument(
        '--log_interval', type=int, default=10, required=False,
        help='how many batches to wait before logging training status',
    )
    parser.add_argument(
        '--fp16_allreduce', action='store_true', default=False, required=False,
        help='use fp16 compression during allreduce',
    )
    parser.add_argument(
        '--device', default='cpu', choices=['cpu', 'gpu'], required=False,
        help='whether this is running on gpu or cpu'
    )
    parser.add_argument(
        '--num_threads', type=int, default=0, required=False,
        help='set number of threads per worker'
    )
    args = parser.parse_args()
    args.__dict__['cuda'] = torch.cuda.is_available()

    return args



if __name__ == '__main__':
    args = parse_args()
    _ = main(args)
