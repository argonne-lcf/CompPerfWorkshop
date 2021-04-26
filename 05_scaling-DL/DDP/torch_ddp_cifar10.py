"""
pytorch10_cifar10.pyhvision.datasets import

Contains end-to-end training example (CIFAR10) on GPU or CPU with optional DDP.
"""
from __future__ import print_function
import argparse
from dataclasses import dataclass
import os
import shutil
import socket
import sys
import time
from typing import Callable

import numpy as np
import torch
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, models, transforms

# Ensure modules from `05_scaling-DL/utils/` is a are accessible
modulepath = os.path.dirname(os.path.dirname(__file__))
if modulepath not in sys.path:
    sys.path.append(modulepath)

from utils.io import Logger
from utils.parse_args import parse_args_torch as parse_args

# Set global variables for rank, local_rank and world_size
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
    print(f'Using DDP. Rank: {RANK} of {SIZE}')

except (ImportError, ModuleNotFoundError) as err:
    WITH_DDP = False
    LOCAL_RANK = 0
    SIZE = 1
    RANK = 0
    print(f'WARNING: MPI Initialization Failed!\n Exception: {err}')


# pylint:disable=invalid-name
logger = Logger()


# -----------------------------------------------------
# Helper object for aggregating relevant data objects
# -----------------------------------------------------
class DistributedDataObject:
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            batch_size: int,
            **kwargs: dict
    ):
        self.dataset = dataset
        self.sampler = torch.utils.data.distributed.DistributedSampler(
            self.dataset, num_replicas=SIZE, rank=RANK
        )
        self.loader = torch.utils.data.DataLoader(
            self.dataset, batch_size, sampler=self.sampler, **kwargs
        )



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

    @autocast()
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

    # Build datasets
    train_dataset = datasets.CIFAR10(
        'datasets', train=True, download=True, transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        'datasets', train=False, transform=transform
    )
    train_data = DistributedDataObject(train_dataset,
                                       args.batch_size, **kwargs)
    test_data = DistributedDataObject(test_dataset,
                                      batch_size=args.batch_size, **kwargs)

    return {'training': train_data, 'testing': test_data}


def build_model(args: dict) -> (nn.Module):
    """Helper method for building model using hyperparams from `args`."""
    model = AlexNet(num_classes=10)

    if args.device == 'gpu' or args.device.find('gpu') != -1:
        model.cuda()  # move model to GPU
    if WITH_DDP:
        model = DDP(model)

    return model


def metric_average(x: torch.tensor) -> (torch.tensor):
    """Compute global averages across all workers if using DDP. """
    if WITH_DDP:
        # Sum everything and divide by total size
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= SIZE
    else:
        pass

    return x


def evaluate(
    data: DistributedDataObject,
    model: nn.Module,
    loss_fn: Callable[[torch.tensor], torch.tensor],
    args: dict
):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data.loader:
            images, labels = data[0].to(args.device), data[1].to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


def train(
        epoch: int,
        data: DistributedDataObject,
        model: nn.Module,
        loss_fn: Callable[[torch.tensor], torch.tensor],
        optimizer: optim.Optimizer,
        args: dict,
        scaler: GradScaler=None,
):
    model.train()
    # Horovod: set epoch to sampler for shuffling
    data.sampler.set_epoch(epoch)
    running_loss = torch.tensor(0.0)
    training_acc = torch.tensor(0.0)

    if args.device == 'gpu':
        running_loss = running_loss.cuda()
        training_acc = training_acc.cuda()

    for batch_idx, (batch, target) in enumerate(data.loader):
        if args.cuda:
            batch, target = batch.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(batch)
        loss = loss_fn(output, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        acc = pred.eq(target.data.view_as(pred)).cpu().float().sum()

        training_acc += acc
        running_loss += loss.item()

        #  running_loss = running_loss / len(data.sampler)
        #  running_acc = running_acc / len(data.sampler)
        #  loss_avg = metric_average(running_loss)
        #  acc_avg = metric_average(running_acc)
        #  if RANK == 0 and batch_idx % args.log_interval == 0:
        if batch_idx % args.log_interval == 0:
            metrics_ = {
                'epoch': epoch,
                'batch_loss': loss.item() / args.batch_size,
                'running_loss': running_loss / len(data.sampler),
                'batch_acc': acc / args.batch_size,
                'training_acc': training_acc / len(data.sampler),
            }
            jdx = batch_idx * len(batch)
            frac = 100. * batch_idx / len(data.loader)
            pre = [f'[{RANK}]',
                   f'[{jdx:05}/{len(data.sampler):05} ({frac:>4.3g}%)]']
            mstr = ' '.join([
                f'{str(k):>5}: {v:<7.4g}' for k, v in metrics_.items()
            ])
            logger.log(' '.join([*pre, mstr]))

            #  str0 = f'[{jdx:5<}/{len(data.sampler):5<} ({frac:>3.3g}%)]'
            #  str1 = ' '.join([
            #      f'{str(k):>5}: {v:<7.4g}' for k, v in metrics_.items()
            #  ])
            #  logger.log(' '.join([str0, str1]))

    running_loss = running_loss / len(data.sampler)
    training_acc = training_acc / len(data.sampler)
    loss_avg = metric_average(running_loss)
    training_acc = metric_average(training_acc)
    if RANK == 0:
        logger.log('\n'.join([
            f'training set, avg loss: {loss_avg:.4g}',
            f'training set, accuracy: {training_acc * 100:.2f}%'
        ]))


def test(
        data: DistributedDataObject,
        model: nn.Module,
        loss_fn: Callable[[torch.tensor], torch.tensor],
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
        test_loss += loss_fn(output, target).item()
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
    loss_fn = nn.CrossEntropyLoss()
    # Horovod: scale learning rate by the number of GPUs
    #optimizer = optim.SGD(model.parameters(),
    #                      lr=args.lr * SIZE,
    #                      momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr * SIZE)
    epoch_times = []

    logger.log(f'Training on {args.device}, cuda: {args.cuda}')
    scaler = GradScaler(enabled=args.cuda)
    train_data = data['training']
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train(args=args,
              epoch=epoch,
              model=model,
              optimizer=optimizer,
              data=train_data,
              loss_fn=loss_fn,
              scaler=scaler)
        test(data['testing'], model, loss_fn, args)
        epoch_times.append(time.time() - t0)

    end = time.time()
    avg_dt = np.mean(epoch_times[-5:])
    if RANK == 0:
        logger.log(
            f'Total training time: {end - start:.5g}\n'
            f'Average time per epoch in the last 5: {avg_dt:.5g}'
        )


if __name__ == '__main__':
    args = parse_args()
    _ = main(args)
