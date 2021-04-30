"""
torch_ddp.py

Modified from original: https://leimao.github.io/blog/PyTorch-Distributed-Training/
"""
import sys
import torch
import time
import json
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from typing import Callable
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import os
import random
import numpy as np

here = os.path.abspath(os.path.dirname(__file__))
modulepath = os.path.dirname(here)
if modulepath not in sys.path:
    sys.path.append(modulepath)

import utils.io as io
from utils.io import Logger
from utils.parse_args import parse_args_ddp

logger = Logger()

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
        data: str = 'MNIST',
) -> (dict):
    """Build `train_data`, `test_data` as `DataObject`'s for easy access."""

    kwargs = {'rank': rank, 'num_workers': num_workers, 'pin_memory': False}
    #  if args.device.find('gpu') != -1:
    #  if not torch.cuda.is_available():
        #  kwargs = {'rank': 0, 'num_workers': 1, 'pin_memory': True}

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


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def cleanup():
    dist.destroy_process_group()


def get_backend():
    if dist.is_nccl_available():
        return 'nccl'
    if dist.is_mpi_available():
        return 'mpi'
    if dist.is_gloo_available():
        return 'gloo'

    raise ValueError('No backend found.')


def setup(
        rank: str = '0',
        master_addr: str = 'localhost',
        master_port: str = '4921',
        backend: str = 'gloo'
):
    os.environ['MASTER_ADDR'] = str(master_addr)
    os.environ['MASTER_PORT'] = str(master_port) # can be anything
    os.environ['RANK'] = str(rank)
    # initialize the process group
    #  dist.init_process_group(backend, rank=rank, world_size=world_size)
    #  dist.init_process_group(backend, rank=rank, world_size=world_size)
    dist.init_process_group(backend=backend, init_method='env://')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


def metric_average(x: torch.Tensor) -> (torch.Tensor):
    """Compute global averages across all workers if using DDP. """
    x = torch.tensor(x)
    #  if with_ddp:
    if dist.is_initialized():
        # Sum everything and divide by total size
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    else:
        pass

    return x.item()


def train(
        epoch: int,
        data: DistributedDataObject,
        device: torch.device,
        rank: int,
        model: nn.Module,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        optimizer: optim.Optimizer,
        args: dict,
        scaler: GradScaler=None,
):
    model.train()
    # Horovod: set epoch to sampler for shuffling
    data.sampler.set_epoch(epoch)
    running_loss = torch.tensor(0.0)
    training_acc = torch.tensor(0.0)
    if torch.cuda.is_available():
        running_loss = running_loss.to(device)
        training_acc = training_acc.to(device)
    #  running_loss = 0.0
    #  training_acc = 0.0

    for batch_idx, (batch, target) in enumerate(data.loader):
        if torch.cuda.is_available():
            batch, target = batch.to(device), target.to(device)

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

        if batch_idx % args.log_interval == 0:
            metrics_ = {
                'epoch': epoch,
                'batch_loss': loss.item() / args.batch_size,
                'running_loss': running_loss / len(data.sampler),
                'batch_acc': acc.item() / args.batch_size,
                'training_acc': training_acc / len(data.sampler),
            }

            jdx = batch_idx * len(batch)
            frac = 100. * batch_idx / len(data.loader)
            pre = [f'[{rank}]',
                   f'[{jdx:>5}/{len(data.sampler):<5} ({frac:>03.1f}%)]']
            io.print_metrics(metrics_, pre=pre, logger=logger)


    running_loss = running_loss / len(data.sampler)
    training_acc = training_acc / len(data.sampler)
    loss_avg = metric_average(running_loss)
    training_acc = metric_average(training_acc)
    if rank == 0:
        logger.log(f'training set; avg loss: {loss_avg:.4g}, '
                   f'accuracy: {training_acc * 100:.2f}%')


def main():
    args = parse_args_ddp()
    args.__dict__['cuda'] = torch.cuda.is_available()
    with_cuda = torch.cuda.is_available()

    backend = 'nccl' if with_cuda else 'gloo'
    dist.init_process_group(backend=backend)
    #  setup(args.local_rank)
    #  setup(args.local_rank, os.en

    args.cuda = with_cuda

    local_rank = args.local_rank
    #  epochs = args.epochs
    #  batch_size = args.batch_size
    #  lr = args.lr
    #  random_seed = args.random_seed
    #  model_dir = args.model_dir
    #  model_filename = args.model_filename
    resume = args.resume

    world_size = 1 if not dist.is_available() else dist.get_world_size()
    backend = 'nccl' if with_cuda else 'gloo'
    #  setup(local_rank, world_size, backend=args.backend)

    # Create directories outside the PyTorch program
    # Do not create directory here because it is not multiprocess safe
    '''
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    '''

    model_filepath = os.path.join(args.model_dir, args.model_filename)

    # We need to use seeds to make sure that the models initialized in different processes are the same
    set_random_seeds(random_seed=args.random_seed)

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #  backend = 'nccl' if with_cuda else 'gloo'
    #  dist.init_process_group(backend=backend)
    #  torch.distributed.init_process_group(backend=backend)
    # torch.distributed.init_process_group(backend="gloo")

    # Encapsulate the model on the GPU assigned to the current process
    #model = torchvision.models.resnet18(pretrained=False)
    model = Net()

    if with_cuda:
        if world_size > 1:
            device = torch.device(f'cuda:{args.local_rank}')
        else:
            device = torch.device('cuda')
        model = model.to(device)
        ddp_model = DDP(model,
                        device_ids=[args.local_rank],
                        output_device=args.local_rank)

    else:
        device = torch.device('cpu')
        ddp_model = DDP(model)
                    #  device_ids=[args.local_rank],
                    #  device_ids=[args.local_rank]
                    #  output_device=args.local_rank)

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if resume == True:
        if with_cuda:
            map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        #  else:
        #      map_location = {'0': f'{local_rank}'}

        state_dict = torch.load(model_filepath, map_location=map_location)
        ddp_model.load_state_dict(state_dict)

    # Prepare dataset and dataloader
    data = prepare_datasets(args, rank=local_rank,
                            num_workers=world_size,
                            data='mnist')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(),
                           lr=args.lr, weight_decay=1e-5)
    #  optimizer = optim.SGD(ddp_model.parameters(), lr=lr, momentum=0.9,
    #                        weight_decay=1e-5)

    # Loop over the dataset multiple times
    epoch_times = []
    for epoch in range(args.epochs):
        t0 = time.time()
        train(epoch, data['training'], device=device, rank=local_rank,
              model=ddp_model, loss_fn=criterion,
              optimizer=optimizer, args=args, scaler=None)

        if epoch > 2:
            epoch_times.append(time.time() - t0)

        if epoch % 10 == 0:
            if local_rank == 0:
                accuracy = evaluate(model=ddp_model, device=device,
                                    test_loader=data['testing'].loader)
                torch.save(ddp_model.state_dict(), model_filepath)
                logger.log('-' * 75)
                logger.log(f'Epoch: {epoch}, Accuracy: {accuracy}')
                logger.log('-' * 75)


    if local_rank == 0:
        epoch_times_str = ', '.join(str(x) for x in epoch_times)
        logger.log('Epoch times:')
        logger.log(epoch_times_str)

        #outdir = os.path.join(os.getcwd(), f'results_mnist_size{world_size}')
        outdir = os.path.join(os.getcwd(), 'results_mnist', f'size{world_size}')
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        args_file = os.path.join(outdir, f'args_size{world_size}.json')
        logger.log(f'Saving args to: {args_file}.')

        with open(args_file, 'at') as f:
            json.dump(args.__dict__, f, indent=4)

        times_file = os.path.join(outdir,
                                  f'epoch_times_size{world_size}.csv')
        logger.log(f'Saving epoch times to: {times_file}')
        with open(times_file, 'a') as f:
            f.write(epoch_times_str + '\n')



if __name__ == "__main__":
    main()
