"""
torch_ddp.py

Modified from original: https://leimao.github.io/blog/PyTorch-Distributed-Training/
"""
import sys
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributed as dist
from typing import Callable
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import argparse
import os
import random
import numpy as np

here = os.path.abspath(os.path.dirname(__file__))
modulepath = os.path.dirname(here)
if modulepath not in sys.path:
    sys.path.append(modulepath)

from utils.io import Logger, DistributedDataObject, prepare_datasets
from utils.parse_args import parse_args_ddp

logger = Logger()

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


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


def metric_average(x: torch.tensor, with_ddp: bool) -> (torch.tensor):
    """Compute global averages across all workers if using DDP. """
    x = torch.tensor(x)
    if with_ddp:
        # Sum everything and divide by total size
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= SIZE
    else:
        pass

    return x.item()


def train(
        epoch: int,
        data: DistributedDataObject,
        device: torch.device,
        rank: int,
        model: nn.Module,
        loss_fn: Callable[[torch.tensor], torch.tensor],
        optimizer: optim.Optimizer,
        args: dict,
        scaler: GradScaler=None,
):
    model.train()
    # Horovod: set epoch to sampler for shuffling
    data.sampler.set_epoch(epoch)
    running_loss = 0.0
    training_acc = 0.0
    #  running_loss = torch.tensor(0.0)
    #  training_acc = torch.tensor(0.0)

    for batch_idx, (batch, target) in enumerate(data.loader):
        if args.cuda:
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
                   f'[{jdx}/{len(data.sampler)} ({frac}%)]']
                   #  f'[{jdx:05}/{len(data.sampler):05} ({frac:>4.3g}%)]']
            mstr = ' '.join([
                #  f'{str(k):>5}: {v:<7.4g}' for k, v in metrics_.items()
                f'{k}: {v}' for k, v in metrics_.items()
            ])
            logger.log(' '.join([*pre, mstr]))

            #  str0 = f'[{jdx:5<}/{len(data.sampler):5<} ({frac:>3.3g}%)]'
            #  str1 = ' '.join([
            #      f'{str(k):>5}: {v:<7.4g}' for k, v in metrics_.items()
            #  ])
            #  logger.log(' '.join([str0, str1]))

    running_loss = running_loss / len(data.sampler)
    training_acc = training_acc / len(data.sampler)
    #  loss_avg = metric_average(running_loss, args.cuda)
    #  training_acc = metric_average(training_acc, args.cuda)
    if rank == 0:
        logger.log(f'training set; avg loss: {running_loss:.4g}, '
                   f'accuracy: {training_acc * 100:.2f}%')


def main():
    argv = parse_args_ddp()
    with_cuda = torch.cuda.is_available()
    argv.cuda = with_cuda

    local_rank = argv.local_rank
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir
    model_filename = argv.model_filename
    resume = argv.resume

    # Create directories outside the PyTorch program
    # Do not create directory here because it is not multiprocess safe
    '''
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    '''

    model_filepath = os.path.join(model_dir, model_filename)

    # We need to use seeds to make sure that the models initialized in different processes are the same
    set_random_seeds(random_seed=random_seed)

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    backend = 'nccl' if with_cuda else 'gloo'
    dist.init_process_group(backend=backend)
    #  torch.distributed.init_process_group(backend=backend)
    # torch.distributed.init_process_group(backend="gloo")

    # Encapsulate the model on the GPU assigned to the current process
    model = torchvision.models.resnet18(pretrained=False)

    if argv.cuda:
        device = torch.device("cuda:{}".format(local_rank))
        num_workers = torch.cuda.device_count()
    else:
        device = torch.device(int(local_rank))

    model = model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if resume == True:
        if with_cuda:
            map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        else:
            map_location = {'0': f'{local_rank}'}

        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))

    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Data should be prefetched
    # Download should be set to be False, because it is not multiprocess safe
    #  train_set = torchvision.datasets.CIFAR10(root="data", train=True,
    #                                           download=True, transform=transform)
    #  test_set = torchvision.datasets.CIFAR10(root="data", train=False,
    #                                          download=True, transform=transform)
    data = prepare_datasets(argv, rank=local_rank,
                            num_workers=num_workers,
                            data='cifar10')

    # Restricts data loading to a subset of the dataset exclusive to the current process
    #  train_sampler = DistributedSampler(dataset=train_set)

    #  train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, num_workers=8)
    #  # Test loader does not have to follow distributed sampling strategy
    #  test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False, num_workers=8)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    # Loop over the dataset multiple times
    for epoch in range(num_epochs):
        #  logger.log("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))
        # Save and evaluate model routinely
        train(epoch, data['training'], device=device, rank=local_rank,
              model=ddp_model, loss_fn=criterion,
              optimizer=optimizer, args=argv, scaler=None)

        if epoch % 10 == 0:
            if local_rank == 0:
                accuracy = evaluate(model=ddp_model, device=device,
                                    test_loader=data['testing'].loader)
                torch.save(ddp_model.state_dict(), model_filepath)
                logger.log('-' * 75)
                logger.log(f'Epoch: {epoch}, Accuracy: {accuracy}')
                logger.log('-' * 75)
        #  if epoch % 10 == 0:
        #      if local_rank == 0:
        #          accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
        #          torch.save(ddp_model.state_dict(), model_filepath)
        #          logger.log("-" * 75)
        #          logger.log("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
        #          logger.log("-" * 75)
        #
        #  ddp_model.train()
        #
        #  for data in train_loader:
        #      inputs, labels = data[0].to(device), data[1].to(device)
        #      optimizer.zero_grad()
        #      outputs = ddp_model(inputs)
        #      loss = criterion(outputs, labels)
        #      loss.backward()
        #      optimizer.step()


if __name__ == "__main__":
    main()
