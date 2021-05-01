import sys
import argparse
import os

import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.utils.data.distributed
import horovod.torch as hvd
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
import json
import time
from typing import Callable

here = os.path.abspath(os.path.dirname(__file__))
modulepath = os.path.dirname(os.path.dirname(here))
if modulepath not in sys.path:
    sys.path.append(modulepath)

import utils.io as io
from utils.parse_args import parse_args_torch_hvd as parse_args
from utils.data_torch import DistributedDataObject, prepare_datasets

logger = io.Logger()
DATA_PATH = os.path.join(modulepath, 'datasets')


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
        return F.log_softmax(x, 1)


def metric_average(x, name):
    if isinstance(x, torch.Tensor):
        tensor = x.clone().detach()
    else:
        tensor = torch.tensor(x)

    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

def train(
        epoch: int,
        data: DistributedDataObject,
        rank: int,
        model: nn.Module,
        loss_fn: LossFunction,
        optimizer: optim.Optimizer,
        args: dict,
        scaler: GradScaler=None,
):
    model.train()
    # Horovod: set epoch to sampler for shuffling
    data.sampler.set_epoch(epoch)
    running_loss = torch.tensor(0.0)
    training_acc = torch.tensor(0.0)

    with_cuda = torch.cuda.is_available()

    if with_cuda:
        running_loss = running_loss.cuda()
        training_acc = training_acc.cuda()

    for batch_idx, (batch, target) in enumerate(data.loader):
        #  if torch.cuda.is_available():
        if with_cuda:
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
    loss_avg = metric_average(running_loss, 'running_loss')
    training_acc = metric_average(training_acc, 'training_acc')
    if rank == 0:
        logger.log(f'training set; avg loss: {loss_avg:.4g}, '
                   f'accuracy: {training_acc * 100:.2f}%')


def main():
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Horovod: initialize library.
    hvd.init()
    torch.manual_seed(args.seed)
    local_rank = hvd.local_rank()
    world_size = hvd.size()

    if args.cuda:
        device = torch.device(f'cuda:{local_rank}')
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(args.seed)

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    # Horovod: use DistributedSampler to partition the training data.
    data = prepare_datasets(args, rank=local_rank, num_workers=world_size, data='mnist')
    model = Net()

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    # Horovod: scale learning rate by lr_scaler.
    optimizer = optim.SGD(model.parameters(), lr=args.lr * lr_scaler,
                          momentum=args.momentum)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = (
        hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    )

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=compression,
        op=hvd.Adasum if args.use_adasum else hvd.Average,
        gradient_predivide_factor=args.gradient_predivide_factor
    )

    loss_fn = nn.CrossEntropyLoss()
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train(epoch, data['training'], rank=local_rank,
              model=model, loss_fn=loss_fn,
              optimizer=optimizer, args=args, scaler=None)

        if epoch > 2:
            epoch_times.append(time.time() - t0)

        if epoch % 10 == 0:
            if hvd.local_rank() == 0:
                accuracy = evaluate(model=model,
                                    test_loader=data['testing'].loader)
                logger.log('-' * 75)
                logger.log(f'Epoch: {epoch}, Accuracy: {accuracy}')
                logger.log('-' * 75)


    if local_rank == 0:
        epoch_times_str = ', '.join(str(x) for x in epoch_times)
        logger.log('Epoch times:')
        logger.log(epoch_times_str)

        outdir = os.path.join(os.getcwd(), 'results_mnist', f'size{world_size}')
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        modeldir = os.path.join(outdir, 'saved_models')
        modelfile = os.path.join(modeldir, 'hvd_model_mnist.pth')
        if not os.path.isdir(modeldir):
            os.makedirs(modeldir)

        logger.log(f'Saving model to: {modelfile}')
        torch.save(model.state_dict(), modelfile)

        args_file = os.path.join(outdir, f'args_size{world_size}.json')
        logger.log(f'Saving args to: {args_file}.')

        with open(args_file, 'at') as f:
            json.dump(args.__dict__, f, indent=4)

        times_file = os.path.join(outdir,
                                  f'epoch_times_size{world_size}.csv')
        logger.log(f'Saving epoch times to: {times_file}')
        with open(times_file, 'a') as f:
            f.write(epoch_times_str + '\n')


if __name__ == '__main__':
    main()
