"""
torch_mnist_hvd.py

Example  using Horovod + PyTorch for distributed data parallel training.

Computational PErformance Workshop -- May, 2021 @ ALCF
"""
from __future__ import print_function
import json
import os
import sys
import time

import horovod.torch as hvd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import torchvision

here = os.path.abspath(os.path.dirname(__file__))
modulepath = os.path.abspath(os.path.dirname(os.path.dirname(here)))
if modulepath not in sys.path:
    sys.path.append(modulepath)

from utils.io import Logger, prepare_datasets, DistributedDataObject
from utils.parse_args import parse_args_torch as parse_args

hvd.init()
RANK = hvd.rank()
SIZE = hvd.size()
LOCAL_RANK = hvd.local_rank()
CUDA = torch.cuda.is_available()


logger = Logger()

class DistributedDataObject:
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            batch_size: int,
            **kwargs: dict
    ):
        self.dataset = dataset
        self.sampler = torch.utils.data.distributed.DistributedSampler(
            self.dataset, num_replicas=hvd.size(), rank=hvd.rank()
        )
        self.loader = torch.utils.data.DataLoader(
            self.dataset, batch_size, sampler=self.sampler, **kwargs
        )


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
        return F.log_softmax(x, dim=-1)


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

    def forward(self, x: torch.Tensor) -> (torch.Tensor):
        """Call the model on input data `x`."""
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def train(
    epoch: int,
    data: DistributedDataObject,
    model: nn.Module,
    optimizer: hvd.DistributedOptimizer,
    args: dict,
):
    model.train()
    running_loss = 0.0
    training_acc = 0.0
    loss_fn = nn.CrossEntropyLoss()
    # Horovod: set epoch to sampler for shuffling
    data.sampler.set_epoch(epoch)
    for batch_idx, (batch, target) in enumerate(data.loader):
        if args.cuda:
            batch, target = batch.cuda(), target.cuda()
        optimizer.zero_grad()

        output = model(batch)
        #  loss= F.nll_loss(output, target)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        batch_acc = pred.eq(target.data.view_as(pred)).cpu().float().sum()
        training_acc += batch_acc
        #  training_acc += pred.eq(target.data.view_as(pred)).cpu().float().sum()
        running_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            metrics_ = {
                'epoch': epoch,
                'batch_loss': loss.item() / args.batch_size,
                'running_loss': running_loss / len(data.sampler),
                'batch_acc': batch_acc / args.batch_size,
                'training_acc': training_acc / len(data.sampler) ,
            }

            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition
            jdx = batch_idx * len(batch)
            frac = 100. * batch_idx / len(data.loader)

            pre = [f'[{hvd.rank()}]',
                   f'[{jdx:05}/{len(data.sampler):05} ({frac:>4.3g}%)]']
            mstr = ' '.join([
                f'{str(k):<5}: {v:<7.4g}' for k, v in metrics_.items()
            ])
            logger.log(' '.join([*pre, mstr]))

    running_loss = running_loss / len(data.sampler)
    training_acc = training_acc / len(data.sampler)
    loss_avg = metric_average(running_loss, 'running_loss')
    training_acc = metric_average(training_acc, 'training_acc')
    if hvd.rank() == 0:
        logger.log(f'training set; avg loss: {loss_avg:.4g}, accuracy: {training_acc * 100:.2f}%')


def evaluate(
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: str
):
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


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        #  for images, labels in test_loader:
        #      images, labels = images.to(device), labels.to(device)
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy

def test1(
        data: DistributedDataObject,
        model: nn.Module,
        args: dict,
):
    model.eval()
    test_loss = 0.
    test_acc = 0.
    loss_fn = nn.CrossEntropy()
    n = 0
    for batch, target in data.loader:
        if args.cuda:
            batch, target = batch.cuda(), target.cuda()
        output = model(batch)

        # sum up batch loss
        loss = loss_fn(output, target)
        test_loss += loss.item()
        #  test_loss += F.nll_loss(output, target).item()
        #  test_loss += nn.CrossEntropy()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        batch_acc = pred.eq(target.data.view_as(pred)).cpu().float().sum()
        test_acc += batch_acc
        n = n + 1

    # Horovod: use test_sampler to determine the number of examples in this
    # worker's partition
    test_loss /= len(data.sampler)
    test_acc /= len(data.sampler)

    # Horovod: print output only on first rank
    if hvd.rank() == 0:
        logger.log('\n'.join([
            f'test set, avg loss: {test_loss:.4g}',
            f'test set, accuracy: {test_acc:.2f}'
        ]))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    logger.log(f'Horovod: I am worker {RANK} of {SIZE}')
    if args.device.find('gpu') != -1:
        # Horovod: pin GPU to local rank
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    if args.num_threads != 0:
        torch.set_num_threads(args.num_threads)

    if hvd.rank() == 0:
        num_threads = torch.get_num_threads()
        logger.log(f'(torch thread setup) num threads: {num_threads}')

    kwargs = {}
    if args.device.find('gpu') != -1:
        kwargs = {'num_workers': 1, 'pin_memory': True}

    data = prepare_datasets(args, rank=RANK, num_workers=SIZE, data=args.dataset)
    #  model = Net()
    #  model = AlexNet(num_classes=10)
    model = torchvision.models.resnet18(pretrained=False)

    if args.device.find('gpu') != -1:
        model.cuda()

    # Horovod: scale learning rate by the number of GPUs
    optimizer = optim.Adam(model.parameters(), lr=args.lr * hvd.size())

    # Horovod: broadcast parameters & optimizer stat
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm
    compression = hvd.Compression.none
    if args.fp16_allreduce:
        compression = hvd.Compression.fp16

    # Horovod: wrap optimizer with DistributedOptimizer
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression)
    epoch_times = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train(epoch, data['training'], model, optimizer, args)
        if epoch > 3:
            epoch_times.append(time.time() - t0)

        test(model, data['testing'].loader)

        #  test(test_data, model, args)

    if hvd.rank() == 0:
        epoch_times_str = ', '.join(str(x) for x in epoch_times)
        logger.log('Epoch times:')
        logger.log(epoch_times_str)

        args_file = os.path.join(os.getcwd(), f'args_size{SIZE}.json')
        logger.log(f'Saving args to: {args_file}.')

        with open(args_file, 'at') as f:
            json.dump(args.__dict__, f, indent=4)

        times_file = os.path.join(os.getcwd(), f'epoch_times_size{SIZE}.csv')
        logger.log(f'Saving epoch times to: {times_file}')
        with open(times_file, 'a') as f:
            f.write(epoch_times_str + '\n')


if __name__ == '__main__':
    main()
