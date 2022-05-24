"""
horovod/torch/main.py

Contains simple implementation of using Hoorovod for data parallel distributed
training.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import time
from typing import Optional, Union

import hydra
from omegaconf import DictConfig
import torch
try:
    import horovod.torch as hvd
    hvd.init()
    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())

    SIZE = hvd.size()
    RANK = hvd.local_rank()
except (ImportError, ModuleNotFoundError):
    RANK = 0
    SIZE = 1

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms


log = logging.getLogger(__name__)

Tensor = torch.Tensor
WITH_CUDA = torch.cuda.is_available()

if RANK == 0:
    log.info(f'RANK: {RANK} of {SIZE}')


def metrics_to_str(m: dict[str, Union[Tensor, float]]) -> str:
    strs = {}
    for key, val in m.items():
        if isinstance(val, Tensor):
            if len(val.shape) > 1:
                val = val.mean()

        strs[key] = f'{val:.5f}'

    return ', '.join([f'{k}: {v}' for (k, v) in strs.items()])


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


def metric_average(x: Tensor | float, name: Optional[str] = None):
    tensor = torch.tensor(x).detach()
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.rank = hvd.local_rank()
        self.fp16 = cfg.get('fp16', False)
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'

        self.backend = self.cfg.get('backend', None)
        self.data = self.setup_data()
        self.model = self.build_model()
        if self.device == 'gpu':
            self.model.cuda()

        self.loss_fn = nn.CrossEntropyLoss()
        optimizer = self.build_optimizer(self.model)
        if SIZE > 1:
            self.optimizer = hvd.DistributedOptimizer(
                optimizer,
                named_parameters=self.model.named_parameters(),
            )
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

    def build_model(self) -> nn.Module:
        model = Net()
        return model

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        # Horovod: scale learning rate by the number of GPUs
        optimizer = optim.Adam(model.parameters(),
                               lr=SIZE * self.cfg.lr_init)
        return optimizer

    def setup_torch(self):
        torch.manual_seed(self.cfg.seed)
        if self.device == 'gpu':
            torch.cuda.set_device(int(RANK))
            torch.cuda.manual_seed(self.cfg.seed)

        if (
                self.cfg.num_threads is not None
                and isinstance(self.cfg.num_threads, int)
                and self.cfg.num_threads > 0
        ):
            torch.set_num_threads(self.cfg.num_threads)

        if self.rank == 0:
            log.info('\n'.join([
                'Torch Thread Setup:',
                f' Number of threads: {torch.get_num_threads()}',
            ]))

    def setup_data(self):
        kwargs = {}

        if self.device == 'gpu':
            kwargs = {'num_workers': 1, 'pin_memory': True}

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = (
            datasets.MNIST(
                self.cfg.data_dir,
                train=True,
                download=True,
                transform=transform,
            )
        )
        test_dataset = (
            datasets.MNIST(
                self.cfg.data_dir,
                train=False,
                download=True,
                transform=transform
            )
        )

        # Horovod: use DistributedSampler to partition training data
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=SIZE, rank=hvd.rank(),
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=SIZE, rank=hvd.rank(),
        )

        # Horovod: use DistributedSampler to partition the test data
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            sampler=train_sampler,
            **kwargs
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size,
            sampler=test_sampler,
            **kwargs
        )

        return {
            'train': {
                'sampler': train_sampler,
                'loader': train_loader,
            },
            'test': {
                'sampler': test_sampler,
                'loader': test_loader,
            }
        }

    def train_step(
        self,
        data: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.device == 'gpu':
            data, target = data.cuda(), target.cuda()

        self.optimizer.zero_grad()
        probs = self.model(data)
        loss = self.loss_fn(probs, target)
        loss.backward()
        self.optimizer.step()

        _, pred = probs.data.max(1)
        acc = (pred == target).sum()

        return loss, acc

    def train_epoch(self, epoch: int):
        self.model.train()
        # Horovod: set epoch to sampler for shuffling.
        train_sampler = self.data['train']['sampler']
        train_loader = self.data['train']['loader']
        train_sampler.set_epoch(epoch)
        loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            if WITH_CUDA:
                data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.cfg.logfreq == 0:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_sampler),
                           100. * batch_idx / len(train_loader), loss.item()))

        return loss


    def train_epoch1(
            self,
            epoch: int,
    ) -> dict:
        self.model.train()
        start = time.time()
        training_acc = torch.tensor(0.)
        running_loss = torch.tensor(0.)
        if WITH_CUDA:
            training_acc = training_acc.cuda()
            running_loss = running_loss.cuda()

        train_sampler = self.data['train']['sampler']
        train_loader = self.data['train']['loader']
        sampler_len = len(train_sampler)
        # HOROVOD: set epoch to sampler for shuffling
        train_sampler.set_epoch(epoch)
        for bidx, (data, target) in enumerate(train_loader):
            loss, acc = self.train_step(data, target)
            training_acc += acc
            running_loss += loss
            if bidx % self.cfg.logfreq == 0 and RANK == 0:
                # HOROVOD: use train_sampler to determine the number of
                # examples in this workers partition
                metrics = {
                    'epoch': epoch,
                    'dt': time.time() - start,
                    'batch_acc': acc.item() / self.cfg.batch_size,
                    'batch_loss': loss.item() / self.cfg.batch_size,
                    'running_loss': running_loss.item() / sampler_len,
                    'acc': training_acc.item() / sampler_len,
                }
                pre = [
                    f'[{RANK}]',
                    (   # looks like: [num_processed/total (% complete)]
                        f'[{epoch}/{self.cfg.epochs}:'
                        f' {bidx * len(data)}/{len(train_sampler)}'
                        f' ({100. * bidx / len(train_loader):.0f}%)]'
                    ),
                ]
                log.info(' '.join([
                    *pre, *[f'{k}={v:.4f}' for k, v in metrics.items()]
                ]))

        running_loss = running_loss / len(train_sampler)
        training_acc = training_acc / len(train_sampler)
        training_acc = metric_average(training_acc)
        loss_avg = metric_average(running_loss)

        return {'loss': loss_avg, 'acc': training_acc}

    def test(self):
        self.model.eval()
        test_loss = 0.
        test_accuracy = 0.
        for data, target in self.data['test']['loader']:
            if WITH_CUDA:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

        # Horovod: use test_sampler to determine the number of examples in
        # this worker's partition.
        test_loss /= len(self.data['test']['sampler'])
        test_accuracy /= len(self.data['test']['sampler'])

        # Horovod: average metric values across workers.
        test_loss = metric_average(test_loss, 'avg_loss')
        test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

        # Horovod: print output only on first rank.
        if hvd.rank() == 0:
            log.info('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
                test_loss, 100. * test_accuracy))

        return test_accuracy


    def test1(self) -> float:
        total = 0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for data, target in self.data['test']['loader']:
                if WITH_CUDA:
                    data, target = data.cuda(), target.cuda()

                probs = self.model(data)
                _, predicted = probs.data.max(1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return correct / total


@hydra.main(config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    if RANK == 0:
        log.info(f'RANK: {RANK} of {SIZE}')

    trainer = Trainer(cfg)
    epoch_times = []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        metrics = trainer.train_epoch1(epoch)
        epoch_times.append(time.time() - t0)
        if epoch % cfg.logfreq and RANK == 0:
            acc = trainer.test1()
            astr = f'[TEST] Accuracy: {100.0 * acc:.0f}%'
            sepstr = '-' * len(astr)
            log.info(sepstr)
            log.info(astr)
            log.info(sepstr)
            summary = '  '.join([
                '[TRAIN]',
                f'loss={metrics["loss"]:.4f}',
                f'acc={metrics["acc"] * 100.0:.0f}%'
            ])
            log.info((sep := '-' * len(summary)))
            log.info(summary)
            log.info(sep)


if __name__ == '__main__':
    main()
