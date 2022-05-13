"""
horovod/torch/main.py

Contains simple implementation of using Hoorovod for data parallel distributed
training.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import time
from typing import Union

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from torch.cuda.amp.grad_scaler import GradScaler
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms

try:
    import horovod.torch as hvd
    hvd.init()
    RANK = hvd.local_rank()
    SIZE = hvd.size()
except (ImportError, ModuleNotFoundError):
    RANK = 0
    SIZE = 1


log = logging.getLogger(__name__)

Tensor = torch.Tensor
WITH_CUDA = torch.cuda.is_available()


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


def metric_average(x: Tensor, name):
    avg_tensor = hvd.allreduce(x, name=name)
    return avg_tensor.item()


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.rank = RANK
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.backend = self.cfg.get('backend', None)
        self.scaler = self.cfg.get('scaler', None)

        self.data = self.setup_data()
        self.model = self.build_model()
        if self.device == 'gpu':
            self.model.cuda()

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = self.build_optimizer(self.model)
        # if WITH_CUDA:
        #    self.loss_fn = self.loss_fn.cuda()

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

        if RANK == 0:
            log.info('\n'.join([
                'Torch Thread Setup:',
                f' Number of threads: {torch.get_num_threads()}',
            ]))

    def setup_data(self):
        kwargs = {}
        # log.warning(f'os.getcwd(): {os.getcwd()}')
        # log.warning(f'cfg.work_dir: {self.cfg.work_dir}')
        # log.warning(f'cfg.data_dir: {self.cfg.data_dir}')

        if self.device == 'gpu':
            kwargs = {'num_workers': 1, 'pin_memory': True}

        train_dataset = (
            datasets.MNIST(
                self.cfg.data_dir,
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ])
            )
        )

        # Horovod: use DistributedSampler to partition training data
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=SIZE, rank=RANK,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            sampler=train_sampler,
            **kwargs
        )

        test_dataset = (
            datasets.MNIST(
                self.cfg.data_dir,
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            )
        )
        # Horovod: use DistributedSampler to partition the test data
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=SIZE, rank=RANK
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.cfg.batch_size
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
        probs = self.model(data)
        loss = self.loss_fn(probs, target)
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        _, pred = probs.data.max(1, keepdim=True)
        acc = pred.eq(target.data.view_as(pred)).float().sum()
        # acc = (pred == target).sum() / pred.shape[0]
        # acc = (pred == target.data.view_as(pred)).float().sum()
        # acc = pred.eq(target.data.view_as(pred)).float().sum()
        # _, pred = probs.data.max(1)

        return loss, acc

    def test_step(
        self,
        data: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if WITH_CUDA:
            data, target = data.cuda(), target.cuda()

        probs = self.model(data)
        loss = self.loss_fn(probs, target)
        pred = probs.data.max(1, keepdim=True)[1]
        acc = pred.eq(target.data.view_as(pred)).cpu().float().sum()

        # _, pred = probs.data.max(1)
        # acc = (pred == target).sum() / pred.shape[0]

        return loss, acc

    def train_epoch(
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
        # Horovod: set epoch to sampler for shuffling
        train_sampler.set_epoch(epoch)
        nsamples = len(train_sampler)
        # ntrain = len(train_loader.dataset) // SIZE // self.cfg.batch_size
        # size = len(train_loader.dataset)
        metrics_ = {}
        for bidx, (data, target) in enumerate(train_loader):
            loss, acc = self.train_step(data, target)
            training_acc = training_acc + acc
            running_loss = running_loss + loss.item()
            # training_acc += acc
            # running_loss += loss.item()
            metrics_ = {
                'epoch': epoch,
                'elapsed:': time.time() - start,
                'batch_loss': loss.item() / self.cfg.batch_size,
                'running_loss': running_loss / nsamples,
                'batch_acc': acc.item() / self.cfg.batch_size,
                'training_acc': training_acc / nsamples,
            }

            if bidx % self.cfg.logfreq == 0 and RANK == 0:
                jdx = bidx * len(data)
                frac = 100. * bidx / len(self.data['train']['sampler'])
                pre = [f'[{RANK}]',
                       f'[{jdx:>5}/{nsamples:<5} ({frac:>03.1f}%)]']
                mstr = ' '.join([
                    f'{k}: {v:.4f}' for (k, v) in metrics_.items()
                ])
                log.info(' '.join([*pre, mstr]))
                # mstr = [f'{v[0]}: {v[1]:.5f}' for v in metrics_.items()]
                # log.info(f'{pre} {metrics_to_str(metrics_)}')

        return metrics_

    def train_epoch1(
            self,
            epoch: int,
    ) -> dict:
        self.model.train()
        start = time.time()
        running_acc = torch.tensor(0.)
        running_loss = torch.tensor(0.)
        if WITH_CUDA:
            running_acc = running_acc.cuda()
            running_loss = running_loss.cuda()

        train_sampler = self.data['train']['sampler']
        train_loader = self.data['train']['loader']
        # Horovod: set epoch to sampler for shuffling
        train_sampler.set_epoch(epoch)
        # ntrain = len(train_loader.dataset) // SIZE // self.cfg.batch_size
        # size = len(train_loader.dataset)
        for bidx, (data, target) in enumerate(train_loader):
            loss, acc = self.train_step(data, target)
            running_acc += acc
            running_loss += loss

            if bidx % self.cfg.logfreq == 0 and RANK == 0:
                # Horovod: use train_sampler to determine the number of
                # examples in this workers partition
                log.info(' '.join([
                    f'[{RANK}]',
                    # f'({epoch}/{self.cfg.epochs})',
                    (   # looks like: [num_processed/total (% complete)]
                        f'[{epoch}/{self.cfg.epochs}:'
                        f' {bidx * len(data)}/{len(train_sampler)}'
                        f' ({100. * bidx / len(train_loader):.0f}%)]'
                    ),
                    f'dt={time.time() - start:.4f}',
                    f'loss={loss.item():.6f}',
                    f'acc={acc.item():.2f}%',
                ]))

        running_loss /= len(train_sampler)
        running_acc /= len(train_sampler)
        training_acc = metric_average(running_acc, 'acc')
        loss_avg = metric_average(running_loss, 'loss')

        if RANK == 0:
            summary = '  '.join([
                '[TRAIN]',
                # f'dt={dt_train:.6f}',
                f'loss={loss_avg:.4f}',
                f'acc={training_acc * 100:.2f}%'
                # '[TRAINING SET]',
                # f'Average loss: {(loss_avg):.4f}',
                # f'Accuracy: {(training_acc * 100):.2f}%'
            ])
            log.info((sep := '-' * len(summary)))
            log.info(summary)
            log.info(sep)

        return {'loss': loss_avg, 'acc': running_acc}

    def test(self) -> dict:
        self.model.eval()
        running_loss = torch.tensor(0.0)
        running_acc = torch.tensor(0.0)
        if WITH_CUDA:
            running_loss = running_loss.cuda()
            running_acc = running_acc.cuda()

        n = 0
        # ntest = (
        #     len(self.data['test']['loader']) // SIZE // self.cfg.batch_size
        # )
        for data, target in self.data['test']['loader']:
            loss, acc = self.test_step(data, target)
            running_acc = running_acc + acc
            running_loss = running_loss + loss.item()
            # running_acc += acc
            # running_loss += loss.item()
            # test_loss += F.nll_loss(output, target).item()
            # get the index of the max log-probability
            # running_acc += pred.eq(target.data.view_as(pred)).float().sum()
            # running_loss += self.loss_fn(output, target).item()
            n = n + 1

        # DDP: use test_sampler to determine
        # the number of examples in this workers partition
        running_loss = running_loss / len(self.data['test']['sampler'])
        running_acc = running_acc / len(self.data['test']['sampler'])

        # DDP: average metric values across workers
        running_loss = metric_average(running_loss, 'loss')
        running_acc = metric_average(running_acc, 'acc')

        if RANK == 0:
            summary = ' '.join([
                '[TEST]',
                f'loss={(running_loss):.4f}',
                f'acc={(running_acc):.2f}%'
            ])
            log.info((sep := '-' * len(summary)))
            log.info(summary)
            log.info(sep)

        return {'loss': running_loss, 'acc': running_acc}


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


@hydra.main(config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    start = time.time()

    epoch_times = []
    trainer = Trainer(cfg)
    for epoch in range(cfg.epochs):
        t0 = time.time()
        _ = trainer.train_epoch(epoch)
        _ = trainer.test()
        epoch_times.append(time.time() - t0)
        # if RANK == 0:
        #    trainer.save_checkpoint()

    log.info(f'Total training time: {time.time() - start} seconds')
    log.info(
        f'Average time per epoch in the last 5: {np.mean(epoch_times[-5])}'
    )


if __name__ == '__main__':
    main()
