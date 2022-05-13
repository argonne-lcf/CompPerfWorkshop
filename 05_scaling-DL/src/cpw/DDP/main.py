"""
DDP/main.py

Conains simple implementation illustrating how to use PyTorch DDP for
distributed data parallel training.
"""
from __future__ import absolute_import, division, print_function, annotations
import os
import socket
import logging

from typing import Optional, Union

import hydra
import time
import torch
import torch.utils.data
import torch.utils.data.distributed

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torch.multiprocessing as mp
from torchvision import datasets, transforms
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.data_torch import DistributedDataObject, prepare_datasets



log = logging.getLogger(__name__)

# Get MPI:
try:
    from mpi4py import MPI
    WITH_DDP = True
    LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
    # LOCAL_RANK = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()

    WITH_CUDA = torch.cuda.is_available()
    DEVICE = 'gpu' if WITH_CUDA else 'CPU'

    # pytorch will look for these
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(SIZE)
    # -----------------------------------------------------------
    # NOTE: Get the hostname of the master node, and broadcast
    # it to all other nodes It will want the master address too,
    # which we'll broadcast:
    # -----------------------------------------------------------
    MASTER_ADDR = socket.gethostname() if RANK == 0 else None
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = str(2345)

except (ImportError, ModuleNotFoundError) as e:
    WITH_DDP = False
    SIZE = 1
    RANK = 0
    LOCAL_RANK = 0
    MASTER_ADDR = 'localhost'
    log.warning('MPI Initialization failed!')
    log.warning(e)


Tensor = torch.Tensor


def init_process_group(
    rank: Union[int, str],
    world_size: Union[int, str],
    backend: Optional[str] = None,
) -> None:
    if WITH_CUDA:
        backend = 'nccl' if backend is None else str(backend)

    else:
        backend = 'gloo' if backend is None else str(backend)

    dist.init_process_group(
        backend,
        rank=int(rank),
        world_size=int(world_size),
        init_method='env://',
    )


def cleanup():
    dist.destroy_process_group()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # in, out, kernel_size, stride
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)  # in_features, out_features
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.fc1(torch.flatten(x, 1)))
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


def metric_average(val: Tensor):
    if (WITH_DDP):
        # Sum everything and divide by the total size
        dist.all_reduce(val, op=dist.reduce_op.SUM)
        return val / SIZE

    return val


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.rank = RANK
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.backend = self.cfg.backend
        if WITH_DDP:
            init_process_group(RANK, SIZE, backend=self.backend)

        self.setup_torch()
        self.data = self.setup_data()
        self.model = self.build_model()
        if self.device == 'gpu':
            self.model.cuda()

        if WITH_DDP and SIZE > 1:
            self.model = DDP(self.model)

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
            # DDP: pin GPU to local rank
            torch.cuda.set_device(int(LOCAL_RANK))
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
        if WITH_CUDA:
            data, target = data.cuda(), target.cuda()
        probs = self.model(data)
        loss = self.loss_fn(probs, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        _, pred = probs.data.max(1)
        acc = (pred == target).sum() / pred.shape[0]
        # acc = (pred == target.data.view_as(pred)).float().sum()
        # acc = pred.eq(target.data.view_as(pred)).float().sum()

        return loss, acc

    def test_step(
        self,
        data: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if WITH_CUDA:
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            probs = self.model(data)
            loss = self.loss_fn(probs, target)
            _, pred = probs.data.max(1)
            acc = (pred == target).sum() / pred.shape[0]

        return loss, acc

    def train_epoch(
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
        training_acc = metric_average(running_acc)
        loss_avg = metric_average(running_loss / self.cfg.batch_size)

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
            running_acc += acc
            running_loss += loss
            # test_loss += F.nll_loss(output, target).item()
            # get the index of the max log-probability
            # running_acc += pred.eq(target.data.view_as(pred)).float().sum()
            # running_loss += self.loss_fn(output, target).item()
            n = n + 1

        # DDP: use test_sampler to determine
        # the number of examples in this workers partition
        running_loss /= len(self.data['test']['sampler'])
        running_acc /= len(self.data['test']['sampler'])

        # DDP: average metric values across workers
        running_loss = metric_average(running_loss)
        running_acc = metric_average(running_acc)

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


# def run_demo(demo_fn: Callable, world_size: int | str) -> None:
#     mp.spawn(demo_fn,
#              args=(world_size,),
#              nprocs=int(world_size),
#              join=True)


@hydra.main(config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    start = time.time()
    trainer = Trainer(cfg)
    epoch_times = []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        _ = trainer.train_epoch(epoch)
        _ = trainer.test()
        epoch_times.append(time.time() - t0)

    log.info(f'Total training time: {time.time() - start} seconds')
    log.info(
        f'Average time per epoch in the last 5: {np.mean(epoch_times[-5])}'
    )

    cleanup()


if __name__ == '__main__':
    main()
