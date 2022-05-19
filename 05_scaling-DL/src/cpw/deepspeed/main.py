"""
src/cpw/deepspeed/main.py


Contains simple implementation demonstrating how to use Microsoft's 
DeepSpeed library for data-parallel distributed training.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms

try:
    import deepspeed
    deepspeed.init_distributed()
    from mpi4py import MPI
    WITH_DEEPSPEED = True
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()
    LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
except (ImportError, ModuleNotFoundError):
    WITH_DEEPSPEED = False
    # RANK = 0
    # SIZE = 1
    # LOCAL_RANK = 0


log = logging.getLogger(__name__)
Tensor = torch.Tensor
WITH_CUDA = torch.cuda.is_available()
# SIZE = torch.cuda.device_count()

# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.


def create_moe_param_groups(model):
    from deepspeed.moe.utils import is_moe_param

    params_with_weight_decay = {'params': [], 'name': 'weight_decay_params'}
    moe_params_with_weight_decay = {
        'params': [],
        'moe': True,
        'name': 'weight_decay_moe_params'
    }

    for module_ in model.modules():
        moe_params_with_weight_decay['params'].extend([
            p for n, p in list(module_._parameters.items())
            if p is not None and is_moe_param(p)
        ])
        params_with_weight_decay['params'].extend([
            p for n, p in list(module_._parameters.items())
            if p is not None and not is_moe_param(p)
        ])

    return params_with_weight_decay, moe_params_with_weight_decay


########################################################################
# Let us show some of the training images, for fun.

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


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



# net = Net()


# parameters = filter(lambda p: p.requires_grad, net.parameters())
# if args.moe_param_group:
#     parameters = create_moe_param_groups(net)

# # Initialize DeepSpeed to use the following features
# # 1) Distributed model
# # 2) Distributed data loader
# # 3) DeepSpeed optimizer
# model_engine, optimizer, trainloader, __ = deepspeed.initialize(
#     args=args, model=net, model_parameters=parameters, training_data=trainset)

# fp16 = model_engine.fp16_enabled()
# print(f'fp16={fp16}')

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#net.to(device)
########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.


criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(model_engine.local_rank), data[1].to(
            model_engine.local_rank)
        if fp16:
            inputs = inputs.half()
        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()

        # print statistics
        running_loss += loss.item()
        if i % args.log_interval == (
                args.log_interval -
                1):  # print every log_interval mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / args.log_interval))
            running_loss = 0.0

print('Finished Training')

########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:
if fp16:
    images = images.half()
outputs = net(images.to(model_engine.local_rank))

########################################################################
# The outputs are energies for the 10 classes.
# The higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if fp16:
            images = images.half()
        outputs = net(images.to(model_engine.local_rank))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(
            model_engine.local_rank)).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' %
      (100 * correct / total))

########################################################################
# That looks way better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if fp16:
            images = images.half()
        outputs = net(images.to(model_engine.local_rank))
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels.to(model_engine.local_rank)).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' %
          (classes[i], 100 * class_correct[i] / class_total[i]))


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.rank = RANK
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.backend = self.cfg.backend
        # self.setup_torch()
        self.data = self.setup_data()
        self.model = self.build_model()

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = self.build_optimizer(self.model)
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.cfg.moe_param_group:
            parameters = create_moe_param_groups(self.model)
        # if WITH_CUDA:
        #    self.loss_fn = self.loss_fn.cuda()
        self._fp16 = False
        if WITH_DEEPSPEED:
            self.model, self.optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                config=self.cfg.deepspeed,
                model_parameters=parameters,
            )
            self._fp16 = self.model.fp16_enabled()

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
                'dataset': train_dataset,
                'sampler': train_sampler,
                'loader': train_loader,
            },
            'test': {
                'dataset': test_dataset,
                'sampler': test_sampler,
                'loader': test_loader,
            }
        }

    def save_checkpoint(self, step: int):
        outdir = Path(os.getcwd()).joinpath('checkpoints')
        outdir.mkdir(exist_ok=True, parents=True)
        if WITH_DEEPSPEED:
            self.model.save_checkpoint(
                save_dir=outdir.as_posix(),
                client_state={'checkpoint_step': step}
            )

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
        # DeepSpeed: Replace `loss.backward` with `model.backward(loss)`
        # loss.backward()
        if WITH_DEEPSPEED:
            self.model.backward(loss)
            self.model.step()
        else:
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
            # data, target = data.cuda(), target.cuda()
            data, target = data.to(self.model)

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


@hydra.main(config_path='./conf', config_name='config')
def main(cfg: DictConfig):
    if WITH_DEEPSPEED:
        if cfg.get('moe', False):
            deepspeed.utils.groups.initialize(ep_size=cfg.ep_world_size)

        # Initialize DeepSpeed to use the following features
        # 1. Distributed model
        # 2. Distributed data loader
        # 3. DeepSpeed optimizer
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        if cfg.moe_param_group:
            parameters = create_moe_param_groups(net)


        fp16 = model_engine.fp16_enabled()



