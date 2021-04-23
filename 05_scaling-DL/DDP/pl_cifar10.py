"""
pl_cifar10.py

Contains example using pytorch-lightning to handle DDP.
"""
from __future__ import print_function

import os
import sys
import torch
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

modulepath = os.path.dirname(os.path.dirname(__file__))
if modulepath not in sys.path:
    sys.path.append(modulepath)

from utils.parse_args import parse_args_torch as parse_args


class LitAlexNet(pl.LightningModule):
    def __init__(self, num_classes=10, criterion=None, scaler=None):
        self.automatic_optimization = False
        super().__init__()

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.scaler = scaler
        self.criterion = criterion
        self.num_classes = num_classes

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
    def forward(self, x: torch.tensor):
        """Call the model on input data `x`."""
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.tensor(0.0)
        acc = torch.tensor(0.0)
        opt = self.optimizers(use_pl_optimizer=True)
        opt.zero_grad()
        x = self.features(x)
        x = x.view(x.size(0), -1)
        z = self.classifier(x)
        loss = self.criterion(z, y)
        #  loss = nn.CrossEntropyLoss()(z, x)
        #  loss = F.cross_entropy(z, x)
        if self.scaler is not None:
            self.manual_backward(self.scaler.scale(loss))
            #  self.scaler.scale(loss).backward()
            self.scaler.step(opt)
            self.scaler.update()
        else:
            self.manual_backward(loss)
        # Logging to tensorboard by default
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    args = parse_args()
    scaler = GradScaler(enabled=args.cuda)
    model = LitAlexNet(10, nn.CrossEntropyLoss(), scaler=scaler)
    model.automatic_optimization = False
    trainer = pl.Trainer()
    dataset = CIFAR10(os.getcwd(), download=True,
                      transform=transforms.ToTensor())
    if args.device == 'cpu':
        num_workers = args.num_threads
    else:
        num_workers = torch.cuda.device_count()

    train_loader = DataLoader(dataset, num_workers=num_workers)
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    main()
