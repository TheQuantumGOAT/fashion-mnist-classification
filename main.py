# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# Hydra
import hydra
from omegaconf import DictConfig

# other
import os
from collections import OrderedDict
import pytorch_lightning as pl
import numpy as np

optimizer = 'adam'
LR = 1e-4
Epochs = 25
Batch_size = 10


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_idx = None
        self.val_idx = None
        self.train_ds = None
        self.test_ds = None
        self.test_dl = None
        self.train_dl = None
        self.val_dl = None

    def prepare_data(self):
        datasets.FashionMNIST('F_MNIST_data', download=True, train=True)
        datasets.FashionMNIST('F_MNIST_data', download=True, train=False)

    def setup(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.test_ds = datasets.FashionMNIST('F_MNIST_data', train=False, transform=transform)
        self.train_ds = datasets.FashionMNIST('F_MNIST_data', train=True, transform=transform)
        train_num = len(self.train_ds)
        indices = list(range(train_num))
        np.random.shuffle(indices)
        split = int(np.floor(0.2 * train_num))
        self.val_idx, self.train_idx = indices[:split], indices[split:]

    def train_dataloader(self):
        if self.train_dl is None:
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_idx)
            self.train_dl = torch.utils.data.DataLoader(self.train_ds, batch_size=Batch_size, sampler=train_sampler)
        return self.train_dl

    def val_dataloader(self):
        if self.val_dl is None:
            val_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.val_idx)
            self.val_dl = torch.utils.data.DataLoader(self.train_ds, batch_size=Batch_size, sampler=val_sampler)
        return self.val_dl

    def test_dataloader(self):
        if self.test_dl is None:
            self.test_dl = torch.utils.data.DataLoader(self.test_ds, batch_size=Batch_size, shuffle=True)
        return self.test_dl


class MyModel1(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(OrderedDict([('fc1', nn.Linear(784, 128)),
                                              ('relu1', nn.ReLU()),
                                              ('drop1', nn.Dropout(0.25)),
                                              ('fc2', nn.Linear(128, 64)),
                                              ('relu2', nn.ReLU()),
                                              ('drop1', nn.Dropout(0.25)),
                                              ('output', nn.Linear(64, 10)),
                                              ('logsoftmax', nn.LogSoftmax(dim=1))]))

    def forward(self, x):
        return self.fc1(x)

    def configure_optimizers(self):
        if optimizer == 'adam':
            return optim.Adam(self.parameters(), lr=LR)
        else:
            return optim.SGD(self.parameters(), lr=LR)


@hydra.main(config_path="config", config_name="config.yaml", version_base="1.1")
def hydra_data_parse(cfg):
    global optimizer, LR, Epochs, Batch_size
    optimizer = cfg.optimizer
    LR = cfg['optimizer.lr']
    Epochs = cfg.epochs
    Batch_size = cfg.batch_size

if __name__ == "__main__":
    hydra_data_parse()
    print(f"The optimizer is {optimizer}")
    print(f"The learning rate is {LR}")
    print(f"The number of epochs is {Epochs}")
    print(f"The batch size is {Batch_size}")

    model = MyModel1()
    datamodule = MNISTDataModule()

    trainer = pl.Trainer(devices="auto", accelerator="auto", max_epochs=Epochs)
    trainer.fit(model, datamodule=datamodule)