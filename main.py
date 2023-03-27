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

Adam = True
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
        if Adam:
            return optim.Adam(self.parameters(), lr=LR)
        else:
            return optim.SGD(self.parameters(), lr=LR)


def network():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyModel1().to(device)
    loss_fn = nn.NLLLoss()
    return model, device, loss_fn


def train_validate(model, dataset_module, device, loss_fn):
    optimizer = model.configure_optimizers()
    train_loader = dataset_module.train_dataloader()
    test_loader = dataset_module.test_dataloader()

    for epoch in range(Epochs):
        model.train()
        train_epoch_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # flatten the images to batch_size x 784
            images = images.view(images.shape[0], -1)
            # forward pass
            outputs = model(images)
            # back propagation
            train_batch_loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            train_batch_loss.backward()
            # Weight updates
            optimizer.step()
            train_epoch_loss += train_batch_loss.item()

        # One epoch of training complete
        # calculate average training epoch loss
        train_epoch_loss = train_epoch_loss / len(train_loader)

        with torch.no_grad():
            test_epoch_acc = 0
            test_epoch_loss = 0
            model.eval()
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                # flatten images to batch_size x 784
                images = images.view(images.shape[0], -1)
                # make predictions
                test_outputs = model(images)
                # calculate test loss
                test_batch_loss = loss_fn(test_outputs, labels)
                test_epoch_loss += test_batch_loss

                # get probabilities, extract the class associated with highest probability
                proba = torch.exp(test_outputs)
                _, pred_labels = proba.topk(1, dim=1)

                # compare actual labels and predicted labels
                result = pred_labels == labels.view(pred_labels.shape)
                batch_acc = torch.mean(result.type(torch.FloatTensor))
                test_epoch_acc += batch_acc.item()

            test_epoch_loss = test_epoch_loss / len(test_loader)
            test_epoch_acc = test_epoch_acc / len(test_loader)
            print(f'Epoch: {epoch} -> train_loss: {train_epoch_loss:.19f}, val_loss: {test_epoch_loss:.19f}, ',
                  f'val_acc: {test_epoch_acc * 100:.2f}%')



@hydra.main(config_path="config", config_name="config.yaml", version_base="1.1")
def hydra_data_parse(cfg):
    global Adam, LR, Epochs, Batch_size
    Adam = cfg.adam
    LR = cfg.lr
    Epochs = cfg.epochs
    Batch_size = cfg.batch_size

if __name__ == "__main__":
    hydra_data_parse()
    print(f"Is it Adam {Adam}")
    print(f"The learning rate is {LR}")
    print(f"The number of epochs is {Epochs}")
    print(f"The batch size is {Batch_size}")

    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup()

    model, device, loss_fn = network()
    train_validate(model, dm, device, loss_fn)
