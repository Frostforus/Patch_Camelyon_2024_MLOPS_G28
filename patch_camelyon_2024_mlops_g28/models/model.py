from pytorch_lightning import LightningModule
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
import wandb
import numpy as np


class SimpleCNN(LightningModule):
    def __init__(self, lr=1e-3, batch_size=64):
        super().__init__()
        # Save hyperparams and log them in wandb
        self.save_hyperparameters("lr", "batch_size")
        self.lr = lr
        self.batch_size = batch_size

        self.loss_fn = nn.CrossEntropyLoss()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * 24 * 128, 1024),
            nn.Linear(1024, 2),
            # nn.LogSoftmax(dim=-1)
        )

        self.train_epoch_losses = []
        self.train_epoch_accuracies = []
        self.val_epoch_losses = []
        self.val_epoch_accuracies = []

    def forward(self, x):
        return self.linear(self.convnet(x))

    def training_step(self, batch):
        data, labels = batch
        preds = self(data)
        loss = self.loss_fn(preds, labels)
        acc = self.accuracy(preds, labels)
        self.train_epoch_losses.append(loss.detach().cpu())
        self.train_epoch_accuracies.append(acc)

        wandb.log({"trainer/global_step": self.global_step, "train loss": loss, "train accuracy": acc})
        return loss

    def validation_step(self, batch):
        self.eval()
        data, labels = batch
        preds = self(data)
        loss = self.loss_fn(preds, labels)
        acc = self.accuracy(preds, labels)
        self.val_epoch_losses.append(loss.cpu())
        self.val_epoch_accuracies.append(acc)

        wandb.log({"trainer/global_step": self.global_step, "validation loss": loss, "validation accuracy": acc})
        return loss

    def on_train_epoch_end(self):
        # To log average train metrics for all batches
        epoch_acc = np.mean(self.train_epoch_accuracies)
        epoch_loss = np.mean(self.train_epoch_losses)
        wandb.log(
            {"trainer/epoch": self.current_epoch, "epoch train accuracy": epoch_acc, "epoch train loss": epoch_loss}
        )

        self.train_epoch_accuracies.clear()
        self.train_epoch_losses.clear()

    def on_validation_epoch_end(self):
        # To log average validation metrics for all batches
        epoch_acc = np.mean(self.val_epoch_accuracies)
        epoch_loss = np.mean(self.val_epoch_losses)
        wandb.log({"trainer/epoch": self.current_epoch, "epoch val accuracy": epoch_acc, "epoch val loss": epoch_loss})

        self.val_epoch_accuracies.clear()
        self.val_epoch_losses.clear()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        train_ds = torch.load("../data/processed/train_dataset.pkl")
        return DataLoader(train_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        test_ds = torch.load("../data/processed/test_dataset.pkl")
        return DataLoader(test_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        val_ds = torch.load("../data/processed/validation_dataset.pkl")
        return DataLoader(val_ds, batch_size=self.batch_size)

    def accuracy(self, preds, labels):
        pred_labels = preds.argmax(dim=-1)
        return torch.sum(pred_labels == labels).item() / len(labels)
