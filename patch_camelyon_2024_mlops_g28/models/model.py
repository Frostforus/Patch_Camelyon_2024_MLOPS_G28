from pytorch_lightning import LightningModule
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
import wandb
import numpy as np


class SimpleCNN(LightningModule):
    """
    CNN model implmentation based on pytorch_lightning's LightningModule class
     ...

    Attributes
    ----------
    lr : float
        learning rate

    batch_size: int
        batch size for training, balifation and testimg of the model

    loss_fn: torch.nn.LossFunction
        loss function for the model

    convnet: torch.nn.Sequential
        encoder of the CNN model

    linear: torch.nn.Sequential
        fully connected layers of the CNN model including output layer

    train_epoch_losses: List
        list of losses for batches on the training process

    train_epoch_accuracies: List
        list of accuracies for batches on the training process

    val_epoch_losses: List
        list of losses for batches on the validation process

    val_epoch_accuracies: List
        list of accuracies for batches on the validation process

    Methods
    -------
    forward(x: torch.Tensor):
        Applies a forward pass of the model to a given input and outputs the model's predictions for it.

    training_step(batch: torch.Tensor):
        Contains the code application for a step of the training process.

    validation_step(batch: torch.Tensor):
        Contains the code application for a step of the validation process.

    on_train_epoch_end():
        Logs in wandb the average train metrics, accuracy and loss, for all batches and clears their values afterwards.

    on_validation_epoch_end():
        Logs in wandb the average validation metrics, accuracy and loss, for all batches and clears their values afterwards.

    configure_optimizers():
        Returns the model's optimizer allways Adam's.

    train_dataloader():
        Returns a torch's Dataloader class with the training data.

    test_dataloader():
        Returns a torch's Dataloader class with the test data.

    val_dataloader():
        Returns a torch's Dataloader class with the validation data.

    accuracy(preds: torch.Tensor, labels: torch.Tensor):
        Calculates and returns the accuracy of the model.
    """

    def __init__(self, lr: float = 1e-3, batch_size: int = 64, seed: int = 42) -> None:
        """
        Defined the classes attributes as well as the shape and parameters of the
        CNN model. Also manages the logging of hyperparameters and metrics on wandb.

        Args:
            lr: learning rate for the model's optimizer
            batch_size: batch size for training, balifation and testimg of the model
            seed: seed for random initiallization od the weigths and bias
        """
        torch.manual_seed(seed)
        super().__init__()
        # Save hyperparams and log them in wandb
        self.save_hyperparameters("lr", "batch_size", "seed")
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a forward pass of the model to a given input and outputs the
        model's predictions for it.

        Args:
            x: input tensor of shape (N,3,96,96) where x >= 1

        Returns
            A prediction's tensor of shape (N,1) where N is the first dimension of the input tensor.
        """
        return self.linear(self.convnet(x))

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """
        This function contains the code application for a step of the training process.

        Args:
            batch: input tensor of shape (N,3,96,96) where x >= 1

        Returns
            A tensor of size (1) representing the average loss of the model for
            each element of the input batch.
        """
        data, labels = batch
        preds = self(data)
        loss = self.loss_fn(preds, labels)
        acc = self.accuracy(preds, labels)
        self.train_epoch_losses.append(loss.detach().cpu())
        self.train_epoch_accuracies.append(acc)

        wandb.log({"trainer/global_step": self.global_step, "train loss": loss, "train accuracy": acc})
        return loss

    def validation_step(self, batch):
        """
        This function contains the code application for a step of the validation process.

        Args:
            batch: input tensor of shape (N,3,96,96) where x >= 1

        Returns
            A tensor of size (1) representing the average loss of the model for
            each element of the input batch.
        """
        self.eval()
        data, labels = batch
        preds = self(data)
        loss = self.loss_fn(preds, labels)
        acc = self.accuracy(preds, labels)
        self.val_epoch_losses.append(loss.cpu())
        self.val_epoch_accuracies.append(acc)

        wandb.log({"trainer/global_step": self.global_step, "validation loss": loss, "validation accuracy": acc})
        return loss

    def on_train_epoch_end(self) -> None:
        """Logs in wandb the average train metrics, accuracy and loss, for all batches and clears their values afterwards."""
        # To log average train metrics for all batches
        epoch_acc = np.mean(self.train_epoch_accuracies)
        epoch_loss = np.mean(self.train_epoch_losses)
        wandb.log(
            {"trainer/epoch": self.current_epoch, "epoch train accuracy": epoch_acc, "epoch train loss": epoch_loss}
        )

        self.train_epoch_accuracies.clear()
        self.train_epoch_losses.clear()

    def on_validation_epoch_end(self) -> None:
        """Logs in wandb the average validation metrics, accuracy and loss, for all batches and clears their values afterwards."""
        # To log average validation metrics for all batches
        epoch_acc = np.mean(self.val_epoch_accuracies)
        epoch_loss = np.mean(self.val_epoch_losses)
        wandb.log({"trainer/epoch": self.current_epoch, "epoch val accuracy": epoch_acc, "epoch val loss": epoch_loss})

        self.val_epoch_accuracies.clear()
        self.val_epoch_losses.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Returns the model's optimizer allways Adam's."""
        return optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns a torch's Dataloader class with the training data."""
        train_ds = torch.load("./data/processed/train_dataset.pkl")
        return DataLoader(train_ds, batch_size=self.batch_size)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns a torch's Dataloader class with the test data."""
        test_ds = torch.load("./data/processed/test_dataset.pkl")
        return DataLoader(test_ds, batch_size=self.batch_size)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns a torch's Dataloader class with the validation data."""
        val_ds = torch.load("./data/processed/validation_dataset.pkl")
        return DataLoader(val_ds, batch_size=self.batch_size)

    def accuracy(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculates the accuracy of the model.

        Args:
            preds: tensor of shape (N,1) where N >= 1 and N == X
            labels: tensor of shape (X,1) where N >= 1 and X == N

        Returns
            A float type tensor with a number between 0 and 1, representing the accuracy of the model.
        """
        pred_labels = preds.argmax(dim=-1)
        return torch.sum(pred_labels == labels).item() / len(labels)
