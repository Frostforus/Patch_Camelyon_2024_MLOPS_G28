from pytorch_lightning import LightningModule
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import PCAM

class SimpleCNN(LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24*24*128,1024),
            nn.Linear(1024, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.linear(self.convnet(x))
    
    def training_step(self, batch):
        data, label = batch
        preds = self(data)
        loss = self.loss_fn(preds, label)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
    
    def train_dataloader(self):
        train_ds = torch.load('./data/processed/train_dataset.pkl')
        return DataLoader(train_ds, batch_size=32)
    
    def test_dataloader(self):
        test_ds = torch.load('./data/processed/test_dataset.pkl')
        return DataLoader(test_ds, batch_size=32)

    def val_dataloader(self):
        val_ds = torch.load('./data/processed/validation_dataset.pkl')
        return DataLoader(val_ds, batch_size=32)
