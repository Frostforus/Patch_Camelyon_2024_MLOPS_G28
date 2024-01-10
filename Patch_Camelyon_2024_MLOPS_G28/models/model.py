from typing import Any
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, optim

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
