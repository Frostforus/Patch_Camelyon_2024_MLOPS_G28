from pytorch_lightning import Trainer
from torchvision.datasets import PCAM

from models.model import SimpleCNN

def train():
    model = SimpleCNN()
    trainer = Trainer(accelerator='gpu', max_epochs=50, limit_train_batches=0.2)
    trainer.fit(model)

if __name__ == '__main__':
    train()