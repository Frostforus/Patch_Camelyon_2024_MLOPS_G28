from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torchvision.datasets import PCAM

from models.model import SimpleCNN

def train():
    model = SimpleCNN()
    wandb_logger = WandbLogger(log_model=True,project='Patch_Camelyon_MLOps_WandB')
    trainer = Trainer(accelerator='gpu', logger=wandb_logger, max_epochs=50, limit_train_batches=0.5, check_val_every_n_epoch=2)
    trainer.fit(model)

if __name__ == '__main__':
    train()