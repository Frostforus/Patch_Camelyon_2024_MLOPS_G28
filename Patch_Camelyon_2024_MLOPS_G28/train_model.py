from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torchvision.datasets import PCAM
import hydra

from models.model import SimpleCNN

@hydra.main(config_path="conf", config_name="config")
def train(cfg):
    model = SimpleCNN()
    wandb_logger = WandbLogger(log_model=True,project='Patch_Camelyon_MLOps_WandB')
    trainer = Trainer(accelerator=cfg.trainer.accelerator, logger=wandb_logger,
        max_epochs=cfg.trainer.max_epochs,
        limit_train_batches=cfg.trainer.limit_train_batches,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch)
    trainer.fit(model)

if __name__ == '__main__':
    train()