from pytorch_lightning import Trainer
import torch
from pytorch_lightning.loggers import WandbLogger
from torchvision.datasets import PCAM
import hydra

from models.model import SimpleCNN

@hydra.main(config_path="conf", config_name="config", version_base= '1.2')
def train(cfg):

    torch.manual_seed(cfg.model.random_seed)

    model = SimpleCNN(
        lr=cfg.model.lr,
        batch_size=cfg.model.batch_size
    )
    wandb_logger = WandbLogger(log_model=True,project='Patch_Camelyon_MLOps_WandB')
    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        logger=wandb_logger,
        max_epochs=cfg.trainer.max_epochs,
        limit_train_batches=cfg.trainer.limit_train_batches,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val
    )
    
    trainer.fit(model)

    trainer.save_checkpoint("trained_model_out.ckpt")

if __name__ == '__main__':
    train()