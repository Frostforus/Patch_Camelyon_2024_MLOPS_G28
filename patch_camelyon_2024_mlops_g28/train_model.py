from pytorch_lightning import Trainer
import torch
from pytorch_lightning.loggers import WandbLogger
from torchvision.datasets import PCAM
import hydra
from google.cloud import storage
import os

from patch_camelyon_2024_mlops_g28.models.model import SimpleCNN


# config hydra
@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def train(cfg) -> None:
    """
    Defined a model, trains it based on the parameters on the config file and
    stores it in a google cloud data bucket.

    Args:
        cfg: hydra specific class with all parameters from the config.yaml file.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(cfg.model.random_seed)

    #Set hyperaparameters
    model = SimpleCNN(
        lr=cfg.model.lr,
        batch_size=cfg.model.batch_size,
        seed=cfg.model.random_seed
    )
    # Set up logging with WandB
    wandb_logger = WandbLogger(log_model=True, project='Patch_Camelyon_MLOps_WandB')

    # Set up PyTorch Lightning Trainer with config parameters
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() and cfg.trainer.gpu_accelerator else "cpu",
        logger=wandb_logger,
        max_epochs=cfg.trainer.max_epochs,
        limit_train_batches=cfg.trainer.limit_train_batches,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val
    )
    wandb_logger.experiment.config['Dataset_fraction_used'] = cfg.trainer.limit_train_batches
    wandb_logger.experiment.config['Accumulated_grad_batches'] = cfg.trainer.accumulate_grad_batches
    wandb_logger.experiment.config['Gradient_clip_val'] = cfg.trainer.gradient_clip_val
    # Train the model
    trainer.fit(model)

    # Save the model locally first
    local_checkpoint_path = "trained_model_out.pt"
    torch.save(model, local_checkpoint_path)

    # Upload the model to Google Cloud Storage bucket
    bucket_name = "prediction-model-bucket"
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob("trained_model_out.pt")
    blob.upload_from_filename(local_checkpoint_path)

    # Delete the locally created file after uploading to the bucket
    os.remove(local_checkpoint_path)

    

if __name__ == '__main__':
    train()
