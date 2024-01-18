from pytorch_lightning import Trainer
import torch
from pytorch_lightning.loggers import WandbLogger
import hydra
from google.cloud import storage
import os

from models.model import SimpleCNN


# config hydra
@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def train(cfg):
    # Set random seeds for reproducibility
    torch.manual_seed(cfg.model.random_seed)

    # Set hyperaparameters
    model = SimpleCNN(lr=cfg.model.lr, batch_size=cfg.model.batch_size, seed=cfg.model.random_seed)
    # Set up logging with WandB
    wandb_logger = WandbLogger(log_model=True, project="Patch_Camelyon_MLOps_WandB")

    if not torch.cuda.is_available() and cfg.trainer.gpu_accelerator:
        print("As no GPU is available the CPU will be used instead.")

    # Set up PyTorch Lightning Trainer with config parameters
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() and cfg.trainer.gpu_accelerator else "cpu",
        logger=wandb_logger,
        max_epochs=cfg.trainer.max_epochs,
        limit_train_batches=cfg.trainer.limit_train_batches,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
    )

    # Train the model
    trainer.fit(model)

    # Save the model locally first
    local_checkpoint_path = "trained_model_out.ckpt"
    trainer.save_checkpoint(local_checkpoint_path)

    # Upload the model to Google Cloud Storage bucket
    bucket_name = "prediction-model-bucket"
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob("trained_model_out.ckpt")
    blob.upload_from_filename(local_checkpoint_path)

    # Delete the locally created file after uploading to the bucket
    os.remove(local_checkpoint_path)

    

if __name__ == '__main__':
    train()
