import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path

from src.dataloader.vehicle_datamodule import VehicleDataModule
from src.models.vehicle_classifier import VehicleClassifier

def main():
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Paths
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data"
    
    # DataModule
    dm = VehicleDataModule(
        data_dir=str(data_dir),
        batch_size=16, # Small batch size for small dataset
        num_workers=4
    )
    
    # Model
    model = VehicleClassifier(learning_rate=1e-6)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=root_dir / 'trained_models',
        filename='vehicle-classifier-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=2,
        verbose=True,
        mode='min'
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=20,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto", # Use GPU if available
        devices=1,
        log_every_n_steps=5
    )
    
    # Train
    print("Starting training...")
    trainer.fit(model, dm)
    
    # Test on Test Set
    print("Starting testing on Test Set...")
    trainer.test(model, dm, ckpt_path="best")


if __name__ == "__main__":
    main()
