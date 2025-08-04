"""
Training CLI for DGDM Histopath Lab.

Command-line interface for training DGDM models on histopathology data.
"""

import typer
import logging
from pathlib import Path
from typing import Optional, List
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from dgdm_histopath.models.dgdm_model import DGDMModel
from dgdm_histopath.data.datamodule import HistopathDataModule
from dgdm_histopath.training.trainer import DGDMTrainer
from dgdm_histopath.utils.config import load_config, save_config


app = typer.Typer(help="Train DGDM models on histopathology data")


@app.command()
def train(
    data_dir: str = typer.Option(..., help="Directory containing training data"),
    config: Optional[str] = typer.Option(None, help="Path to configuration file"),
    output_dir: str = typer.Option("./outputs", help="Output directory for results"),
    
    # Model parameters
    node_features: int = typer.Option(768, help="Node feature dimension"),
    hidden_dims: str = typer.Option("512,256,128", help="Hidden dimensions (comma-separated)"),
    num_diffusion_steps: int = typer.Option(10, help="Number of diffusion steps"),
    attention_heads: int = typer.Option(8, help="Number of attention heads"),
    dropout: float = typer.Option(0.1, help="Dropout rate"),
    
    # Training parameters
    max_epochs: int = typer.Option(100, help="Maximum training epochs"),
    batch_size: int = typer.Option(4, help="Batch size"),
    learning_rate: float = typer.Option(1e-4, help="Learning rate"),
    weight_decay: float = typer.Option(1e-5, help="Weight decay"),
    
    # Data parameters
    dataset_type: str = typer.Option("slide", help="Dataset type (slide/graph/patch)"),
    augmentations: str = typer.Option("light", help="Augmentation strategy"),
    num_workers: int = typer.Option(8, help="Number of data loading workers"),
    
    # Training strategy
    pretrain_epochs: int = typer.Option(50, help="Pretraining epochs"),
    finetune_epochs: int = typer.Option(50, help="Finetuning epochs"),
    masking_ratio: float = typer.Option(0.15, help="Masking ratio for self-supervision"),
    
    # Hardware
    gpus: int = typer.Option(1, help="Number of GPUs"),
    precision: str = typer.Option("16-mixed", help="Training precision"),
    
    # Logging
    logger_type: str = typer.Option("tensorboard", help="Logger type (tensorboard/wandb)"),
    experiment_name: str = typer.Option("dgdm_experiment", help="Experiment name"),
    
    # Checkpointing
    save_top_k: int = typer.Option(3, help="Number of best checkpoints to save"),
    monitor_metric: str = typer.Option("val_loss", help="Metric to monitor for checkpointing"),
    
    # Other options
    seed: int = typer.Option(42, help="Random seed"),
    resume_from_checkpoint: Optional[str] = typer.Option(None, help="Checkpoint to resume from"),
    fast_dev_run: bool = typer.Option(False, help="Run single batch for debugging"),
    debug: bool = typer.Option(False, help="Enable debug logging")
):
    """Train DGDM model on histopathology data."""
    
    # Setup logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Set random seed
    pl.seed_everything(seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting DGDM training experiment: {experiment_name}")
    logger.info(f"Output directory: {output_path}")
    
    # Parse hidden dimensions
    hidden_dims_list = [int(x.strip()) for x in hidden_dims.split(",")]
    
    # Load configuration if provided
    if config:
        config_dict = load_config(config)
        logger.info(f"Loaded configuration from {config}")
    else:
        config_dict = {}
        
    # Create data module
    logger.info("Setting up data module...")
    data_module = HistopathDataModule(
        data_dir=data_dir,
        dataset_type=dataset_type,
        batch_size=batch_size,
        num_workers=num_workers,
        augmentations=augmentations,
        **config_dict.get("data", {})
    )
    
    # Create model
    logger.info("Creating DGDM model...")
    model_config = {
        "node_features": node_features,
        "hidden_dims": hidden_dims_list,
        "num_diffusion_steps": num_diffusion_steps,
        "attention_heads": attention_heads,
        "dropout": dropout,
        **config_dict.get("model", {})
    }
    
    model = DGDMModel(**model_config)
    
    # Create trainer wrapper
    trainer_wrapper = DGDMTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        masking_ratio=masking_ratio,
        **config_dict.get("training", {})
    )
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path / "checkpoints",
        filename="dgdm-{epoch:02d}-{val_loss:.2f}",
        monitor=monitor_metric,
        mode="min" if "loss" in monitor_metric else "max",
        save_top_k=save_top_k,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        mode="min" if "loss" in monitor_metric else "max",
        patience=10,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Setup logger
    if logger_type == "wandb":
        pl_logger = WandbLogger(
            project="dgdm-histopath",
            name=experiment_name,
            save_dir=output_path
        )
    else:
        pl_logger = TensorBoardLogger(
            save_dir=output_path,
            name=experiment_name
        )
        
    # Create PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else "auto",
        precision=precision,
        callbacks=callbacks,
        logger=pl_logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        fast_dev_run=fast_dev_run,
        deterministic=True,
        **config_dict.get("trainer", {})
    )
    
    # Save configuration
    final_config = {
        "model": model_config,
        "data": data_module.get_dataset_info(),
        "training": {
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "pretrain_epochs": pretrain_epochs,
            "finetune_epochs": finetune_epochs,
            "masking_ratio": masking_ratio
        },
        "hardware": {
            "gpus": gpus,
            "precision": precision
        },
        "experiment": {
            "name": experiment_name,
            "seed": seed,
            "output_dir": str(output_path)
        }
    }
    
    save_config(final_config, output_path / "config.yaml")
    logger.info(f"Saved configuration to {output_path / 'config.yaml'}")
    
    # Training
    logger.info("Starting training...")
    try:
        trainer.fit(
            trainer_wrapper, 
            datamodule=data_module,
            ckpt_path=resume_from_checkpoint
        )
        
        # Test after training
        if not fast_dev_run:
            logger.info("Running final evaluation...")
            trainer.test(datamodule=data_module)
            
        logger.info("Training completed successfully!")
        logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise typer.Exit(1)
        
    # Save final model
    final_model_path = output_path / "final_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")


@app.command()
def resume(
    checkpoint_path: str = typer.Option(..., help="Path to checkpoint to resume from"),
    data_dir: str = typer.Option(..., help="Directory containing training data"),
    max_epochs: int = typer.Option(100, help="Maximum epochs to continue training"),
    output_dir: str = typer.Option("./outputs", help="Output directory"),
):
    """Resume training from a checkpoint."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Extract configuration from checkpoint
    # This is a simplified version - full implementation would extract all parameters
    data_module = HistopathDataModule(data_dir=data_dir)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
    )
    
    # Resume training
    model = DGDMTrainer.load_from_checkpoint(checkpoint_path)
    trainer.fit(model, datamodule=data_module)


@app.command()
def validate(
    checkpoint_path: str = typer.Option(..., help="Path to trained model checkpoint"),
    data_dir: str = typer.Option(..., help="Directory containing validation data"),
    batch_size: int = typer.Option(4, help="Batch size for validation"),
    output_dir: str = typer.Option("./validation_results", help="Output directory"),
):
    """Validate a trained DGDM model."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Validating model: {checkpoint_path}")
    
    # Load model
    model = DGDMTrainer.load_from_checkpoint(checkpoint_path)
    
    # Setup data
    data_module = HistopathDataModule(
        data_dir=data_dir,
        batch_size=batch_size
    )
    
    # Create trainer for validation
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
        logger=False
    )
    
    # Run validation
    results = trainer.validate(model, datamodule=data_module)
    
    logger.info("Validation results:")
    for result in results:
        for key, value in result.items():
            logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    app()