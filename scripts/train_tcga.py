#!/usr/bin/env python3
"""
Example training script for TCGA dataset using DGDM.

This script demonstrates how to train a DGDM model on TCGA breast cancer data
for molecular subtype classification.
"""

import argparse
import logging
from pathlib import Path
import pytorch_lightning as pl

from dgdm_histopath.models.dgdm_model import DGDMModel
from dgdm_histopath.data.datamodule import HistopathDataModule
from dgdm_histopath.training.trainer import DGDMTrainer
from dgdm_histopath.utils.config import load_config
from dgdm_histopath.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train DGDM on TCGA dataset")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing TCGA slide data")
    parser.add_argument("--config", type=str, default="configs/dgdm_base.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="./outputs/tcga_experiment",
                       help="Output directory for results")
    
    # Model arguments
    parser.add_argument("--num_classes", type=int, default=4,
                       help="Number of molecular subtypes (4 for TCGA-BRCA)")
    parser.add_argument("--pretrain_epochs", type=int, default=50,
                       help="Number of pretraining epochs")
    parser.add_argument("--finetune_epochs", type=int, default=50,
                       help="Number of finetuning epochs")
    
    # Hardware arguments
    parser.add_argument("--gpus", type=int, default=1,
                       help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--fast_dev_run", action="store_true",
                       help="Run single batch for debugging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="DEBUG" if args.debug else "INFO")
    logger = logging.getLogger(__name__)
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting TCGA DGDM training")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Configuration: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config["model"]["num_classes"] = args.num_classes
    config["training"]["pretrain_epochs"] = args.pretrain_epochs
    config["training"]["finetune_epochs"] = args.finetune_epochs
    config["data"]["batch_size"] = args.batch_size
    config["hardware"]["gpus"] = args.gpus
    config["experiment"]["seed"] = args.seed
    
    # Setup data module for TCGA
    logger.info("Setting up TCGA data module...")
    data_module = HistopathDataModule(
        data_dir=args.data_dir,
        dataset_type="slide",
        batch_size=args.batch_size,
        num_workers=8,
        augmentations="strong",  # Strong augmentation for small dataset
        **config.get("data", {})
    )
    
    # Create DGDM model for classification
    logger.info("Creating DGDM model for molecular subtype classification...")
    model_config = config["model"]
    model_config["num_classes"] = args.num_classes  # 4 subtypes: Luminal A, Luminal B, Her2, Basal
    
    model = DGDMModel(**model_config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer wrapper
    trainer_wrapper = DGDMTrainer(
        model=model,
        learning_rate=5e-5,  # Lower learning rate for finetuning
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        **config.get("training", {})
    )
    
    # Setup PyTorch Lightning trainer
    callbacks = []
    
    # Model checkpointing
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path / "checkpoints",
        filename="tcga-dgdm-{epoch:02d}-{val_accuracy:.3f}",
        monitor="val_accuracy",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=15,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Logger
    from pytorch_lightning.loggers import TensorBoardLogger
    pl_logger = TensorBoardLogger(
        save_dir=output_path,
        name="tcga_dgdm_logs"
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.pretrain_epochs + args.finetune_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else "auto",
        precision="16-mixed",
        callbacks=callbacks,
        logger=pl_logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        fast_dev_run=args.fast_dev_run,
        deterministic=True
    )
    
    # Training
    logger.info("Starting DGDM training on TCGA dataset...")
    try:
        trainer.fit(trainer_wrapper, datamodule=data_module)
        
        # Test the best model
        if not args.fast_dev_run:
            logger.info("Running final evaluation on test set...")
            trainer.test(ckpt_path="best", datamodule=data_module)
            
        logger.info("Training completed successfully!")
        logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        logger.info(f"Best validation accuracy: {checkpoint_callback.best_model_score:.3f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
        
    # Save example predictions
    logger.info("Generating example predictions...")
    try:
        predictions = trainer.predict(datamodule=data_module, ckpt_path="best")
        
        # Save predictions (simplified version)
        import torch
        prediction_file = output_path / "example_predictions.pt"
        torch.save(predictions[:10], prediction_file)
        logger.info(f"Saved example predictions to {prediction_file}")
        
    except Exception as e:
        logger.warning(f"Failed to generate predictions: {e}")


if __name__ == "__main__":
    main()