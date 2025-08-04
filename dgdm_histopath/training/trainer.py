"""
PyTorch Lightning trainer for DGDM models.

Implements training logic for self-supervised pretraining and 
supervised finetuning of Dynamic Graph Diffusion Models.
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from typing import Dict, Any, Optional, Union
import logging
from torch_geometric.data import Batch

from dgdm_histopath.models.dgdm_model import DGDMModel
from dgdm_histopath.training.losses import DiffusionLoss, ContrastiveLoss


class DGDMTrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for DGDM models.
    
    Handles both self-supervised pretraining with diffusion and entity masking,
    and supervised finetuning for downstream tasks.
    """
    
    def __init__(
        self,
        model: DGDMModel,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        pretrain_epochs: int = 50,
        finetune_epochs: int = 50,
        masking_ratio: float = 0.15,
        diffusion_noise_schedule: str = "cosine",
        use_contrastive_loss: bool = True,
        contrastive_temperature: float = 0.1,
        scheduler_type: str = "cosine",
        warmup_steps: int = 1000,
        **kwargs
    ):
        """
        Initialize DGDM trainer.
        
        Args:
            model: DGDM model to train
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            pretrain_epochs: Number of pretraining epochs
            finetune_epochs: Number of finetuning epochs
            masking_ratio: Ratio of nodes to mask during pretraining
            diffusion_noise_schedule: Noise schedule for diffusion
            use_contrastive_loss: Whether to use contrastive loss
            contrastive_temperature: Temperature for contrastive loss
            scheduler_type: Type of learning rate scheduler
            warmup_steps: Number of warmup steps
        """
        super().__init__()
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.masking_ratio = masking_ratio
        self.use_contrastive_loss = use_contrastive_loss
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        
        # Loss functions
        self.diffusion_loss = DiffusionLoss()
        if use_contrastive_loss:
            self.contrastive_loss = ContrastiveLoss(temperature=contrastive_temperature)
        else:
            self.contrastive_loss = None
            
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
        # Training phase tracking
        self.current_phase = "pretrain"  # or "finetune"
        
        self.logger_obj = logging.getLogger(__name__)
        
    def forward(self, batch: Batch, mode: str = "inference") -> Dict[str, torch.Tensor]:
        """Forward pass through model."""
        return self.model(batch, mode=mode, return_attention=True, return_embeddings=True)
        
    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Training step for both pretraining and finetuning."""
        
        # Determine training phase based on current epoch
        if self.current_epoch < self.pretrain_epochs:
            return self._pretrain_step(batch, batch_idx)
        else:
            return self._finetune_step(batch, batch_idx)
            
    def _pretrain_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Pretraining step with self-supervised objectives."""
        
        # Forward pass in pretrain mode
        outputs = self.model.pretrain_step(batch, mask_ratio=self.masking_ratio)
        
        # Compute losses
        total_loss = outputs["total_pretrain_loss"]
        diffusion_loss = outputs["diffusion_loss"]
        
        # Add contrastive loss if enabled
        if self.contrastive_loss is not None and "node_embeddings" in outputs:
            contrastive_loss = self.contrastive_loss(
                outputs["node_embeddings"], batch.batch
            )
            total_loss = total_loss + contrastive_loss
            self.log("train/contrastive_loss", contrastive_loss, sync_dist=True)
            
        # Log losses
        self.log("train/total_loss", total_loss, sync_dist=True)
        self.log("train/diffusion_loss", diffusion_loss, sync_dist=True)
        
        if "reconstruction_loss" in outputs:
            self.log("train/reconstruction_loss", outputs["reconstruction_loss"], sync_dist=True)
            
        # Log phase
        self.log("train/phase", 0.0, sync_dist=True)  # 0 for pretrain
        
        return total_loss
        
    def _finetune_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Finetuning step with supervised objectives."""
        
        # Forward pass in finetune mode
        outputs = self.forward(batch, mode="finetune")
        
        total_loss = 0.0
        num_losses = 0
        
        # Classification loss
        if "classification_logits" in outputs and hasattr(batch, 'y'):
            if self.model.classification_head is not None:
                cls_loss = self.model.classification_head.compute_loss(
                    outputs["classification_logits"], batch.y
                )
                total_loss += cls_loss
                num_losses += 1
                self.log("train/classification_loss", cls_loss, sync_dist=True)
                
                # Compute accuracy
                preds = torch.argmax(outputs["classification_logits"], dim=1)
                acc = (preds == batch.y).float().mean()
                self.log("train/accuracy", acc, sync_dist=True)
                
        # Regression loss
        if "regression_outputs" in outputs and hasattr(batch, 'regression_targets'):
            if self.model.regression_head is not None:
                reg_loss = self.model.regression_head.compute_loss(
                    outputs["regression_outputs"], batch.regression_targets
                )
                total_loss += reg_loss
                num_losses += 1
                self.log("train/regression_loss", reg_loss, sync_dist=True)
                
        # Fallback to diffusion loss if no supervised targets
        if num_losses == 0:
            diffusion_outputs = self.model._compute_diffusion_loss(
                outputs["node_embeddings"], batch
            )
            total_loss = diffusion_outputs["diffusion_loss"]
            self.log("train/diffusion_loss", total_loss, sync_dist=True)
            
        self.log("train/total_loss", total_loss, sync_dist=True)
        self.log("train/phase", 1.0, sync_dist=True)  # 1 for finetune
        
        return total_loss
        
    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        
        # Forward pass
        outputs = self.forward(batch, mode="inference")
        
        val_metrics = {}
        
        # Validation losses
        if hasattr(batch, 'y') and "classification_logits" in outputs:
            if self.model.classification_head is not None:
                val_loss = self.model.classification_head.compute_loss(
                    outputs["classification_logits"], batch.y
                )
                val_metrics["val_loss"] = val_loss
                
                # Accuracy
                preds = torch.argmax(outputs["classification_logits"], dim=1)
                acc = (preds == batch.y).float().mean()
                val_metrics["val_accuracy"] = acc
                
        # Regression metrics
        if hasattr(batch, 'regression_targets') and "regression_outputs" in outputs:
            if self.model.regression_head is not None:
                reg_loss = self.model.regression_head.compute_loss(
                    outputs["regression_outputs"], batch.regression_targets
                )
                val_metrics["val_regression_loss"] = reg_loss
                
                # MAE
                mae = F.l1_loss(outputs["regression_outputs"], batch.regression_targets)
                val_metrics["val_mae"] = mae
                
        self.log_dict(val_metrics, sync_dist=True)
        return val_metrics
        
    def test_step(self, batch: Batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        return self.validation_step(batch, batch_idx)
        
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        
        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        total_steps = self.trainer.estimated_stepping_batches
        
        if self.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=self.learning_rate * 0.01
            )
        elif self.scheduler_type == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                pct_start=0.1
            )
        else:
            # No scheduler
            return optimizer
            
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        
    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        
        # Update training phase
        if self.current_epoch < self.pretrain_epochs:
            if self.current_phase != "pretrain":
                self.current_phase = "pretrain"
                self.logger_obj.info(f"Entering pretraining phase at epoch {self.current_epoch}")
        else:
            if self.current_phase != "finetune":
                self.current_phase = "finetune"
                self.logger_obj.info(f"Entering finetuning phase at epoch {self.current_epoch}")
                
                # Optionally reduce learning rate for finetuning
                for param_group in self.optimizers().param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
                    
    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch."""
        
        # Log current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", current_lr, sync_dist=True)
        
    def predict_step(self, batch: Batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step."""
        
        outputs = self.forward(batch, mode="inference")
        
        predictions = {
            "graph_embeddings": outputs["graph_embedding"],
            "node_embeddings": outputs.get("node_embeddings"),
        }
        
        # Add task-specific predictions
        if "classification_probs" in outputs:
            predictions["classification_probs"] = outputs["classification_probs"]
            predictions["predicted_classes"] = torch.argmax(
                outputs["classification_logits"], dim=1
            )
            
        if "regression_outputs" in outputs:
            predictions["regression_predictions"] = outputs["regression_outputs"]
            
        if "attention_weights" in outputs:
            predictions["attention_weights"] = outputs["attention_weights"]
            
        return predictions
        
    def generate_embeddings(self, dataloader) -> Dict[str, torch.Tensor]:
        """Generate embeddings for a dataset."""
        
        self.eval()
        all_embeddings = []
        all_labels = []
        all_slide_ids = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                outputs = self.forward(batch, mode="inference")
                
                all_embeddings.append(outputs["graph_embedding"].cpu())
                
                if hasattr(batch, 'y'):
                    all_labels.append(batch.y.cpu())
                    
                if hasattr(batch, 'slide_id'):
                    all_slide_ids.extend(batch.slide_id)
                    
        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0) if all_labels else None
        
        return {
            "embeddings": embeddings,
            "labels": labels,
            "slide_ids": all_slide_ids
        }
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DGDMTrainer':
        """Create trainer from configuration dictionary."""
        
        # Create model
        model_config = config.get("model", {})
        model = DGDMModel(**model_config)
        
        # Create trainer
        trainer_config = config.get("training", {})
        return cls(model=model, **trainer_config)
        
    def save_model(self, filepath: str):
        """Save model weights and configuration."""
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "hyperparameters": self.hparams,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
        }
        
        torch.save(checkpoint, filepath)
        self.logger_obj.info(f"Saved model to {filepath}")