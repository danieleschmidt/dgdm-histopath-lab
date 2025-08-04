"""Training modules for DGDM."""

from dgdm_histopath.training.trainer import DGDMTrainer
from dgdm_histopath.training.losses import DiffusionLoss, ContrastiveLoss

__all__ = ["DGDMTrainer", "DiffusionLoss", "ContrastiveLoss"]