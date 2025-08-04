"""Command-line interface for DGDM Histopath Lab."""

from dgdm_histopath.cli.train import train
from dgdm_histopath.cli.predict import predict  
from dgdm_histopath.cli.preprocess import preprocess

__all__ = ["train", "predict", "preprocess"]