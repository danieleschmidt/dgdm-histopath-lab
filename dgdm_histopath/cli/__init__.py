"""Command-line interface for DGDM Histopath Lab."""

from dgdm_histopath.cli.train import train
from dgdm_histopath.cli.predict import predict  
from dgdm_histopath.cli.preprocess import preprocess
from dgdm_histopath.cli.quality_gates import app as quality_gates

__all__ = ["train", "predict", "preprocess", "quality_gates"]