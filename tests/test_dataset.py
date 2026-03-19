"""Tests for SyntheticHistoDataset."""

import numpy as np
import pytest
from dgdm_histopath.dataset import SyntheticHistoDataset


class TestSyntheticHistoDataset:
    def test_length(self):
        ds = SyntheticHistoDataset(n_samples=20)
        assert len(ds) == 20

    def test_class_balance(self):
        ds = SyntheticHistoDataset(n_samples=20)
        labels = ds.labels()
        assert labels.count(0) == 10
        assert labels.count(1) == 10

    def test_patch_shape(self):
        ds = SyntheticHistoDataset(n_samples=4, patch_size=64)
        for i in range(len(ds)):
            item = ds[i]
            assert item["image"].shape == (64, 64)

    def test_dtype_uint8(self):
        ds = SyntheticHistoDataset(n_samples=4)
        for item in [ds[0], ds[1]]:
            assert item["image"].dtype == np.uint8

    def test_pixel_range(self):
        ds = SyntheticHistoDataset(n_samples=10)
        for i in range(len(ds)):
            img = ds[i]["image"]
            assert img.min() >= 0
            assert img.max() <= 255

    def test_label_names(self):
        ds = SyntheticHistoDataset(n_samples=4)
        for i in range(len(ds)):
            item = ds[i]
            assert item["label_name"] in ("NORMAL", "TUMOR")
            if item["label"] == 0:
                assert item["label_name"] == "NORMAL"
            else:
                assert item["label_name"] == "TUMOR"

    def test_reproducibility(self):
        ds1 = SyntheticHistoDataset(n_samples=10, seed=99)
        ds2 = SyntheticHistoDataset(n_samples=10, seed=99)
        for i in range(len(ds1)):
            np.testing.assert_array_equal(ds1[i]["image"], ds2[i]["image"])
            assert ds1[i]["label"] == ds2[i]["label"]

    def test_different_seeds(self):
        ds1 = SyntheticHistoDataset(n_samples=10, seed=1)
        ds2 = SyntheticHistoDataset(n_samples=10, seed=2)
        # At least some images should differ
        diffs = sum(
            not np.array_equal(ds1[i]["image"], ds2[i]["image"])
            for i in range(len(ds1))
        )
        assert diffs > 0

    def test_tumor_is_darker(self):
        """On average, tumor patches should have lower mean intensity than NORMAL."""
        ds = SyntheticHistoDataset(n_samples=100, seed=7)
        normal_means = [ds[i]["image"].mean() for i in range(len(ds)) if ds[i]["label"] == 0]
        tumor_means  = [ds[i]["image"].mean() for i in range(len(ds)) if ds[i]["label"] == 1]
        assert np.mean(tumor_means) < np.mean(normal_means)

    def test_images_helper(self):
        ds = SyntheticHistoDataset(n_samples=6)
        imgs = ds.images()
        assert len(imgs) == 6
        assert all(isinstance(img, np.ndarray) for img in imgs)
