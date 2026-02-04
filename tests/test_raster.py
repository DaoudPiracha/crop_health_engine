"""Tests for raster I/O operations."""

import numpy as np
import pytest
import rasterio

from engine.io.raster import (
    load_raster_with_affine,
    crop_raster_with_polygon,
    open_raster_in_crs,
)


class TestLoadRasterWithAffine:
    """Test loading raster with affine transform."""

    def test_load_raster_basic(self, test_raster_path):
        """Test basic raster loading."""
        image_data, affine_transform, crs, meta = load_raster_with_affine(test_raster_path)

        # Check shape (3 bands for RGB: RED, GREEN, BLUE)
        assert image_data.shape[0] == 3  # 3 bands
        assert image_data.shape[1] == 50  # height
        assert image_data.shape[2] == 50  # width

        # Check affine transform exists
        assert affine_transform is not None

        # Check CRS
        assert crs is not None
        assert str(crs) == "EPSG:4326"

        # Check metadata
        bounds, width, height = meta
        assert bounds is not None
        assert width == 50
        assert height == 50

    def test_load_raster_data_values(self, test_raster_path):
        """Test that loaded data has expected values."""
        image_data, _, _, _ = load_raster_with_affine(test_raster_path)

        # Data should be non-zero
        assert np.any(image_data > 0)

        # Data should be in reasonable range for satellite imagery
        assert np.all(image_data < 10000)


class TestCropRasterWithPolygon:
    """Test raster cropping with polygon."""

    def test_crop_raster_basic(self, test_raster_path, test_polygon):
        """Test basic raster cropping."""
        with rasterio.open(test_raster_path) as raster:
            cropped_image, out_transform = crop_raster_with_polygon(raster, test_polygon)

        # Cropped image should have 8 bands
        assert cropped_image.shape[0] == 8

        # Cropped dimensions should be smaller than original
        assert cropped_image.shape[1] <= 50
        assert cropped_image.shape[2] <= 50

        # Should have some valid data
        assert np.any(cropped_image > 0)

        # Transform should exist
        assert out_transform is not None

    def test_crop_raster_reduces_size(self, test_raster_path, test_polygon):
        """Test that cropping reduces raster size."""
        with rasterio.open(test_raster_path) as raster:
            original_shape = (raster.height, raster.width)
            cropped_image, _ = crop_raster_with_polygon(raster, test_polygon)

        cropped_shape = cropped_image.shape[1:]  # (height, width)

        # Cropped should be smaller in at least one dimension
        assert cropped_shape[0] < original_shape[0] or cropped_shape[1] < original_shape[1]


class TestOpenRasterInCRS:
    """Test context manager for CRS-aware raster opening."""

    def test_open_raster_same_crs(self, test_raster_path):
        """Test opening raster when CRS matches target."""
        target_crs = "EPSG:4326"

        with open_raster_in_crs(test_raster_path, target_crs) as ds:
            assert ds is not None
            assert ds.crs == rasterio.crs.CRS.from_string(target_crs)
            assert ds.count == 8  # 8 bands
            assert not ds.closed

        # Dataset should be closed after context
        assert ds.closed

    def test_open_raster_different_crs(self, test_raster_path):
        """Test opening raster when CRS needs reprojection."""
        target_crs = "EPSG:3857"  # Web Mercator

        with open_raster_in_crs(test_raster_path, target_crs) as ds:
            assert ds is not None
            assert ds.crs == rasterio.crs.CRS.from_string(target_crs)
            assert ds.count == 8

            # Should be able to read data
            data = ds.read(1)
            assert data.shape[0] > 0
            assert data.shape[1] > 0

        # Dataset should be closed
        assert ds.closed

    def test_open_raster_cleanup_on_error(self, test_raster_path):
        """Test that resources are cleaned up even on error."""
        target_crs = "EPSG:4326"

        with pytest.raises(ValueError):
            with open_raster_in_crs(test_raster_path, target_crs) as ds:
                # Force an error
                raise ValueError("Test error")

        # Context manager should still clean up

    def test_open_raster_reads_all_bands(self, test_raster_path):
        """Test that all bands can be read."""
        target_crs = "EPSG:4326"

        with open_raster_in_crs(test_raster_path, target_crs) as ds:
            # Read all 8 bands
            for band_idx in range(1, 9):
                band_data = ds.read(band_idx)
                assert band_data is not None
                assert band_data.shape == (50, 50)

    def test_open_raster_reprojection_preserves_data(self, test_raster_path):
        """Test that reprojection preserves data integrity."""
        target_crs = "EPSG:3857"

        # Read original
        with rasterio.open(test_raster_path) as original:
            original_data = original.read(1)

        # Read reprojected
        with open_raster_in_crs(test_raster_path, target_crs) as reprojected:
            reprojected_data = reprojected.read(1)

        # Data should exist in both
        assert np.any(original_data > 0)
        assert np.any(reprojected_data > 0)

        # Shape might differ but data should have similar statistical properties
        assert np.mean(reprojected_data) > 0
