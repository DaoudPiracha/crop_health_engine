"""Pytest configuration and shared fixtures."""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import tempfile
import pytest


@pytest.fixture
def test_raster_path(tmp_path):
    """Create a small 8-band test raster."""
    # Create a small 50x50 pixel raster with 8 bands
    width, height = 50, 50
    bands = 8

    # Define bounds (lat/lon around Wattoo farms area)
    bounds = (73.67, 30.67, 73.68, 30.68)
    transform = from_bounds(*bounds, width, height)

    # Create synthetic data for each band
    # Typical satellite imagery values range from 0-3000
    data = np.zeros((bands, height, width), dtype=np.uint16)

    for band_idx in range(bands):
        # Create a gradient pattern with some variation per band
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)

        # Different pattern for each band
        data[band_idx] = ((xx + yy) * 1000 + band_idx * 200 + 500).astype(np.uint16)

    # Write to temporary file
    raster_path = tmp_path / "test_raster.tif"

    with rasterio.open(
        raster_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=np.uint16,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        for band_idx in range(bands):
            dst.write(data[band_idx], band_idx + 1)

    return str(raster_path)


@pytest.fixture
def test_raster_epsg3857_path(tmp_path):
    """Create a small test raster in EPSG:3857 (Web Mercator)."""
    width, height = 50, 50
    bands = 8

    # Bounds in EPSG:3857 (meters)
    bounds = (8200000, 3550000, 8210000, 3560000)
    transform = from_bounds(*bounds, width, height)

    data = np.zeros((bands, height, width), dtype=np.uint16)

    for band_idx in range(bands):
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)
        data[band_idx] = ((xx + yy) * 1000 + band_idx * 200 + 500).astype(np.uint16)

    raster_path = tmp_path / "test_raster_3857.tif"

    with rasterio.open(
        raster_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=np.uint16,
        crs='EPSG:3857',
        transform=transform,
    ) as dst:
        for band_idx in range(bands):
            dst.write(data[band_idx], band_idx + 1)

    return str(raster_path)


@pytest.fixture
def test_polygon():
    """Create a test polygon in EPSG:4326."""
    from shapely.geometry import box
    # Small polygon within test raster bounds
    return box(73.671, 30.671, 73.675, 30.675)


@pytest.fixture
def test_polygon_epsg3857():
    """Create a test polygon in EPSG:3857."""
    from shapely.geometry import box
    return box(8201000, 3551000, 8205000, 3555000)
