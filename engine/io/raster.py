# engine/io/raster.py

from __future__ import annotations

import logging
from typing import Iterator, Tuple
from contextlib import contextmanager

import rasterio

logger = logging.getLogger(__name__)
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

from engine.constants import RED, GREEN, BLUE


def load_raster_with_affine(image_path: str):
    """
    Load raster RGB data and affine transform.

    Returns:
        image_data: np.ndarray (3, H, W) for [RED, GREEN, BLUE]
        affine_transform: Affine
        crs: rasterio CRS
        meta: (bounds, width, height)
    """
    with rasterio.open(image_path) as src:
        image_data = src.read([RED, GREEN, BLUE])
        affine_transform = src.transform
        crs = src.crs
        bounds = src.bounds
        width, height = src.width, src.height
    return image_data, affine_transform, crs, (bounds, width, height)


def reproject_raster_to_match_gdf_crs(raster_path: str, target_crs: str):
    """
    Reproject a raster to match a target CRS.

    NOTE:
        Kept for backward compatibility / minimizing diffs.
        Returns a dataset handle, but beware lifetime if you return src from inside `with`.
        Prefer `open_raster_in_crs` for safe lifetime management.
    """
    with rasterio.open(raster_path) as src:
        if src.crs == target_crs:
            logger.info("Raster already matches target CRS")
            return src

        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        profile = src.profile
        profile.update(
            {
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        with MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                for i in range(1, src.count + 1):
                    src_array = src.read(i)
                    reproject(
                        source=src_array,
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest,
                    )
            return memfile.open()


@contextmanager
def open_raster_in_crs(raster_path: str, target_crs: str) -> Iterator[rasterio.io.DatasetReader]:
    """
    Always yields an OPEN dataset in target_crs, and guarantees cleanup.

    - If CRS matches: yields rasterio.open(...) directly.
    - If CRS differs: reprojects into MemoryFile and yields that dataset.
    """
    src = rasterio.open(raster_path)
    try:
        if src.crs == target_crs:
            yield src
            return

        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        profile = src.profile.copy()
        profile.update(
            {
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        memfile = MemoryFile()
        try:
            with memfile.open(**profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=src.read(i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest,
                    )

            ds = memfile.open()
            try:
                yield ds
            finally:
                ds.close()
        finally:
            memfile.close()
    finally:
        src.close()


def crop_raster_with_polygon(raster, polygon):
    """Mask the image with the polygon (clipping)."""
    out_image, out_transform = mask(raster, [polygon], crop=True)
    return out_image, out_transform
