# engine/compute/indices.py

from __future__ import annotations

import numpy as np

from engine.constants import (
    RED,
    BLUE,
    NIR,
    RED_EDGE,
    GREEN_I,
)

from engine.vegetation_indices import (
    calculate_cire_mask,
    calculate_msavi_mask,
    calculate_mcari_mask,
    calculate_ndvi_mask,
    create_evi,
)


LOG_COLUMNS = [
    "date", "name",
    "ndvi_mean", "ndvi_std",
    "ndre_mean", "ndre_std",
    "evi_mean", "evi_std",
    "cire_mean", "cire_std",
    "mcari_mean", "mcari_std",
    "msavi_mean", "msavi_std",
]


def calculate_band_stats(band_values: np.ndarray) -> tuple[float, float]:
    mean = np.nanmean(band_values)
    std = np.nanstd(band_values)
    return float(mean), float(std)


def compute_indices_stats(cropped_image: np.ndarray) -> dict[str, float]:
    """
    Pure compute:
    - Input: cropped_image array as returned by rasterio.mask.mask (bands-first).
    - Output: dict with means/stds for each index.
    """
    red = cropped_image[RED - 1]
    nir = cropped_image[NIR - 1]
    blue = cropped_image[BLUE - 1]
    red_edge = cropped_image[RED_EDGE - 1]
    green_1 = cropped_image[GREEN_I - 1]

    ndvi = calculate_ndvi_mask(red, nir)
    ndre = calculate_ndvi_mask(red_edge, nir)
    evi = create_evi(cropped_image)
    cire = calculate_cire_mask(red_edge, nir)
    mcari = calculate_mcari_mask(red_band=red, blue_band=blue, green_1_band=green_1)
    msavi = calculate_msavi_mask(red_band=red, nir_band=nir)

    ndvi_mean, ndvi_std = calculate_band_stats(ndvi)
    ndre_mean, ndre_std = calculate_band_stats(ndre)
    evi_mean, evi_std = calculate_band_stats(evi)
    cire_mean, cire_std = calculate_band_stats(cire)
    mcari_mean, mcari_std = calculate_band_stats(mcari)
    msavi_mean, msavi_std = calculate_band_stats(msavi)

    return {
        "ndvi_mean": ndvi_mean,
        "ndvi_std": ndvi_std,
        "ndre_mean": ndre_mean,
        "ndre_std": ndre_std,
        "evi_mean": evi_mean,
        "evi_std": evi_std,
        "cire_mean": cire_mean,
        "cire_std": cire_std,
        "mcari_mean": mcari_mean,
        "mcari_std": mcari_std,
        "msavi_mean": msavi_mean,
        "msavi_std": msavi_std,
    }
