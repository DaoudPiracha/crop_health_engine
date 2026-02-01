
import numpy as np
from engine.constants import *

def normalize_band(band):
    """Normalize the band to range [0, 1] for visualization."""
    return (band - band.min()) / (band.max() - band.min())


def clip_extremes(band_data, lower_percentile=1, upper_percentile=99):
    """
    Clip extreme values at the specified lower and upper percentiles.

    Args:
        band_data: array
        lower_percentile: float
        upper_percentile: float

    Returns:
        clipped array
    """
    p1, p99 = np.percentile(band_data, [lower_percentile, upper_percentile])
    return np.clip(band_data, p1, p99)


def create_rgb_composite(image_data):
    """Create an RGB composite using the Red, Green, and Blue bands."""
    red_band = image_data[RED]
    green_band = image_data[GREEN]
    blue_band = image_data[BLUE]

    red_norm = normalize_band(red_band)
    green_norm = normalize_band(green_band)
    blue_norm = normalize_band(blue_band)

    red_norm = np.power(red_norm, 0.8)
    green_norm = np.power(green_norm, 0.8)
    blue_norm = np.power(blue_norm, 0.8)

    return np.dstack((red_norm, green_norm, blue_norm))


def create_ndvi(image_data, clip=True):
    """Create NDVI using the Near-Infrared and Red bands."""
    red_band = image_data[RED]
    nir_band = image_data[NIR_1]

    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi = np.nan_to_num(ndvi, nan=0.0)
    return ndvi


def create_red_edge(image_data):
    """Create the Red Edge composite."""
    red_edge_band = image_data[RED_EDGE]
    return normalize_band(red_edge_band)


def create_evi(image_data):
    """Create EVI using the Near-Infrared, Red, and Blue bands."""
    red_band = image_data[RED]
    nir_band = image_data[NIR_2]
    blue_band = image_data[BLUE]

    return 2.5 * ((nir_band - red_band) / (nir_band + 6 * red_band - 7.5 * blue_band + 1))


def create_ndre(image_data):
    """Create NDRE using the Red Edge and Near-Infrared bands."""
    red_edge_band = image_data[RED_EDGE]
    nir_band = image_data[NIR_1]

    ndre = (nir_band - red_edge_band) / (nir_band + red_edge_band)
    ndre = np.nan_to_num(ndre, nan=0.0)
    return ndre


def calculate_ndvi_mask(red_band, nir_band):
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    return ndvi


def calculate_mcari_mask(red_band, blue_band, green_1_band):
    mcari = ((red_band - green_1_band) - 0.2 * (red_band - blue_band)) * (red_band / green_1_band)
    return mcari


def calculate_msavi_mask(red_band, nir_band):
    msavi = ((2 * (nir_band - red_band) + 1 - (2 * (nir_band - red_band + 1) ** 2 - 8 * (nir_band - red_band)) * 0.5)) / 2
    return msavi


def calculate_cire_mask(red_edge, nir_band):
    cire = (nir_band / red_edge) - 1
    return cire