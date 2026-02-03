import os
import glob
from typing import Optional, Iterator
from contextlib import contextmanager

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling

from shapely.geometry import box

from engine.constants import (
    RED,
    GREEN,
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

from engine.pipeline_config import PipelineConfig


def load_raster_with_affine(image_path):
    """Load raster data and affine transform."""
    with rasterio.open(image_path) as src:
        image_data = src.read([RED, GREEN, BLUE])  # Red, Green, Blue bands
        affine_transform = src.transform
        crs = src.crs
        bounds = src.bounds
        width, height = src.width, src.height
    return image_data, affine_transform, crs, (bounds, width, height)


def reproject_raster_to_match_gdf_crs(raster_path, target_crs):
    """
    Reproject a raster to match the CRS of a given target CRS.

    NOTE:
        This returns `src` directly when CRS matches (handle will be closed after with-block).
        We are keeping this function for now to minimize diffs, but it is no longer used.
    """
    with rasterio.open(raster_path) as src:
        if src.crs == target_crs:
            print("The raster already matches the target CRS.")
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


def visualize_raster_and_gdf(
    raster_image,
    affine_transform,
    gdf,
    raster_crs,
    img_date,
    id_subset=4277,
    bounding_box=None,
):
    fig, ax = plt.subplots(figsize=(8, 8))
    show(raster_image / 2000, ax=ax, transform=affine_transform, with_bounds=bounding_box)

    gdf_overlapping = gdf.to_crs(raster_crs)
    gdf_overlapping.boundary.plot(ax=ax, color=gdf_overlapping["color"], linewidth=2)

    plt.title(f"{img_date}")
    plt.show()


def calculate_band_stats(band_values):
    mean = np.nanmean(band_values)
    std = np.nanstd(band_values)
    return mean, std


def get_date(filename):
    file_name = filename.split("/")[-1]
    if "SKY" in file_name:
        return file_name.split("_")[3][:8]
    return file_name[:8]


LOG_COLUMNS = [
    "date", "name",
    "ndvi_mean", "ndvi_std",
    "ndre_mean", "ndre_std",
    "evi_mean", "evi_std",
    "cire_mean", "cire_std",
    "mcari_mean", "mcari_std",
    "msavi_mean", "msavi_std",
]


def build_ndvi_log(
    img_file_paths: list[str],
    gdf_overlapping: gpd.GeoDataFrame,
    target_crs: str,
    only_visual: bool,
) -> pd.DataFrame:
    rows: list[dict] = []

    gdf_proj = gdf_overlapping.to_crs(target_crs)

    for img_file in img_file_paths:
        img_date = get_date(img_file)
        print(">>>", img_date)

        if only_visual:
            continue

        with open_raster_in_crs(img_file, target_crs) as img_file_raster:
            for idx, row in gdf_proj.iterrows():
                geom = row["geometry"]
                print(row["Name"])
                try:
                    cropped_image, _ = crop_raster_with_polygon(img_file_raster, geom)

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

                    rows.append(
                        {
                            "date": img_date,
                            "name": row["Name"],
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
                    )

                    print(
                        f"Polygon {idx} (Date: {img_date}): NDVI Mean = {ndvi_mean:.4f}, NDVI Std Dev = {ndvi_std:.4f}"
                    )
                except Exception as e:
                    print(f"Error in {img_date}/{idx}: {e}")

    return pd.DataFrame.from_records(rows, columns=LOG_COLUMNS)


def load_boundaries(boundaries_file: str, reset_names: bool) -> tuple[str, gpd.GeoDataFrame]:
    print(boundaries_file)
    gdf_boundaries = gpd.read_file(boundaries_file)

    if reset_names:
        gdf_boundaries["Name"] = gdf_boundaries.index
        gdf_boundaries.to_file(boundaries_file)

    gdf_boundaries.boundary.plot(edgecolor="red")
    return boundaries_file, gdf_boundaries


def collect_image_files(file_dir: str) -> list[str]:
    img_files_skywatch = glob.glob(file_dir + "SKY*.tif")
    img_files_planet = glob.glob(file_dir + "*SR_8b_clip.tif")
    img_files = img_files_skywatch + img_files_planet
    img_files.sort(key=get_date)
    return img_files


def prepare_overlapping_gdf_and_bbox(
    gdf_boundaries: gpd.GeoDataFrame,
    bbox_latlon: tuple[float, float, float, float],
) -> tuple[gpd.GeoDataFrame, object]:
    gdf = gdf_boundaries.to_crs("epsg:4326")
    lat_min, lat_max, lon_min, lon_max = bbox_latlon
    bounding_box = box(lon_min, lat_min, lon_max, lat_max)

    gdf_overlapping = gdf  # keep behavior the same (filter disabled)
    gdf_overlapping.boundary.plot()
    return gdf_overlapping, bounding_box


def preview_images(
    img_files: list[str],
    gdf_overlapping: gpd.GeoDataFrame,
    bounding_box,
    target_crs: str,
):
    print(f"analysing {len(img_files)} images")

    for img_file in img_files:
        img_date = get_date(img_file)
        print(">>>", img_date)

        with open_raster_in_crs(img_file, target_crs) as img_file_raster:
            print(f"re-proj raster {img_file_raster.crs}")
            print(f"re-proj gdf {gdf_overlapping.crs}")

            _ = img_file_raster.read()
            raster_image, affine_transform, raster_crs, _meta = load_raster_with_affine(img_file)

            id_subset = 3826
            visualize_raster_and_gdf(
                raster_image,
                affine_transform,
                gdf_overlapping,
                raster_crs,
                img_date,
                id_subset,
                bounding_box=bounding_box,
            )


def write_ndvi_log(ndvi_log: pd.DataFrame, season: str, crop_id: str, write_to_file: bool) -> str:
    ndvi_log["date"] = pd.to_datetime(ndvi_log["date"], format="%Y%m%d")
    log_file_name = f"./{season}_{crop_id}_field_logs_new_1.csv"
    if write_to_file:
        ndvi_log.to_csv(log_file_name)
    return log_file_name


def main(cfg: PipelineConfig) -> None:
    if cfg.only_visual and cfg.write_to_file:
        raise ValueError("only_visual=True but write_to_file=True; refusing to write empty logs.")

    boundaries_file, gdf_boundaries = load_boundaries(cfg.boundaries_file, reset_names=cfg.reset_names)
    img_files = collect_image_files(cfg.file_dir)
    gdf_overlapping, bounding_box = prepare_overlapping_gdf_and_bbox(gdf_boundaries, cfg.bbox_latlon)

    ndvi_log = build_ndvi_log(img_files, gdf_overlapping, cfg.target_crs, cfg.only_visual)
    out_path = write_ndvi_log(ndvi_log, cfg.season, cfg.crop_id, cfg.write_to_file)
    print(f"Wrote log to: {out_path}")

    show_images_at_each_ts = True
    if show_images_at_each_ts:
        preview_images(
            img_files=img_files,
            gdf_overlapping=gdf_overlapping,
            bounding_box=bounding_box,
            target_crs=cfg.target_crs,
        )


if __name__ == "__main__":
    asset_dir = "/Users/daoud/PycharmAssets/wattoo_farms"

    cfg = PipelineConfig(
        asset_dir=asset_dir,
        season="kharif",
        crop_id="wattoo",

        file_dir=f"{asset_dir}/*/PSScene/",
        boundaries_file=f"{asset_dir}/wattoo_farms.geojson",

        cluster_file="/Users/daoud/PycharmProjects/DISA/ai4h_disa/experimental/sufi/cluster_cire.csv",
        z_score_ts_file="/Users/daoud/PycharmProjects/DISA/ai4h_disa/experimental/wattoo_cire_z_scores_ts.csv",
        z_score_glob="/Users/daoud/PycharmProjects/DISA/ai4h_disa/experimental/wattoo/wattoo_cire_z_scores_norm.csv",

        target_crs="epsg:3857",

        only_visual=False,
        write_to_file=True,

        color_clusters=False,
        color_z_scores=True,
        show_z_ts_plots=False,
        reset_names=False,

        unwanted_ids=["301", "302", "304", "153", "176", "170", "175", "172"],

        bbox_latlon=(30.6655, 30.676, 73.675377, 73.6815427),
    )

    main(cfg)
