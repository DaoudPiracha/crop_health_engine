import os
from typing import Optional

import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from engine.compute.indices import compute_indices_stats
from engine.pipeline_config import PipelineConfig

from engine.io.raster import (
    open_raster_in_crs,
    crop_raster_with_polygon,
)

from engine.io.assets import (
    get_date,
    collect_image_files,
    load_boundaries,
)

from engine.viz.preview import preview_images


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

                    stats = compute_indices_stats(cropped_image)

                    rows.append(
                        {
                            "date": img_date,
                            "name": row["Name"],
                            **stats,
                        }
                    )

                    print(
                        f"Polygon {idx} (Date: {img_date}): NDVI Mean = {stats['ndvi_mean']:.4f}, "
                        f"NDVI Std Dev = {stats['ndvi_std']:.4f}"
                    )
                except Exception as e:
                    print(f"Error in {img_date}/{idx}: {e}")

    return pd.DataFrame.from_records(rows, columns=LOG_COLUMNS)


def prepare_overlapping_gdf_and_bbox(
    gdf_boundaries: gpd.GeoDataFrame,
    bbox_latlon: tuple[float, float, float, float],
) -> tuple[gpd.GeoDataFrame, object]:
    """
    bbox_latlon: (lat_min, lat_max, lon_min, lon_max)
    """
    gdf = gdf_boundaries.to_crs("epsg:4326")
    lat_min, lat_max, lon_min, lon_max = bbox_latlon
    bounding_box = box(lon_min, lat_min, lon_max, lat_max)

    # Keeping your behavior: no filtering currently applied
    gdf_overlapping = gdf
    gdf_overlapping.boundary.plot()
    return gdf_overlapping, bounding_box


def write_ndvi_log(ndvi_log: pd.DataFrame, season: str, crop_id: str, write_to_file: bool) -> str:
    ndvi_log["date"] = pd.to_datetime(ndvi_log["date"], format="%Y%m%d")
    log_file_name = f"./{season}_{crop_id}_field_logs_new_1.csv"
    if write_to_file:
        ndvi_log.to_csv(log_file_name, index=False)
    return log_file_name


def main(cfg: PipelineConfig) -> None:
    if cfg.only_visual and cfg.write_to_file:
        raise ValueError("only_visual=True but write_to_file=True; refusing to write empty logs.")

    _boundaries_file, gdf_boundaries = load_boundaries(cfg.boundaries_file, reset_names=cfg.reset_names)
    img_files = collect_image_files(cfg.file_dir)
    gdf_overlapping, bounding_box = prepare_overlapping_gdf_and_bbox(gdf_boundaries, cfg.bbox_latlon)

    ndvi_log = build_ndvi_log(img_files, gdf_overlapping, cfg.target_crs, cfg.only_visual)
    out_path = write_ndvi_log(ndvi_log, cfg.season, cfg.crop_id, cfg.write_to_file)
    print(f"Wrote log to: {out_path}")

    show_images_at_each_ts = False
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
