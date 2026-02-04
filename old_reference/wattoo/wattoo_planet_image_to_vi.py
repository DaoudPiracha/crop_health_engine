
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from engine.pipeline_config import PipelineConfig
from engine.compute.stats import build_field_veg_index_stats, write_field_veg_index_stats
from engine.io.assets import collect_image_files, load_boundaries
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


def main(cfg: PipelineConfig) -> None:
    if cfg.only_visual and cfg.write_to_file:
        raise ValueError("only_visual=True but write_to_file=True; refusing to write empty logs.")

    _boundaries_file, gdf_boundaries = load_boundaries(cfg.boundaries_file, reset_names=cfg.reset_names)
    gdf_boundaries = gdf_boundaries.to_crs("epsg:4326")
    img_files = collect_image_files(cfg.file_dir)

    lat_min, lat_max, lon_min, lon_max = cfg.bbox_latlon
    bounding_box = box(lon_min, lat_min, lon_max, lat_max)

    ndvi_log = build_field_veg_index_stats(img_files, gdf_boundaries, cfg.target_crs, cfg.only_visual,
                                           log_columns=LOG_COLUMNS)
    out_path = write_field_veg_index_stats(ndvi_log, cfg.season, cfg.crop_id, cfg.write_to_file)
    print(f"Wrote log to: {out_path}")

    show_images_at_each_ts = False
    if show_images_at_each_ts:
        preview_images(
            img_files=img_files,
            gdf_overlapping=gdf_boundaries,
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
