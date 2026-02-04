import logging
from shapely.geometry import box

from engine.pipeline_config import PipelineConfig
from engine.compute.stats import build_field_veg_index_stats, write_field_veg_index_stats
from engine.io.assets import collect_image_files, load_boundaries
from engine.viz.preview import preview_images

logger = logging.getLogger(__name__)

def _build_assets(cfg:PipelineConfig):
    _boundaries_file, gdf_boundaries = load_boundaries(cfg.boundaries_file, reset_names=cfg.reset_names)
    gdf_boundaries = gdf_boundaries.to_crs("epsg:4326")
    img_files = collect_image_files(cfg.file_dir)

    lat_min, lat_max, lon_min, lon_max = cfg.bbox_latlon
    bounding_box = box(lon_min, lat_min, lon_max, lat_max)
    return img_files, gdf_boundaries, bounding_box

def run_pipeline(cfg: PipelineConfig) -> None:
    if cfg.only_visual and cfg.write_to_file:
        raise ValueError("only_visual=True but write_to_file=True; refusing to write empty logs.")

    img_files, gdf_boundaries, bounding_box = _build_assets(cfg)
    ndvi_log = build_field_veg_index_stats(img_files, gdf_boundaries, cfg.target_crs, cfg.only_visual)
    out_path = write_field_veg_index_stats(ndvi_log, cfg.season, cfg.crop_id, cfg.write_to_file)
    logger.info(f"Results written to: {out_path}")

    if cfg.show_images_at_each_ts:
        preview_images(
            img_files=img_files,
            gdf_overlapping=gdf_boundaries,
            bounding_box=bounding_box,
            target_crs=cfg.target_crs,
        )
