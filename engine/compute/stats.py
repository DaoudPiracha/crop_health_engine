import logging
import geopandas as gpd
import pandas as pd

from engine.compute.indices import compute_indices_stats
from engine.io.assets import get_date
from engine.io.raster import open_raster_in_crs, crop_raster_with_polygon

logger = logging.getLogger(__name__)


# todo: move the crop function out of this
LOG_COLUMNS = [
    "date", "name",
    "ndvi_mean", "ndvi_std",
    "ndre_mean", "ndre_std",
    "evi_mean", "evi_std",
    "cire_mean", "cire_std",
    "mcari_mean", "mcari_std",
    "msavi_mean", "msavi_std",
]

def build_field_veg_index_stats(
    img_file_paths: list[str],
    gdf_overlapping: gpd.GeoDataFrame,
    target_crs: str,
    only_visual: bool,
    log_columns: list[str] = LOG_COLUMNS,
) -> pd.DataFrame:
    """
    Build a per-(date, field/polygon) vegetation index statistics table.
    """

    if only_visual:
        return pd.DataFrame(columns=log_columns)

    rows: list[dict] = []
    gdf_proj = gdf_overlapping.to_crs(target_crs)

    for img_file in img_file_paths:
        img_date = get_date(img_file)
        logger.info(f"Processing image date: {img_date}")

        with open_raster_in_crs(img_file, target_crs) as ds:
            for idx, row in gdf_proj.iterrows():

                geom = row["geometry"]
                name = row.get("Name", idx)

                try:
                    cropped_image, _ = crop_raster_with_polygon(ds, geom)
                    stats = compute_indices_stats(cropped_image)

                    rows.append({
                        "date": img_date,
                        "name": name,
                        **stats,
                    })

                except Exception as e:
                    logger.error(f"Error processing field {name} on {img_date}: {e}", exc_info=True)

    return pd.DataFrame.from_records(rows, columns=log_columns)

def write_field_veg_index_stats(
    df: pd.DataFrame,
    season: str,
    crop_id: str,
    write_to_file: bool,
    out_dir: str = ".",
) -> str:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    file_name = f"{out_dir}/{season}_{crop_id}_field_veg_index_stats.csv"

    if write_to_file:
        df.to_csv(file_name, index=False)

    return file_name