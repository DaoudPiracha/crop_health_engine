import logging
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.plot import show

from engine.io.assets import get_date
from engine.io.raster import open_raster_in_crs, load_raster_with_affine

logger = logging.getLogger(__name__)


def visualize_raster_and_gdf(
    raster_image,
    affine_transform,
    gdf: gpd.GeoDataFrame,
    raster_crs,
    img_date: str,
    id_subset: int = 4277,
    bounding_box=None,
):
    fig, ax = plt.subplots(figsize=(8, 8))
    show(raster_image / 2000, ax=ax, transform=affine_transform, with_bounds=bounding_box)

    gdf_overlapping = gdf.to_crs(raster_crs)
    gdf_overlapping.boundary.plot(ax=ax, color=gdf_overlapping.get("color", "#808080"), linewidth=2)

    plt.title(f"{img_date}")
    plt.show()


def preview_images(
    img_files: list[str],
    gdf_overlapping: gpd.GeoDataFrame,
    bounding_box,
    target_crs: str,
):
    logger.info(f"Analyzing {len(img_files)} images")

    for img_file in img_files:
        img_date = get_date(img_file)
        logger.info(f"Processing image date: {img_date}")

        with open_raster_in_crs(img_file, target_crs) as img_file_raster:
            logger.debug(f"Reprojected raster CRS: {img_file_raster.crs}")
            logger.debug(f"Reprojected GDF CRS: {gdf_overlapping.crs}")

            _ = img_file_raster.read()
            raster_image, affine_transform, raster_crs, _meta = load_raster_with_affine(img_file)

            visualize_raster_and_gdf(
                raster_image=raster_image,
                affine_transform=affine_transform,
                gdf=gdf_overlapping,
                raster_crs=raster_crs,
                img_date=img_date,
                id_subset=3826,
                bounding_box=bounding_box,
            )
