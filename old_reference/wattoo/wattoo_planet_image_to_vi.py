import os.path
import glob

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling

from shapely.geometry import box, Polygon

import matplotlib.colors as mcolors
from matplotlib import cm

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


# -------------------------
# Existing helpers (unchanged)
# -------------------------

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
        We'll fix lifetime/return-type consistency in a later pass.
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

        img_file_raster = rasterio.open(img_file)
        if img_file_raster.crs != target_crs:
            img_file_raster = reproject_raster_to_match_gdf_crs(img_file, target_crs)

        if only_visual:
            continue

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


# -------------------------
# New orchestration helpers (extracted from __main__)
# -------------------------

def load_z_ts_and_cumulative(z_score_ts_file: str, unwanted_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if os.path.exists(z_score_ts_file):
        z_ts_df = pd.read_csv(z_score_ts_file)
        z_ts_df = z_ts_df.drop(columns=unwanted_ids)
    else:
        z_ts_df = pd.DataFrame()

    cumulative_avg_df = pd.DataFrame(index=z_ts_df.index)
    for column in z_ts_df.columns[1:]:
        cumulative_avg_df[column] = z_ts_df[column].cumsum() / z_ts_df[column].notnull().cumsum()

    return z_ts_df, cumulative_avg_df


def load_boundaries(asset_dir: str, reset_names: bool) -> tuple[str, gpd.GeoDataFrame]:
    boundaries_file = f"{asset_dir}/wattoo_farms.geojson"
    print(boundaries_file)
    gdf_boundaries = gpd.read_file(boundaries_file)

    if reset_names:
        gdf_boundaries["Name"] = gdf_boundaries.index
        gdf_boundaries.to_file(boundaries_file)

    gdf_boundaries.boundary.plot(edgecolor="red")
    return boundaries_file, gdf_boundaries


def load_z_score_df(z_score_glob: str) -> pd.DataFrame:
    z_score_file_n = glob.glob(z_score_glob)
    z_score_df_n = [pd.read_csv(p) for p in z_score_file_n]

    if not z_score_df_n:
        return None

    z_score_df = pd.concat(z_score_df_n)
    z_score_df = z_score_df[:94]
    z_score_df["Unnamed: 0"] = z_score_df["Unnamed: 0"].astype(np.int64)
    return z_score_df


def plot_z_score_ts(
    z_ts_df: pd.DataFrame,
    cumulative_avg_df: pd.DataFrame,
    boundaries_file: str,
):

    for time in z_ts_df.index:
        z_ts_df = cumulative_avg_df
        time_data = z_ts_df.loc[time]

        gdf_boundaries = gpd.read_file(boundaries_file)
        gdf_boundaries = gdf_boundaries.copy()
        gdf_boundaries["z_score"] = gdf_boundaries["Name"].map(time_data)

        gdf_boundaries_with_z = gdf_boundaries

        norm = mcolors.Normalize(
            vmin=gdf_boundaries_with_z["z_score"].min(),
            vmax=gdf_boundaries_with_z["z_score"].max(),
        )
        cmap = cm.plasma
        gdf_boundaries_with_z["color"] = gdf_boundaries_with_z["z_score"].apply(
            lambda z: mcolors.to_hex(cmap(norm(z)))
        )

        gdf_boundaries_with_z["color"] = gdf_boundaries_with_z["color"].apply(
            lambda color: "#e3e3e3" if color == "#000000" else color
        )

        fig, ax = plt.subplots(figsize=(10, 10))
        plt.title(f"Health Score / Days since emergence: {time}")
        gdf_boundaries_with_z.plot(
            ax=ax,
            facecolor=gdf_boundaries_with_z["color"],
            edgecolor=gdf_boundaries_with_z["color"],
            linewidth=0.5,
        )
        plt.show()


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


def apply_cluster_color(
    gdf_overlapping: gpd.GeoDataFrame,
    color_clusters: bool,
    cluster_file: str,
) -> gpd.GeoDataFrame:
    if color_clusters:
        clusters = pd.read_csv(cluster_file)
        gdf_overlapping = pd.merge(gdf_overlapping, clusters, on="id")
        color_mapping = {
            0: "black",
            1: "blue",
            2: "black",
            3: "white",
            4: "black",
            5: "cyan",
            6: "black",
            7: "black",
            8: "red",
            9: "black",
        }
        gdf_overlapping["color"] = gdf_overlapping["cluster"].map(color_mapping)
    else:
        gdf_overlapping["color"] = "#808080"

    return gdf_overlapping


def apply_z_score_coloring_and_export(
    gdf_overlapping: gpd.GeoDataFrame,
    z_score_df: pd.DataFrame,
    unwanted_ids_int: list[int],
    color_z_scores: bool,
) -> gpd.GeoDataFrame:
    if not color_z_scores:
        return gdf_overlapping
    if z_score_df is None:
        # keep behavior: if missing, it would crash later anyway; here we just return unchanged
        return gdf_overlapping

    z_scores = z_score_df.rename(columns={"Unnamed: 0": "id", "0": "z_score"})
    z_scores = z_scores[~z_scores["id"].isin(unwanted_ids_int)]

    max_z_score = 3
    min_z_score = -1.5
    z_scores = z_scores[z_scores["z_score"] < max_z_score]
    z_scores = z_scores[z_scores["z_score"] > min_z_score]
    z_scores["z_score"] = z_scores["z_score"].clip(lower=-2, upper=2)

    gdf_overlapping["Name"] = gdf_overlapping["Name"].astype(np.int64)
    gdf_overlapping = pd.merge(gdf_overlapping, z_scores, left_on="Name", right_on="id", how="left")

    norm = mcolors.Normalize(vmin=gdf_overlapping["z_score"].min(), vmax=gdf_overlapping["z_score"].max())
    cmap = cm.RdYlGn
    gdf_overlapping["color"] = gdf_overlapping["z_score"].apply(lambda z: mcolors.to_hex(cmap(norm(z))))
    gdf_overlapping["color"] = gdf_overlapping["color"].apply(
        lambda color: "#e3e3e3" if color == "#000000" else color
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_overlapping.plot(
        ax=ax,
        facecolor=gdf_overlapping["color"],
        edgecolor=gdf_overlapping["color"],
        linewidth=0.5,
    )

    gdf_overlapping["predicted_yield"] = (18.81 * gdf_overlapping["z_score"] + 46.918)

    norm = mcolors.Normalize(
        vmin=gdf_overlapping["predicted_yield"].min(),
        vmax=gdf_overlapping["predicted_yield"].max(),
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(gdf_overlapping["predicted_yield"])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Predicted Yield (man/ac)")

    plt.title("Corn Fields colored by Health Score")
    plt.show()
    gdf_overlapping.to_file("cire_z_score.geojson", driver="GeoJSON")

    return gdf_overlapping


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

        img_file_raster = rasterio.open(img_file)
        print(f"preproj raster {img_file_raster.crs}")
        print(f"preproj gdf {gdf_overlapping.crs}")

        if img_file_raster.crs != target_crs:
            img_file_raster = reproject_raster_to_match_gdf_crs(img_file, target_crs)

        print(f"re-proj raster {img_file_raster.crs}")
        print(f"re-proj gdf {gdf_overlapping.crs}")

        # Keep behavior: you were reading the full raster and also reading RGB separately
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




def write_ndvi_log(ndvi_log: pd.DataFrame, season: str, crop_id: str, write_to_file: bool):
    ndvi_log["date"] = pd.to_datetime(ndvi_log["date"], format="%Y%m%d")
    log_file_name = f"./{season}_{crop_id}_field_logs_new.csv"
    if write_to_file:
        ndvi_log.to_csv(log_file_name)
    return log_file_name

def main():
    asset_dir = "/Users/daoud/PycharmAssets/wattoo_farms"
    season = "kharif"
    crop_id = "wattoo"
    file_dir = f"{asset_dir}/*/PSScene/"

    color_clusters = False
    color_z_scores = True
    show_z_ts_plots = False

    unwanted_ids = ["301", "302", "304", "153", "176", "170", "175", "172"]
    unwanted_ids_int = [int(x) for x in unwanted_ids]

    cluster_file = "/Users/daoud/PycharmProjects/DISA/ai4h_disa/experimental/sufi/cluster_cire.csv"
    z_score_ts_file = "/Users/daoud/PycharmProjects/DISA/ai4h_disa/experimental/wattoo_cire_z_scores_ts.csv"

    z_score_glob = "/Users/daoud/PycharmProjects/DISA/ai4h_disa/experimental/wattoo/wattoo_cire_z_scores_norm.csv"

    target_crs = "epsg:3857"
    only_visual = False
    write_to_file = True

    if only_visual and write_to_file:
        raise ValueError

    # 1) Load time series + cumulative (even if not used later, keep behavior)
    z_ts_df, cumulative_avg_df = load_z_ts_and_cumulative(z_score_ts_file, unwanted_ids)

    # 2) Load boundaries
    boundaries_file, gdf_boundaries = load_boundaries(asset_dir, reset_names=False)

    # 3) Load z-score df
    z_score_df = load_z_score_df(z_score_glob)

    # 4) Optional: show z-ts plots
    if show_z_ts_plots:
        plot_z_score_ts(z_ts_df, cumulative_avg_df, boundaries_file)

    # 5) Collect images
    img_files = collect_image_files(file_dir)

    # 6) Prepare gdf + bbox
    # Note: keeping your bbox values exactly
    gdf_overlapping, bounding_box = prepare_overlapping_gdf_and_bbox(
        gdf_boundaries,
        bbox_latlon=(30.6655, 30.676, 73.675377, 73.6815427),
    )

    # 7) Apply cluster/default coloring
    gdf_overlapping = apply_cluster_color(gdf_overlapping, color_clusters, cluster_file)

    # 8) Apply z-score coloring + export
    gdf_overlapping = apply_z_score_coloring_and_export(
        gdf_overlapping,
        z_score_df=z_score_df,
        unwanted_ids_int=unwanted_ids_int,
        color_z_scores=color_z_scores,
    )

    # 9) Preview images (your previous loop)
    preview_images(
        img_files=img_files,
        gdf_overlapping=gdf_overlapping,
        bounding_box=bounding_box,
        target_crs=target_crs,
    )

    # 10) Compute NDVI log
    ndvi_log = build_ndvi_log(img_files, gdf_overlapping, target_crs, only_visual)

    # 11) Write log
    out_path = write_ndvi_log(ndvi_log, season, crop_id, write_to_file)
    print(f"Wrote log to: {out_path}")


if __name__ == "__main__":
    main()
