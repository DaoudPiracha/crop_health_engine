
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

from engine.constants import (RED,
                              GREEN,
                              BLUE,
                              NIR,
                              RED_EDGE,
                              GREEN_I)

from engine.vegetation_indices import (
    calculate_cire_mask,
    calculate_msavi_mask,
    calculate_mcari_mask,
    calculate_ndvi_mask,
    create_evi,
)


def load_raster_with_affine(image_path):
    """Load raster data and affine transform."""
    with rasterio.open(image_path) as src:
        image_data = src.read([RED, GREEN, BLUE])  # Red, Green, Blue bands (adjust as needed)
        affine_transform = src.transform
        crs = src.crs
        bounds = src.bounds
        width, height = src.width, src.height
    return image_data, affine_transform, crs, (bounds, width, height)

def reproject_raster_to_match_gdf_crs(raster_path, target_crs):
    """
    Reproject a raster to match the CRS of a given target CRS (usually from a GeoDataFrame).

    Parameters:
        raster_path (str): Path to the input raster file.
        target_crs (str or rasterio CRS): CRS of the target GeoDataFrame, either as a string
            (e.g., "EPSG:4326") or a rasterio CRS object.

    Returns:
        rasterio.io.DatasetReader: An open dataset of the reprojected raster.

    NOTE:
        This function returns `src` directly when CRS matches. That dataset handle will be closed
        when exiting the `with rasterio.open(...)` context. We will fix the lifetime/return-type
        consistency in a later refactor pass (Pass C). For Pass A, behavior is unchanged.
    """
    with rasterio.open(raster_path) as src:
        # Check if the raster already matches the target CRS
        if src.crs == target_crs:
            print("The raster already matches the target CRS.")
            return src

        # Calculate the transform and dimensions for the new projection
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        # Update profile for the new raster
        profile = src.profile
        profile.update(
            {
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        # Create an in-memory raster with the reprojected data
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


def plot_composite(image_data, composite_data, title, cmap="viridis"):
    """Plot a composite image or index."""
    plt.figure(figsize=(10, 10))
    plt.imshow(composite_data, cmap=cmap)
    plt.colorbar(label=title)
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_rgb_with_kml(raster_image, affine_transform, kml_gdf, ax):
    """Plot the RGB image with KML polygons overlayed, using affine transform."""
    rgb_image = raster_image.transpose(1, 2, 0)  # Convert to (height, width, 3)

    # NOTE: original code had ax.imshow commented out; keep behavior the same.
    # ax.imshow(rgb_image)

    ax.set_title("RGB with Overlaid Polygons")
    ax.axis("off")

    for idx, row in kml_gdf.iterrows():
        geom = row["geometry"]
        if isinstance(geom, Polygon) and geom.is_valid:
            ax.plot(*geom.exterior.xy, color="red", linewidth=2)
        else:
            print(f"error in {idx}")


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
    # gdf_overlapping[gdf_overlapping['id'] == id_subset].boundary.plot(ax=ax, edgecolor="yellow", linewidth=5)
    gdf_overlapping.boundary.plot(ax=ax, color=gdf_overlapping["color"], linewidth=2)

    plt.title(f"{img_date}")
    plt.show()

# misc. utils

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
    "date","name",
    "ndvi_mean","ndvi_std",
    "ndre_mean","ndre_std",
    "evi_mean","evi_std",
    "cire_mean","cire_std",
    "mcari_mean","mcari_std",
    "msavi_mean","msavi_std",
]

def build_ndvi_log(
    img_file_paths: list[str],
    gdf_overlapping: gpd.GeoDataFrame,
    target_crs: str,
    only_visual: bool,
) -> pd.DataFrame:
    rows: list[dict] = []

    # Important: project once, not per image
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



if __name__ == "__main__":
    # file_dir = "/Users/daoud/PycharmAssets/wattoo_farms/"
    asset_dir = "/Users/daoud/PycharmAssets/wattoo_farms"
    season = "kharif"
    crop_id = "wattoo"
    file_dir = f"{asset_dir}/*/PSScene/"
    color_clusters = False
    color_z_scores = True
    show_z_ts_plots = False

    ultra_high_ids = []

    unwanted_ids = ["301", "302", "304", "153", "176", "170", "175", "172"]  # unwanted_ids[crop_id]

    unwanted_ids_int = [int(id) for id in unwanted_ids]

    cluster_file = '/Users/daoud/PycharmProjects/DISA/ai4h_disa/experimental/sufi/cluster_cire.csv'
    z_score_file = '/Users/daoud/PycharmProjects/DISA/ai4h_disa/experimental/sufi/cire_z_scores_cluster_8.csv'
    z_score_ts_file = "/Users/daoud/PycharmProjects/DISA/ai4h_disa/experimental/wattoo_cire_z_scores_ts.csv"
    if os.path.exists(z_score_ts_file):
        z_ts_df = pd.read_csv(z_score_ts_file)
        z_ts_df = z_ts_df.drop(columns=unwanted_ids)
        # z_ts_df = z_ts_df.drop(columns=['830', '515', '509', '554'])
    else:
        z_ts_df = pd.DataFrame()

    cumulative_avg_df = pd.DataFrame(index=z_ts_df.index)
    for column in z_ts_df.columns[1:]:
        cumulative_avg_df[column] = z_ts_df[column].cumsum() / z_ts_df[column].notnull().cumsum()

    boundaries_file = f"{asset_dir}/wattoo_farms.geojson"
    print(boundaries_file)
    gdf_boundaries = gpd.read_file(boundaries_file)

    reset_names = False
    if reset_names:
        gdf_boundaries["Name"] = gdf_boundaries.index
        gdf_boundaries.to_file(boundaries_file)

    gdf_boundaries.boundary.plot(edgecolor="red")

    z_score_file_n = glob.glob("/Users/daoud/PycharmProjects/DISA/ai4h_disa/experimental/wattoo/wattoo_cire_z_scores_norm.csv")
    z_score_df_n = []

    for z_score_file in z_score_file_n:
        z_score_df_n.append(pd.read_csv(z_score_file))

    if z_score_df_n:
        z_score_df = pd.concat(z_score_df_n)
        z_score_df = z_score_df[:94]
        z_score_df["Unnamed: 0"] = z_score_df["Unnamed: 0"].astype(np.int64)

    target_crs = "epsg:3857"
    only_visual = False  # set false if you want to calculate scores
    write_to_file = True

    if only_visual and write_to_file:
        raise ValueError

    if show_z_ts_plots:
        for time in z_ts_df.index:  # Replace with your desired timestamp
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
            # plt.savefig(f'{asset_dir}/gifs/frames/cumulative_avg/z_score_ts_{time}.png')
            plt.show()

    img_files_skywatch = glob.glob(file_dir + "SKY*.tif")
    img_files_planet = glob.glob(file_dir + "*SR_8b_clip.tif")
    img_files = img_files_skywatch + img_files_planet

    gdf = gdf_boundaries
    gdf = gdf.to_crs("epsg:4326")

    lat_min, lat_max = 30.6655, 30.676
    lon_min, lon_max = 73.675377, 73.6815427

    bounding_box = box(lon_min, lat_min, lon_max, lat_max)
    gdf_overlapping = gdf  # gdf[gdf.intersects(bounding_box)]
    gdf = gdf_overlapping
    gdf.boundary.plot()

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

    if color_z_scores:
        z_scores = z_score_df

        z_scores = z_scores.rename(columns={"Unnamed: 0": "id", "0": "z_score"})
        z_scores = z_scores[~z_scores["id"].isin(unwanted_ids_int)]

        # todo: remove hardcoded threshold used for cleaner plots below
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

    img_files.sort(key=lambda file: get_date(file))
    print(f"analysing {len(img_files)} images")

    for img_file in img_files:
        img_date = get_date(img_file)
        print(">>>", img_date)

        img_file_raster = rasterio.open(img_file)
        print(f"preproj raster {img_file_raster.crs}")
        print(f"preproj gdf {gdf.crs}")

        if img_file_raster.crs != target_crs:
            reprojected_raster = reproject_raster_to_match_gdf_crs(img_file, target_crs)
            img_file_raster = reprojected_raster

        print(f"re-proj raster {img_file_raster.crs}")
        print(f"re-proj gdf {gdf.crs}")

        image_data = img_file_raster.read()
        raster_image, affine_transform, raster_crs, (raster_bounds, raster_width, raster_hieght) = load_raster_with_affine(img_file)

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

        if only_visual:
            continue

        gdf_overlapping = gdf_overlapping.to_crs(target_crs)

    ndvi_log = build_ndvi_log(img_files, gdf_overlapping, target_crs, only_visual)
    ndvi_log["date"] = pd.to_datetime(ndvi_log["date"], format="%Y%m%d")
    log_file_name = f"./{season}_{crop_id}_field_logs_new.csv"
    if write_to_file:
        ndvi_log.to_csv(log_file_name)

    assert False

    fig, ax = plt.subplots(figsize=(10, 10))
    for idx, row in kml_gdf.iterrows():
        geom = row["geometry"]
        if isinstance(geom, Polygon) and geom.is_valid:
            ax.plot(*geom.exterior.xy, color="red", linewidth=2)
        else:
            print(f"error in {idx}")

    rgb_composite = create_rgb_composite(image_data)
    ndvi_composite = create_ndvi(image_data)
    red_edge_composite = create_red_edge(image_data)
    evi_composite = create_evi(image_data)
    ndre_composite = create_ndre(image_data)

    show(clip_extremes(ndre_composite), transform=img_file_raster.transform, ax=ax)

    plt.title(img_date)
    plt.show()
