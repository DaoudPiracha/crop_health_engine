# engine/io/assets.py

import glob
import geopandas as gpd


def get_date(filename: str) -> str:
    file_name = filename.split("/")[-1]
    if "SKY" in file_name:
        return file_name.split("_")[3][:8]
    return file_name[:8]


def collect_image_files(file_dir: str) -> list[str]:
    img_files_skywatch = glob.glob(file_dir + "SKY*.tif")
    img_files_planet = glob.glob(file_dir + "*SR_8b_clip.tif")
    img_files = img_files_skywatch + img_files_planet
    img_files.sort(key=get_date)
    return img_files


def load_boundaries(boundaries_file: str, reset_names: bool) -> tuple[str, gpd.GeoDataFrame]:
    print(boundaries_file)
    gdf_boundaries = gpd.read_file(boundaries_file)

    if reset_names:
        gdf_boundaries["Name"] = gdf_boundaries.index
        gdf_boundaries.to_file(boundaries_file)

    # Keeping your side effect
    gdf_boundaries.boundary.plot(edgecolor="red")
    return boundaries_file, gdf_boundaries
