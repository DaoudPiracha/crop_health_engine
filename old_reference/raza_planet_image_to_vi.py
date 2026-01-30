import os.path

import pandas as pd
import rasterio
import glob
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.colors as mcolors

from rasterio.plot import show
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
import pandas as pd
from rasterio.crs import CRS
from shapely.geometry import box
from shapely.affinity import rotate, translate
from rasterio.windows import from_bounds

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from rasterio.mask import mask



def reproject_raster_to_match_gdf_crs(raster_path, target_crs):
    """
    Reproject a raster to match the CRS of a given target CRS (usually from a GeoDataFrame).

    Parameters:
    raster_path (str): Path to the input raster file.
    target_crs (str or CRS): CRS of the target GeoDataFrame, either as a string (e.g., "EPSG:4326") or a rasterio CRS object.

    Returns:
    MemoryFile: A rasterio in-memory file of the reprojected raster.
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
        profile.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

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
                        resampling=Resampling.nearest
                    )
            return memfile.open()


# Constants for band indexing
# COASTAL_BLUE = 0
# BLUE = 1
# GREEN = 2
# YELLOW = 3
# RED = 4
# RED_EDGE = 5
NIR_1 = 6
NIR_2 = 7

COASTAL_BLUE = 1
BLUE = 2
GREEN_I = 3
GREEN = 4

YELLOW = 5
RED = 6
RED_EDGE = 7
NIR = 8

# NIR_1 = 8
# NIR_2 = 7

def load_geojson(geojson_path):
    """
    Load a KML file and return a GeoDataFrame.
    """
    gdf = gpd.read_file(geojson_path)
    return gdf

def load_kml(kml_path):
    """
    Load a KML file and return a GeoDataFrame.
    """
    gdf = gpd.read_file(kml_path, driver='kml')
    return gdf

# Function to normalize a band for display
def normalize_band(band):
    """Normalize the band to range [0, 1] for visualization."""
    # # Clip extreme values at the 1st and 99th percentiles
    # p1, p99 = np.percentile(band_data, [1, 99])
    # band_data_clipped = np.clip(band_data, p1, p99)

    return (band - band.min()) / (band.max() - band.min())

def clip_extremes(band_data, lower_percentile=1, upper_percentile=99):
    """
    Clip the extreme values at the specified lower and upper percentiles.

    :param band_data: The raw band data to be clipped.
    :param lower_percentile: The lower percentile for clipping (default 1%).
    :param upper_percentile: The upper percentile for clipping (default 99%).
    :return: The clipped band data.
    """
    # Calculate the percentiles
    p1, p99 = np.percentile(band_data, [lower_percentile, upper_percentile])

    # Clip the values
    clipped_data = np.clip(band_data, p1, p99)

    return clipped_data

# --- Composite Creation Functions ---

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

    rgb = np.dstack((red_norm, green_norm, blue_norm))

    return rgb


def create_ndvi(image_data, clip=True):
    """Create NDVI using the Near-Infrared and Red bands."""
    red_band = image_data[RED]
    nir_band = image_data[NIR_1]

    # Calculate NDVI
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi = np.nan_to_num(ndvi, nan=0.0)

    return ndvi


def create_red_edge(image_data):
    """Create the Red Edge composite."""
    red_edge_band = image_data[RED_EDGE]

    # Normalize for display
    return normalize_band(red_edge_band)


def create_evi(image_data):
    """Create EVI using the Near-Infrared, Red, and Blue bands."""
    red_band = image_data[RED]
    nir_band = image_data[NIR_2]
    blue_band = image_data[BLUE]

    # Calculate EVI
    return 2.5 * ((nir_band - red_band) / (nir_band + 6 * red_band - 7.5 * blue_band + 1))



def create_ndre(image_data):
    """Create NDRE using the Red Edge and Near-Infrared bands."""
    red_edge_band = image_data[RED_EDGE]
    nir_band = image_data[NIR_1]

    # Calculate NDRE
    ndre =(nir_band - red_edge_band) / (nir_band + red_edge_band)
    ndre = np.nan_to_num(ndre, nan=0.0)

    return ndre

# --- Plotting Functions ---

def plot_composite(image_data, composite_data, title, cmap='viridis'):
    """Plot a composite image or index."""
    plt.figure(figsize=(10, 10))
    plt.imshow(composite_data, cmap=cmap)
    plt.colorbar(label=title)
    plt.title(title)
    plt.axis('off')
    plt.show()


import matplotlib.pyplot as plt
from shapely.geometry import Polygon




def load_raster_with_affine(image_path):
    """Load raster data and affine transform."""
    with rasterio.open(image_path) as src:
        image_data = src.read([RED, GREEN, BLUE])  # Red, Green, Blue bands (adjust as needed)
        affine_transform = src.transform
        crs = src.crs
        bounds = src.bounds
        width, height = src.width, src.height

    return image_data, affine_transform, crs, (bounds, width, height)


def plot_rgb_with_kml(raster_image, affine_transform, kml_gdf, ax):
    """Plot the RGB image with KML polygons overlayed, using affine transform."""
    # Stack the RGB bands into an image (assuming 3 bands for RGB)
    rgb_image = raster_image.transpose(1, 2, 0)  # Convert to (height, width, 3)

    # Plot the RGB image
    # ax.imshow(rgb_image)
    ax.set_title('RGB with Overlaid Polygons')
    ax.axis('off')

    # Plot the KML polygons on top of the RGB image
    for idx, row in kml_gdf.iterrows():
        geom = row['geometry']
        if isinstance(geom, Polygon) and geom.is_valid:
            ax.plot(*geom.exterior.xy, color='red', linewidth=2)
        else:
            print (f'error in {idx}')
            # for poly in geom.geoms:
            #     ax.plot(*poly.exterior.xy, color='red', linewidth=2)


def crop_raster_with_polygon(raster, polygon):
    # Mask the image with the polygon (clipping)
    out_image, out_transform = mask(raster, [polygon], crop=True)
    return out_image, out_transform

def calculate_band_stats(band_values):
    ndvi_mean = np.nanmean(band_values)
    ndvi_std = np.nanstd(band_values)
    return ndvi_mean, ndvi_std

def calculate_ndvi_mask(red_band, nir_band):
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    return ndvi

def calculate_mcari_mask(red_band, blue_band, green_1_band):
    mcari = ((red_band - green_1_band)  - 0.2 * (red_band - blue_band)) * (red_band/green_1_band)
    return mcari

def calculate_msavi_mask(red_band, nir_band):
    msavi = ((2*(nir_band-red_band)+1 - (2*(nir_band - red_band + 1)**2 - 8*(nir_band-red_band))*0.5))/2
    return msavi

def calculate_cire_mask(red_edge, nir_band):
    cire = (nir_band/red_edge) - 1
    return cire

def get_date(filename):
        file_name = filename.split('/')[-1]
        if 'SKY' in file_name:
            return file_name.split('_')[3][:8]
        else:
            return file_name[:8]

def visualize_raster_and_gdf(raster_image, affine_transform, gdf, raster_crs, img_date, id_subset =4277, bounding_box = None ):
    fig, ax = plt.subplots(figsize=(8, 8))
    show(raster_image / 2000, ax=ax, transform=affine_transform, with_bounds=bounding_box)

    gdf_overlapping = gdf.to_crs(raster_crs)

    # gdf_overlapping[gdf_overlapping['id'] == id_subset].boundary.plot(ax=ax, edgecolor="yellow", linewidth=5)
    gdf_overlapping.boundary.plot(ax=ax, color=gdf_overlapping['color'], linewidth=2)

    plt.title(f'{img_date}')
    plt.show()

if __name__ == '__main__':
    crop_id = 'raza_vehari'

    file_dir=f'/Users/daoud/PycharmAssets/{crop_id}_farms/'
    asset_dir =f'/Users/daoud/PycharmAssets/{crop_id}_farms'
    season = 'kharif'
    file_dir = f'{asset_dir}/*/PSScene/'

    ultra_high_ids = []#['349', '717', '269', '774', '855', '790']
    unwanted_ids = {
        'r1': [],
        'k2': [],
        'k4': [],
        'k7': [],
        'r5': [],

        'r9': ['110', '526', '68', '1438', '118', '23', '750', '1290'],
        'r10': ['688', '425'],
    }
    unwanted_ids = ['301', '302', '304', '153', '176', '170', '175', '172'] #unwanted_ids[crop_id]
    # unwanted_ids += ultra_high_ids

    unwanted_ids_int = [int(id) for id in unwanted_ids]

    # cluster_file = '/Users/daoud/PycharmProjects/DISA/ai4h_disa/experimental/sufi/cluster_cire.csv'
    # z_score_file = '/Users/daoud/PycharmProjects/DISA/ai4h_disa/experimental/sufi/cire_z_scores_cluster_8.csv'
    z_score_ts_file = f'/Users/daoud/PycharmProjects/DISA/ai4h_disa/experimental/{crop_id}_cire_z_scores_ts.csv'
    if os.path.exists(z_score_ts_file):
        z_ts_df = pd.read_csv(z_score_ts_file)

        z_ts_df = z_ts_df.drop(columns=unwanted_ids)

        # z_ts_df = z_ts_df.drop(columns=['830', '515', '509', '554'])
    else:
        z_ts_df = pd.DataFrame()
    # (z_ts_df['62'].cumsum()) / (z_ts_df['62'].index + 1)
    cumulative_avg_df = pd.DataFrame(index=z_ts_df.index)
    for column in z_ts_df.columns[1:]:
        cumulative_avg_df[column] = z_ts_df[column].cumsum() / z_ts_df[column].notnull().cumsum()

    boundaries_file = f'{asset_dir}/{crop_id}_farms.geojson'
    print (boundaries_file)
    gdf_boundaries = gpd.read_file(boundaries_file)

    demo_ids = {
        'k2': [],
        'k4': [],
        'k7': [],
        'r1': [],
        'r5': [],

        'r9': [],
        'r10': ['2000', '2001'],
    }

    # demo_ids = demo_ids[crop_id]
    # gdf_boundaries['color'] = gdf_boundaries['Name'].apply(lambda id: 'red' if id in demo_ids else '#1f77b4')
    reset_names = False
    if reset_names:
        gdf_boundaries["Name"] = gdf_boundaries.index
        gdf_boundaries.to_file(boundaries_file)

    gdf_boundaries.boundary.plot(
        edgecolor = 'red'#gdf_boundaries['color']
    )

    z_score_file_n = glob.glob(f'/Users/daoud/PycharmProjects/DISA/ai4h_disa/experimental/{crop_id}/{crop_id}_cire_z_scores_norm.csv')
    z_score_df_n = []

    for z_score_file in z_score_file_n:
        z_score_df_n.append(pd.read_csv(z_score_file))

    if z_score_df_n:
        z_score_df = pd.concat(z_score_df_n)
        z_score_df = z_score_df[:94]
        z_score_df['Unnamed: 0'] = z_score_df['Unnamed: 0'].astype(np.int64)

    target_crs = 'epsg:3857'
    only_visual = False # set false if you want to calculate scores
    write_to_file = True

    if only_visual and write_to_file:
        raise ValueError

    color_clusters = False
    color_z_scores = True
    show_z_ts_plots = False

    if show_z_ts_plots:
        for time in z_ts_df.index:  # Replace with your desired timestamp
            z_ts_df = cumulative_avg_df
            time_data = z_ts_df.loc[time]

            gdf_boundaries = gpd.read_file(boundaries_file)


            gdf_boundaries = gdf_boundaries.copy()  # Make a copy to avoid modifying the original
            gdf_boundaries["z_score"] = gdf_boundaries["Name"].map(time_data)

            gdf_boundaries_with_z = gdf_boundaries#[~gdf_boundaries['z_score'].isna()]

            norm = mcolors.Normalize(vmin=gdf_boundaries_with_z['z_score'].min(),
                                     vmax=gdf_boundaries_with_z['z_score'].max())
            cmap = cm.plasma  # Colormap for positive and negative values
            gdf_boundaries_with_z['color'] = gdf_boundaries_with_z['z_score'].apply(
                lambda z: mcolors.to_hex(cmap(norm(z))))

            gdf_boundaries_with_z['color'] = gdf_boundaries_with_z['color'].apply(
                lambda color: '#e3e3e3' if color == '#000000' else color)

            fig, ax = plt.subplots(figsize=(10, 10))
            plt.title(f'Health Score / Days since emergence: {time}')
            gdf_boundaries_with_z.plot(
                ax=ax,
                facecolor=gdf_boundaries_with_z['color'],  # Transparent fill
                edgecolor= gdf_boundaries_with_z['color'],  # Use precomputed colors
                linewidth=0.5
            )
            # plt.savefig(f'{asset_dir}/gifs/frames/cumulative_avg/z_score_ts_{time}.png')
            plt.show()

    img_files_skywatch = glob.glob(file_dir+'SKY*.tif')
    img_files_planet = glob.glob(file_dir+'*SR_8b_clip.tif')
    img_files = img_files_skywatch + img_files_planet
    ndvi_log = pd.DataFrame(columns=['date', 'name',
                                     'ndvi_mean', 'ndvi_std',
                                     'ndre_mean', 'ndre_std',
                                     'evi_mean', 'evi_std',
                                     'cire_mean', 'cire_std'])
    # kml_gdf = load_geojson(file_dir + '6_Nov_visit.geojson')  # Your KML polygons as GeoDataFrame


    # Example usage
    # kml_file = file_dir + 'kot_ishaq_filter.kml'
    # gdf_chunk_n = []
    # for chunk_id in range(1,6):
    #     gdf_chunk = gpd.read_file(file_dir + f'kot_ishaq_regions_{chunk_id}.geojson')
    #     gdf_chunk_n.append(gdf_chunk)
    # gdf_kot_ishaq = pd.concat(gdf_chunk_n)
    # gdf_kot_ishaq =  gpd.read_file(file_dir + 'kot_ishaq.geojson')  # Your KML polygons as GeoDataFrame
    # gdf_syedwala_1 = gpd.read_file(file_dir + 'syedwala_filter_1.geojson')
    # gdf_syedwala_2 = gpd.read_file(file_dir + 'syedwala_filter_2.geojson')
    # gdf_kot_ishaq = gdf_kot_ishaq.to_crs(gdf_syedwala_1.crs)

    gdf = gdf_boundaries
    # gdf = pd.concat([gdf_kot_ishaq, gdf_syedwala_1, gdf_syedwala_2], axis=0)
    # gdf = gpd.read_file(file_dir + 'kot_ishaq.geojson')  # Your KML polygons as GeoDataFrame
    # gdf['area_acre'] = gdf.geometry.area / 4046
    # gdf_filter = gdf[gdf['area_acre']>0.5]
    # gdf_filter = gdf_filter[gdf_filter['area_acre']<10]
    # gdf_filter = gdf_filter.reset_index(drop=True)
    # gdf_filter['id'] = gdf_filter.index
    # gdf = gdf_filter
    # print (gdf_filter.head())
    gdf = gdf.to_crs('epsg:4326')
    lat_min, lat_max = 30.6655, 30.676
    lon_min, lon_max = 73.675377, 73.6815427



    bounding_box = box(lon_min, lat_min, lon_max, lat_max)
    gdf_overlapping = gdf#gdf[gdf.intersects(bounding_box)]
    gdf = gdf_overlapping
    gdf.boundary.plot()

    # if gdf_overlapping.crs != img_file_raster.crs:
    #     gdf_overlapping = gdf_overlapping.to_crs(img_file_raster.crs)
    if color_clusters:
        clusters = pd.read_csv(cluster_file)
        gdf_overlapping = pd.merge(gdf_overlapping, clusters, on='id')
        color_mapping = {
            # 0: 'red', 1: 'blue', 2: 'yellow', 3:'white', 4:'purple',
            0: 'black', 1: 'blue', 2: 'black', 3: 'white', 4: 'black',

            5: 'cyan', 6: 'black', 7: 'black', 8: 'red', 9: 'black'

                         # 5: 'cyan', 6: 'pink', 7: 'brown', 8: 'orange', 9: 'green'
                         }
        gdf_overlapping['color'] = gdf_overlapping['cluster'].map(color_mapping)
    else:
        gdf_overlapping['color'] = '#808080'

    if color_z_scores:
        z_scores = z_score_df

        z_scores = z_scores.rename(columns={'Unnamed: 0': 'id',
                                            '0': 'z_score'})

        z_scores = z_scores[~z_scores['id'].isin(unwanted_ids_int)]
        # todo: remove hardcoded threshold used for cleaner plots below
        # max_z_score = 3
        # min_z_score = -1.5
        #
        # z_scores = z_scores[z_scores['z_score'] < max_z_score]
        # z_scores = z_scores[z_scores['z_score'] > min_z_score]

        # z_scores = z_scores.rename(columns={'name':'id',
        #                          '0': 'z_score'})
        z_scores['z_score'] = z_scores['z_score'].clip(lower= -2, upper = 2)
        gdf_overlapping['Name'] = gdf_overlapping['Name'].astype(np.int64)
        gdf_overlapping = pd.merge(gdf_overlapping, z_scores, left_on='Name', right_on='id', how='left')
        norm = mcolors.Normalize(vmin=gdf_overlapping['z_score'].min(), vmax=gdf_overlapping['z_score'].max())
        cmap = cm.RdYlGn# plasma  # Colormap for positive and negative values
        gdf_overlapping['color'] = gdf_overlapping['z_score'].apply(lambda z: mcolors.to_hex(cmap(norm(z))))
        gdf_overlapping['color'] = gdf_overlapping['color'].apply(lambda color: '#e3e3e3' if color == '#000000' else color)

        fig, ax = plt.subplots(figsize=(10, 10))

        gdf_overlapping.plot(
            ax=ax,
            facecolor=gdf_overlapping['color'],  # Transparent fill
            edgecolor=gdf_overlapping['color'],  # Use precomputed colors
            linewidth=0.5
        )

        gdf_overlapping['predicted_yield'] = (18.81 * gdf_overlapping['z_score'] + 46.918)
        # Add a colorbar for reference
        norm = mcolors.Normalize(vmin=gdf_overlapping['predicted_yield'].min(), vmax=gdf_overlapping['predicted_yield'].max())

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(gdf_overlapping['predicted_yield'])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Predicted Yield (man/ac)")

        # Show the plot
        plt.title("Corn Fields colored by Health Score")
        plt.show()
        # color = cmap(norm(z_scores[field]))  # Map Z-score to color
        # gdf['color'] = color
        gdf_overlapping.to_file('cire_z_score.geojson', driver = 'GeoJSON')
    # fig, ax = plt.subplots(figsize = (5,5))
    # gdf_overlapping.boundary.plot(ax=ax, color=gdf_overlapping['color'])

    # bounds = rasterio.open(raster_path).bounds
    img_files.sort(key = lambda file:get_date(file))
    print (f'analysing {len(img_files)} images')
    for img_file in img_files:
        img_date =get_date(img_file)
        print ('>>>', img_date)
        # need to reproj to standard crs
        img_file_raster = rasterio.open(img_file)
        print(f'preproj raster {img_file_raster.crs}')
        print(f'preproj gdf {gdf.crs}')

        if img_file_raster.crs != target_crs:
            reprojected_raster = reproject_raster_to_match_gdf_crs(img_file, target_crs)
            img_file_raster = reprojected_raster

        print(f're-proj raster {img_file_raster.crs}')
        print(f're-proj gdf {gdf.crs}')


        image_data = img_file_raster.read()
        raster_image, affine_transform, raster_crs, (raster_bounds, raster_width, raster_hieght) = load_raster_with_affine(img_file)
        id_subset = 3826

        visualize_raster_and_gdf(raster_image, affine_transform, gdf_overlapping, raster_crs,
                                 img_date, id_subset, bounding_box = bounding_box)

        if only_visual: continue
        gdf_overlapping = gdf_overlapping.to_crs(target_crs)

        for idx, row in gdf_overlapping.iterrows():
            geom = row['geometry']
            # print (row['id'])
            print (row['Name'])

            # if gdf.crs != img_file_raster.crs:
            #     gdf = gdf.to_crs(img_file_raster.crs)
            # Crop the raster using the current polygon
            try:
                cropped_image, cropped_transform = crop_raster_with_polygon(img_file_raster, geom)



                # Read Red and NIR bands (assuming Band 3 = Red, Band 7 = NIR)
                red = cropped_image[RED-1]  # Band 3 = Red (index 2)
                nir = cropped_image[NIR-1]  # Band 7 = NIR (index 6)
                blue = cropped_image[BLUE-1]
                red_edge = cropped_image[RED_EDGE-1]
                green_1 = cropped_image[GREEN_I - 1]
                ndvi = calculate_ndvi_mask(red, nir)
                ndre = calculate_ndvi_mask(red_edge, nir)
                evi = create_evi(cropped_image)
                cire = calculate_cire_mask(red_edge, nir)
                mcari = calculate_mcari_mask(red_band=red, blue_band=blue, green_1_band=green_1)
                msavi = calculate_msavi_mask(red_band=red, nir_band = nir)

                # fig, ax = plt.subplots(figsize=(10, 10))
                # show(ndvi)
                # plt.show()
                # Calculate NDVI for the cropped image

                # Calculate mean and standard deviation of NDVI for the pixels inside the polygon
                ndvi_mean, ndvi_std = calculate_band_stats(ndvi)
                ndre_mean, ndre_std = calculate_band_stats(ndre)
                evi_mean, evi_std = calculate_band_stats(evi)
                cire_mean, cire_std = calculate_band_stats(cire)
                mcari_mean, mcari_std = calculate_band_stats(mcari)
                msavi_mean, msavi_std = calculate_band_stats(msavi)

                new_row = pd.DataFrame(
                    {'date': [img_date],
                     'name': [row['Name']], #id when id used
                     'ndvi_mean': [ndvi_mean],
                     'ndvi_std': [ndvi_std],
                     'ndre_mean': [ndre_mean],
                     'ndre_std': [ndre_std],
                     'evi_mean': [evi_mean],
                     'evi_std': [evi_std],
                     'cire_mean': [cire_mean],
                     'cire_std': [cire_std],
                     'mcari_mean': [mcari_mean],
                     'mcari_std': [mcari_std],
                     'msavi_mean': [msavi_mean],
                     'msavi_std': [msavi_std]
                     })

                ndvi_log = pd.concat([ndvi_log, new_row], ignore_index=True)


                # Print the results
                print(f'Polygon {idx} (Date: {img_date}): NDVI Mean = {ndvi_mean:.4f}, NDVI Std Dev = {ndvi_std:.4f}')

            except Exception as e:
                print(f"Error in {img_date}/{idx}: {e}")




    ndvi_log['date'] = pd.to_datetime(ndvi_log['date'], format='%Y%m%d')
    log_file_name = f'./{season}_{crop_id}_field_logs.csv'
    if write_to_file:
        ndvi_log.to_csv(log_file_name)



    # print (ndvi_log[0])
    assert False


    fig, ax = plt.subplots(figsize=(10, 10))
    for idx, row in kml_gdf.iterrows():
        geom = row['geometry']
        if isinstance(geom, Polygon) and geom.is_valid:
            ax.plot(*geom.exterior.xy, color='red', linewidth=2)
        else:
            print(f'error in {idx}')

    rgb_composite = create_rgb_composite(image_data)
    ndvi_composite = create_ndvi(image_data)
    red_edge_composite = create_red_edge(image_data)
    evi_composite = create_evi(image_data)
    ndre_composite = create_ndre(image_data)  # New NDRE composite

    show(clip_extremes(ndre_composite), transform=img_file_raster.transform, ax = ax)

    plt.title(img_date)
    plt.show()