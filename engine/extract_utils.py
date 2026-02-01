import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box

# --- Raster Utilities ---
def load_raster(image_path):
    """Load raster and return the dataset."""
    return rasterio.open(image_path)

def reproject_raster(raster_path, target_crs):
    """Reproject a raster to match target CRS and return MemoryFile dataset."""
    with rasterio.open(raster_path) as src:
        if src.crs == target_crs:
            return src
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile
        profile.update({'crs': target_crs, 'transform': transform, 'width': width, 'height': height})

        with MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=src.read(i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest
                    )
            return memfile.open()

# --- Polygon Utilities ---
def load_geojson(path):
    return gpd.read_file(path)

def load_kml(path):
    return gpd.read_file(path, driver='kml')

def crop_raster(raster, polygon):
    """Mask and crop raster using a polygon geometry."""
    out_image, out_transform = mask(raster, [polygon], crop=True)
    return out_image, out_transform

def bounding_box_from_coords(lat_min, lat_max, lon_min, lon_max):
    return box(lon_min, lat_min, lon_max, lat_max)
