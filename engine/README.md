# Engine Module

A modular pipeline for computing vegetation indices and crop health statistics from satellite imagery.

## Overview

The engine module processes multispectral satellite imagery (Planet, SkyWatch) to extract vegetation health metrics for agricultural fields. It computes time-series statistics of various vegetation indices (NDVI, EVI, CIRE, etc.) for each field polygon and outputs structured CSV data.

## Architecture

```
engine/
├── constants.py           # Band indices for multispectral imagery
├── pipeline_config.py     # Configuration dataclass
├── main.py               # Entry point with example configuration
├── io/                   # Input/Output operations
│   ├── raster.py        # Raster loading, reprojection, cropping
│   └── assets.py        # File collection, boundary loading
├── compute/             # Core computations
│   ├── indices.py       # Vegetation index calculations
│   └── stats.py         # Field statistics aggregation
├── viz/                 # Visualization
│   └── preview.py       # Raster and boundary preview
└── pipeline/            # Orchestration
    └── run.py           # Main pipeline runner
```

## Key Components

### Configuration

**PipelineConfig** (pipeline_config.py) defines all pipeline parameters:
- Input paths (imagery directory, boundaries file)
- Output configuration (season, crop_id, write_to_file)
- Processing options (target CRS, preview mode)
- Visualization flags

### I/O Operations

**Raster Operations** (io/raster.py):
- `load_raster_with_affine()` - Load raster with geospatial transform
- `open_raster_in_crs()` - Context manager for safe CRS-matched raster access
- `crop_raster_with_polygon()` - Clip raster to field geometry

**Asset Management** (io/assets.py):
- `collect_image_files()` - Gather and sort satellite imagery by date
- `load_boundaries()` - Load field polygons from GeoJSON
- `get_date()` - Extract date from filename

### Vegetation Indices

**Supported Indices** (compute/indices.py):
- **NDVI** - Normalized Difference Vegetation Index
- **NDRE** - Normalized Difference Red Edge
- **EVI** - Enhanced Vegetation Index
- **CIRE** - Chlorophyll Index Red Edge
- **MCARI** - Modified Chlorophyll Absorption Ratio Index
- **MSAVI** - Modified Soil-Adjusted Vegetation Index

**Statistics**:
- `compute_indices_stats()` - Compute mean and std for all indices
- `calculate_band_stats()` - Extract mean/std from band array

### Pipeline

**Main Runner** (pipeline/run.py):
- `run_pipeline()` - Orchestrates full processing workflow
  1. Load boundaries and imagery
  2. Build bounding box from lat/lon
  3. Compute field statistics for each date
  4. Write results to CSV
  5. Optional: preview imagery

## Usage

### Basic Example

```python
from engine.pipeline_config import PipelineConfig
from engine.pipeline.run import run_pipeline

cfg = PipelineConfig(
    asset_dir="/path/to/assets",
    season="kharif",
    crop_id="farm_name",
    file_dir="/path/to/assets/*/PSScene/",
    boundaries_file="/path/to/boundaries.geojson",
    target_crs="epsg:3857",
    only_visual=False,
    write_to_file=True,
    reset_names=False,
    show_images_at_each_ts=False,
    unwanted_ids=[],
    bbox_latlon=(lat_min, lat_max, lon_min, lon_max),
    # Visualization options
    color_clusters=False,
    color_z_scores=False,
    show_z_ts_plots=False,
    # Placeholder paths for future features
    cluster_file="",
    z_score_ts_file="",
    z_score_glob="",
)

run_pipeline(cfg)
```

### Preview Mode

Set `only_visual=True` to skip computations and only preview imagery:

```python
cfg = PipelineConfig(
    # ... other params
    only_visual=True,
    write_to_file=False,
    show_images_at_each_ts=True,
)
```

### Output Format

The pipeline generates a CSV with columns:
```
date, name, ndvi_mean, ndvi_std, ndre_mean, ndre_std, evi_mean, evi_std,
cire_mean, cire_std, mcari_mean, mcari_std, msavi_mean, msavi_std
```

Output filename: `{season}_{crop_id}_field_veg_index_stats.csv`

## Dependencies

- **rasterio** - Geospatial raster I/O and operations
- **geopandas** - Vector data handling
- **numpy** - Numerical computations
- **pandas** - Data structuring and CSV output
- **matplotlib** - Visualization
- **shapely** - Geometric operations

## Band Indices

The module expects 8-band imagery with the following band order (constants.py):
1. Coastal Blue
2. Blue
3. Green I
4. Green
5. Yellow
6. Red
7. Red Edge
8. NIR (Near-Infrared)

## Coordinate Reference Systems

- Input boundaries are expected in any CRS supported by GeoPandas
- Pipeline reprojects to `target_crs` (commonly "epsg:3857" or "epsg:4326")
- Bounding box coordinates use lat/lon (EPSG:4326)

## Error Handling

- Field processing errors are caught and logged per field/date
- Pipeline validates `only_visual` and `write_to_file` compatibility
- CRS mismatches are handled via automatic reprojection

## Extending the Pipeline

### Adding New Vegetation Indices

1. Add computation function to `compute/indices.py`:
```python
def calculate_new_index(band1, band2):
    return (band1 - band2) / (band1 + band2)
```

2. Update `compute_indices_stats()` to include the new index

3. Add corresponding columns to `LOG_COLUMNS` in `compute/stats.py`

### Custom Statistics

Extend `build_field_veg_index_stats()` in `compute/stats.py` to add custom aggregations beyond mean/std.

## Notes

- Images are sorted chronologically by date extracted from filename
- Supports both SkyWatch (`SKY*.tif`) and Planet (`*SR_8b_clip.tif`) naming conventions
- Memory-efficient: uses context managers for raster handling
- CRS reprojection happens in-memory via MemoryFile
