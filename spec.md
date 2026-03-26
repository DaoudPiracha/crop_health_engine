# Crop Health Scoring Engine — Specification

## Overview

**Kisaan AI - Crop Health Scoring** is a Python pipeline that processes multispectral satellite imagery from Planet Labs and SkyWatch to compute vegetation indices and aggregate health metrics for agricultural fields. It is designed to support precision farming by tracking crop health over time across polygonal field boundaries.

The system produces time-series CSV reports of vegetation indices per field, and includes advanced analysis capabilities (clustering, anomaly detection) for identifying problematic fields and unusual growth patterns.

---

## Goals

- Ingest 8-band satellite GeoTIFFs and GeoJSON field boundary files
- Compute 6 vegetation indices per field per image date
- Aggregate per-field statistics (mean, std) for each index
- Write time-series reports to CSV
- Support visualization of imagery overlaid with field boundaries
- Support advanced temporal analysis (clustering, z-score anomaly detection)

---

## Architecture

```
engine/
├── constants.py            # Band index mappings for 8-band imagery
├── pipeline_config.py      # Immutable configuration dataclass
├── main.py                 # Entry point with example config
│
├── io/
│   ├── raster.py           # Raster loading, CRS reprojection, polygon cropping
│   └── assets.py           # Image file discovery, boundary loading, date parsing
│
├── compute/
│   ├── indices.py          # Vegetation index calculations
│   └── stats.py            # Per-field statistics aggregation, CSV writing
│
├── viz/
│   └── preview.py          # Raster + boundary visualization
│
├── pipeline/
│   └── run.py              # Main pipeline orchestrator
│
└── vi_analysis.py          # Advanced time-series analysis (clustering, z-scores)
```

### Layers

| Layer | Responsibility |
|---|---|
| Configuration | Immutable `PipelineConfig` dataclass; all parameters defined at call site |
| I/O | Load rasters, reproject CRS, crop to polygons, discover files, load boundaries |
| Compute | Pure functions for vegetation indices and field statistics |
| Orchestration | Coordinate I/O + compute, handle per-field errors, write output |
| Visualization | Optional rendering of rasters and boundaries |

---

## Input Formats

### Satellite Imagery

- **Format:** GeoTIFF, 8 spectral bands (Planet Labs SuperDove standard)
- **Band order:**

  | Band # | Name | Wavelength |
  |---|---|---|
  | 1 | Coastal Blue | 443 nm |
  | 2 | Blue | 490 nm |
  | 3 | Green I | 531 nm |
  | 4 | Green | 565 nm |
  | 5 | Yellow | 610 nm |
  | 6 | Red | 665 nm |
  | 7 | Red Edge | 705 nm |
  | 8 | NIR | 865 nm |

- **Typical value range:** 0–3000 digital numbers (surface reflectance)
- **Supported filename conventions:**
  - Planet: `YYYYMMDD_*_SR_8b_clip.tif`
  - SkyWatch: `SKY_*_*_YYYYMMDD_*.tif`
- **CRS:** Any EPSG code; auto-reprojected to `target_crs` during processing

### Field Boundaries

- **Format:** GeoJSON `FeatureCollection` of `Polygon` features
- **Required:** `geometry` field (polygon)
- **Optional:** `Name` property — used as field identifier; defaults to integer index if absent

---

## Configuration

All pipeline parameters are specified via `PipelineConfig`, a frozen Python dataclass.

```python
@dataclass(frozen=True)
class PipelineConfig:
    asset_dir: str               # Base directory for assets
    file_dir: str                # Glob pattern for image directories (supports {asset_dir})
    boundaries_file: str         # Path to GeoJSON field boundaries
    season: str                  # "kharif" or "rabi" — used in output filename
    crop_id: str                 # Farm/crop identifier — used in output filename
    target_crs: str              # Target EPSG code, e.g. "epsg:3857"
    only_visual: bool            # If True, skip computation and only preview images
    write_to_file: bool          # If True, write CSV to disk
    reset_names: bool            # If True, replace field names with integer indices
    show_images_at_each_ts: bool # Display imagery at each timestamp during processing
    color_clusters: bool         # Color fields by cluster assignment
    color_z_scores: bool         # Color fields by z-score anomaly level
    show_z_ts_plots: bool        # Show z-score time-series plots
    unwanted_ids: List[str]      # Field IDs to exclude from processing
    bbox_latlon: Tuple[float, float, float, float]  # (lat_min, lat_max, lon_min, lon_max)
    cluster_file: str            # (reserved)
    z_score_ts_file: str         # (reserved)
    z_score_glob: str            # (reserved)
```

**Constraint:** `only_visual=True` and `write_to_file=True` cannot both be set — raises `ValueError`.

---

## Pipeline Execution

### `run_pipeline(cfg: PipelineConfig)`

**Steps:**

1. **Build assets** — load field boundaries (reprojected to EPSG:4326), discover and sort image files chronologically, build bounding box from `bbox_latlon`
2. **Compute statistics** — for each image × field pair:
   - Open raster and reproject to `target_crs`
   - Crop raster to field polygon
   - Extract 8 bands as numpy arrays
   - Compute all 6 vegetation indices
   - Compute mean and std for each index
   - Append result row: `(date, field_name, 12 metrics)`
3. **Write output** — convert date strings to datetime, write CSV if `write_to_file=True`
4. **Preview** (optional) — display imagery with field boundaries overlaid

**Error handling:** Per-field computation errors are caught and logged; they do not halt the pipeline.

---

## Vegetation Indices

All indices are computed in `engine/compute/indices.py` using numpy array operations on normalized band reflectance values (DN / 3000).

| Index | Formula | Interpretation |
|---|---|---|
| **NDVI** | `(NIR - RED) / (NIR + RED)` | General vegetation density; >0.5 = healthy canopy |
| **NDRE** | `(NIR - RED_EDGE) / (NIR + RED_EDGE)` | Chlorophyll content; more sensitive than NDVI mid-season |
| **EVI** | `2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)` | Vegetation vigor with atmospheric correction |
| **CIRE** | `(NIR / RED_EDGE) - 1` | Chlorophyll/nitrogen status |
| **MCARI** | `((RED - GREEN_I) - 0.2*(RED - BLUE)) * (RED / GREEN_I)` | Chlorophyll absorption ratio |
| **MSAVI** | modified soil-adjusted formula | Vegetation index with reduced soil background signal |

For each index, two statistics are computed per field per date:
- `{index}_mean` — spatial mean across all pixels within the polygon
- `{index}_std` — spatial standard deviation (uniformity measure)

---

## Output Format

### CSV Report

**Filename:** `{season}_{crop_id}_field_veg_index_stats.csv`
**Example:** `kharif_shahmeer_field_veg_index_stats.csv`

**Schema:**

| Column | Type | Description |
|---|---|---|
| `date` | datetime | Image acquisition date |
| `name` | str | Field identifier |
| `ndvi_mean` | float | Mean NDVI across field |
| `ndvi_std` | float | Std NDVI across field |
| `ndre_mean` | float | Mean NDRE |
| `ndre_std` | float | Std NDRE |
| `evi_mean` | float | Mean EVI |
| `evi_std` | float | Std EVI |
| `cire_mean` | float | Mean CIRE |
| `cire_std` | float | Std CIRE |
| `mcari_mean` | float | Mean MCARI |
| `mcari_std` | float | Std MCARI |
| `msavi_mean` | float | Mean MSAVI |
| `msavi_std` | float | Std MSAVI |

Each row represents one field on one date. Multiple fields and dates produce a long-format time-series table.

---

## Advanced Analysis (`vi_analysis.py`)

Post-processing analysis on the CSV output. **Not part of the core pipeline.**

### K-Shape Time Series Clustering

- **Library:** `tslearn`
- **Method:** K-Shape clustering — groups fields with similar vegetation growth trajectories across the season
- **Input:** NDVI (or other index) time series per field
- **Output:** Cluster assignments, optional colorized field map

### Z-Score Anomaly Detection

- **Method:** Z-score normalization of each field's index values relative to the seasonal population of all fields at the same date
- **Purpose:** Identify fields that are significantly above or below seasonal norms
- **Output:** `{crop_id}_{index}_z_scores_norm.csv`, optional time-series plots

### PCA (optional)

- Dimensionality reduction prior to clustering to reduce noise from multiple indices

---

## CRS Handling

- Field boundaries are loaded and stored in **EPSG:4326** (lat/lon)
- Processing is performed in **`target_crs`** (configurable; typically EPSG:3857 or EPSG:4326)
- Reprojection uses in-memory `MemoryFile` via rasterio — non-destructive, no files written
- All spatial operations require consistent CRS between raster and vector data

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `rasterio` | 1.3.10 | Raster I/O and CRS reprojection |
| `geopandas` | 1.0.1 | GeoJSON loading and vector operations |
| `shapely` | 2.0.7 | Polygon geometry |
| `pyproj` | 3.5.0 | Coordinate projection |
| `numpy` | 2.0.2 | Band array computations |
| `pandas` | 2.3.3 | DataFrame and CSV I/O |
| `matplotlib` | 3.9.4 | Visualization |
| `tslearn` | — | K-Shape time-series clustering (vi_analysis only) |
| `scikit-learn` | — | PCA (vi_analysis only) |
| `pytest` | ≥7.0.0 | Test framework |

---

## Testing

**Framework:** pytest with coverage enforcement (≥75% line coverage required)

**Test modules:**

| File | Scope |
|---|---|
| `tests/conftest.py` | Shared fixtures: synthetic rasters, test polygons in multiple CRS |
| `tests/test_indices.py` | Unit tests for all 6 vegetation index functions and edge cases |
| `tests/test_stats.py` | Unit tests for statistics aggregation and CSV writing |
| `tests/test_raster.py` | Unit tests for raster loading, reprojection, and polygon cropping |
| `tests/test_io_assets.py` | Unit tests for date parsing, file discovery, boundary loading |
| `tests/test_pipeline.py` | Integration tests for end-to-end pipeline execution |
| `tests/test_main.py` | Smoke tests for entry point, config structure, immutability |

**Coverage target:** 75% (`--cov-fail-under=75`)
**Reports:** terminal (missing lines) + HTML

---

## Known Limitations / TODO

- CIRE mean normalization has been identified as a desired post-processing step but is not yet implemented
- `vi_analysis.py` is currently WIP (active development on K-Shape clustering)
- `cluster_file`, `z_score_ts_file`, and `z_score_glob` config fields are reserved but not yet used
- `main.py` contains hardcoded paths for Shahmeer farms; no CLI interface exists yet
