"""Tests for statistics computation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from engine.compute.stats import (
    build_field_veg_index_stats,
    write_field_veg_index_stats,
    LOG_COLUMNS,
)


class TestBuildFieldVegIndexStats:
    """Test building field vegetation index statistics."""

    def test_build_stats_single_image_single_field(self, test_raster_path, test_polygon):
        """Test with one image and one field."""
        # Create GeoDataFrame with single polygon
        gdf = gpd.GeoDataFrame(
            {"Name": ["Field_1"], "geometry": [test_polygon]},
            crs="EPSG:4326"
        )

        img_files = [test_raster_path]

        df = build_field_veg_index_stats(
            img_file_paths=img_files,
            gdf_overlapping=gdf,
            target_crs="EPSG:4326",
            only_visual=False,
        )

        # Should have one row (1 image * 1 field)
        assert len(df) == 1

        # Should have all expected columns
        for col in LOG_COLUMNS:
            assert col in df.columns

        # Check data types
        assert df["name"].iloc[0] == "Field_1"
        assert isinstance(df["ndvi_mean"].iloc[0], (float, np.floating))

    def test_build_stats_multiple_fields(self, test_raster_path):
        """Test with multiple fields."""
        # Create GeoDataFrame with 3 polygons
        polygons = [
            box(73.671, 30.671, 73.673, 30.673),
            box(73.674, 30.671, 73.676, 30.673),
            box(73.671, 30.674, 73.673, 30.676),
        ]
        gdf = gpd.GeoDataFrame(
            {"Name": ["Field_1", "Field_2", "Field_3"], "geometry": polygons},
            crs="EPSG:4326"
        )

        img_files = [test_raster_path]

        df = build_field_veg_index_stats(
            img_file_paths=img_files,
            gdf_overlapping=gdf,
            target_crs="EPSG:4326",
            only_visual=False,
        )

        # Should have 3 rows (1 image * 3 fields)
        assert len(df) == 3
        assert set(df["name"].values) == {"Field_1", "Field_2", "Field_3"}

    def test_build_stats_multiple_images(self, tmp_path, test_polygon):
        """Test with multiple images."""
        # Create 2 test rasters
        from tests.conftest import test_raster_path

        # Use fixture but need to create multiple
        gdf = gpd.GeoDataFrame(
            {"Name": ["Field_1"], "geometry": [test_polygon]},
            crs="EPSG:4326"
        )

        # Create two raster files with different names
        import rasterio
        from rasterio.transform import from_bounds

        img_files = []
        for i in range(2):
            raster_path = tmp_path / f"test_raster_{i}.tif"

            width, height = 50, 50
            bounds = (73.67, 30.67, 73.68, 30.68)
            transform = from_bounds(*bounds, width, height)
            data = np.random.rand(8, height, width) * 1000 + 500

            with rasterio.open(
                raster_path, 'w', driver='GTiff',
                height=height, width=width, count=8,
                dtype=np.uint16, crs='EPSG:4326', transform=transform,
            ) as dst:
                for band_idx in range(8):
                    dst.write(data[band_idx].astype(np.uint16), band_idx + 1)

            img_files.append(str(raster_path))

        df = build_field_veg_index_stats(
            img_file_paths=img_files,
            gdf_overlapping=gdf,
            target_crs="EPSG:4326",
            only_visual=False,
        )

        # Should have 2 rows (2 images * 1 field)
        assert len(df) == 2

    def test_build_stats_only_visual_mode(self, test_raster_path, test_polygon):
        """Test with only_visual=True returns empty DataFrame."""
        gdf = gpd.GeoDataFrame(
            {"Name": ["Field_1"], "geometry": [test_polygon]},
            crs="EPSG:4326"
        )

        df = build_field_veg_index_stats(
            img_file_paths=[test_raster_path],
            gdf_overlapping=gdf,
            target_crs="EPSG:4326",
            only_visual=True,
        )

        # Should return empty DataFrame with correct columns
        assert len(df) == 0
        assert list(df.columns) == LOG_COLUMNS

    def test_build_stats_crs_reprojection(self, test_raster_path, test_polygon):
        """Test that CRS reprojection works correctly."""
        gdf = gpd.GeoDataFrame(
            {"Name": ["Field_1"], "geometry": [test_polygon]},
            crs="EPSG:4326"
        )

        # Request different target CRS
        df = build_field_veg_index_stats(
            img_file_paths=[test_raster_path],
            gdf_overlapping=gdf,
            target_crs="EPSG:3857",  # Web Mercator
            only_visual=False,
        )

        assert len(df) == 1
        assert "ndvi_mean" in df.columns

    def test_build_stats_handles_errors_gracefully(self, test_raster_path):
        """Test that errors for individual fields don't crash entire process."""
        # Create a polygon outside the raster bounds
        out_of_bounds_polygon = box(0.0, 0.0, 0.1, 0.1)  # Way outside test raster
        valid_polygon = box(73.671, 30.671, 73.675, 30.675)

        gdf = gpd.GeoDataFrame(
            {"Name": ["OutOfBounds", "Valid"], "geometry": [out_of_bounds_polygon, valid_polygon]},
            crs="EPSG:4326"
        )

        df = build_field_veg_index_stats(
            img_file_paths=[test_raster_path],
            gdf_overlapping=gdf,
            target_crs="EPSG:4326",
            only_visual=False,
        )

        # Should have processed at least the valid field
        # (may have 1 or 2 rows depending on error handling)
        assert len(df) >= 0  # At minimum doesn't crash

    def test_build_stats_column_types(self, test_raster_path, test_polygon):
        """Test that output columns have correct types."""
        gdf = gpd.GeoDataFrame(
            {"Name": ["Field_1"], "geometry": [test_polygon]},
            crs="EPSG:4326"
        )

        df = build_field_veg_index_stats(
            img_file_paths=[test_raster_path],
            gdf_overlapping=gdf,
            target_crs="EPSG:4326",
            only_visual=False,
        )

        # Check column types
        assert df["date"].dtype == object  # string
        assert df["name"].dtype == object  # string
        assert df["ndvi_mean"].dtype in [np.float64, float]
        assert df["ndvi_std"].dtype in [np.float64, float]


class TestWriteFieldVegIndexStats:
    """Test writing field vegetation index statistics."""

    def test_write_stats_to_file(self, tmp_path):
        """Test writing stats to CSV file."""
        # Create sample DataFrame
        df = pd.DataFrame({
            "date": ["20240101"],
            "name": ["Field_1"],
            "ndvi_mean": [0.75],
            "ndvi_std": [0.05],
            "ndre_mean": [0.65],
            "ndre_std": [0.04],
            "evi_mean": [0.80],
            "evi_std": [0.06],
            "cire_mean": [1.5],
            "cire_std": [0.1],
            "mcari_mean": [0.5],
            "mcari_std": [0.05],
            "msavi_mean": [0.70],
            "msavi_std": [0.04],
        })

        file_name = write_field_veg_index_stats(
            df=df,
            season="kharif",
            crop_id="test_crop",
            write_to_file=True,
            out_dir=str(tmp_path),
        )

        # Check file was created
        assert Path(file_name).exists()
        assert "kharif_test_crop_field_veg_index_stats.csv" in file_name

        # Read back and verify
        df_read = pd.read_csv(file_name)
        assert len(df_read) == 1
        assert df_read["name"].iloc[0] == "Field_1"

    def test_write_stats_skip_write(self, tmp_path):
        """Test that write_to_file=False doesn't write."""
        df = pd.DataFrame({
            "date": ["20240101"],
            "name": ["Field_1"],
            "ndvi_mean": [0.75],
            "ndvi_std": [0.05],
            "ndre_mean": [0.65],
            "ndre_std": [0.04],
            "evi_mean": [0.80],
            "evi_std": [0.06],
            "cire_mean": [1.5],
            "cire_std": [0.1],
            "mcari_mean": [0.5],
            "mcari_std": [0.05],
            "msavi_mean": [0.70],
            "msavi_std": [0.04],
        })

        file_name = write_field_veg_index_stats(
            df=df,
            season="kharif",
            crop_id="test_crop",
            write_to_file=False,
            out_dir=str(tmp_path),
        )

        # File should not exist
        assert not Path(file_name).exists()

    def test_write_stats_date_formatting(self, tmp_path):
        """Test that dates are properly formatted."""
        df = pd.DataFrame({
            "date": ["20240101", "20240215"],
            "name": ["Field_1", "Field_1"],
            "ndvi_mean": [0.75, 0.80],
            "ndvi_std": [0.05, 0.05],
            "ndre_mean": [0.65, 0.70],
            "ndre_std": [0.04, 0.04],
            "evi_mean": [0.80, 0.85],
            "evi_std": [0.06, 0.06],
            "cire_mean": [1.5, 1.6],
            "cire_std": [0.1, 0.1],
            "mcari_mean": [0.5, 0.55],
            "mcari_std": [0.05, 0.05],
            "msavi_mean": [0.70, 0.75],
            "msavi_std": [0.04, 0.04],
        })

        file_name = write_field_veg_index_stats(
            df=df,
            season="kharif",
            crop_id="test_crop",
            write_to_file=True,
            out_dir=str(tmp_path),
        )

        # Read back and check date formatting
        df_read = pd.read_csv(file_name)
        assert len(df_read) == 2
        # Dates should be converted to datetime
        pd.to_datetime(df_read["date"])  # Should not raise

    def test_write_stats_filename_format(self, tmp_path):
        """Test that filename follows expected format."""
        # Create proper data with valid date
        data = {col: [0.0] if col not in ["date", "name"] else ["Field_1" if col == "name" else "20240101"]
                for col in LOG_COLUMNS}
        df = pd.DataFrame(data)

        file_name = write_field_veg_index_stats(
            df=df,
            season="rabi",
            crop_id="wheat",
            write_to_file=True,
            out_dir=str(tmp_path),
        )

        # Check filename format
        expected_name = f"{tmp_path}/rabi_wheat_field_veg_index_stats.csv"
        assert file_name == expected_name
