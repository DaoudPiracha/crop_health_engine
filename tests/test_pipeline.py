"""Integration tests for the full pipeline."""

import tempfile
from pathlib import Path

import pytest
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from engine.pipeline_config import PipelineConfig
from engine.pipeline.run import run_pipeline, _build_assets


class TestBuildAssets:
    """Test asset building helper function."""

    def test_build_assets(self, tmp_path):
        """Test that _build_assets loads and processes assets correctly."""
        # Create test boundaries file
        import geopandas as gpd
        gdf = gpd.GeoDataFrame(
            {"Name": ["Field_1"]},
            geometry=[box(73.67, 30.67, 73.68, 30.68)],
            crs="EPSG:4326"
        )
        boundaries_file = tmp_path / "boundaries.geojson"
        gdf.to_file(boundaries_file)

        # Create test image files
        img_dir = tmp_path / "images" / "date1" / "PSScene"
        img_dir.mkdir(parents=True)

        width, height = 50, 50
        bounds = (73.67, 30.67, 73.68, 30.68)
        transform = from_bounds(*bounds, width, height)
        data = np.random.rand(8, height, width) * 1000 + 500

        img_file = img_dir / "20240101_SR_8b_clip.tif"
        with rasterio.open(
            img_file, 'w', driver='GTiff',
            height=height, width=width, count=8,
            dtype=np.uint16, crs='EPSG:4326', transform=transform,
        ) as dst:
            for band_idx in range(8):
                dst.write(data[band_idx].astype(np.uint16), band_idx + 1)

        # Create config
        cfg = PipelineConfig(
            asset_dir=str(tmp_path),
            season="test",
            crop_id="test_crop",
            file_dir=str(tmp_path / "images" / "*" / "PSScene") + "/",
            boundaries_file=str(boundaries_file),
            cluster_file="",
            z_score_ts_file="",
            z_score_glob="",
            target_crs="EPSG:4326",
            only_visual=False,
            write_to_file=False,
            color_clusters=False,
            color_z_scores=False,
            show_z_ts_plots=False,
            reset_names=False,
            show_images_at_each_ts=False,
            unwanted_ids=[],
            bbox_latlon=(30.67, 30.68, 73.67, 73.68),
        )

        img_files, gdf_boundaries, bounding_box = _build_assets(cfg)

        # Verify outputs
        assert len(img_files) == 1
        assert img_files[0].endswith("20240101_SR_8b_clip.tif")
        assert len(gdf_boundaries) == 1
        assert gdf_boundaries.crs.to_string() == "EPSG:4326"
        assert bounding_box is not None


class TestRunPipeline:
    """Test the full pipeline execution."""

    def test_run_pipeline_complete_workflow(self, tmp_path):
        """Test complete pipeline workflow with file output."""
        # Setup: Create test data
        import geopandas as gpd

        # Create boundaries
        gdf = gpd.GeoDataFrame(
            {"Name": ["Field_1", "Field_2"]},
            geometry=[
                box(73.671, 30.671, 73.675, 30.675),
                box(73.676, 30.671, 73.679, 30.675),
            ],
            crs="EPSG:4326"
        )
        boundaries_file = tmp_path / "boundaries.geojson"
        gdf.to_file(boundaries_file)

        # Create test images
        img_dir = tmp_path / "images" / "date1" / "PSScene"
        img_dir.mkdir(parents=True)

        for date in ["20240101", "20240215"]:
            width, height = 50, 50
            bounds = (73.67, 30.67, 73.68, 30.68)
            transform = from_bounds(*bounds, width, height)
            data = np.random.rand(8, height, width) * 1000 + 500

            img_file = img_dir / f"{date}_SR_8b_clip.tif"
            with rasterio.open(
                img_file, 'w', driver='GTiff',
                height=height, width=width, count=8,
                dtype=np.uint16, crs='EPSG:4326', transform=transform,
            ) as dst:
                for band_idx in range(8):
                    dst.write(data[band_idx].astype(np.uint16), band_idx + 1)

        # Create output directory
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        # Configure and run pipeline
        cfg = PipelineConfig(
            asset_dir=str(tmp_path),
            season="kharif",
            crop_id="test_farm",
            file_dir=str(tmp_path / "images" / "*" / "PSScene") + "/",
            boundaries_file=str(boundaries_file),
            cluster_file="",
            z_score_ts_file="",
            z_score_glob="",
            target_crs="EPSG:4326",
            only_visual=False,
            write_to_file=True,
            color_clusters=False,
            color_z_scores=False,
            show_z_ts_plots=False,
            reset_names=False,
            show_images_at_each_ts=False,
            unwanted_ids=[],
            bbox_latlon=(30.67, 30.68, 73.67, 73.68),
        )

        # Change to output dir for writing
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(out_dir)
            run_pipeline(cfg)
        finally:
            os.chdir(original_dir)

        # Verify output file was created
        output_file = out_dir / "kharif_test_farm_field_veg_index_stats.csv"
        assert output_file.exists()

        # Read and verify output
        import pandas as pd
        df = pd.read_csv(output_file)

        # Should have 4 rows (2 dates * 2 fields)
        assert len(df) == 4

        # Check columns exist
        expected_cols = [
            "date", "name",
            "ndvi_mean", "ndvi_std",
            "ndre_mean", "ndre_std",
            "evi_mean", "evi_std",
            "cire_mean", "cire_std",
            "mcari_mean", "mcari_std",
            "msavi_mean", "msavi_std",
        ]
        for col in expected_cols:
            assert col in df.columns

        # Check field names
        assert set(df["name"].unique()) == {"Field_1", "Field_2"}

    def test_run_pipeline_only_visual_mode(self, tmp_path):
        """Test pipeline in only_visual mode doesn't write output."""
        # Setup minimal data
        import geopandas as gpd

        gdf = gpd.GeoDataFrame(
            {"Name": ["Field_1"]},
            geometry=[box(73.67, 30.67, 73.68, 30.68)],
            crs="EPSG:4326"
        )
        boundaries_file = tmp_path / "boundaries.geojson"
        gdf.to_file(boundaries_file)

        img_dir = tmp_path / "images" / "date1" / "PSScene"
        img_dir.mkdir(parents=True)

        width, height = 50, 50
        bounds = (73.67, 30.67, 73.68, 30.68)
        transform = from_bounds(*bounds, width, height)
        data = np.random.rand(8, height, width) * 1000 + 500

        img_file = img_dir / "20240101_SR_8b_clip.tif"
        with rasterio.open(
            img_file, 'w', driver='GTiff',
            height=height, width=width, count=8,
            dtype=np.uint16, crs='EPSG:4326', transform=transform,
        ) as dst:
            for band_idx in range(8):
                dst.write(data[band_idx].astype(np.uint16), band_idx + 1)

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        cfg = PipelineConfig(
            asset_dir=str(tmp_path),
            season="kharif",
            crop_id="test_farm",
            file_dir=str(tmp_path / "images" / "*" / "PSScene") + "/",
            boundaries_file=str(boundaries_file),
            cluster_file="",
            z_score_ts_file="",
            z_score_glob="",
            target_crs="EPSG:4326",
            only_visual=True,
            write_to_file=False,
            color_clusters=False,
            color_z_scores=False,
            show_z_ts_plots=False,
            reset_names=False,
            show_images_at_each_ts=False,
            unwanted_ids=[],
            bbox_latlon=(30.67, 30.68, 73.67, 73.68),
        )

        import os
        original_dir = os.getcwd()
        try:
            os.chdir(out_dir)
            run_pipeline(cfg)
        finally:
            os.chdir(original_dir)

        # Output file should NOT exist
        output_file = out_dir / "kharif_test_farm_field_veg_index_stats.csv"
        assert not output_file.exists()

    def test_run_pipeline_error_on_invalid_config(self, tmp_path):
        """Test that pipeline raises error for invalid config."""
        import geopandas as gpd

        gdf = gpd.GeoDataFrame(
            {"Name": ["Field_1"]},
            geometry=[box(73.67, 30.67, 73.68, 30.68)],
            crs="EPSG:4326"
        )
        boundaries_file = tmp_path / "boundaries.geojson"
        gdf.to_file(boundaries_file)

        img_dir = tmp_path / "images" / "date1" / "PSScene"
        img_dir.mkdir(parents=True)

        # Invalid config: only_visual=True but write_to_file=True
        cfg = PipelineConfig(
            asset_dir=str(tmp_path),
            season="kharif",
            crop_id="test_farm",
            file_dir=str(tmp_path / "images" / "*" / "PSScene") + "/",
            boundaries_file=str(boundaries_file),
            cluster_file="",
            z_score_ts_file="",
            z_score_glob="",
            target_crs="EPSG:4326",
            only_visual=True,
            write_to_file=True,  # Invalid combination!
            color_clusters=False,
            color_z_scores=False,
            show_z_ts_plots=False,
            reset_names=False,
            show_images_at_each_ts=False,
            unwanted_ids=[],
            bbox_latlon=(30.67, 30.68, 73.67, 73.68),
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="only_visual=True but write_to_file=True"):
            run_pipeline(cfg)

    def test_run_pipeline_crs_reprojection(self, tmp_path):
        """Test pipeline with CRS reprojection."""
        import geopandas as gpd

        # Create boundaries in EPSG:4326
        gdf = gpd.GeoDataFrame(
            {"Name": ["Field_1"]},
            geometry=[box(73.67, 30.67, 73.68, 30.68)],
            crs="EPSG:4326"
        )
        boundaries_file = tmp_path / "boundaries.geojson"
        gdf.to_file(boundaries_file)

        img_dir = tmp_path / "images" / "date1" / "PSScene"
        img_dir.mkdir(parents=True)

        width, height = 50, 50
        bounds = (73.67, 30.67, 73.68, 30.68)
        transform = from_bounds(*bounds, width, height)
        data = np.random.rand(8, height, width) * 1000 + 500

        img_file = img_dir / "20240101_SR_8b_clip.tif"
        with rasterio.open(
            img_file, 'w', driver='GTiff',
            height=height, width=width, count=8,
            dtype=np.uint16, crs='EPSG:4326', transform=transform,
        ) as dst:
            for band_idx in range(8):
                dst.write(data[band_idx].astype(np.uint16), band_idx + 1)

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        # Request different target CRS
        cfg = PipelineConfig(
            asset_dir=str(tmp_path),
            season="kharif",
            crop_id="test_farm",
            file_dir=str(tmp_path / "images" / "*" / "PSScene") + "/",
            boundaries_file=str(boundaries_file),
            cluster_file="",
            z_score_ts_file="",
            z_score_glob="",
            target_crs="EPSG:3857",  # Web Mercator
            only_visual=False,
            write_to_file=True,
            color_clusters=False,
            color_z_scores=False,
            show_z_ts_plots=False,
            reset_names=False,
            show_images_at_each_ts=False,
            unwanted_ids=[],
            bbox_latlon=(30.67, 30.68, 73.67, 73.68),
        )

        import os
        original_dir = os.getcwd()
        try:
            os.chdir(out_dir)
            run_pipeline(cfg)
        finally:
            os.chdir(original_dir)

        # Should still produce output
        output_file = out_dir / "kharif_test_farm_field_veg_index_stats.csv"
        assert output_file.exists()
