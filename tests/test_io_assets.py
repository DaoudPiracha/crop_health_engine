"""Tests for I/O asset management functions."""

import os
import tempfile
from pathlib import Path

import pytest
import geopandas as gpd

from engine.io.assets import get_date, collect_image_files, load_boundaries


class TestGetDate:
    """Test date extraction from filenames."""

    def test_get_date_skywatch_format(self):
        """Test date extraction from SkyWatch filename."""
        filename = "path/to/SKY_xxx_xxx_20240315_something.tif"
        date = get_date(filename)
        assert date == "20240315"

    def test_get_date_planet_format(self):
        """Test date extraction from Planet filename."""
        filename = "path/to/20240315_clip_SR_8b_clip.tif"
        date = get_date(filename)
        assert date == "20240315"

    def test_get_date_with_full_path(self):
        """Test that full paths are handled correctly."""
        filename = "/full/path/to/data/SKY_abc_def_20231201_xxx.tif"
        date = get_date(filename)
        assert date == "20231201"

    def test_get_date_planet_at_start(self):
        """Test Planet format with date at start of filename."""
        filename = "20240101_SR_8b_clip.tif"
        date = get_date(filename)
        assert date == "20240101"

    def test_get_date_different_dates(self):
        """Test that different dates are correctly extracted."""
        test_cases = [
            ("SKY_x_y_20230515_z.tif", "20230515"),
            ("20230515_SR_8b_clip.tif", "20230515"),
            ("/data/SKY_a_b_20221225_c.tif", "20221225"),
            ("path/20220101_clip.tif", "20220101"),
        ]

        for filename, expected_date in test_cases:
            assert get_date(filename) == expected_date


class TestCollectImageFiles:
    """Test image file collection."""

    def test_collect_skywatch_files(self):
        """Test collecting SkyWatch format files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_files = [
                "SKY_abc_def_20240101_xxx.tif",
                "SKY_abc_def_20240215_xxx.tif",
                "SKY_abc_def_20240330_xxx.tif",
            ]
            for f in test_files:
                Path(tmpdir, f).touch()

            # Also create a non-matching file
            Path(tmpdir, "other_file.tif").touch()

            result = collect_image_files(tmpdir + "/")

            assert len(result) == 3
            # Should be sorted by date
            assert "20240101" in result[0]
            assert "20240215" in result[1]
            assert "20240330" in result[2]

    def test_collect_planet_files(self):
        """Test collecting Planet format files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_files = [
                "20240101_xxx_SR_8b_clip.tif",
                "20240215_xxx_SR_8b_clip.tif",
                "20240330_xxx_SR_8b_clip.tif",
            ]
            for f in test_files:
                Path(tmpdir, f).touch()

            result = collect_image_files(tmpdir + "/")

            assert len(result) == 3
            # Should be sorted by date
            assert "20240101" in result[0]
            assert "20240215" in result[1]
            assert "20240330" in result[2]

    def test_collect_mixed_files(self):
        """Test collecting mixed SkyWatch and Planet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mixed test files
            test_files = [
                "SKY_abc_def_20240101_xxx.tif",
                "20240215_xxx_SR_8b_clip.tif",
                "SKY_abc_def_20240330_xxx.tif",
            ]
            for f in test_files:
                Path(tmpdir, f).touch()

            result = collect_image_files(tmpdir + "/")

            assert len(result) == 3
            # Should be sorted by date regardless of format
            assert "20240101" in result[0]
            assert "20240215" in result[1]
            assert "20240330" in result[2]

    def test_collect_empty_directory(self):
        """Test collecting from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = collect_image_files(tmpdir + "/")
            assert len(result) == 0


class TestLoadBoundaries:
    """Test boundary loading."""

    def test_load_boundaries_basic(self):
        """Test loading boundaries from GeoJSON."""
        fixtures_path = Path(__file__).parent / "fixtures" / "test_boundaries.geojson"

        boundaries_file, gdf = load_boundaries(str(fixtures_path), reset_names=False)

        assert boundaries_file == str(fixtures_path)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 2
        assert "Name" in gdf.columns
        assert gdf["Name"].tolist() == ["Field_1", "Field_2"]

    def test_load_boundaries_reset_names(self):
        """Test loading boundaries with name reset."""
        fixtures_path = Path(__file__).parent / "fixtures" / "test_boundaries.geojson"

        # Create a copy to avoid modifying the original
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_boundaries.geojson"
            import shutil
            shutil.copy(fixtures_path, test_file)

            _, gdf = load_boundaries(str(test_file), reset_names=True)

            # Names should be reset to indices
            assert gdf["Name"].tolist() == [0, 1]

    def test_load_boundaries_has_geometry(self):
        """Test that loaded boundaries have valid geometry."""
        fixtures_path = Path(__file__).parent / "fixtures" / "test_boundaries.geojson"

        _, gdf = load_boundaries(str(fixtures_path), reset_names=False)

        assert "geometry" in gdf.columns
        assert all(gdf.geometry.is_valid)
        assert gdf.geometry.geom_type.tolist() == ["Polygon", "Polygon"]
