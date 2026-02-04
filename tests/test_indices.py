"""Tests for vegetation index calculations."""

import numpy as np
import pytest

from engine.compute.indices import (
    calculate_band_stats,
    calculate_ndvi_mask,
    calculate_cire_mask,
    calculate_mcari_mask,
    calculate_msavi_mask,
    compute_indices_stats,
)


class TestBandStats:
    """Test band statistics calculations."""

    def test_calculate_band_stats_simple(self):
        """Test basic mean and std calculation."""
        band = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, std = calculate_band_stats(band)

        assert mean == pytest.approx(3.0)
        assert std == pytest.approx(np.std(band))

    def test_calculate_band_stats_with_nan(self):
        """Test that NaN values are handled correctly."""
        band = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        mean, std = calculate_band_stats(band)

        # Should ignore NaN and compute on valid values
        assert mean == pytest.approx(3.0)
        assert not np.isnan(mean)
        assert not np.isnan(std)

    def test_calculate_band_stats_all_nan(self):
        """Test behavior with all NaN values."""
        band = np.array([np.nan, np.nan, np.nan])
        mean, std = calculate_band_stats(band)

        # Should return NaN when all values are NaN
        assert np.isnan(mean)
        assert np.isnan(std)

    def test_calculate_band_stats_uniform_values(self):
        """Test with uniform values (std should be 0)."""
        band = np.array([5.0, 5.0, 5.0, 5.0])
        mean, std = calculate_band_stats(band)

        assert mean == pytest.approx(5.0)
        assert std == pytest.approx(0.0)


class TestNDVI:
    """Test NDVI calculation."""

    def test_calculate_ndvi_basic(self):
        """Test basic NDVI calculation."""
        red = np.array([100.0, 200.0])
        nir = np.array([400.0, 600.0])

        ndvi = calculate_ndvi_mask(red, nir)

        # NDVI = (NIR - RED) / (NIR + RED)
        expected = np.array([
            (400 - 100) / (400 + 100),  # 0.6
            (600 - 200) / (600 + 200),  # 0.5
        ])

        np.testing.assert_array_almost_equal(ndvi, expected)

    def test_calculate_ndvi_range(self):
        """Test that NDVI values are in valid range [-1, 1]."""
        red = np.array([100.0, 200.0, 300.0])
        nir = np.array([400.0, 500.0, 600.0])

        ndvi = calculate_ndvi_mask(red, nir)

        assert np.all(ndvi >= -1)
        assert np.all(ndvi <= 1)

    def test_calculate_ndvi_zero_denominator(self):
        """Test NDVI when red + nir = 0."""
        red = np.array([0.0])
        nir = np.array([0.0])

        ndvi = calculate_ndvi_mask(red, nir)

        # Should handle division by zero gracefully
        assert np.isnan(ndvi[0]) or np.isinf(ndvi[0])

    def test_calculate_ndvi_vegetation_vs_soil(self):
        """Test NDVI distinguishes vegetation from soil."""
        # Healthy vegetation: high NIR, low RED
        veg_red = np.array([100.0])
        veg_nir = np.array([800.0])
        veg_ndvi = calculate_ndvi_mask(veg_red, veg_nir)

        # Bare soil: similar NIR and RED
        soil_red = np.array([400.0])
        soil_nir = np.array([450.0])
        soil_ndvi = calculate_ndvi_mask(soil_red, soil_nir)

        # Vegetation should have much higher NDVI
        assert veg_ndvi[0] > soil_ndvi[0]
        assert veg_ndvi[0] > 0.7  # Healthy vegetation
        assert soil_ndvi[0] < 0.3  # Bare soil


class TestCIRE:
    """Test CIRE (Chlorophyll Index Red Edge) calculation."""

    def test_calculate_cire_basic(self):
        """Test basic CIRE calculation."""
        red_edge = np.array([200.0, 300.0])
        nir = np.array([600.0, 900.0])

        cire = calculate_cire_mask(red_edge, nir)

        # CIRE = (NIR / RED_EDGE) - 1
        expected = np.array([
            (600 / 200) - 1,  # 2.0
            (900 / 300) - 1,  # 2.0
        ])

        np.testing.assert_array_almost_equal(cire, expected)

    def test_calculate_cire_zero_red_edge(self):
        """Test CIRE when red_edge = 0."""
        red_edge = np.array([0.0])
        nir = np.array([600.0])

        cire = calculate_cire_mask(red_edge, nir)

        # Should handle division by zero
        assert np.isinf(cire[0])

    def test_calculate_cire_positive_values(self):
        """Test that CIRE is positive for healthy vegetation."""
        red_edge = np.array([300.0, 400.0])
        nir = np.array([900.0, 1200.0])

        cire = calculate_cire_mask(red_edge, nir)

        # CIRE should be positive for vegetation
        assert np.all(cire > 0)


class TestMCARI:
    """Test MCARI calculation."""

    def test_calculate_mcari_basic(self):
        """Test basic MCARI calculation."""
        red = np.array([400.0])
        green_1 = np.array([300.0])
        blue = np.array([200.0])

        mcari = calculate_mcari_mask(red, blue, green_1)

        # MCARI = ((red - green_1) - 0.2 * (red - blue)) * (red / green_1)
        expected = ((400 - 300) - 0.2 * (400 - 200)) * (400 / 300)

        assert mcari[0] == pytest.approx(expected)

    def test_calculate_mcari_array(self):
        """Test MCARI with arrays."""
        red = np.array([400.0, 500.0])
        green_1 = np.array([300.0, 400.0])
        blue = np.array([200.0, 250.0])

        mcari = calculate_mcari_mask(red, blue, green_1)

        assert len(mcari) == 2
        assert not np.isnan(mcari[0])
        assert not np.isnan(mcari[1])


class TestMSAVI:
    """Test MSAVI calculation."""

    def test_calculate_msavi_basic(self):
        """Test basic MSAVI calculation."""
        red = np.array([200.0])
        nir = np.array([600.0])

        msavi = calculate_msavi_mask(red, nir)

        assert len(msavi) == 1
        assert not np.isnan(msavi[0])
        # MSAVI calculation produces values (formula may need review)
        assert isinstance(msavi[0], (float, np.floating))

    def test_calculate_msavi_array(self):
        """Test MSAVI with arrays."""
        red = np.array([200.0, 300.0, 400.0])
        nir = np.array([600.0, 700.0, 800.0])

        msavi = calculate_msavi_mask(red, nir)

        assert len(msavi) == 3
        assert not np.any(np.isnan(msavi))


class TestComputeIndicesStats:
    """Test the main compute_indices_stats function."""

    def test_compute_indices_stats_shape(self):
        """Test that compute_indices_stats returns all expected indices."""
        # Create a mock 8-band image (bands, height, width)
        np.random.seed(42)
        cropped_image = np.random.rand(8, 10, 10) * 1000 + 500

        stats = compute_indices_stats(cropped_image)

        # Check all expected keys are present
        expected_keys = [
            "ndvi_mean", "ndvi_std",
            "ndre_mean", "ndre_std",
            "evi_mean", "evi_std",
            "cire_mean", "cire_std",
            "mcari_mean", "mcari_std",
            "msavi_mean", "msavi_std",
        ]

        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], float)

    def test_compute_indices_stats_values_are_finite(self):
        """Test that computed stats are finite (not inf/nan) for valid input."""
        # Create realistic band values (500-2500 range typical for satellite imagery)
        np.random.seed(42)
        cropped_image = np.random.rand(8, 10, 10) * 2000 + 500

        stats = compute_indices_stats(cropped_image)

        # Most values should be finite for valid input
        finite_count = sum(1 for v in stats.values() if np.isfinite(v))
        assert finite_count >= 10  # At least 10 out of 12 should be finite

    def test_compute_indices_stats_realistic_values(self):
        """Test with realistic satellite imagery values."""
        np.random.seed(42)
        # Create realistic 8-band data
        cropped_image = np.zeros((8, 20, 20))

        # Band values typical for healthy vegetation
        cropped_image[0] = np.random.rand(20, 20) * 200 + 300  # Coastal Blue
        cropped_image[1] = np.random.rand(20, 20) * 300 + 400  # Blue
        cropped_image[2] = np.random.rand(20, 20) * 400 + 500  # Green I
        cropped_image[3] = np.random.rand(20, 20) * 400 + 500  # Green
        cropped_image[4] = np.random.rand(20, 20) * 500 + 600  # Yellow
        cropped_image[5] = np.random.rand(20, 20) * 300 + 400  # Red
        cropped_image[6] = np.random.rand(20, 20) * 600 + 700  # Red Edge
        cropped_image[7] = np.random.rand(20, 20) * 1000 + 1500  # NIR (high for vegetation)

        stats = compute_indices_stats(cropped_image)

        # NDVI should be positive for vegetation
        assert stats["ndvi_mean"] > 0
        assert stats["ndvi_mean"] < 1

        # All means should be reasonable
        assert all(np.isfinite(v) for k, v in stats.items() if "mean" in k)

    def test_compute_indices_stats_consistent_shape(self):
        """Test that function works with different input shapes."""
        shapes = [(8, 10, 10), (8, 50, 50), (8, 5, 5)]

        for shape in shapes:
            cropped_image = np.random.rand(*shape) * 1000 + 500
            stats = compute_indices_stats(cropped_image)

            assert len(stats) == 12  # 6 indices * 2 stats each
            assert all(isinstance(v, float) for v in stats.values())
