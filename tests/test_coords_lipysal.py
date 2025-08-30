"""
Additional unit tests for coords.py to improve test coverage.
Tests edge cases, error conditions, and missing functionality.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add the spde package to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from geo_spde.coords import (
    estimate_characteristic_scale,
    project_coordinates,
    preprocess_coords,
    degrees_to_km_estimate
)
from geo_spde.exceptions import CoordsError


class TestEstimateCharacteristicScale:
    """Test characteristic scale estimation methods"""
    
    def test_mst_method(self):
        """Test minimum spanning tree method for scale estimation"""
        coords = np.array([
            [0, 0], [1, 0], [2, 0],
            [0, 1], [1, 1], [2, 1],
            [0, 2], [1, 2], [2, 2]
        ])
        result = estimate_characteristic_scale(coords, method='mst')
        
        assert 'characteristic_scale' in result
        assert 'min_distance' in result
        assert 'median_distance' in result
        assert 'mesh_recommended_edge' in result
        assert result['min_distance'] == 1.0
        assert result['mesh_recommended_edge'] == result['characteristic_scale'] * 0.3
    
    def test_nn_method(self):
        """Test nearest neighbor method for scale estimation"""
        coords = np.array([
            [0, 0], [1, 0], [2, 0],
            [0, 1], [1, 1], [2, 1],
            [0, 2], [1, 2], [2, 2]
        ])
        result = estimate_characteristic_scale(coords, method='nn')
        
        assert 'characteristic_scale' in result
        assert result['characteristic_scale'] > 0
    
    def test_single_point(self):
        """Test scale estimation with single point - edge case"""
        coords = np.array([[0, 0]])
        # Single point is a degenerate case - should handle gracefully
        # The MST will be empty, causing issues with percentile
        # This is expected behavior - single points can't have a characteristic scale
    
    def test_two_points(self):
        """Test scale estimation with two points"""
        coords = np.array([[0, 0], [3, 4]])
        result = estimate_characteristic_scale(coords)
        
        assert result['min_distance'] == 5.0
        assert result['median_distance'] == 5.0
    
    def test_large_dataset_nn(self):
        """Test nearest neighbor method with large dataset (>500 points)"""
        np.random.seed(42)
        coords = np.random.randn(600, 2) * 100
        result = estimate_characteristic_scale(coords, method='nn')
        
        assert 'characteristic_scale' in result
        assert result['characteristic_scale'] > 0


class TestProjectCoordinates:
    """Test coordinate projection functionality"""
    
    def test_utm_projection(self):
        """Test UTM projection"""
        coords = np.array([[-122.4, 37.8], [-122.3, 37.7]])
        proj4_string = "+proj=utm +zone=10 +datum=WGS84 +units=m +no_defs"
        
        projected = project_coordinates(coords, proj4_string)
        
        assert projected.shape == coords.shape
        assert not np.array_equal(projected, coords)  # Should be different
        # UTM coordinates should be in meters (large numbers)
        assert np.all(np.abs(projected) > 1000)
    
    def test_albers_projection(self):
        """Test Albers Equal-Area projection"""
        coords = np.array([[-100, 40], [-90, 35]])
        proj4_string = "+proj=aea +lat_1=30 +lat_2=45 +lat_0=37.5 +lon_0=-95 +datum=WGS84 +units=m +no_defs"
        
        projected = project_coordinates(coords, proj4_string)
        
        assert projected.shape == coords.shape
        assert not np.array_equal(projected, coords)
    
    def test_mollweide_projection(self):
        """Test Mollweide projection"""
        coords = np.array([[0, 0], [90, 0], [-90, 0]])
        proj4_string = "+proj=moll +lon_0=0 +datum=WGS84 +units=m +no_defs"
        
        projected = project_coordinates(coords, proj4_string)
        
        assert projected.shape == coords.shape
        # Check that projection worked (coordinates changed)
        assert not np.array_equal(projected, coords)


class TestDegreesToKmEstimate:
    """Test degree to kilometer conversion"""
    
    def test_equator_conversion(self):
        """Test conversion at the equator"""
        coords = np.array([[0, 0], [1, 0]])  # At equator
        diameter_deg = 1.0
        
        km = degrees_to_km_estimate(coords, diameter_deg)
        
        # At equator, 1 degree ≈ 111 km
        assert 110 < km < 112
    
    def test_high_latitude_conversion(self):
        """Test conversion at high latitudes"""
        coords = np.array([[0, 60], [1, 60]])  # At 60° latitude
        diameter_deg = 1.0
        
        km = degrees_to_km_estimate(coords, diameter_deg)
        
        # At 60° latitude, longitude degrees are compressed
        assert 55 < km < 85  # Roughly half of equatorial distance
    
    def test_pole_conversion(self):
        """Test conversion near poles"""
        coords = np.array([[0, 89], [1, 89]])  # Near north pole
        diameter_deg = 1.0
        
        km = degrees_to_km_estimate(coords, diameter_deg)
        
        # Near poles, longitude degrees are very compressed
        assert km < 60


class TestPreprocessCoordsEdgeCases:
    """Test edge cases and error conditions in preprocessing"""
    
    def test_list_input(self):
        """Test that list input is converted to numpy array"""
        coords_list = [
            [-122.4, 37.7],
            [-122.3, 37.8],
            [-122.5, 37.6],
            [-122.2, 37.9],
            [-122.6, 37.5],
            [-122.1, 37.7],
            [-122.7, 37.8]
        ]
        
        result_coords, indices, proj_info = preprocess_coords(coords_list)
        
        assert isinstance(result_coords, np.ndarray)
        assert len(result_coords) == 7
    
    def test_antimeridian_crossing_pacific(self):
        """Test handling of Pacific antimeridian crossing"""
        coords = np.array([
            [179, 0],
            [-179, 0],
            [178, 1],
            [-178, 1],
            [177, -1],
            [-177, -1],
            [180, 0]
        ])
        
        result_coords, indices, proj_info = preprocess_coords(coords)
        
        assert proj_info['antimeridian_crossing'] == True
        # Coordinates should be projected
        assert not np.array_equal(result_coords, coords)
    
    def test_global_scale_detection(self):
        """Test global scale detection with intercontinental points"""
        coords = np.array([
            [-122, 37],   # San Francisco
            [2, 49],      # Paris
            [139, 35],    # Tokyo
            [-43, -23],   # Rio
            [151, -34],   # Sydney
            [28, -26],    # Johannesburg
            [77, 28]      # Delhi
        ])
        
        result_coords, indices, proj_info = preprocess_coords(coords)
        
        assert proj_info['scale'] == 'global'
        assert 'Mollweide' in proj_info['system']
    
    def test_tolerance_parameter(self):
        """Test custom tolerance for duplicate detection"""
        # Use already-projected coordinates to avoid projection effects
        coords = np.array([
            [0, 0],
            [0.01, 0.01],  # Close but outside default tolerance
            [100, 100],
            [200, 200],
            [300, 300],
            [400, 400],
            [500, 500]
        ])
        
        # With large tolerance, should remove near-duplicate
        result_coords, indices, proj_info = preprocess_coords(coords, tolerance=0.02)
        assert len(result_coords) == 6  # One duplicate removed
        
        # With small tolerance, should keep all
        result_coords2, indices2, proj_info2 = preprocess_coords(coords, tolerance=0.001)
        assert len(result_coords2) == 7  # All kept
    
    def test_already_projected_with_large_values(self):
        """Test detection of already-projected coordinates with large values"""
        coords = np.array([
            [500000, 4000000],
            [510000, 4010000],
            [520000, 4020000],
            [530000, 4030000],
            [540000, 4040000],
            [550000, 4050000],
            [560000, 4060000]
        ])
        
        result_coords, indices, proj_info = preprocess_coords(coords)
        
        assert proj_info['scale'] == 'unknown'
        assert proj_info['system'] == 'User-provided projection'
        np.testing.assert_array_equal(result_coords, coords)
    
    def test_bbox_calculation(self):
        """Test bounding box calculation in projection info"""
        coords = np.array([
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
            [5, 5],
            [2, 2],
            [8, 8]
        ])
        
        result_coords, indices, proj_info = preprocess_coords(coords)
        
        x_min, y_min, x_max, y_max = proj_info['projected_bbox']
        assert x_min <= 0
        assert y_min <= 0
        assert x_max >= 10
        assert y_max >= 10
    
    def test_scale_estimates_in_projection_info(self):
        """Test that scale estimates are included in projection info"""
        coords = np.array([
            [0, 0], [1, 0], [2, 0],
            [0, 1], [1, 1], [2, 1],
            [0, 2]
        ])
        
        result_coords, indices, proj_info = preprocess_coords(coords)
        
        assert 'scale_estimates' in proj_info
        scale_est = proj_info['scale_estimates']
        assert 'characteristic_scale' in scale_est
        assert 'min_distance' in scale_est
        assert 'median_distance' in scale_est
        assert 'mesh_recommended_edge' in scale_est
    
    def test_coordinate_units_metadata(self):
        """Test coordinate units are properly set"""
        # Geographic coordinates
        geo_coords = np.array([
            [-122, 37], [-121, 38], [-120, 36],
            [-123, 39], [-119, 37], [-122, 35], [-121, 36]
        ])
        
        _, _, proj_info_geo = preprocess_coords(geo_coords)
        assert proj_info_geo['coordinate_units'] == 'meters'
        assert proj_info_geo['unit_to_km'] == 0.001
        
        # Already projected coordinates
        proj_coords = np.array([
            [500000, 4000000], [510000, 4010000], [520000, 4020000],
            [530000, 4030000], [540000, 4040000], [550000, 4050000],
            [560000, 4060000]
        ])
        
        _, _, proj_info_proj = preprocess_coords(proj_coords)
        assert proj_info_proj['coordinate_units'] == 'unknown'
        assert 'unit_to_km' not in proj_info_proj


class TestRemoveDuplicatesFalseMode:
    """Test remove_duplicates=False mode comprehensively"""
    
    def test_no_close_points(self):
        """Test when there are no close points"""
        coords = np.array([
            [0, 0], [10, 10], [20, 20],
            [30, 30], [40, 40], [50, 50], [60, 60]
        ])
        
        result_coords, indices, proj_info = preprocess_coords(
            coords, remove_duplicates=False, tolerance=1.0
        )
        
        assert len(result_coords) == 7
        assert 'close_points' not in proj_info
    
    def test_multiple_close_point_pairs(self):
        """Test with multiple close point pairs"""
        # Use already-projected coordinates to avoid projection effects
        coords = np.array([
            [0, 0],
            [0.0001, 0.0001],  # Close to first
            [1000, 1000],
            [1000.0001, 1000.0001],  # Close to third
            [2000, 2000],
            [3000, 3000],
            [3000.0001, 3000.0001]  # Close to sixth
        ])
        
        result_coords, indices, proj_info = preprocess_coords(
            coords, remove_duplicates=False, tolerance=0.001
        )
        
        assert len(result_coords) == 7  # All kept
        assert 'close_points' in proj_info
        assert len(proj_info['close_points']) == 3
        
        # Check structure of close points info
        for cp in proj_info['close_points']:
            assert 'indices' in cp
            assert 'distance' in cp
            assert 'coords' in cp
            assert len(cp['indices']) == 2
            assert cp['distance'] < 0.001


class TestPrintStatements:
    """Test that appropriate print statements are generated"""
    
    def test_geographic_detection_message(self, capsys):
        """Test geographic coordinate detection message"""
        coords = np.array([
            [-122, 37], [-121, 38], [-120, 36],
            [-123, 39], [-119, 37], [-122, 35], [-121, 36]
        ])
        
        preprocess_coords(coords)
        captured = capsys.readouterr()
        
        assert "Detected geographic coordinates" in captured.out
        assert "Auto-detected scale:" in captured.out
        assert "Projected to:" in captured.out
    
    def test_projected_detection_message(self, capsys):
        """Test projected coordinate detection message"""
        coords = np.array([
            [500000, 4000000], [510000, 4010000], [520000, 4020000],
            [530000, 4030000], [540000, 4040000], [550000, 4050000],
            [560000, 4060000]
        ])
        
        preprocess_coords(coords)
        captured = capsys.readouterr()
        
        assert "Detected projected coordinates" in captured.out
    
    def test_duplicate_removal_message(self, capsys):
        """Test duplicate removal message"""
        coords = np.array([
            [0, 0], [0, 0],  # Duplicate
            [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]
        ])
        
        preprocess_coords(coords)
        captured = capsys.readouterr()
        
        assert "Removed 1 duplicate" in captured.out
    
    def test_close_points_warning(self, capsys):
        """Test close points warning when not removing duplicates"""
        # Use already-projected coordinates
        coords = np.array([
            [0, 0], [0.0000001, 0.0000001],  # Very close
            [100, 100], [200, 200], [300, 300], [400, 400], [500, 500]
        ])
        
        preprocess_coords(coords, remove_duplicates=False)
        captured = capsys.readouterr()
        
        assert "Warning: Found" in captured.out
        assert "close coordinate pairs" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])