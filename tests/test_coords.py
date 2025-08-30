import pytest
import numpy as np
import sys
from pathlib import Path
import pyproj

# Add the spde package to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from geo_spde.coords import (
    is_geographic,
    detect_antimeridian_crossing,
    unwrap_antimeridian,
    compute_convex_hull_diameter,
    degrees_to_km_estimate,
    determine_projection_scale,
    determine_utm_zone,
    compute_albers_standard_parallels,
    create_projection_string,
    remove_duplicate_coords,
    preprocess_coords
)
from geo_spde.exceptions import CoordsError


class TestGeographicDetection:
    """Test geographic coordinate detection"""
    
    def test_clearly_geographic_coords(self):
        """Test detection of obvious lon/lat coordinates"""
        coords = np.array([
            [-122.4, 37.8],  # San Francisco
            [-74.0, 40.7],   # New York
            [2.3, 48.9]      # Paris
        ])
        assert is_geographic(coords) == True
    
    def test_clearly_projected_coords(self):
        """Test detection of obvious projected coordinates"""
        coords = np.array([
            [552000, 4182000],  # UTM coordinates
            [654000, 4283000],
            [456000, 4184000]
        ])
        assert is_geographic(coords) == False
    
    def test_edge_case_small_geographic_region(self):
        """Test small geographic region that might be confused with projected"""
        coords = np.array([
            [-122.40, 37.80],
            [-122.41, 37.81],
            [-122.39, 37.79]
        ])
        assert is_geographic(coords) == True
    
    def test_edge_case_projected_in_geographic_range(self):
        """Test projected coords that happen to fall in geographic range"""
        # Simulating projected coordinates with large variance
        coords = np.array([
            [50, 30],
            [150, 80],
            [-100, -50],
            [170, 85]
        ])
        # This should be detected as projected due to large variance
        assert is_geographic(coords) == False
    
    def test_coordinates_outside_geographic_bounds(self):
        """Test coordinates clearly outside lon/lat bounds"""
        coords = np.array([
            [200, 100],  # Longitude > 180
            [0, -100]    # Latitude < -90
        ])
        assert is_geographic(coords) == False


class TestAntimeridianHandling:
    """Test antimeridian crossing detection and handling"""
    
    def test_no_antimeridian_crossing(self):
        """Test coordinates that don't cross antimeridian"""
        coords = np.array([
            [-150, 60],
            [-140, 65],
            [-160, 55]
        ])
        assert detect_antimeridian_crossing(coords) == False
    
    def test_antimeridian_crossing_detected(self):
        """Test detection of antimeridian crossing"""
        coords = np.array([
            [179, 60],   # Just east of antimeridian
            [-179, 65],  # Just west of antimeridian
            [175, 55]
        ])
        assert detect_antimeridian_crossing(coords) == True
    
    def test_antimeridian_unwrapping(self):
        """Test unwrapping of antimeridian coordinates"""
        coords = np.array([
            [179, 60],
            [-179, 65],
            [-170, 55]
        ])
        unwrapped = unwrap_antimeridian(coords)
        
        # Check that negative longitudes are converted to 0-360
        expected = np.array([
            [179, 60],
            [181, 65],  # -179 + 360 = 181
            [190, 55]   # -170 + 360 = 190
        ])
        np.testing.assert_allclose(unwrapped, expected)


class TestConvexHullDiameter:
    """Test convex hull diameter calculations"""
    
    def test_normal_convex_hull(self):
        """Test diameter calculation for normal point set"""
        coords = np.array([
            [0, 0],
            [3, 0],
            [3, 4],
            [0, 4],
            [1.5, 2]  # Interior point
        ])
        diameter = compute_convex_hull_diameter(coords)
        # Should be distance from (0,0) to (3,4) = 5
        assert abs(diameter - 5.0) < 1e-6
    
    def test_two_points_only(self):
        """Test diameter with only two points"""
        coords = np.array([
            [0, 0],
            [3, 4]
        ])
        diameter = compute_convex_hull_diameter(coords)
        assert abs(diameter - 5.0) < 1e-6
    
    def test_single_point(self):
        """Test diameter with single point"""
        coords = np.array([[0, 0]])
        diameter = compute_convex_hull_diameter(coords)
        assert diameter == 0.0
    
    def test_collinear_points(self):
        """Test diameter with collinear points (degenerate hull)"""
        coords = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3]
        ])
        diameter = compute_convex_hull_diameter(coords)
        # Should fallback to maximum pairwise distance
        expected = np.sqrt((3-0)**2 + (3-0)**2)  # ≈ 4.24
        assert abs(diameter - expected) < 1e-6


class TestScaleDetection:
    """Test scale detection and projection selection"""
    
    def test_single_region_scale(self):
        """Test detection of single region scale"""
        # Small region around San Francisco (< 1000 km)
        coords = np.array([
            [-122.5, 37.7],
            [-122.3, 37.9],
            [-122.1, 37.5],
            [-122.7, 37.6],
            [-122.4, 37.8],
            [-122.6, 37.4],
            [-122.2, 37.7]
        ])
        hull_diameter = compute_convex_hull_diameter(coords)
        scale = determine_projection_scale(coords, hull_diameter)
        assert scale == 'single_region'
    
    def test_multi_region_scale(self):
        """Test detection of multi-region scale"""
        # Continental US (1000-8000 km) - updated threshold
        coords = np.array([
            [-125, 49],  # Pacific Northwest
            [-67, 45],   # Atlantic Northeast
            [-97, 25],   # Gulf Coast
            [-120, 35],  # California
            [-80, 26],   # Florida
            [-110, 45],  # Mountain West
            [-90, 40]    # Midwest
        ])
        hull_diameter = compute_convex_hull_diameter(coords)
        scale = determine_projection_scale(coords, hull_diameter)
        assert scale == 'multi_region'
    
    def test_global_scale(self):
        """Test detection of global scale"""
        # Intercontinental points (> 8000 km) - updated threshold
        coords = np.array([
            [-122, 37],  # San Francisco
            [2, 49],     # Paris
            [139, 35],   # Tokyo
            [-43, -23],  # Rio de Janeiro
            [151, -34],  # Sydney
            [77, 28],    # Delhi
            [-99, 19]    # Mexico City
        ])
        hull_diameter = compute_convex_hull_diameter(coords)
        scale = determine_projection_scale(coords, hull_diameter)
        assert scale == 'global'


class TestUTMZoneSelection:
    """Test UTM zone determination"""
    
    def test_utm_zone_calculation(self):
        """Test UTM zone calculation for various longitudes"""
        # Test known UTM zones
        test_cases = [
            ([-122, 37], 10, 'north'),  # San Francisco → Zone 10N
            ([2, 49], 31, 'north'),     # Paris → Zone 31N  
            ([139, 35], 54, 'north'),   # Tokyo → Zone 54N
            ([-43, -23], 23, 'south'),  # Rio → Zone 23S
        ]
        
        for (lon, lat), expected_zone, expected_hemisphere in test_cases:
            coords = np.array([[lon, lat]])
            zone, hemisphere = determine_utm_zone(coords)
            assert zone == expected_zone
            assert hemisphere == expected_hemisphere
    
    def test_utm_zone_edge_cases(self):
        """Test UTM zone edge cases"""
        # Test longitude at zone boundary
        coords = np.array([[-180, 45]])  # Should be zone 1
        zone, hemisphere = determine_utm_zone(coords)
        assert zone == 1
        assert hemisphere == 'north'
        
        # Test longitude at other extreme
        coords = np.array([[180, 45]])  # Should be zone 60
        zone, hemisphere = determine_utm_zone(coords)
        assert zone == 60
        assert hemisphere == 'north'


class TestAlbersProjection:
    """Test Albers projection parameter calculation"""
    
    def test_albers_standard_parallels(self):
        """Test calculation of Albers standard parallels"""
        lat_min, lat_max = 25.0, 49.0  # Continental US extent
        parallel_1, parallel_2 = compute_albers_standard_parallels(lat_min, lat_max)
        
        lat_range = lat_max - lat_min  # 24°
        expected_p1 = lat_min + lat_range / 6  # 25 + 4 = 29°
        expected_p2 = lat_max - lat_range / 6  # 49 - 4 = 45°
        
        assert abs(parallel_1 - expected_p1) < 1e-6
        assert abs(parallel_2 - expected_p2) < 1e-6


class TestProjectionStrings:
    """Test Proj4 string generation"""
    
    def test_utm_projection_string(self):
        """Test UTM projection string generation"""
        coords = np.array([[-122, 37]])  # Zone 10N
        proj4_string = create_projection_string('single_region', coords)
        
        assert '+proj=utm' in proj4_string
        assert '+zone=10' in proj4_string
        assert '+datum=WGS84' in proj4_string
        assert '+south' not in proj4_string  # Northern hemisphere
    
    def test_utm_southern_hemisphere(self):
        """Test UTM projection for southern hemisphere"""
        coords = np.array([[-43, -23]])  # Zone 23S
        proj4_string = create_projection_string('single_region', coords)
        
        assert '+zone=23' in proj4_string
        assert '+south' in proj4_string
    
    def test_albers_projection_string(self):
        """Test Albers projection string generation"""
        coords = np.array([
            [-125, 25],
            [-67, 49]
        ])
        proj4_string = create_projection_string('multi_region', coords)
        
        assert '+proj=aea' in proj4_string
        assert '+lat_1=' in proj4_string
        assert '+lat_2=' in proj4_string
        assert '+lon_0=' in proj4_string
        assert '+lat_0=' in proj4_string
    
    def test_mollweide_projection_string(self):
        """Test Mollweide projection string generation"""
        coords = np.array([
            [-122, 37],
            [139, 35]
        ])
        proj4_string = create_projection_string('global', coords)
        
        assert '+proj=moll' in proj4_string
        assert '+lon_0=' in proj4_string


class TestDuplicateRemoval:
    """Test duplicate coordinate removal"""
    
    def test_no_duplicates(self):
        """Test with coordinates that have no duplicates"""
        coords = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3]
        ])
        unique_coords, indices, n_duplicates = remove_duplicate_coords(coords)
        
        np.testing.assert_array_equal(unique_coords, coords)
        np.testing.assert_array_equal(indices, np.arange(4))
        assert n_duplicates == 0
    
    def test_exact_duplicates(self):
        """Test removal of exact duplicate coordinates"""
        coords = np.array([
            [0, 0],
            [1, 1],
            [0, 0],  # Exact duplicate
            [2, 2]
        ])
        unique_coords, indices, n_duplicates = remove_duplicate_coords(coords)
        
        expected_coords = np.array([[0, 0], [1, 1], [2, 2]])
        expected_indices = np.array([0, 1, 3])
        
        np.testing.assert_array_equal(unique_coords, expected_coords)
        np.testing.assert_array_equal(indices, expected_indices)
        assert n_duplicates == 1
    
    def test_near_duplicates_within_tolerance(self):
        """Test removal of near-duplicate coordinates within tolerance"""
        coords = np.array([
            [0, 0],
            [1, 1],
            [0.0000005, 0.0000005],  # Within default tolerance
            [2, 2]
        ])
        unique_coords, indices, n_duplicates = remove_duplicate_coords(coords)
        
        assert len(unique_coords) == 3
        assert n_duplicates == 1
    
    def test_near_duplicates_outside_tolerance(self):
        """Test that near-duplicates outside tolerance are kept"""
        coords = np.array([
            [0, 0],
            [1, 1],
            [0.001, 0.001],  # Outside default tolerance
            [2, 2]
        ])
        unique_coords, indices, n_duplicates = remove_duplicate_coords(coords, tolerance=1e-6)
        
        assert len(unique_coords) == 4
        assert n_duplicates == 0


class TestInputValidation:
    """Test input validation and error handling"""
    
    def test_invalid_array_shape(self):
        """Test error for invalid coordinate array shape"""
        coords = np.array([1, 2, 3])  # 1D array
        
        with pytest.raises(CoordsError, match="Expected coords shape \\(n_obs, 2\\)"):
            preprocess_coords(coords)
    
    def test_wrong_number_of_columns(self):
        """Test error for wrong number of columns"""
        coords = np.array([[1, 2, 3], [4, 5, 6]])  # 3 columns
        
        with pytest.raises(CoordsError, match="Expected coords shape \\(n_obs, 2\\)"):
            preprocess_coords(coords)
    
    def test_too_few_coordinates(self):
        """Test error for too few coordinates"""
        coords = np.array([
            [0, 0],
            [1, 1],
            [2, 2]  # Only 3 points, need minimum 7
        ])
        
        with pytest.raises(CoordsError, match="Minimum 7 coordinates required"):
            preprocess_coords(coords)
    
    def test_nan_coordinates(self):
        """Test error for NaN coordinates"""
        coords = np.array([
            [0, 0],
            [1, np.nan],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [6, 6]
        ])
        
        with pytest.raises(CoordsError, match="contain NaN or infinite values"):
            preprocess_coords(coords)
    
    def test_infinite_coordinates(self):
        """Test error for infinite coordinates"""
        coords = np.array([
            [0, 0],
            [1, 1],
            [np.inf, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [6, 6]
        ])
        
        with pytest.raises(CoordsError, match="contain NaN or infinite values"):
            preprocess_coords(coords)


class TestFullPreprocessing:
    """Test the full preprocessing pipeline"""
    
    @pytest.fixture
    def valid_geographic_coords(self):
        """Fixture for valid geographic coordinates"""
        return np.array([
            [-122.4, 37.7],  # San Francisco area
            [-122.3, 37.8],
            [-122.5, 37.6],
            [-122.2, 37.9],
            [-122.6, 37.5],
            [-122.1, 37.7],
            [-122.7, 37.8]
        ])
    
    @pytest.fixture
    def valid_projected_coords(self):
        """Fixture for valid projected coordinates"""
        return np.array([
            [552000, 4182000],
            [554000, 4184000],
            [556000, 4186000],
            [558000, 4188000],
            [560000, 4190000],
            [562000, 4192000],
            [564000, 4194000]
        ])
    
    def test_geographic_coordinates_processing(self, valid_geographic_coords):
        """Test processing of geographic coordinates"""
        result_coords, indices, proj_info = preprocess_coords(valid_geographic_coords)
        
        # Check return values
        assert len(result_coords) == 7
        assert len(indices) == 7
        assert 'proj4_string' in proj_info
        assert 'system' in proj_info
        assert 'scale' in proj_info
        assert proj_info['scale'] == 'single_region'
    
    def test_projected_coordinates_processing(self, valid_projected_coords):
        """Test processing of already projected coordinates"""
        result_coords, indices, proj_info = preprocess_coords(valid_projected_coords)
        
        # Should use coordinates as-is
        np.testing.assert_array_equal(result_coords, valid_projected_coords)
        assert len(indices) == 7
        assert proj_info['scale'] == 'unknown'
        assert proj_info['system'] == 'User-provided projection'
    
    def test_coordinates_with_duplicates(self):
        """Test processing coordinates with duplicates using already-projected coords"""
        # Use already-projected coordinates to avoid mocking
        coords = np.array([
            [552000, 4182000],
            [554000, 4184000],
            [552000, 4182000],  # Duplicate
            [556000, 4186000],
            [558000, 4188000],
            [560000, 4190000],
            [562000, 4192000],
            [564000, 4194000]
        ])
        
        result_coords, indices, proj_info = preprocess_coords(coords)
        
        # Should have 7 unique coordinates (one duplicate removed)
        assert len(result_coords) == 7
        assert len(indices) == 7
        # Index 2 (duplicate) should not be in kept indices
        assert 2 not in indices
    
    def test_remove_duplicates_false(self):
        """Test processing with remove_duplicates=False keeps all points"""
        # Use already-projected coordinates
        coords = np.array([
            [552000, 4182000],
            [554000, 4184000],
            [552000, 4182000],  # Duplicate
            [556000, 4186000],
            [558000, 4188000],
            [560000, 4190000],
            [562000, 4192000],
            [564000, 4194000]
        ])
        
        result_coords, indices, proj_info = preprocess_coords(coords, remove_duplicates=False)
        
        # Should keep all 8 coordinates
        assert len(result_coords) == 8
        assert len(indices) == 8
        np.testing.assert_array_equal(indices, np.arange(8))
        
        # Should have close_points info in projection_info
        assert 'close_points' in proj_info
        assert len(proj_info['close_points']) == 1
        
        # Check the close point pair
        close_point = proj_info['close_points'][0]
        assert close_point['indices'] == (0, 2)
        assert close_point['distance'] < 1e-6
    
    def test_remove_duplicates_false_with_near_points(self):
        """Test processing with remove_duplicates=False identifies near points"""
        # Create projected coordinates with small differences
        coords = np.array([
            [552000.0, 4182000.0],
            [552000.0005, 4182000.0005],  # Very close but not identical
            [554000, 4184000],
            [556000, 4186000],
            [558000, 4188000],
            [560000, 4190000],
            [562000, 4192000],
            [564000, 4194000]
        ])
        
        result_coords, indices, proj_info = preprocess_coords(
            coords, 
            tolerance=0.001,  # 1mm tolerance
            remove_duplicates=False
        )
        
        # Should keep all 8 coordinates
        assert len(result_coords) == 8
        assert len(indices) == 8
        
        # Should identify the close points
        assert 'close_points' in proj_info
        assert len(proj_info['close_points']) == 1
        
        close_point = proj_info['close_points'][0]
        assert close_point['indices'] == (0, 1)
        # Distance should be approximately 0.0007 (sqrt(0.0005^2 + 0.0005^2))
        assert 0.0006 < close_point['distance'] < 0.0008
    
    def test_remove_duplicates_true_default_behavior(self):
        """Test that remove_duplicates=True is the default"""
        # Use already-projected coordinates
        coords = np.array([
            [552000, 4182000],
            [554000, 4184000],
            [552000, 4182000],  # Duplicate
            [556000, 4186000],
            [558000, 4188000],
            [560000, 4190000],
            [562000, 4192000],
            [564000, 4194000]
        ])
        
        # Test without specifying remove_duplicates (should default to True)
        result_coords_default, indices_default, proj_info_default = preprocess_coords(coords)
        
        # Test with explicit remove_duplicates=True
        result_coords_explicit, indices_explicit, proj_info_explicit = preprocess_coords(
            coords, remove_duplicates=True
        )
        
        # Both should have same behavior
        np.testing.assert_array_equal(result_coords_default, result_coords_explicit)
        np.testing.assert_array_equal(indices_default, indices_explicit)
        assert len(result_coords_default) == 7
        assert 2 not in indices_default


class TestReturnValueStructure:
    """Test the structure and content of return values"""
    
    def test_projection_info_structure(self):
        """Test that projection_info contains all required keys"""
        # Use already-projected coordinates
        coords = np.array([
            [552000, 4182000],
            [554000, 4184000],
            [556000, 4186000],
            [558000, 4188000],
            [560000, 4190000],
            [562000, 4192000],
            [564000, 4194000]
        ])
        
        result_coords, indices, proj_info = preprocess_coords(coords)
        
        # Check all required keys are present
        required_keys = {
            'proj4_string', 'system', 'scale', 'projected_bbox', 
            'hull_diameter_km', 'antimeridian_crossing',
            'scale_estimates', 'transform_info', 'normalized', 
            'coordinate_units'
        }
        assert set(proj_info.keys()) == required_keys
        
        # Check data types
        assert isinstance(proj_info['proj4_string'], str)
        assert isinstance(proj_info['system'], str)
        assert isinstance(proj_info['scale'], str)
        assert isinstance(proj_info['projected_bbox'], tuple)
        assert isinstance(proj_info['hull_diameter_km'], (int, float))
        assert isinstance(proj_info['antimeridian_crossing'], bool)
        
        # Check bbox structure
        assert len(proj_info['projected_bbox']) == 4
        x_min, y_min, x_max, y_max = proj_info['projected_bbox']
        assert x_min <= x_max
        assert y_min <= y_max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
