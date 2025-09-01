import pytest
import numpy as np
import sys
from pathlib import Path

# Add the spde package to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from geo_spde.mesh import SPDEMesh
from geo_spde.exceptions import MeshError


class TestSPDEMeshInitialization:
    """Test SPDEMesh initialization and validation"""
    
    @pytest.fixture
    def valid_coords(self):
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
    
    @pytest.fixture
    def valid_projection_info(self):
        """Fixture for projection metadata"""
        return {
            'proj4_string': '+proj=utm +zone=10 +datum=WGS84 +units=m +no_defs',
            'system': 'UTM Zone 10N',
            'scale': 'single_region',
            'projected_bbox': (552000, 4182000, 564000, 4194000),
            'hull_diameter_km': 16.97,
            'antimeridian_crossing': False
        }
    
    def test_valid_initialization(self, valid_coords, valid_projection_info):
        """Test successful initialization with valid inputs"""
        mesh = SPDEMesh(valid_coords, valid_projection_info)
        
        assert mesh.coords.shape == (7, 2)
        assert mesh.projection_info == valid_projection_info
        assert mesh.mesh_params is None
        assert mesh.vertices is None
        assert mesh.triangles is None
        assert mesh.diagnostics is None
    
    def test_initialization_without_projection_info(self, valid_coords):
        """Test initialization without projection info"""
        mesh = SPDEMesh(valid_coords)
        
        assert mesh.coords.shape == (7, 2)
        assert mesh.projection_info == {}
    
    def test_initialization_with_list_input(self, valid_projection_info):
        """Test initialization with list input (should convert to numpy)"""
        coords_list = [
            [552000, 4182000],
            [554000, 4184000],
            [556000, 4186000],
            [558000, 4188000]
        ]
        
        mesh = SPDEMesh(coords_list, valid_projection_info)
        assert isinstance(mesh.coords, np.ndarray)
        assert mesh.coords.shape == (4, 2)
    
    def test_invalid_coord_shape_1d(self):
        """Test error for 1D coordinate array"""
        coords = np.array([1, 2, 3, 4])
        
        with pytest.raises(MeshError, match="Expected coords shape \\(n_obs, 2\\)"):
            SPDEMesh(coords)
    
    def test_invalid_coord_shape_3d(self):
        """Test error for 3D coordinate array"""
        coords = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        
        with pytest.raises(MeshError, match="Expected coords shape \\(n_obs, 2\\)"):
            SPDEMesh(coords)
    
    def test_invalid_coord_columns(self):
        """Test error for wrong number of columns"""
        coords = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        
        with pytest.raises(MeshError, match="Expected coords shape \\(n_obs, 2\\)"):
            SPDEMesh(coords)
    
    def test_too_few_coordinates(self):
        """Test error for insufficient coordinates"""
        coords = np.array([
            [552000, 4182000],
            [554000, 4184000]  # Only 2 points
        ])
        
        with pytest.raises(MeshError, match="Need at least 3 coordinates"):
            SPDEMesh(coords)



class TestBoundaryCreation:
    """Test boundary polygon creation"""
    
    def test_boundary_square_coords(self):
        """Test boundary creation with square coordinates"""
        coords = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        mesh = SPDEMesh(coords)
        
        boundary = mesh._create_extended_boundary(extension_factor=0.2)
        
        # Should have 4 points (convex hull of square)
        assert boundary.shape[0] == 4
        assert boundary.shape[1] == 2
        
        # Check that boundary is extended beyond original points
        assert boundary[:, 0].min() < coords[:, 0].min()
        assert boundary[:, 0].max() > coords[:, 0].max()
        assert boundary[:, 1].min() < coords[:, 1].min()
        assert boundary[:, 1].max() > coords[:, 1].max()
    
    def test_boundary_extension_factor(self):
        """Test different extension factors"""
        coords = np.array([
            [0, 0],
            [2, 0],
            [2, 2],
            [0, 2]
        ])
        mesh = SPDEMesh(coords)
        
        boundary1 = mesh._create_extended_boundary(extension_factor=0.1)
        boundary2 = mesh._create_extended_boundary(extension_factor=0.5)
        
        # Larger extension should create larger boundary
        range1 = boundary1.max() - boundary1.min()
        range2 = boundary2.max() - boundary2.min()
        assert range2 > range1
    
    def test_boundary_triangle_coords(self):
        """Test boundary creation with triangular coordinates"""
        coords = np.array([
            [0, 0],
            [2, 0],
            [1, 2]
        ])
        mesh = SPDEMesh(coords)
        
        boundary = mesh._create_extended_boundary(extension_factor=0.2)
        
        # Should have 3 points (convex hull of triangle)
        assert boundary.shape[0] == 3
        assert boundary.shape[1] == 2


class TestMeshGeneration:
    """Test mesh generation functionality"""
    
    @pytest.fixture
    def simple_mesh(self):
        """Fixture for simple square mesh"""
        coords = np.array([
            [1, 1],
            [3, 1],
            [3, 3],
            [1, 3],
            [2, 2]  # Interior point
        ])
        return SPDEMesh(coords)
    
    def test_basic_mesh_generation(self, simple_mesh):
        """Test basic mesh generation"""
        vertices, triangles = simple_mesh.create_adaptive_mesh(verbose=False)
        
        # Check output types and shapes
        assert isinstance(vertices, np.ndarray)
        assert isinstance(triangles, np.ndarray)
        assert vertices.shape[1] == 2
        assert triangles.shape[1] == 3
        assert triangles.dtype == np.int32
        
        # Should have at least the original coordinates
        assert len(vertices) >= len(simple_mesh.coords)
        
        # Check that mesh contains original points (within tolerance)
        for orig_point in simple_mesh.coords:
            distances = np.linalg.norm(vertices - orig_point, axis=1)
            assert np.min(distances) < 1e-6
    
    def test_mesh_generation_with_custom_boundary(self, simple_mesh):
        """Test mesh generation with custom boundary"""
        custom_boundary = np.array([
            [0, 0],
            [4, 0],
            [4, 4],
            [0, 4]
        ])
        
        vertices, triangles = simple_mesh.create_adaptive_mesh(
            boundary=custom_boundary,
            verbose=False
        )
        
        # Check that boundary points are included
        for boundary_point in custom_boundary:
            distances = np.linalg.norm(vertices - boundary_point, axis=1)
            assert np.min(distances) < 1e-6
    
    def test_mesh_generation_different_parameters(self, simple_mesh):
        """Test adaptive mesh generation with different parameters"""
        # Mesh with larger edges far from observations
        vertices1, triangles1 = simple_mesh.create_adaptive_mesh(
            max_edge_far=10.0,
            verbose=False
        )
        
        # Mesh with smaller edges far from observations
        vertices2, triangles2 = simple_mesh.create_adaptive_mesh(
            max_edge_far=2.0,
            verbose=False
        )
        
        # Should both complete successfully
        assert len(vertices1) > 0
        assert len(vertices2) > 0
    
    def test_adaptive_mesh_generation(self, simple_mesh):
        """Test adaptive mesh generation functionality"""
        vertices, triangles = simple_mesh.create_adaptive_mesh(verbose=False)
        
        # Check output types and shapes
        assert isinstance(vertices, np.ndarray)
        assert isinstance(triangles, np.ndarray)
        assert vertices.shape[1] == 2
        assert triangles.shape[1] == 3
        assert triangles.dtype == np.int32
        
        # Should have at least the original coordinates
        assert len(vertices) >= len(simple_mesh.coords)
        
        # Check that mesh contains original points
        for orig_point in simple_mesh.coords:
            distances = np.linalg.norm(vertices - orig_point, axis=1)
            assert np.min(distances) < 1e-6
    
    def test_adaptive_mesh_with_custom_params(self, simple_mesh):
        """Test adaptive mesh with custom parameters"""
        vertices, triangles = simple_mesh.create_adaptive_mesh(
            min_edge_near_obs=0.1,
            max_edge_far=5.0,
            transition_distance=2.0,
            verbose=False
        )
        
        # Should complete successfully
        assert len(vertices) > 0
        assert len(triangles) > 0
        assert vertices.shape[1] == 2
        assert triangles.shape[1] == 3
    
    def test_mesh_generation_min_angle_constraint(self, simple_mesh):
        """Test that minimum angle constraint is respected"""
        vertices, triangles = simple_mesh.create_adaptive_mesh(
            min_angle=25.0,
            verbose=False
        )
        
        # Compute minimum angles in triangles
        min_angles = []
        for tri in triangles:
            v0, v1, v2 = vertices[tri]
            
            # Compute angles
            edges = np.array([v1 - v0, v2 - v1, v0 - v2])
            for i in range(3):
                e1 = edges[i]
                e2 = -edges[(i+1) % 3]
                cos_angle = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                min_angles.append(angle)
        
        min_angles = np.array(min_angles)
        # Allow small numerical tolerance
        assert np.min(min_angles) >= 24.0


class TestMeshDiagnostics:
    """Test mesh diagnostics computation"""
    
    @pytest.fixture
    def mesh_with_triangles(self):
        """Fixture for mesh with generated triangles"""
        coords = np.array([
            [0, 0],
            [2, 0],
            [2, 2],
            [0, 2],
            [1, 1]
        ])
        mesh = SPDEMesh(coords)
        mesh.create_adaptive_mesh(verbose=False)
        return mesh
    
    def test_diagnostics_computation(self, mesh_with_triangles):
        """Test diagnostics computation"""
        diagnostics = mesh_with_triangles.diagnostics
        
        # Check all required keys are present
        expected_keys = {
            'n_vertices', 'n_triangles', 'n_observations',
            'mesh_to_obs_ratio', 'mean_triangle_area', 'total_area',
            'mesh_density', 'memory_estimate_mb'
        }
        assert set(diagnostics.keys()) == expected_keys
        
        # Check data types
        for key, value in diagnostics.items():
            assert isinstance(value, (int, float))
        
        # Check logical constraints
        assert diagnostics['n_vertices'] >= diagnostics['n_observations']
        assert diagnostics['n_triangles'] > 0
        assert diagnostics['mesh_to_obs_ratio'] >= 1.0
        assert diagnostics['mean_triangle_area'] > 0
        assert diagnostics['total_area'] > 0
        assert diagnostics['memory_estimate_mb'] > 0
    
    def test_get_mesh_info_before_generation(self):
        """Test error when getting mesh info before generation"""
        coords = np.array([[0, 0], [1, 1], [2, 0]])
        mesh = SPDEMesh(coords)
        
        with pytest.raises(MeshError, match="Mesh not yet generated"):
            mesh.get_mesh_info()
    
    def test_get_mesh_info_after_generation(self, mesh_with_triangles):
        """Test mesh info retrieval after generation"""
        info = mesh_with_triangles.get_mesh_info()
        
        # Check all required keys are present
        expected_keys = {
            'mesh_params', 'diagnostics', 'projection_info',
            'n_vertices', 'n_triangles', 'n_observations'
        }
        assert set(info.keys()) == expected_keys
        
        # Check consistency
        assert info['n_vertices'] == len(mesh_with_triangles.vertices)
        assert info['n_triangles'] == len(mesh_with_triangles.triangles)
        assert info['n_observations'] == len(mesh_with_triangles.coords)


class TestMeshQuality:
    """Test mesh quality and edge cases"""
    
    def test_mesh_with_collinear_points(self):
        """Test mesh generation with collinear points"""
        coords = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [0, 1]  # Off the line to make triangulation possible
        ])
        mesh = SPDEMesh(coords)
        
        # Should not crash
        vertices, triangles = mesh.create_adaptive_mesh(verbose=False)
        assert len(vertices) >= 5
        assert len(triangles) >= 1
    
    def test_mesh_with_very_close_points(self):
        """Test mesh generation with very close points"""
        coords = np.array([
            [0, 0],
            [0.000001, 0.000001],  # Very close to first point
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        
        # Should remove the very close point during initialization
        with pytest.warns(UserWarning, match="Removed .* coordinates that were closer"):
            mesh = SPDEMesh(coords)
        
        # Should have removed the close point
        assert len(mesh.coords) == 4  # One point removed
        
        # Should handle remaining points gracefully
        vertices, triangles = mesh.create_adaptive_mesh(verbose=False)
        assert len(vertices) >= 4
    
    def test_mesh_with_large_coordinates(self):
        """Test mesh generation with large coordinate values"""
        coords = np.array([
            [1e6, 1e6],
            [1e6 + 1000, 1e6],
            [1e6 + 1000, 1e6 + 1000],
            [1e6, 1e6 + 1000]
        ])
        mesh = SPDEMesh(coords)
        
        vertices, triangles = mesh.create_adaptive_mesh(verbose=False)
        assert len(vertices) >= 4


class TestMeshOutput:
    """Test mesh output formatting and warnings"""
    
    def test_verbose_output(self, capsys):
        """Test verbose output during mesh generation"""
        coords = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        mesh = SPDEMesh(coords)
        
        vertices, triangles = mesh.create_adaptive_mesh(verbose=True)
        
        captured = capsys.readouterr()
        assert "Creating adaptive mesh for 4 observation points" in captured.out
        assert "Min edge (near obs):" in captured.out
        assert "Refinement ratio:" in captured.out
    
    def test_large_mesh_warning(self, capsys):
        """Test warning for large mesh"""
        # Create a mesh that will have many vertices
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        
        mesh = SPDEMesh(coords)
        vertices, triangles = mesh.create_adaptive_mesh(
            min_edge_near_obs=0.01,  # Fine mesh
            verbose=True
        )
        
        captured = capsys.readouterr()
        # Should warn about large mesh or high ratio
        assert ("WARNING" in captured.out and 
                ("Large mesh" in captured.out or "High mesh/observation ratio" in captured.out))


class TestSpatialScaleEstimation:
    """Test spatial scale estimation methods"""
    
    @pytest.fixture
    def regular_grid_mesh(self):
        """Create mesh with regular grid pattern"""
        x = np.linspace(0, 10, 10)
        y = np.linspace(0, 10, 10)
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        return SPDEMesh(coords)
    
    @pytest.fixture
    def sparse_mesh(self):
        """Create mesh with sparse random points"""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (15, 2))
        return SPDEMesh(coords)
    
    @pytest.fixture
    def dense_mesh(self):
        """Create mesh with dense points"""
        np.random.seed(42)
        coords = np.random.uniform(0, 10, (200, 2))
        return SPDEMesh(coords)
    
    def test_estimate_spatial_scale_variogram(self, regular_grid_mesh):
        """Test spatial scale estimation using variogram method"""
        result = regular_grid_mesh.estimate_spatial_scale(method='variogram')
        
        assert 'estimated_range' in result
        assert 'min_distance' in result
        assert 'median_distance' in result
        assert 'practical_range' in result
        assert 'method_used' in result
        assert result['method_used'] == 'variogram'
        
        # Check reasonable values
        assert result['estimated_range'] > 0
        assert result['min_distance'] > 0
        assert result['median_distance'] > result['min_distance']
        assert result['practical_range'] == result['estimated_range'] * 3
    
    def test_estimate_spatial_scale_mst(self, regular_grid_mesh):
        """Test spatial scale estimation using MST method"""
        result = regular_grid_mesh.estimate_spatial_scale(method='mst')
        
        assert 'estimated_range' in result
        assert 'method_used' in result
        assert result['method_used'] == 'mst'
        assert result['estimated_range'] > 0
    
    def test_estimate_spatial_scale_both(self, regular_grid_mesh):
        """Test spatial scale estimation using both methods"""
        result = regular_grid_mesh.estimate_spatial_scale(method='both')
        
        assert 'estimated_range' in result
        assert 'min_distance' in result
        assert 'median_distance' in result
        
        # Should prefer variogram if it works
        if 'method_used' in result:
            assert result['method_used'] in ['variogram', 'mst']
        
        # If variogram worked, should also have mst_range
        if result.get('method_used') == 'variogram':
            assert 'mst_range' in result
    
    def test_estimate_scale_large_dataset(self, dense_mesh):
        """Test spatial scale estimation with large dataset (triggers subsampling)"""
        result = dense_mesh.estimate_spatial_scale(method='variogram', max_pairs=1000)
        
        assert 'estimated_range' in result
        assert result['estimated_range'] > 0
    
    def test_estimate_scale_sparse_data(self, sparse_mesh):
        """Test spatial scale estimation with sparse data"""
        result = sparse_mesh.estimate_spatial_scale(method='both')
        
        assert 'estimated_range' in result
        assert result['estimated_range'] > 0
    
    def test_variogram_insufficient_data(self):
        """Test variogram with insufficient data points"""
        coords = np.array([[0, 0], [1, 1], [2, 2]])  # Only 3 points
        mesh = SPDEMesh(coords)
        
        # Should fallback to MST
        result = mesh.estimate_spatial_scale(method='variogram')
        assert result['method_used'] == 'mst'
    
    def test_estimate_scale_stores_result(self, regular_grid_mesh):
        """Test that spatial scale estimation stores results"""
        result = regular_grid_mesh.estimate_spatial_scale()
        
        assert hasattr(regular_grid_mesh, 'spatial_scale')
        assert regular_grid_mesh.spatial_scale == result


class TestVariogramEstimation:
    """Test variogram-based range estimation"""
    
    def test_variogram_regular_pattern(self):
        """Test variogram with regular spatial pattern"""
        # Create a denser grid to ensure enough data for variogram
        x = np.linspace(0, 10, 8)
        y = np.linspace(0, 10, 8)
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        mesh = SPDEMesh(coords)
        
        range_est = mesh._estimate_range_variogram(max_pairs=1000)
        assert range_est > 0
        assert range_est < 20  # Should be reasonable for this scale
    
    def test_variogram_with_subsampling(self):
        """Test variogram with automatic subsampling"""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (100, 2))
        mesh = SPDEMesh(coords)
        
        # Use enough pairs to avoid insufficient data error
        range_est = mesh._estimate_range_variogram(max_pairs=500)
        assert range_est > 0
    
    def test_variogram_error_handling(self):
        """Test variogram error handling with insufficient data"""
        # Very few points that should cause fitting issues
        coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        mesh = SPDEMesh(coords)
        
        # Should raise ValueError for insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            mesh._estimate_range_variogram()
        
        # Test with slightly more points that might still have fitting issues
        coords2 = np.array([
            [0, 0], [1, 0], [2, 0],
            [0, 1], [1, 1], [2, 1],
            [0, 2], [1, 2], [2, 2]
        ])
        mesh2 = SPDEMesh(coords2)
        # This should work or fallback gracefully
        try:
            range_est = mesh2._estimate_range_variogram()
            assert range_est > 0
        except ValueError:
            # If it still fails, that's ok - the method correctly identifies insufficient data
            pass


class TestMSTEstimation:
    """Test MST-based spatial scale estimation"""
    
    def test_mst_basic(self):
        """Test basic MST estimation"""
        coords = np.array([
            [0, 0], [1, 0], [0, 1], [1, 1],
            [2, 0], [2, 1], [0, 2], [1, 2], [2, 2]
        ])
        mesh = SPDEMesh(coords)
        
        scale = mesh._estimate_range_mst()
        assert scale > 0
        assert scale <= 2  # Should be around grid spacing
    
    def test_mst_large_dataset(self):
        """Test MST with large dataset (triggers subsampling)"""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (600, 2))
        mesh = SPDEMesh(coords)
        
        scale = mesh._estimate_range_mst()
        assert scale > 0
    
    def test_mst_irregular_pattern(self):
        """Test MST with irregular spatial pattern"""
        np.random.seed(42)
        # Create clustered pattern
        cluster1 = np.random.normal(0, 1, (10, 2))
        cluster2 = np.random.normal(10, 1, (10, 2))
        coords = np.vstack([cluster1, cluster2])
        mesh = SPDEMesh(coords)
        
        scale = mesh._estimate_range_mst()
        assert scale > 0


class TestMeshResolutionValidation:
    """Test mesh resolution validation"""
    
    @pytest.fixture
    def mesh_with_scale(self):
        """Create mesh with known spatial scale"""
        coords = np.array([
            [0, 0], [2, 0], [4, 0],
            [0, 2], [2, 2], [4, 2],
            [0, 4], [2, 4], [4, 4]
        ])
        mesh = SPDEMesh(coords)
        mesh.create_adaptive_mesh(min_edge_near_obs=0.5, verbose=False)
        return mesh
    
    def test_validate_resolution_basic(self, mesh_with_scale):
        """Test basic resolution validation"""
        validation = mesh_with_scale.validate_mesh_resolution(verbose=False)
        
        assert 'resolution_ok' in validation
        assert 'extent_ok' in validation
        assert 'edge_to_range_ratio' in validation
        assert 'recommended_edge_factor' in validation
        assert 'mesh_extent' in validation
        
        assert isinstance(validation['resolution_ok'], (bool, np.bool_))
        assert isinstance(validation['extent_ok'], (bool, np.bool_))
        assert validation['edge_to_range_ratio'] > 0
    
    def test_validate_resolution_too_coarse(self):
        """Test validation when mesh is too coarse"""
        coords = np.array([
            [0, 0], [10, 0], [20, 0],
            [0, 10], [10, 10], [20, 10],
            [0, 20], [10, 20], [20, 20]
        ])
        mesh = SPDEMesh(coords)
        mesh.create_adaptive_mesh(min_edge_near_obs=2.0, verbose=False)  # Very coarse
        
        validation = mesh.validate_mesh_resolution(verbose=False)
        
        # Should detect that mesh is too coarse
        if validation['edge_to_range_ratio'] > 0.5:
            assert not validation['resolution_ok']
            assert validation['recommended_edge_factor'] is not None
            assert validation['recommended_edge_factor'] < 0.5
    
    def test_validate_resolution_with_warning(self):
        """Test validation warning output"""
        import warnings
        
        coords = np.array([
            [0, 0], [5, 0], [10, 0],
            [0, 5], [5, 5], [10, 5],
            [0, 10], [5, 10], [10, 10]
        ])
        mesh = SPDEMesh(coords)
        mesh.create_adaptive_mesh(min_edge_near_obs=3.0, verbose=False)  # Too coarse
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validation = mesh.validate_mesh_resolution(verbose=True)
            
            if not validation['resolution_ok']:
                assert len(w) == 1
                assert "Mesh may be too coarse" in str(w[0].message)
    
    def test_validate_before_mesh_creation(self):
        """Test validation before mesh is created"""
        coords = np.array([[0, 0], [1, 1], [2, 2]])
        mesh = SPDEMesh(coords)
        
        with pytest.raises(MeshError, match="Mesh parameters not computed"):
            mesh.validate_mesh_resolution()
    
    def test_validation_stores_result(self, mesh_with_scale):
        """Test that validation stores results"""
        validation = mesh_with_scale.validate_mesh_resolution(verbose=False)
        
        assert hasattr(mesh_with_scale, 'mesh_validation')
        assert mesh_with_scale.mesh_validation == validation


class TestSPDEParameterSuggestions:
    """Test SPDE parameter suggestions"""
    
    @pytest.fixture
    def standard_mesh(self):
        """Create standard mesh for testing"""
        coords = np.array([
            [0, 0], [5, 0], [10, 0], [15, 0],
            [0, 5], [5, 5], [10, 5], [15, 5],
            [0, 10], [5, 10], [10, 10], [15, 10],
            [0, 15], [5, 15], [10, 15], [15, 15]
        ])
        return SPDEMesh(coords)
    
    def test_suggest_parameters_basic(self, standard_mesh):
        """Test basic parameter suggestions"""
        suggestions = standard_mesh.suggest_spde_parameters()
        
        # Check all required keys
        assert 'kappa_suggestion' in suggestions
        assert 'tau_suggestion' in suggestions
        assert 'spatial_range_suggestion' in suggestions
        assert 'spatial_sd_suggestion' in suggestions
        assert 'prior_range_kappa' in suggestions
        assert 'prior_range_tau' in suggestions
        assert 'mesh_extent' in suggestions
        assert 'range_to_extent_ratio' in suggestions
        
        # Check reasonable values
        assert suggestions['kappa_suggestion'] > 0
        assert suggestions['tau_suggestion'] > 0
        assert suggestions['spatial_range_suggestion'] > 0
        assert suggestions['spatial_sd_suggestion'] > 0
        
        # Check prior ranges
        kappa_lower, kappa_upper = suggestions['prior_range_kappa']
        assert kappa_lower < suggestions['kappa_suggestion'] < kappa_upper
        assert kappa_upper / kappa_lower == 9  # 3x range each direction
        
        tau_lower, tau_upper = suggestions['prior_range_tau']
        assert tau_lower < suggestions['tau_suggestion'] < tau_upper
        assert tau_upper / tau_lower == 9
    
    def test_suggest_parameters_small_domain(self):
        """Test parameter suggestions for small domain"""
        coords = np.array([
            [0, 0], [0.5, 0], [1, 0],
            [0, 0.5], [0.5, 0.5], [1, 0.5],
            [0, 1], [0.5, 1], [1, 1]
        ])
        mesh = SPDEMesh(coords)
        
        suggestions = mesh.suggest_spde_parameters()
        
        # Small domain should have appropriate parameters
        assert suggestions['mesh_extent'] < 2
        assert suggestions['range_to_extent_ratio'] > 0
        assert suggestions['range_to_extent_ratio'] < 1
    
    def test_suggest_parameters_large_domain(self):
        """Test parameter suggestions for large domain"""
        coords = np.array([
            [0, 0], [100, 0], [200, 0],
            [0, 100], [100, 100], [200, 100],
            [0, 200], [100, 200], [200, 200]
        ])
        mesh = SPDEMesh(coords)
        
        suggestions = mesh.suggest_spde_parameters()
        
        # Large domain should have appropriate scaling
        assert suggestions['mesh_extent'] > 100
        assert suggestions['spatial_range_suggestion'] <= suggestions['mesh_extent'] * 0.3
        assert suggestions['spatial_range_suggestion'] >= suggestions['mesh_extent'] * 0.05
    
    def test_parameter_suggestions_stores_result(self, standard_mesh):
        """Test that parameter suggestions are stored"""
        suggestions = standard_mesh.suggest_spde_parameters()
        
        assert hasattr(standard_mesh, 'parameter_suggestions')
        assert standard_mesh.parameter_suggestions == suggestions


class TestAdaptiveMesh:
    """Test adaptive mesh generation functionality"""
    
    @pytest.fixture
    def clustered_coords(self):
        """Create coordinates with clustered pattern"""
        np.random.seed(42)
        # Create three clusters
        cluster1 = np.random.normal([10, 10], 2, (10, 2))
        cluster2 = np.random.normal([30, 10], 2, (10, 2))
        cluster3 = np.random.normal([20, 30], 2, (10, 2))
        sparse_points = np.random.uniform(0, 40, (5, 2))
        return np.vstack([cluster1, cluster2, cluster3, sparse_points])
    
    @pytest.fixture
    def uniform_coords(self):
        """Create uniformly distributed coordinates"""
        x = np.linspace(0, 20, 7)
        y = np.linspace(0, 20, 7)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])
    
    @pytest.fixture
    def sparse_coords(self):
        """Create sparse, irregularly distributed coordinates"""
        np.random.seed(42)
        return np.random.uniform(0, 100, (8, 2))
    
    def test_adaptive_mesh_basic(self, clustered_coords):
        """Test basic adaptive mesh generation"""
        mesh = SPDEMesh(clustered_coords)
        vertices, triangles = mesh.create_adaptive_mesh(verbose=False)
        
        assert isinstance(vertices, np.ndarray)
        assert isinstance(triangles, np.ndarray)
        assert vertices.shape[1] == 2
        assert triangles.shape[1] == 3
        assert len(vertices) >= len(clustered_coords)
        
        # Check that original points are included
        for coord in clustered_coords:
            distances = np.linalg.norm(vertices - coord, axis=1)
            assert np.min(distances) < 1e-6
    
    def test_adaptive_mesh_custom_parameters(self, uniform_coords):
        """Test adaptive mesh with custom edge parameters"""
        mesh = SPDEMesh(uniform_coords)
        
        vertices, triangles = mesh.create_adaptive_mesh(
            min_edge_near_obs=0.5,
            max_edge_far=5.0,
            transition_distance=3.0,
            verbose=False
        )
        
        # Compute edge lengths
        edge_lengths = []
        for tri in triangles:
            for i in range(3):
                j = (i + 1) % 3
                edge = np.linalg.norm(vertices[tri[j]] - vertices[tri[i]])
                edge_lengths.append(edge)
        
        edge_lengths = np.array(edge_lengths)
        
        # Check that edge lengths are within reasonable bounds
        assert edge_lengths.min() >= 0.1  # Not too small
        assert edge_lengths.max() <= 10.0  # Not too large
    
    def test_adaptive_mesh_with_boundary(self, sparse_coords):
        """Test adaptive mesh with custom boundary"""
        mesh = SPDEMesh(sparse_coords)
        
        # Create a large rectangular boundary
        boundary = np.array([
            [-10, -10],
            [110, -10],
            [110, 110],
            [-10, 110]
        ])
        
        vertices, triangles = mesh.create_adaptive_mesh(
            boundary=boundary,
            verbose=False
        )
        
        # Check boundary points are included
        for bpoint in boundary:
            distances = np.linalg.norm(vertices - bpoint, axis=1)
            assert np.min(distances) < 1e-6
    
    def test_adaptive_parameter_computation(self, clustered_coords):
        """Test automatic parameter computation for adaptive mesh"""
        mesh = SPDEMesh(clustered_coords)
        params = mesh._compute_adaptive_parameters()
        
        assert 'min_edge_near_obs' in params
        assert 'max_edge_far' in params
        assert 'transition_distance' in params
        
        # Check parameter relationships
        assert params['min_edge_near_obs'] > 0
        assert params['max_edge_far'] > params['min_edge_near_obs']
        assert params['transition_distance'] > 0
        
        # Check reasonable ratio
        ratio = params['max_edge_far'] / params['min_edge_near_obs']
        assert 2 <= ratio <= 50
    
    def test_adaptive_mesh_single_cluster(self):
        """Test adaptive mesh with single dense cluster"""
        np.random.seed(42)
        coords = np.random.normal([50, 50], 1, (20, 2))
        mesh = SPDEMesh(coords)
        
        vertices, triangles = mesh.create_adaptive_mesh(verbose=False)
        
        # Should create reasonable mesh even with single cluster
        assert len(vertices) >= len(coords)
        assert len(triangles) > 0
    
    def test_adaptive_mesh_extreme_spacing(self):
        """Test adaptive mesh with extreme spacing variations"""
        # Very dense cluster and very sparse points
        dense = np.random.normal([0, 0], 0.1, (15, 2))
        sparse = np.array([[100, 100], [100, -100], [-100, 100], [-100, -100]])
        coords = np.vstack([dense, sparse])
        
        mesh = SPDEMesh(coords)
        vertices, triangles = mesh.create_adaptive_mesh(verbose=False)
        
        # Should handle extreme variations gracefully
        assert len(vertices) >= len(coords)
        assert len(triangles) > 0
    
    def test_adaptive_mesh_diagnostics(self, uniform_coords, capsys):
        """Test diagnostic output for adaptive mesh"""
        mesh = SPDEMesh(uniform_coords)
        vertices, triangles = mesh.create_adaptive_mesh(verbose=True)
        
        captured = capsys.readouterr()
        assert "Creating adaptive mesh" in captured.out
        assert "Min edge (near obs)" in captured.out
        assert "Max edge (far)" in captured.out
        assert "Transition distance" in captured.out
        assert "Edge length distribution" in captured.out
        assert "Refinement ratio" in captured.out
    
    def test_adaptive_mesh_edge_distribution(self, clustered_coords):
        """Test that adaptive mesh has appropriate edge length distribution"""
        mesh = SPDEMesh(clustered_coords)
        vertices, triangles = mesh.create_adaptive_mesh(verbose=False)
        
        # Build KDTree for distance queries
        from scipy.spatial import KDTree
        obs_tree = KDTree(clustered_coords)
        
        # Analyze triangles by distance to observations
        near_edges = []
        far_edges = []
        
        for tri in triangles:
            # Get triangle centroid
            tri_verts = vertices[tri]
            centroid = np.mean(tri_verts, axis=0)
            
            # Distance to nearest observation
            dist_to_obs, _ = obs_tree.query([centroid], k=1)
            dist_to_obs = dist_to_obs[0]
            
            # Compute edge lengths
            for i in range(3):
                j = (i + 1) % 3
                edge_length = np.linalg.norm(tri_verts[j] - tri_verts[i])
                
                if dist_to_obs < 5:  # Near observations
                    near_edges.append(edge_length)
                elif dist_to_obs > 15:  # Far from observations
                    far_edges.append(edge_length)
        
        if len(near_edges) > 0 and len(far_edges) > 0:
            # Near edges should generally be smaller than far edges
            assert np.median(near_edges) < np.median(far_edges)
    
    def test_triangle_area_computation(self):
        """Test triangle area calculation"""
        mesh = SPDEMesh(np.array([[0, 0], [1, 0], [0, 1]]))
        
        # Right triangle with base=1, height=1
        vertices = np.array([[0, 0], [1, 0], [0, 1]])
        area = mesh._triangle_area(vertices)
        assert abs(area - 0.5) < 1e-10
        
        # Equilateral triangle
        vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
        area = mesh._triangle_area(vertices)
        expected = np.sqrt(3) / 4
        assert abs(area - expected) < 1e-10
    
    def test_intermediate_points_generation(self, sparse_coords):
        """Test generation of intermediate transition points"""
        from scipy.spatial import KDTree
        
        mesh = SPDEMesh(sparse_coords)
        boundary = mesh._create_extended_boundary(0.2)
        obs_tree = KDTree(sparse_coords)
        
        intermediate = mesh._generate_intermediate_points(
            boundary, obs_tree, transition_dist=10.0, max_edge=5.0
        )
        
        assert isinstance(intermediate, list)
        # Should generate some intermediate points for transition
        if len(intermediate) > 0:
            # Check that points are in transition zone
            for point in intermediate:
                dist_to_obs, _ = obs_tree.query([point], k=1)
                # Should be in reasonable transition range
                assert dist_to_obs[0] > 0
    
    def test_adaptive_mesh_min_angle_constraint(self, uniform_coords):
        """Test that minimum angle constraint is respected in adaptive mesh"""
        mesh = SPDEMesh(uniform_coords)
        vertices, triangles = mesh.create_adaptive_mesh(
            min_angle=25.0,
            verbose=False
        )
        
        # Check minimum angles
        min_angles = []
        for tri in triangles:
            v0, v1, v2 = vertices[tri]
            
            # Compute angles
            edges = np.array([v1 - v0, v2 - v1, v0 - v2])
            for i in range(3):
                e1 = edges[i]
                e2 = -edges[(i+1) % 3]
                cos_angle = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                min_angles.append(angle)
        
        # Allow small tolerance for numerical precision
        assert np.min(min_angles) >= 24.0
    
    def test_adaptive_mesh_error_handling(self):
        """Test error handling in adaptive parameter computation"""
        # Test with too few points - should raise error at initialization
        coords = np.array([[0, 0], [1, 1]])
        
        with pytest.raises(MeshError, match="Need at least 3 coordinates"):
            mesh = SPDEMesh(coords)
    
    def test_adaptive_vs_regular_mesh_comparison(self, clustered_coords):
        """Compare adaptive mesh with regular mesh"""
        mesh = SPDEMesh(clustered_coords)
        
        # Generate regular mesh
        regular_verts, regular_tris = mesh.create_adaptive_mesh(
            min_edge_near_obs=0.5,
            verbose=False
        )
        
        # Generate adaptive mesh
        adaptive_verts, adaptive_tris = mesh.create_adaptive_mesh(verbose=False)
        
        # Adaptive mesh should have different characteristics
        # Generally fewer vertices in sparse areas
        assert len(adaptive_verts) != len(regular_verts)
        
        # Both should cover the observation points
        for coord in clustered_coords:
            reg_distances = np.linalg.norm(regular_verts - coord, axis=1)
            adapt_distances = np.linalg.norm(adaptive_verts - coord, axis=1)
            assert np.min(reg_distances) < 1e-6
            assert np.min(adapt_distances) < 1e-6


class TestScaleDiagnostics:
    """Test comprehensive scale diagnostics"""
    
    @pytest.fixture
    def complete_mesh(self):
        """Create mesh with all components"""
        np.random.seed(42)
        coords = np.random.uniform(0, 50, (30, 2))
        mesh = SPDEMesh(coords)
        mesh.create_adaptive_mesh(verbose=False)
        return mesh
    
    def test_compute_scale_diagnostics_basic(self, complete_mesh):
        """Test basic scale diagnostics computation"""
        diagnostics = complete_mesh.compute_scale_diagnostics(verbose=False)
        
        # Check structure
        assert 'spatial_scale' in diagnostics
        assert 'validation' in diagnostics
        assert 'suggestions' in diagnostics
        
        # Check spatial scale content
        assert 'estimated_range' in diagnostics['spatial_scale']
        assert 'min_distance' in diagnostics['spatial_scale']
        
        # Check validation content
        assert 'resolution_ok' in diagnostics['validation']
        assert 'extent_ok' in diagnostics['validation']
        
        # Check suggestions content
        assert 'kappa_suggestion' in diagnostics['suggestions']
        assert 'tau_suggestion' in diagnostics['suggestions']
    
    def test_scale_diagnostics_verbose(self, complete_mesh, capsys):
        """Test verbose output of scale diagnostics"""
        diagnostics = complete_mesh.compute_scale_diagnostics(verbose=True)
        
        captured = capsys.readouterr()
        assert "Scale Diagnostics:" in captured.out
        assert "Estimated correlation range:" in captured.out
        assert "Mesh resolution adequate:" in captured.out
        assert "Suggested kappa:" in captured.out
        assert "Suggested tau:" in captured.out
        assert "spatial_range:" in captured.out
        assert "spatial_sd:" in captured.out
    
    def test_scale_diagnostics_before_mesh(self):
        """Test error when computing diagnostics before mesh creation"""
        coords = np.array([[0, 0], [1, 1], [2, 2]])
        mesh = SPDEMesh(coords)
        
        with pytest.raises(MeshError, match="Mesh must be created"):
            mesh.compute_scale_diagnostics()
    
    def test_scale_diagnostics_stores_results(self, complete_mesh):
        """Test that scale diagnostics are stored"""
        diagnostics = complete_mesh.compute_scale_diagnostics(verbose=False)
        
        assert hasattr(complete_mesh, 'scale_diagnostics')
        assert complete_mesh.scale_diagnostics == diagnostics
        
        # Also check that individual components are stored
        assert hasattr(complete_mesh, 'spatial_scale')
        assert hasattr(complete_mesh, 'mesh_validation')
        assert hasattr(complete_mesh, 'parameter_suggestions')
    
    def test_scale_diagnostics_integration(self):
        """Test full integration of scale diagnostics pipeline"""
        # Create a realistic scenario
        np.random.seed(42)
        # Simulate clustered spatial data
        cluster_centers = [[10, 10], [30, 10], [20, 30]]
        coords = []
        for center in cluster_centers:
            cluster = np.random.normal(center, 3, (20, 2))
            coords.append(cluster)
        coords = np.vstack(coords)
        
        mesh = SPDEMesh(coords)
        mesh.create_adaptive_mesh(min_edge_near_obs=0.4, verbose=False)
        
        diagnostics = mesh.compute_scale_diagnostics(verbose=False)
        
        # Check that all components worked together
        assert diagnostics['spatial_scale']['estimated_range'] > 0
        assert isinstance(diagnostics['validation']['resolution_ok'], (bool, np.bool_))
        assert diagnostics['suggestions']['kappa_suggestion'] > 0
        
        # Check relationships make sense
        range_est = diagnostics['spatial_scale']['estimated_range']
        kappa = diagnostics['suggestions']['kappa_suggestion']
        # For Matérn nu=1/2: kappa = sqrt(8)/range
        expected_kappa = np.sqrt(8) / diagnostics['suggestions']['spatial_range_suggestion']
        assert abs(kappa - expected_kappa) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
