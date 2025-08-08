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


class TestMeshParameterCalculation:
    """Test mesh parameter calculation"""
    
    @pytest.fixture
    def mesh(self):
        """Fixture for mesh with square grid coordinates"""
        coords = np.array([
            [0, 0],
            [1, 0],
            [2, 0],
            [0, 1],
            [1, 1],
            [2, 1],
            [0, 2],
            [1, 2],
            [2, 2]
        ], dtype=float)
        return SPDEMesh(coords)
    
    def test_parameter_calculation_basic(self, mesh):
        """Test basic parameter calculation"""
        params = mesh._compute_mesh_parameters(target_edge_factor=0.5, max_area=None)
        
        # Check that all required keys are present
        expected_keys = {'max_edge', 'max_area', 'cutoff', 'min_distance', 'median_distance'}
        assert set(params.keys()) == expected_keys
        
        # Check data types
        for value in params.values():
            assert isinstance(value, (int, float))
        
        # Check logical relationships
        assert params['max_area'] == (params['max_edge'] ** 2) / 2.0
        assert params['cutoff'] == params['min_distance'] * 0.1
        assert params['max_edge'] == params['median_distance'] * 0.5
    
    def test_parameter_calculation_with_custom_area(self, mesh):
        """Test parameter calculation with custom max_area"""
        custom_area = 100.0
        params = mesh._compute_mesh_parameters(target_edge_factor=0.5, max_area=custom_area)
        
        assert params['max_area'] == custom_area
    
    def test_parameter_calculation_different_edge_factor(self, mesh):
        """Test parameter calculation with different edge factor"""
        params1 = mesh._compute_mesh_parameters(target_edge_factor=0.3, max_area=None)
        params2 = mesh._compute_mesh_parameters(target_edge_factor=0.7, max_area=None)
        
        assert params1['max_edge'] < params2['max_edge']
        assert params1['max_area'] < params2['max_area']
        # min_distance, median_distance, cutoff should be the same
        assert params1['min_distance'] == params2['min_distance']
        assert params1['median_distance'] == params2['median_distance']
        assert params1['cutoff'] == params2['cutoff']
    
    def test_parameter_calculation_identical_coords(self):
        """Test error when all coordinates are identical"""
        coords = np.array([
            [1, 1],
            [1, 1],
            [1, 1]
        ])
        mesh = SPDEMesh(coords)
        
        with pytest.raises(MeshError, match="All coordinates are identical"):
            mesh._compute_mesh_parameters(0.5, None)


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
        vertices, triangles = simple_mesh.create_mesh(verbose=False)
        
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
        
        vertices, triangles = simple_mesh.create_mesh(
            boundary=custom_boundary,
            verbose=False
        )
        
        # Check that boundary points are included
        for boundary_point in custom_boundary:
            distances = np.linalg.norm(vertices - boundary_point, axis=1)
            assert np.min(distances) < 1e-6
    
    def test_mesh_generation_different_parameters(self, simple_mesh):
        """Test mesh generation with different parameters"""
        # Coarse mesh
        vertices1, triangles1 = simple_mesh.create_mesh(
            target_edge_factor=1.0,
            verbose=False
        )
        
        # Fine mesh
        vertices2, triangles2 = simple_mesh.create_mesh(
            target_edge_factor=0.2,
            verbose=False
        )
        
        # Fine mesh should have more vertices
        assert len(vertices2) > len(vertices1)
        assert len(triangles2) > len(triangles1)
    
    def test_mesh_generation_min_angle_constraint(self, simple_mesh):
        """Test that minimum angle constraint is respected"""
        vertices, triangles = simple_mesh.create_mesh(
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
        mesh.create_mesh(verbose=False)
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
        vertices, triangles = mesh.create_mesh(verbose=False)
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
        mesh = SPDEMesh(coords)
        
        # Should handle close points gracefully
        vertices, triangles = mesh.create_mesh(verbose=False)
        assert len(vertices) >= 5
    
    def test_mesh_with_large_coordinates(self):
        """Test mesh generation with large coordinate values"""
        coords = np.array([
            [1e6, 1e6],
            [1e6 + 1000, 1e6],
            [1e6 + 1000, 1e6 + 1000],
            [1e6, 1e6 + 1000]
        ])
        mesh = SPDEMesh(coords)
        
        vertices, triangles = mesh.create_mesh(verbose=False)
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
        
        vertices, triangles = mesh.create_mesh(verbose=True)
        
        captured = capsys.readouterr()
        assert "Creating mesh for 4 observation points" in captured.out
        assert "Target edge length" in captured.out
        assert "Mesh Generation Complete" in captured.out
    
    def test_large_mesh_warning(self, capsys):
        """Test warning for large mesh"""
        # Create a mesh that will have many vertices
        x = np.linspace(0, 10, 30)
        y = np.linspace(0, 10, 30)
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        
        mesh = SPDEMesh(coords)
        vertices, triangles = mesh.create_mesh(
            target_edge_factor=0.1,  # Fine mesh
            verbose=True
        )
        
        captured = capsys.readouterr()
        # Should warn about large mesh or high ratio
        assert ("WARNING" in captured.out and 
                ("Large mesh" in captured.out or "High mesh/observation ratio" in captured.out))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])