"""
Unit tests for geo_spde.stan_spde module.

Tests the main StanSPDE API including initialization, Stan data preparation,
prior mode suggestions, and diagnostic reporting.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from geo_spde.stan_spde import (
    StanSPDE,
    sparse_to_stan_csr
)
from geo_spde.exceptions import MeshError, CoordsError
from scipy.sparse import csr_matrix


class TestStanSPDEInitialization:
    """Test StanSPDE initialization and setup."""
    
    @pytest.fixture
    def valid_coords_raw(self):
        """Fixture for raw coordinates (geographic)."""
        return np.array([
            [-122.4, 37.8],  # San Francisco
            [-122.3, 37.9],
            [-122.2, 37.7],
            [-122.1, 37.6],
            [-122.0, 37.5],
            [-121.9, 37.4],
            [-121.8, 37.3],
            [-121.7, 37.2]
        ])
    
    @pytest.fixture
    def valid_observations(self):
        """Fixture for observation values."""
        return np.array([50.2, 48.1, 52.3, 49.8, 51.1, 47.5, 53.2, 46.8])
    
    def test_initialization_success(self, valid_coords_raw, valid_observations):
        """Test successful initialization with valid inputs."""
        spde = StanSPDE(
            coords_raw=valid_coords_raw,
            y_obs=valid_observations,
            verbose=False
        )
        
        assert spde.coords_raw.shape == (8, 2)
        assert spde.y_obs.shape == (8,)
        assert spde.alpha == 2  # Default smoothness
        assert spde.extension_factor == 0.2  # Default extension
        assert spde.domain_knowledge is None
        
        # Check processed coordinates exist
        assert spde.coords_clean is not None
        assert spde.y_clean is not None
        assert spde.proj_info is not None
        
        # Check mesh is created
        assert spde.mesh is not None
        assert spde.vertices is not None
        assert spde.triangles is not None
    
    def test_initialization_with_domain_knowledge(self, valid_coords_raw, valid_observations):
        """Test initialization with domain knowledge."""
        spde = StanSPDE(
            coords_raw=valid_coords_raw,
            y_obs=valid_observations,
            domain_knowledge="environmental",
            verbose=False
        )
        
        assert spde.domain_knowledge == "environmental"
        assert spde.suggested_prior in ["tight", "medium", "wide"]
    
    def test_initialization_invalid_coords_shape(self, valid_observations):
        """Test initialization with invalid coordinate shape."""
        invalid_coords = np.array([1, 2, 3])  # 1D array
        
        with pytest.raises((ValueError, MeshError, CoordsError)):
            StanSPDE(invalid_coords, valid_observations, verbose=False)
    
    def test_initialization_mismatched_lengths(self, valid_coords_raw):
        """Test initialization with mismatched coordinate/observation lengths."""
        short_obs = np.array([1, 2])  # Only 2 observations for 5 coordinates
        
        with pytest.raises((ValueError, IndexError)):
            StanSPDE(valid_coords_raw, short_obs, verbose=False)


class TestStanDataPreparation:
    """Test Stan data preparation functionality."""
    
    @pytest.fixture
    def simple_spde(self):
        """Create a simple StanSPDE instance for testing."""
        coords = np.array([
            [-122.4, 37.8], [-122.3, 37.9], [-122.2, 37.7],
            [-122.1, 37.6], [-122.0, 37.5], [-121.9, 37.4],
            [-121.8, 37.3], [-121.7, 37.2]
        ])
        y_obs = np.array([50, 48, 52, 49, 51, 47, 53, 45])
        return StanSPDE(coords, y_obs, verbose=False)
    
    def test_prepare_stan_data_auto_mode(self, simple_spde):
        """Test Stan data preparation with automatic prior mode."""
        stan_data = simple_spde.prepare_stan_data(prior_mode="auto")
        
        # Check required fields exist
        required_fields = [
            'N_obs', 'N_mesh', 'y',
            'A_nnz', 'A_w', 'A_v', 'A_u',
            'Q_nnz', 'Q_w', 'Q_v', 'Q_u', 'log_det_Q_base',
            'estimated_range', 'estimated_sigma',
            'data_sd', 'coordinate_units_to_km', 'prior_mode', 'spatial_fraction'
        ]
        
        for field in required_fields:
            assert field in stan_data, f"Missing required field: {field}"
        
        # Check data types and shapes
        assert isinstance(stan_data['N_obs'], int)
        assert isinstance(stan_data['N_mesh'], int)
        assert isinstance(stan_data['y'], list)
        assert len(stan_data['y']) == stan_data['N_obs']
        assert stan_data['prior_mode'] == 0  # auto = 0
    
    def test_prepare_stan_data_tight_mode(self, simple_spde):
        """Test Stan data preparation with tight prior mode."""
        stan_data = simple_spde.prepare_stan_data(
            prior_mode="tight",
            spatial_fraction=0.7
        )
        
        assert stan_data['prior_mode'] == 1  # tight = 1
        assert stan_data['spatial_fraction'] == 0.7
    
    def test_prepare_stan_data_custom_range(self, simple_spde):
        """Test Stan data preparation with custom range."""
        stan_data = simple_spde.prepare_stan_data(
            prior_mode="custom",
            user_range_km=50.0
        )
        
        assert stan_data['prior_mode'] == 0  # custom maps to 0
        # Custom range should affect estimated_range
        assert stan_data['estimated_range'] > 0
    
    def test_get_prior_report(self, simple_spde):
        """Test prior configuration report generation."""
        simple_spde.prepare_stan_data()
        report = simple_spde.get_prior_report()
        
        assert "SPDE Prior Configuration Report" in report
        assert "Prior Mode:" in report
        assert "Spatial Range (median):" in report
        assert "Mesh Information:" in report
    
    def test_get_diagnostics(self, simple_spde):
        """Test diagnostic information retrieval."""
        mesh_diag = simple_spde.get_mesh_diagnostics()
        scale_diag = simple_spde.get_scale_diagnostics()
        
        assert isinstance(mesh_diag, dict)
        assert isinstance(scale_diag, dict)
        
        # Check key diagnostic fields
        assert 'n_vertices' in mesh_diag
        assert 'n_triangles' in mesh_diag
        assert 'spatial_scale' in scale_diag


class TestPriorModesuggestion:
    """Test prior mode suggestion functionality through StanSPDE API."""
    
    def test_suggest_prior_mode_environmental(self):
        """Test prior mode suggestion for environmental data."""
        coords = np.array([
            [-122.4, 37.8], [-122.3, 37.9], [-122.2, 37.7],
            [-122.1, 37.6], [-122.0, 37.5], [-121.9, 37.4],
            [-121.8, 37.3], [-121.7, 37.2]
        ])
        y_obs = np.random.randn(8)
        
        spde = StanSPDE(coords, y_obs, domain_knowledge="environmental", verbose=False)
        assert spde.suggested_prior == "medium"
    
    def test_suggest_prior_mode_disease(self):
        """Test prior mode suggestion for disease data."""
        coords = np.array([
            [-122.4, 37.8], [-122.3, 37.9], [-122.2, 37.7],
            [-122.1, 37.6], [-122.0, 37.5], [-121.9, 37.4],
            [-121.8, 37.3], [-121.7, 37.2]
        ])
        y_obs = np.random.randn(8)
        
        spde = StanSPDE(coords, y_obs, domain_knowledge="disease", verbose=False)
        assert spde.suggested_prior == "tight"
    
    def test_suggest_prior_mode_data_driven(self):
        """Test data-driven prior mode suggestion."""
        coords = np.array([
            [-122.4, 37.8], [-122.3, 37.9], [-122.2, 37.7],
            [-122.1, 37.6], [-122.0, 37.5], [-121.9, 37.4],
            [-121.8, 37.3], [-121.7, 37.2]
        ])
        y_obs = np.random.randn(8)
        
        spde = StanSPDE(coords, y_obs, verbose=False)
        assert spde.suggested_prior in ["wide", "medium", "tight"]


class TestUtilityFunctions:
    """Test utility functions for Stan data preparation."""
    
    def test_sparse_to_stan_csr(self):
        """Test conversion of scipy sparse matrix to Stan CSR format."""
        # Create simple sparse matrix
        data = np.array([1, 2, 3, 4])
        row = np.array([0, 0, 1, 2])
        col = np.array([0, 2, 1, 2])
        matrix = csr_matrix((data, (row, col)), shape=(3, 3))
        
        row_ptr, col_idx, values = sparse_to_stan_csr(matrix)
        
        # Check 1-based indexing for Stan
        assert np.min(row_ptr) == 1
        assert np.min(col_idx) == 1
        assert len(values) == len(data)
    
    def test_compute_log_det_q_base(self):
        """Test log determinant computation through StanSPDE API."""
        coords = np.array([
            [-122.4, 37.8], [-122.3, 37.9], [-122.2, 37.7],
            [-122.1, 37.6], [-122.0, 37.5], [-121.9, 37.4],
            [-121.8, 37.3], [-121.7, 37.2]
        ])
        y_obs = np.array([50, 48, 52, 49, 51, 47, 53, 46])
        
        spde = StanSPDE(coords, y_obs, verbose=False)
        stan_data = spde.prepare_stan_data()
        
        # The log determinant should be computed and stored in stan_data
        log_det = stan_data['log_det_Q_base']
        assert isinstance(log_det, float)
        assert not np.isnan(log_det)
        assert not np.isinf(log_det)


class TestStanSPDEIntegration:
    """Test the complete StanSPDE workflow integration."""
    
    def test_prepare_stan_data_integration(self):
        """Test the full StanSPDE workflow end-to-end."""
        coords = np.array([
            [-122.4, 37.8], [-122.3, 37.9], [-122.2, 37.7], 
            [-122.1, 37.6], [-122.0, 37.5], [-121.9, 37.4],
            [-121.8, 37.3], [-121.7, 37.2]
        ])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        spde = StanSPDE(coords, y, verbose=False)
        stan_data = spde.prepare_stan_data(prior_mode="auto")
        
        # Check required fields are present
        required_fields = [
            'N_obs', 'N_mesh', 'y', 'prior_mode',
            'A_nnz', 'A_w', 'A_v', 'A_u',
            'Q_nnz', 'Q_w', 'Q_v', 'Q_u', 'log_det_Q_base',
            'estimated_range', 'estimated_sigma',
            'data_sd', 'coordinate_units_to_km', 'spatial_fraction'
        ]
        
        for field in required_fields:
            assert field in stan_data, f"Missing required field: {field}"
        
        assert stan_data['N_obs'] == 8
        assert isinstance(stan_data['prior_mode'], int)
        assert len(stan_data['y']) == 8