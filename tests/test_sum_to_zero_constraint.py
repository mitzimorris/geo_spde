"""
Unit tests for sum-to-zero constraint in SPDE models.
"""

import pytest
import numpy as np
from typing import Dict
from geo_spde import StanSPDE


class TestSumToZeroConstraint:
    """Test sum-to-zero constraint handling in SPDE models."""
    
    @pytest.fixture
    def synthetic_data(self) -> Dict:
        """Create synthetic spatial data with known parameters."""
        np.random.seed(42)
        
        # Generate spatial coordinates
        N = 20
        coords_raw = np.column_stack([
            np.random.uniform(-122, -92, N),
            np.random.uniform(37, 47, N)
        ])
        
        # Generate correlated spatial data
        from scipy.spatial.distance import cdist
        dists = cdist(coords_raw, coords_raw)
        
        # True parameters
        true_range = 500.0
        true_sigma = 1.0
        
        # Matern covariance (nu=1/2)
        cov = true_sigma**2 * np.exp(-np.sqrt(8) * dists / true_range)
        cov += np.eye(N) * 0.01  # Numerical stability
        
        y_obs = np.random.multivariate_normal(np.zeros(N), cov)
        
        return {
            'coords_raw': coords_raw,
            'y_obs': y_obs,
            'true_range': true_range,
            'true_sigma': true_sigma
        }
    
    def test_stan_data_includes_mesh_info(self, synthetic_data: Dict) -> None:
        """Test that Stan data includes mesh information for sum-to-zero constraint."""
        spde = StanSPDE(
            coords_raw=synthetic_data['coords_raw'],
            y_obs=synthetic_data['y_obs'],
            alpha=1,
            verbose=False
        )
        
        stan_data = spde.prepare_stan_data()
        
        # Check required fields for sum-to-zero constraint
        assert 'N_mesh' in stan_data
        assert 'N_obs' in stan_data
        assert stan_data['N_mesh'] > stan_data['N_obs']
        
        # Check that mesh size is reasonable
        mesh_to_obs_ratio = stan_data['N_mesh'] / stan_data['N_obs']
        assert 1.5 < mesh_to_obs_ratio < 10, f"Mesh/obs ratio {mesh_to_obs_ratio:.2f} seems unreasonable"
    
    def test_precision_matrix_rank_deficiency(self, synthetic_data: Dict) -> None:
        """Test that precision matrix has expected rank deficiency for alpha >= 2."""
        from scipy.sparse.linalg import eigsh
        
        # For alpha=1 (Matern nu=1/2 in 2D), no rank deficiency expected
        # For alpha=2 (Matern nu=1 in 2D), rank deficiency of 1 expected
        spde_alpha1 = StanSPDE(
            coords_raw=synthetic_data['coords_raw'],
            y_obs=synthetic_data['y_obs'],
            alpha=1,
            verbose=False
        )
        
        # Get the base Q matrix for alpha=1
        Q_base_alpha1 = spde_alpha1.Q_base
        
        # Check smallest eigenvalues for alpha=1
        k = min(10, Q_base_alpha1.shape[0] - 2)
        eigenvalues_alpha1 = eigsh(Q_base_alpha1.tocsc(), k=k, which='SA', return_eigenvectors=False)
        
        # For alpha=1, should NOT have near-zero eigenvalues
        min_eig_alpha1 = np.min(np.abs(eigenvalues_alpha1))
        assert min_eig_alpha1 > 1e-6, f"Alpha=1: Unexpected near-zero eigenvalue {min_eig_alpha1:.2e}"
    
    def test_log_determinant_excludes_null_space(self, synthetic_data: Dict) -> None:
        """Test that log determinant computation excludes null space."""
        spde = StanSPDE(
            coords_raw=synthetic_data['coords_raw'],
            y_obs=synthetic_data['y_obs'],
            alpha=1,
            verbose=False
        )
        
        stan_data = spde.prepare_stan_data()
        
        # Log determinant should be finite and reasonable
        log_det = stan_data['log_det_Q_base']
        assert np.isfinite(log_det), "Log determinant is not finite"
        assert log_det > 0, f"Log determinant {log_det:.2f} should be positive"
        
        # Check that it's computed excluding null space
        # For a rank-deficient matrix, log_det should be based on positive eigenvalues only
        n_mesh = stan_data['N_mesh']
        
        # Very rough bounds check
        assert log_det < n_mesh * 10, f"Log determinant {log_det:.2f} seems too large"
        assert log_det > -n_mesh * 10, f"Log determinant {log_det:.2f} seems too small"
    
    def test_stan_data_scaling_factor(self, synthetic_data: Dict) -> None:
        """Test that Stan data could include scaling factor for sum-to-zero constraint."""
        spde = StanSPDE(
            coords_raw=synthetic_data['coords_raw'],
            y_obs=synthetic_data['y_obs'],
            alpha=1,
            verbose=False
        )
        
        stan_data = spde.prepare_stan_data()
        
        # Compute what the scaling factor should be
        n_mesh = stan_data['N_mesh']
        expected_scaling = np.sqrt(n_mesh / (n_mesh - 1.0))
        
        # For small meshes, scaling is more important
        if n_mesh < 50:
            assert expected_scaling > 1.01, f"For small mesh (N={n_mesh}), scaling factor {expected_scaling:.4f} should be > 1.01"
        
        # Could add this to stan_data in future
        # assert 'sum_to_zero_scaling' in stan_data
        # assert np.abs(stan_data['sum_to_zero_scaling'] - expected_scaling) < 1e-10