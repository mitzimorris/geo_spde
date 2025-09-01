"""
Unit tests for geo_spde.pc_priors module.

Tests PC (Penalized Complexity) prior computation, validation,
and sampling functionality for SPDE models.
"""

import pytest
import numpy as np
from geo_spde.pc_priors import (
    pc_prior_range,
    pc_prior_variance,
    compute_pc_prior_params,
    validate_pc_priors,
    generate_pc_prior_code,
    sample_from_pc_prior
)


class TestPCPriorRange:
    """Test PC prior computation for spatial range parameter."""
    
    def test_pc_prior_range_alpha_1(self):
        """Test PC prior for range with alpha=1 (Matern nu=1/2)."""
        rho_0 = 10.0
        alpha_rho = 0.5
        alpha = 1
        
        result = pc_prior_range(rho_0, alpha_rho, alpha)
        
        # Check structure
        assert 'lambda_rho' in result
        assert 'kappa_0' in result
        assert 'rho_0' in result
        assert 'alpha_rho' in result
        assert 'scale_factor' in result
        
        # Check values
        assert result['rho_0'] == rho_0
        assert result['alpha_rho'] == alpha_rho
        assert result['scale_factor'] == 2.0  # sqrt(8 * 0.5)
        assert result['kappa_0'] == 2.0 / rho_0  # scale_factor / rho_0
        assert result['lambda_rho'] == -np.log(alpha_rho) / rho_0
    
    def test_pc_prior_range_alpha_2(self):
        """Test PC prior for range with alpha=2 (Matern nu=3/2)."""
        rho_0 = 5.0
        alpha_rho = 0.1
        alpha = 2
        
        result = pc_prior_range(rho_0, alpha_rho, alpha)
        
        assert result['scale_factor'] == np.sqrt(12)  # sqrt(8 * 1.5)
        assert result['kappa_0'] == np.sqrt(12) / rho_0
        assert result['lambda_rho'] == -np.log(alpha_rho) / rho_0
    
    def test_pc_prior_range_invalid_alpha(self):
        """Test error handling for invalid alpha value."""
        with pytest.raises(ValueError, match="Unsupported alpha value"):
            pc_prior_range(10.0, 0.5, alpha=3)
    
    def test_pc_prior_range_boundary_values(self):
        """Test PC prior with boundary values."""
        # Very small alpha_rho
        result = pc_prior_range(10.0, 0.01, alpha=2)
        assert result['lambda_rho'] > 0
        assert not np.isnan(result['lambda_rho'])
        
        # Alpha_rho close to 1
        result = pc_prior_range(10.0, 0.99, alpha=2)
        assert result['lambda_rho'] > 0
        assert not np.isinf(result['lambda_rho'])


class TestPCPriorVariance:
    """Test PC prior computation for spatial variance parameter."""
    
    def test_pc_prior_variance_basic(self):
        """Test basic PC prior computation for variance."""
        sigma_0 = 1.5
        alpha_sigma = 0.05
        
        result = pc_prior_variance(sigma_0, alpha_sigma)
        
        # Check structure
        assert 'lambda_sigma' in result
        assert 'sigma_0' in result
        assert 'alpha_sigma' in result
        
        # Check values
        assert result['sigma_0'] == sigma_0
        assert result['alpha_sigma'] == alpha_sigma
        assert result['lambda_sigma'] == -np.log(alpha_sigma) / sigma_0
        assert result['lambda_sigma'] > 0
    
    def test_pc_prior_variance_edge_cases(self):
        """Test PC prior variance with edge case values."""
        # Very small alpha_sigma
        result = pc_prior_variance(1.0, 0.001)
        assert result['lambda_sigma'] > 0
        assert not np.isinf(result['lambda_sigma'])
        
        # Large sigma_0
        result = pc_prior_variance(100.0, 0.05)
        assert result['lambda_sigma'] > 0
        assert not np.isnan(result['lambda_sigma'])


class TestComputePCPriorParams:
    """Test complete PC prior parameter computation."""
    
    @pytest.fixture
    def mock_mesh_diagnostics(self):
        """Create mock mesh diagnostics for testing."""
        return {
            'spatial_scale': {
                'min_distance': 0.5,
                'median_distance': 2.0,
                'estimated_range': 5.0
            },
            'suggestions': {
                'mesh_extent': 20.0
            }
        }
    
    def test_compute_pc_prior_params_auto(self, mock_mesh_diagnostics):
        """Test PC prior parameter computation in auto mode."""
        estimated_range = 5.0
        data_sd = 2.0
        spatial_fraction = 0.6
        
        result = compute_pc_prior_params(
            estimated_range=estimated_range,
            data_sd=data_sd,
            prior_mode="auto",
            spatial_fraction=spatial_fraction,
            mesh_diagnostics=mock_mesh_diagnostics,
            alpha=2
        )
        
        # Check all required fields
        required_fields = [
            'rho_0', 'alpha_rho', 'lambda_rho', 'kappa_0',
            'sigma_0', 'alpha_sigma', 'lambda_sigma',
            'expected_range', 'expected_sigma'
        ]
        
        for field in required_fields:
            assert field in result
        
        # Check auto mode values
        assert result['rho_0'] == estimated_range
        assert result['alpha_rho'] == 0.5
        assert result['sigma_0'] == data_sd * np.sqrt(spatial_fraction)
        assert result['alpha_sigma'] == 0.05
    
    def test_compute_pc_prior_params_tight(self, mock_mesh_diagnostics):
        """Test PC prior parameter computation in tight mode."""
        result = compute_pc_prior_params(
            estimated_range=5.0,
            data_sd=2.0,
            prior_mode="tight",
            spatial_fraction=0.5,
            mesh_diagnostics=mock_mesh_diagnostics
        )
        
        # Check tight mode values
        min_distance = mock_mesh_diagnostics['spatial_scale']['min_distance']
        assert result['rho_0'] == min_distance * 10
        assert result['alpha_rho'] == 0.9
        assert result['alpha_sigma'] == 0.01
    
    def test_compute_pc_prior_params_medium(self, mock_mesh_diagnostics):
        """Test PC prior parameter computation in medium mode."""
        result = compute_pc_prior_params(
            estimated_range=5.0,
            data_sd=2.0,
            prior_mode="medium",
            spatial_fraction=0.5,
            mesh_diagnostics=mock_mesh_diagnostics
        )
        
        # Check medium mode values
        median_distance = mock_mesh_diagnostics['spatial_scale']['median_distance']
        assert result['rho_0'] == median_distance * 3
        assert result['alpha_rho'] == 0.5
        assert result['alpha_sigma'] == 0.05
    
    def test_compute_pc_prior_params_wide(self, mock_mesh_diagnostics):
        """Test PC prior parameter computation in wide mode."""
        result = compute_pc_prior_params(
            estimated_range=5.0,
            data_sd=2.0,
            prior_mode="wide",
            spatial_fraction=0.5,
            mesh_diagnostics=mock_mesh_diagnostics
        )
        
        # Check wide mode values
        mesh_extent = mock_mesh_diagnostics['suggestions']['mesh_extent']
        assert result['rho_0'] == mesh_extent * 0.3
        assert result['alpha_rho'] == 0.1
        assert result['alpha_sigma'] == 0.1


class TestValidatePCPriors:
    """Test PC prior validation functionality."""
    
    @pytest.fixture
    def mock_pc_params(self):
        """Create mock PC prior parameters."""
        return {
            'rho_0': 5.0,
            'sigma_0': 1.0,
            'alpha_rho': 0.5,
            'alpha_sigma': 0.05
        }
    
    @pytest.fixture
    def mock_mesh_diagnostics(self):
        """Create mock mesh diagnostics."""
        return {
            'edge_lengths': {
                'min': 0.5,
                'max': 2.0
            }
        }
    
    def test_validate_pc_priors_all_good(self, mock_pc_params, mock_mesh_diagnostics):
        """Test validation with all parameters in good ranges."""
        validation = validate_pc_priors(
            mock_pc_params,
            mock_mesh_diagnostics,
            verbose=False
        )
        
        assert validation['range_compatible'] is True
        assert validation['variance_reasonable'] is True
        assert validation['mesh_adequate'] is True
    
    def test_validate_pc_priors_range_too_small(self, mock_mesh_diagnostics):
        """Test validation with range too small for mesh resolution."""
        small_range_params = {
            'rho_0': 0.1,  # Very small range
            'sigma_0': 1.0,
            'alpha_rho': 0.5,
            'alpha_sigma': 0.05
        }
        
        validation = validate_pc_priors(
            small_range_params,
            mock_mesh_diagnostics,
            verbose=False
        )
        
        assert validation['range_compatible'] is False
    
    def test_validate_pc_priors_range_too_large(self, mock_mesh_diagnostics):
        """Test validation with range much larger than mesh extent."""
        large_range_params = {
            'rho_0': 1000.0,  # Very large range
            'sigma_0': 1.0,
            'alpha_rho': 0.5,
            'alpha_sigma': 0.05
        }
        
        validation = validate_pc_priors(
            large_range_params,
            mock_mesh_diagnostics,
            verbose=False
        )
        
        assert validation['mesh_adequate'] is False
    
    def test_validate_pc_priors_variance_too_small(self, mock_mesh_diagnostics):
        """Test validation with very small variance."""
        small_var_params = {
            'rho_0': 5.0,
            'sigma_0': 1e-10,  # Very small variance
            'alpha_rho': 0.5,
            'alpha_sigma': 0.05
        }
        
        validation = validate_pc_priors(
            small_var_params,
            mock_mesh_diagnostics,
            verbose=False
        )
        
        assert validation['variance_reasonable'] is False


class TestGeneratePCPriorCode:
    """Test Stan code generation for PC priors."""
    
    def test_generate_pc_prior_code_structure(self):
        """Test that generated Stan code has correct structure."""
        code = generate_pc_prior_code()
        
        # Check it's a string
        assert isinstance(code, str)
        
        # Check key function names are present
        assert "pc_prior_log_kappa_lpdf" in code
        assert "pc_prior_log_tau_lpdf" in code
        assert "pc_prior_range_lpdf" in code
        assert "pc_prior_sigma_lpdf" in code
        
        # Check functions block structure
        assert "functions {" in code
        assert "}" in code
    
    def test_generate_pc_prior_code_no_unicode(self):
        """Test that generated code contains no unicode characters."""
        code = generate_pc_prior_code()
        
        # Check for ASCII-only content
        try:
            code.encode('ascii')
        except UnicodeEncodeError:
            pytest.fail("Generated Stan code contains non-ASCII characters")
        
        # Check specific replacements
        assert "proportional to" in code
        assert "pi(" in code  # Should use pi instead of π


class TestSampleFromPCPrior:
    """Test sampling from PC priors."""
    
    @pytest.fixture
    def sample_pc_params(self):
        """Create sample PC prior parameters for testing."""
        return {
            'lambda_rho': 0.1,
            'lambda_sigma': 0.2,
            'rho_0': 10.0,
            'sigma_0': 1.0
        }
    
    def test_sample_from_pc_prior_structure(self, sample_pc_params):
        """Test sampling returns correct structure."""
        n_samples = 100
        samples = sample_from_pc_prior(sample_pc_params, n_samples, alpha=2)
        
        # Check structure
        assert 'range' in samples
        assert 'sigma' in samples
        assert 'kappa' in samples
        assert 'tau' in samples
        
        # Check sample sizes
        for key in samples:
            assert len(samples[key]) == n_samples
    
    def test_sample_from_pc_prior_values(self, sample_pc_params):
        """Test that sampled values are reasonable."""
        np.random.seed(42)  # For reproducibility
        samples = sample_from_pc_prior(sample_pc_params, 1000, alpha=2)
        
        # Check all values are positive
        for key in samples:
            assert np.all(samples[key] > 0)
        
        # Check ranges are reasonable (not all identical)
        assert np.std(samples['range']) > 0
        assert np.std(samples['sigma']) > 0
        
        # Check kappa and range relationship for alpha=2
        scale_factor = np.sqrt(12)
        expected_kappa = scale_factor / samples['range']
        np.testing.assert_allclose(samples['kappa'], expected_kappa, rtol=1e-10)
    
    def test_sample_from_pc_prior_alpha_1(self, sample_pc_params):
        """Test sampling with alpha=1 (Matern nu=1/2)."""
        samples = sample_from_pc_prior(sample_pc_params, 100, alpha=1)
        
        # Check kappa-range relationship for alpha=1
        scale_factor = 2.0
        expected_kappa = scale_factor / samples['range']
        np.testing.assert_allclose(samples['kappa'], expected_kappa, rtol=1e-10)
    
    def test_sample_from_pc_prior_reproducibility(self, sample_pc_params):
        """Test that sampling with same seed gives same results."""
        np.random.seed(123)
        samples1 = sample_from_pc_prior(sample_pc_params, 50, alpha=2)
        
        np.random.seed(123)
        samples2 = sample_from_pc_prior(sample_pc_params, 50, alpha=2)
        
        for key in samples1:
            np.testing.assert_array_equal(samples1[key], samples2[key])