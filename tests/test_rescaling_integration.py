"""
Integration tests for coordinate rescaling in the full stan_data_prep pipeline.

Tests ensure that rescaled coordinates work properly with mesh generation,
matrix computation, and prior specification throughout the entire workflow.
"""

import numpy as np
import pytest
from geo_spde import preprocess_coords
from geo_spde.stan_data_prep import (
    prepare_stan_data, compute_prior_specifications, 
    translate_parameters_to_interpretable_units,
    validate_prior_compatibility
)
from geo_spde.mesh import SPDEMesh


class TestStanDataPrepWithRescaling:
    """Test stan_data_prep pipeline with rescaled coordinates"""
    
    def test_prepare_stan_data_with_rescaled_coords(self):
        """Test full stan_data_prep pipeline with rescaled coordinates"""
        # Large region that will be rescaled to km
        coords = np.array([
            [-122.0, 37.0],
            [-121.0, 38.0],
            [-120.0, 39.0],
            [-119.0, 40.0],
            [-118.0, 41.0],
            [-117.0, 42.0],
            [-116.0, 43.0]
        ])
        
        # Create synthetic observations
        np.random.seed(42)
        y_obs = np.random.normal(0, 1, len(coords))
        
        # Run full pipeline
        result = prepare_stan_data(
            coords_raw=coords,
            y_obs=y_obs,
            target_range_km=100.0,  # 100 km correlation range
            mesh_resolution=0.8,
            verbose=False
        )
        
        # Check that result structure is correct
        assert 'stan_data' in result
        assert 'metadata' in result
        
        stan_data = result['stan_data']
        metadata = result['metadata']
        
        # Check stan_data contains required fields
        required_fields = {
            'N_obs', 'N_mesh', 'A_nnz', 'A_w', 'A_v', 'A_u',
            'Q_nnz', 'Q_w', 'Q_v', 'Q_u', 'log_det_Q_base',
            'y', 'prior_kappa_mean', 'prior_kappa_sd', 
            'prior_tau_mean', 'prior_tau_sd'
        }
        assert set(stan_data.keys()) >= required_fields
        
        # Check that coordinates were rescaled
        proj_info = metadata['projection']
        assert proj_info['coordinate_units'] == 'kilometers'
        assert proj_info['rescale_factor'] == 0.001
        
        # Check that priors are reasonable for km coordinates
        priors = metadata['priors']
        assert priors['range_km'] == 100.0
        assert 'range_coord_units' in priors
        assert priors['range_coord_units'] == 100.0  # Should equal range_km for km coords
        
        # Kappa should be appropriate for km units
        kappa_mean = priors['kappa_mean']
        expected_kappa = np.sqrt(8) / 100.0  # sqrt(8) / range_km
        assert abs(kappa_mean - expected_kappa) < 1e-6
    
    def test_prepare_stan_data_with_small_region(self):
        """Test stan_data_prep pipeline with small region (no rescaling)"""
        # Small region that will stay in meters
        coords = np.array([
            [-122.5, 37.7],
            [-122.4, 37.8],
            [-122.3, 37.9],
            [-122.2, 38.0],
            [-122.1, 38.1],
            [-122.0, 38.2],
            [-121.9, 38.3]
        ])
        
        y_obs = np.ones(len(coords))
        
        result = prepare_stan_data(
            coords_raw=coords,
            y_obs=y_obs,
            target_range_km=10.0,
            verbose=False
        )
        
        metadata = result['metadata']
        proj_info = metadata['projection']
        
        # Should not be rescaled
        assert proj_info['coordinate_units'] == 'meters'
        assert proj_info['rescale_factor'] == 1.0
        
        # Priors should account for meter coordinates
        priors = metadata['priors']
        assert priors['range_km'] == 10.0
        assert priors['range_coord_units'] == 10000.0  # 10 km = 10,000 m
        
        # Kappa should be appropriate for meter units
        kappa_mean = priors['kappa_mean']
        expected_kappa = np.sqrt(8) / 10000.0  # sqrt(8) / range_meters
        assert abs(kappa_mean - expected_kappa) < 1e-8


class TestPriorSpecificationsWithRescaling:
    """Test prior specification functions with rescaled coordinates"""
    
    def setup_method(self):
        """Set up test data"""
        # Mock projection info for rescaled coordinates
        self.proj_info_km = {
            'coordinate_units': 'kilometers',
            'unit_to_km': 1.0,
            'rescale_factor': 0.001,
            'hull_diameter_km': 500.0
        }
        
        # Mock projection info for meter coordinates  
        self.proj_info_m = {
            'coordinate_units': 'meters',
            'unit_to_km': 0.001,
            'rescale_factor': 1.0,
            'hull_diameter_km': 50.0
        }
        
        # Mock scale diagnostics
        self.scale_diag = {
            'suggestions': {
                'spatial_range_suggestion': 100.0  # In coordinate units
            }
        }
    
    def test_compute_prior_specifications_km_coords(self):
        """Test prior computation with kilometer coordinates"""
        priors = compute_prior_specifications(
            target_range_km=150.0,
            target_variance=2.0,
            proj_info=self.proj_info_km,
            scale_diag=self.scale_diag,
            verbose=False
        )
        
        # Check computed values
        assert priors['range_km'] == 150.0
        assert priors['range_coord_units'] == 150.0  # km coords: same as range_km
        
        # Kappa should be sqrt(8) / range_coord_units
        expected_kappa = np.sqrt(8) / 150.0
        assert abs(priors['kappa_mean'] - expected_kappa) < 1e-10
        
        # Tau based on variance
        assert priors['tau_mean'] == 1.0 / 2.0
    
    def test_compute_prior_specifications_meter_coords(self):
        """Test prior computation with meter coordinates"""
        priors = compute_prior_specifications(
            target_range_km=50.0,
            target_variance=1.0,
            proj_info=self.proj_info_m,
            scale_diag=self.scale_diag,
            verbose=False
        )
        
        # Check computed values
        assert priors['range_km'] == 50.0
        assert priors['range_coord_units'] == 50000.0  # 50 km = 50,000 m
        
        # Kappa should be sqrt(8) / range_coord_units (in meters)
        expected_kappa = np.sqrt(8) / 50000.0
        assert abs(priors['kappa_mean'] - expected_kappa) < 1e-10
        
        # Tau based on variance
        assert priors['tau_mean'] == 1.0
    
    def test_auto_range_selection_km_coords(self):
        """Test automatic range selection with km coordinates"""
        priors = compute_prior_specifications(
            target_range_km=None,  # Auto-select
            target_variance=1.0,
            proj_info=self.proj_info_km,
            scale_diag=self.scale_diag,  # suggests 100.0 coord units
            verbose=False
        )
        
        # Should use scale_diag suggestion * unit_to_km
        expected_range_km = 100.0 * 1.0  # 100 coord units * 1.0 km/unit
        assert priors['range_km'] == expected_range_km
        assert priors['range_coord_units'] == expected_range_km
    
    def test_auto_range_selection_meter_coords(self):
        """Test automatic range selection with meter coordinates"""
        priors = compute_prior_specifications(
            target_range_km=None,  # Auto-select
            target_variance=1.0,
            proj_info=self.proj_info_m,
            scale_diag=self.scale_diag,  # suggests 100.0 coord units (meters)
            verbose=False
        )
        
        # Should use scale_diag suggestion * unit_to_km
        expected_range_km = 100.0 * 0.001  # 100 meters * 0.001 km/m = 0.1 km
        assert priors['range_km'] == expected_range_km
        assert priors['range_coord_units'] == 100.0


class TestParameterTranslation:
    """Test parameter translation with rescaled coordinates"""
    
    def test_translate_parameters_km_coords(self):
        """Test parameter translation with kilometer coordinates"""
        metadata = {
            'projection': {
                'coordinate_units': 'kilometers',
                'unit_to_km': 1.0
            }
        }
        
        kappa = 0.1  # 1/km
        tau = 2.0
        
        result = translate_parameters_to_interpretable_units(
            kappa=kappa, tau=tau, metadata=metadata
        )
        
        # Check range calculation
        expected_range_coord = np.sqrt(8) / kappa
        assert abs(result['range_coord_units'] - expected_range_coord) < 1e-10
        
        # Range in km should equal range in coord units (since coords are in km)
        assert abs(result['range_km'] - expected_range_coord) < 1e-10
        
        # Check variance/sd
        assert result['spatial_variance'] == 1.0 / tau
        assert result['spatial_sd'] == np.sqrt(1.0 / tau)
    
    def test_translate_parameters_meter_coords(self):
        """Test parameter translation with meter coordinates"""
        metadata = {
            'projection': {
                'coordinate_units': 'meters',
                'unit_to_km': 0.001
            }
        }
        
        kappa = 0.0001  # 1/m
        tau = 1.0
        
        result = translate_parameters_to_interpretable_units(
            kappa=kappa, tau=tau, metadata=metadata
        )
        
        # Check range calculation
        expected_range_coord = np.sqrt(8) / kappa  # In meters
        assert abs(result['range_coord_units'] - expected_range_coord) < 1e-6
        
        # Range in km should be range_coord * unit_to_km
        expected_range_km = expected_range_coord * 0.001
        assert abs(result['range_km'] - expected_range_km) < 1e-9
        
        # Check variance/sd
        assert result['spatial_variance'] == 1.0 / tau
        assert result['spatial_sd'] == np.sqrt(1.0 / tau)


class TestValidationWithRescaling:
    """Test validation functions with rescaled coordinates"""
    
    def test_validate_prior_compatibility_km_coords(self):
        """Test prior validation with kilometer coordinates"""
        priors = {
            'range_coord_units': 100.0,  # 100 km
            'kappa_mean': np.sqrt(8) / 100.0,
            'range_km': 100.0
        }
        
        mesh_params = {
            'max_edge': 30.0,  # 30 km mesh edge
            'domain_extent': 500.0  # 500 km domain
        }
        
        proj_info = {
            'coordinate_units': 'kilometers'
        }
        
        validation = validate_prior_compatibility(
            priors=priors,
            mesh_params=mesh_params, 
            proj_info=proj_info,
            verbose=False
        )
        
        # Edge/range ratio should be 30/100 = 0.3 (good)
        assert abs(validation['edge_to_range_ratio'] - 0.3) < 1e-10
        assert validation['resolution_ok'] == True  # 0.3 < 0.5
        
        # Range/domain ratio should be 100/500 = 0.2 (good)
        assert abs(validation['range_to_domain_ratio'] - 0.2) < 1e-10
        assert validation['range_ok'] == True  # 0.02 < 0.2 < 0.8
    
    def test_validate_prior_compatibility_meter_coords(self):
        """Test prior validation with meter coordinates"""
        priors = {
            'range_coord_units': 50000.0,  # 50 km = 50,000 m
            'kappa_mean': np.sqrt(8) / 50000.0,
            'range_km': 50.0
        }
        
        mesh_params = {
            'max_edge': 10000.0,  # 10 km = 10,000 m mesh edge
            'domain_extent': 200000.0  # 200 km = 200,000 m domain
        }
        
        proj_info = {
            'coordinate_units': 'meters'
        }
        
        validation = validate_prior_compatibility(
            priors=priors,
            mesh_params=mesh_params,
            proj_info=proj_info,
            verbose=False
        )
        
        # Edge/range ratio should be 10000/50000 = 0.2 (good)
        assert abs(validation['edge_to_range_ratio'] - 0.2) < 1e-10
        assert validation['resolution_ok'] == True  # 0.2 < 0.5
        
        # Range/domain ratio should be 50000/200000 = 0.25 (good)
        assert abs(validation['range_to_domain_ratio'] - 0.25) < 1e-10
        assert validation['range_ok'] == True  # 0.02 < 0.25 < 0.8


class TestMeshWithRescaling:
    """Test mesh generation with rescaled coordinates"""
    
    def test_mesh_creation_with_rescaled_coords(self):
        """Test that mesh generation works with rescaled coordinates"""
        # Large region coordinates - will be rescaled
        coords = np.array([
            [-122.0, 37.0],
            [-121.0, 38.0],
            [-120.0, 39.0],
            [-119.0, 40.0],
            [-118.0, 41.0],
            [-117.0, 42.0],
            [-116.0, 43.0]
        ])
        
        # Preprocess coordinates (will be rescaled to km)
        result_coords, indices, proj_info = preprocess_coords(coords)
        
        # Create mesh with rescaled coordinates
        mesh = SPDEMesh(result_coords, proj_info)
        vertices, triangles = mesh.create_mesh(
            target_edge_factor=0.5,
            verbose=False
        )
        
        # Mesh should be created successfully
        assert vertices.shape[1] == 2
        assert triangles.shape[1] == 3
        assert len(vertices) > len(result_coords)  # Should have more vertices than obs
        
        # Vertices should be in km range (not meter range)
        vertex_extent_x = vertices[:, 0].max() - vertices[:, 0].min()
        vertex_extent_y = vertices[:, 1].max() - vertices[:, 1].min()
        max_vertex_extent = max(vertex_extent_x, vertex_extent_y)
        
        # Should be hundreds of km, not hundreds of thousands of meters
        assert 300 < max_vertex_extent < 1000
        
        # Scale diagnostics should work
        scale_diag = mesh.compute_scale_diagnostics(verbose=False)
        assert 'spatial_scale' in scale_diag
        assert 'suggestions' in scale_diag
        
        # Suggested range should be reasonable for km coordinates
        suggested_range = scale_diag['suggestions']['spatial_range_suggestion']
        assert 10 < suggested_range < 500  # Should be tens to hundreds of km


if __name__ == "__main__":
    pytest.main([__file__, "-v"])