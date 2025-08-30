"""
Stan data preparation for SPDE models.

This module handles the complete pipeline from raw coordinates to Stan-ready data,
always working in normalized [0,1] coordinates for optimal HMC sampling.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from typing import Tuple, Dict, Optional, Any
import warnings
import json

from geo_spde import preprocess_coords, SPDEMesh
from geo_spde.matrices import compute_fem_matrices
from geo_spde.exceptions import ParameterScaleError, ConditioningError


def prepare_stan_data(
    coords_raw: np.ndarray,
    y_obs: np.ndarray,
    target_range_km: Optional[float] = None,
    target_variance: Optional[float] = 1.0,
    mesh_resolution: float = 0.5,
    extension_factor: float = 0.2,
    alpha: int = 1,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Complete pipeline from raw coordinates to Stan data.
    Always normalizes to [0,1] for optimal sampling geometry.
    
    Parameters
    ----------
    coords_raw : np.ndarray
        Raw coordinates (can be lon/lat or projected)
    y_obs : np.ndarray
        Observations at coordinates
    target_range_km : float, optional
        Expected correlation range in km (for prior specification)
        If None, estimated from data
    target_variance : float
        Expected marginal variance
    mesh_resolution : float
        Target edge factor for mesh
    extension_factor : float
        Boundary extension
    alpha : int
        1 for Matérn ν=1/2, 2 for ν=3/2
    verbose : bool
        Print progress
        
    Returns
    -------
    Dict containing:
        - All Stan data fields
        - 'metadata': Transformation and scaling information
        - 'priors': Suggested prior specifications
    """
    if verbose:
        print("="*60)
        print("SPDE Stan Data Preparation")
        print("="*60)
    
    # Step 1: Preprocess and normalize to [0,1]
    coords_proj, kept_idx, proj_info = preprocess_coords(
        coords_raw, 
        normalize=True,  # Always normalize!
        normalize_method='unit_square'
    )
    y_clean = y_obs[kept_idx]
    
    # Step 2: Create mesh in normalized space
    mesh = SPDEMesh(coords_proj, proj_info)
    vertices, triangles = mesh.create_mesh(
        target_edge_factor=mesh_resolution,
        extension_factor=extension_factor,
        verbose=verbose
    )
    
    # Step 3: Compute scale diagnostics
    scale_diag = mesh.compute_scale_diagnostics(verbose=False)
    
    # Step 4: Compute FEM matrices in normalized space
    C, G, A, Q_base = compute_fem_matrices(
        vertices, triangles, coords_proj,
        kappa=1.0,  # Reference value
        alpha=alpha,
        verbose=verbose
    )
    
    # Step 5: Compute reference log determinant
    log_det_Q_base = compute_log_det_sparse(Q_base)
    
    # Step 6: Determine priors
    priors = compute_prior_specifications(
        target_range_km, target_variance, 
        proj_info, scale_diag, verbose
    )
    
    # Step 7: Convert matrices to Stan format
    A_csr = sparse_to_stan_csr(A)
    Q_csr = sparse_to_stan_csr(Q_base)
    
    # Step 8: Assemble Stan data
    stan_data = {
        'N_obs': len(y_clean),
        'N_mesh': vertices.shape[0],
        
        # A matrix (projector)
        'A_nnz': A_csr['nnz'],
        'A_w': A_csr['w'],
        'A_v': A_csr['v'],
        'A_u': A_csr['u'],
        
        # Q matrix (precision at reference)
        'Q_nnz': Q_csr['nnz'],
        'Q_w': Q_csr['w'],
        'Q_v': Q_csr['v'],
        'Q_u': Q_csr['u'],
        'log_det_Q_base': log_det_Q_base,
        
        # Observations
        'y': y_clean.tolist(),
        
        # Prior parameters (for Stan model)
        'prior_kappa_mean': priors['kappa_mean'],
        'prior_kappa_sd': priors['kappa_sd'],
        'prior_tau_mean': priors['tau_mean'],
        'prior_tau_sd': priors['tau_sd']
    }
    
    # Add metadata for interpretation
    metadata = {
        'coordinate_system': 'normalized_unit_square',
        'original_projection': proj_info,
        'scaling': scale_diag,
        'priors': priors,
        'mesh_info': mesh.get_mesh_info()
    }
    
    if verbose:
        print("\n" + "="*60)
        print("Data Preparation Complete")
        print(f"  Mesh: {stan_data['N_mesh']} vertices in [0,1]²")
        print(f"  Observations: {stan_data['N_obs']}")
        print(f"  Prior range (normalized): {priors['range_normalized']:.3f}")
        print(f"  Prior range (km): {priors['range_km']:.1f}")
        print("="*60)
    
    return {
        'stan_data': stan_data,
        'metadata': metadata
    }


def compute_prior_specifications(
    target_range_km: Optional[float],
    target_variance: float,
    proj_info: Dict,
    scale_diag: Dict,
    verbose: bool = True
) -> Dict:
    """
    Compute prior specifications in normalized space.
    
    Parameters
    ----------
    target_range_km : float or None
        Desired correlation range in km
    target_variance : float
        Desired marginal variance
    proj_info : Dict
        From coordinate preprocessing
    scale_diag : Dict
        From mesh scale diagnostics
    verbose : bool
        Print prior information
        
    Returns
    -------
    Dict with prior specifications in normalized space
    """
    # Get scale conversion factor
    if 'transform_info' in proj_info:
        # We have normalized coordinates
        transform = proj_info['transform_info']
        if transform['method'] == 'unit_square':
            # Original extent in meters
            original_extent_m = np.max(1.0 / transform['scale_factors'])
            original_extent_km = original_extent_m / 1000
        else:
            # Approximate from scale factors
            original_extent_km = np.max(1.0 / np.abs(transform['scale_factors'])) / 1000
    else:
        # Use hull diameter
        original_extent_km = proj_info.get('hull_diameter_km', 100)
    
    # Determine target range
    if target_range_km is None:
        # Use data-driven estimate
        suggested_range_norm = scale_diag['suggestions']['spatial_range_suggestion']
        target_range_km = suggested_range_norm * original_extent_km
        if verbose:
            print(f"Auto-selected correlation range: {target_range_km:.1f} km")
    
    # Convert to normalized space
    # In [0,1], a range of 0.1 means 10% of domain
    range_normalized = target_range_km / original_extent_km
    
    # Ensure reasonable bounds
    range_normalized = np.clip(range_normalized, 0.05, 0.5)
    
    # Convert to kappa (for Matérn ν=1/2)
    kappa_mean = np.sqrt(8) / range_normalized
    kappa_sd = kappa_mean / 3  # Allow variation within factor of 3
    
    # Tau based on desired variance
    tau_mean = 1.0 / target_variance
    tau_sd = tau_mean / 3
    
    priors = {
        'range_km': target_range_km,
        'range_normalized': range_normalized,
        'kappa_mean': kappa_mean,
        'kappa_sd': kappa_sd,
        'tau_mean': tau_mean,
        'tau_sd': tau_sd,
        'variance': target_variance,
        'original_extent_km': original_extent_km
    }
    
    if verbose:
        print(f"\nPrior Specifications (normalized space):")
        print(f"  Correlation range: {target_range_km:.1f} km = {range_normalized:.3f} normalized")
        print(f"  Kappa prior: Normal({kappa_mean:.2f}, {kappa_sd:.2f})")
        print(f"  Tau prior: Normal({tau_mean:.2f}, {tau_sd:.2f})")
    
    return priors


def translate_parameters_to_original_scale(
    kappa_normalized: float,
    tau_normalized: float,
    metadata: Dict
) -> Dict:
    """
    Convert parameters from normalized space back to original units.
    
    Parameters
    ----------
    kappa_normalized : float
        Kappa in [0,1] space
    tau_normalized : float
        Tau in [0,1] space
    metadata : Dict
        From prepare_stan_data
        
    Returns
    -------
    Dict with parameters in interpretable units
    """
    # Get scale factor
    original_extent_km = metadata['priors']['original_extent_km']
    
    # Convert kappa to range
    range_normalized = np.sqrt(8) / kappa_normalized
    range_km = range_normalized * original_extent_km
    
    # Variance from tau
    variance = 1.0 / tau_normalized
    sd = np.sqrt(variance)
    
    return {
        'range_km': range_km,
        'range_normalized': range_normalized,
        'spatial_sd': sd,
        'spatial_variance': variance,
        'kappa_original': kappa_normalized * original_extent_km * 1000,  # in meters
        'tau_original': tau_normalized / (original_extent_km * 1000)**2
    }


def validate_prior_compatibility(
    priors: Dict,
    mesh_params: Dict,
    verbose: bool = True
) -> Dict[str, bool]:
    """
    Check if priors are compatible with mesh resolution.
    
    Parameters
    ----------
    priors : Dict
        Prior specifications
    mesh_params : Dict
        Mesh parameters
    verbose : bool
        Print warnings
        
    Returns
    -------
    Dict with validation results
    """
    range_norm = priors['range_normalized']
    mesh_edge = mesh_params['max_edge']
    
    # Check if mesh is fine enough
    edge_to_range = mesh_edge / range_norm
    resolution_ok = edge_to_range < 0.5
    
    # Check if range is reasonable
    range_ok = 0.02 < range_norm < 0.8
    
    validation = {
        'resolution_ok': resolution_ok,
        'range_ok': range_ok,
        'edge_to_range_ratio': edge_to_range
    }
    
    if verbose:
        if not resolution_ok:
            warnings.warn(
                f"Mesh may be too coarse for prior range.\n"
                f"  Edge/Range ratio: {edge_to_range:.2f} (should be < 0.5)\n"
                f"  Consider finer mesh or longer correlation range"
            )
        if not range_ok:
            warnings.warn(
                f"Prior range {range_norm:.3f} may be extreme.\n"
                f"  Consider range in [0.05, 0.5] for normalized coordinates"
            )
    
    return validation


def generate_synthetic_data(
    mesh: SPDEMesh,
    kappa: float,
    tau: float, 
    sigma: float,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data in normalized space for testing.
    
    Parameters
    ----------
    mesh : SPDEMesh
        Mesh object with vertices in [0,1]
    kappa : float
        Range parameter in normalized space
    tau : float
        Precision parameter
    sigma : float
        Observation noise SD
    seed : int, optional
        Random seed
        
    Returns
    -------
    y_obs : np.ndarray
        Synthetic observations
    u_true : np.ndarray
        True field at mesh vertices
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get matrices
    C, G, A, _ = compute_fem_matrices(
        mesh.vertices, mesh.triangles, mesh.coords,
        kappa=1.0, alpha=1
    )
    
    # Build precision matrix
    Q = tau * (kappa**2 * C + G)
    
    # Generate field
    z = np.random.randn(mesh.vertices.shape[0])
    from scipy.sparse.linalg import spsolve
    u_true = spsolve(Q, z)
    
    # Project to observations
    y_true = A @ u_true
    y_obs = y_true + sigma * np.random.randn(len(y_true))
    
    return y_obs, u_true


def compute_log_det_sparse(Q: csr_matrix) -> float:
    """Compute log determinant of sparse positive definite matrix."""
    try:
        lu = splu(Q.tocsc())
        diagL = lu.L.diagonal()
        diagU = lu.U.diagonal()
        return np.sum(np.log(np.abs(diagL))) + np.sum(np.log(np.abs(diagU)))
    except:
        raise ConditioningError("Failed to compute log determinant - Q may be singular")


def sparse_to_stan_csr(mat: csr_matrix) -> Dict:
    """Convert scipy CSR matrix to Stan format."""
    return {
        'nnz': mat.nnz,
        'w': mat.data.tolist(),
        'v': (mat.indices + 1).tolist(),  # 1-indexed
        'u': (mat.indptr + 1).tolist()     # 1-indexed
    }


def create_stan_init(stan_data: Dict, metadata: Dict) -> Dict:
    """
    Generate reasonable initial values for Stan.
    
    Parameters
    ----------
    stan_data : Dict
        Stan data dictionary
    metadata : Dict
        Metadata with priors
        
    Returns
    -------
    Dict with initial values
    """
    priors = metadata['priors']
    
    return {
        'u_field': np.zeros(stan_data['N_mesh']).tolist(),
        'kappa': priors['kappa_mean'],
        'tau': priors['tau_mean'],
        'sigma': np.std(stan_data['y']) * 0.5
    }
