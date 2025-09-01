"""
Stan SPDE API with automatic PC priors and adaptive mesh.

This module provides a streamlined interface for SPDE modeling in Stan with:
- Automatic adaptive mesh generation
- PC prior configuration based on mesh diagnostics
- Integrated Stan data preparation
"""

import numpy as np
from typing import Dict, Tuple, Optional, Literal
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
import warnings

from geo_spde import preprocess_coords, SPDEMesh
from geo_spde.matrices import compute_fem_matrices
from geo_spde.exceptions import ParameterScaleError


PriorMode = Literal["auto", "tight", "medium", "wide", "custom"]
DomainKnowledge = Literal["environmental", "disease", "economic", "weather", "soil"]


def prepare_stan_data_with_priors(
    y: np.ndarray,
    coords: np.ndarray,
    mesh: SPDEMesh,
    A_matrix_csr: Tuple,
    Q_matrix_csr: Tuple,
    prior_mode: PriorMode = "auto",
    spatial_fraction: float = 0.5,
    user_range_km: Optional[float] = None,
    alpha: int = 2
) -> Dict:
    """
    Prepare Stan data with automatic prior configuration using mesh diagnostics.
    
    :param y: Observations at coordinate locations
    :param coords: Cleaned coordinates from preprocess_coords()
    :param mesh: Mesh object after create_adaptive_mesh()
    :param A_matrix_csr: CSR format projection matrix from FEM computation
    :param Q_matrix_csr: CSR format precision matrix from FEM computation
    :param prior_mode: Prior configuration mode
    :param spatial_fraction: Expected proportion of variance from spatial field (0-1)
    :param user_range_km: User-specified range in km (for custom mode)
    :param alpha: Smoothness parameter (1 for Matérn ν=1/2, 2 for ν=3/2)
    
    :returns: Stan data with automatic prior configuration
    """
    # Ensure scale diagnostics are computed
    if not hasattr(mesh, 'scale_diagnostics'):
        scale_diag = mesh.compute_scale_diagnostics(verbose=False)
    else:
        scale_diag = mesh.scale_diagnostics
    
    # Extract key quantities
    spatial_scale = scale_diag['spatial_scale']
    suggestions = scale_diag['suggestions']
    
    # Convert units if needed
    unit_to_km = mesh.projection_info.get('unit_to_km', 1.0)
    
    # Handle custom range specification
    if prior_mode == "custom" and user_range_km is not None:
        # Convert user's km specification to model units
        user_range_units = user_range_km / unit_to_km
        estimated_range = user_range_units
    else:
        estimated_range = spatial_scale['estimated_range']
    
    # Map prior modes
    mode_map = {"auto": 0, "tight": 1, "medium": 2, "wide": 3, "custom": 0}
    
    # Compute log determinant
    log_det_Q_base = compute_log_det_Q_base(Q_matrix_csr)
    
    # Prepare Stan data
    stan_data = {
        # Observation data
        'N_obs': len(y),
        'N_mesh': mesh.vertices.shape[0],
        'y': y.tolist(),
        
        # CSR matrices
        'A_nnz': len(A_matrix_csr[2]),
        'A_w': A_matrix_csr[2].tolist(),
        'A_v': A_matrix_csr[1].tolist(),
        'A_u': A_matrix_csr[0].tolist(),
        
        'Q_nnz': len(Q_matrix_csr[2]),
        'Q_w': Q_matrix_csr[2].tolist(),
        'Q_v': Q_matrix_csr[1].tolist(),
        'Q_u': Q_matrix_csr[0].tolist(),
        'log_det_Q_base': log_det_Q_base,
        
        # Scale information from mesh diagnostics
        'estimated_range': estimated_range,
        'min_distance': spatial_scale['min_distance'],
        'median_distance': spatial_scale['median_distance'],
        'mesh_extent': suggestions['mesh_extent'],
        
        # Data characteristics
        'data_sd': np.std(y),
        'coordinate_units_to_km': unit_to_km,
        
        # Prior control
        'prior_mode': mode_map[prior_mode],
        'spatial_fraction': spatial_fraction,
        
        # SPDE parameters
        'alpha': alpha
    }
    
    return stan_data


def suggest_prior_mode(
    mesh: SPDEMesh, 
    y: np.ndarray, 
    domain_knowledge: Optional[DomainKnowledge] = None
) -> PriorMode:
    """
    Suggest appropriate prior mode based on mesh and data characteristics.
    
    :param mesh: Configured mesh object
    :param y: Observation values
    :param domain_knowledge: Domain hint ("environmental", "disease", "economic", etc.)
    
    :returns: Suggested prior_mode
    """
    # Get mesh diagnostics
    if not hasattr(mesh, 'scale_diagnostics'):
        scale_diag = mesh.compute_scale_diagnostics(verbose=False)
    else:
        scale_diag = mesh.scale_diagnostics
    
    # Analyze observation density
    n_obs = len(mesh.coords)
    mesh_area = scale_diag['diagnostics']['total_area']
    obs_density = n_obs / mesh_area if mesh_area > 0 else 0
    
    # Check mesh resolution adequacy
    validation = scale_diag['validation']
    resolution_ok = validation['resolution_ok']
    
    # Domain-specific defaults
    domain_defaults = {
        'environmental': 'medium',  # Moderate spatial correlation expected
        'disease': 'tight',         # Strong local clustering
        'economic': 'wide',         # Broader regional patterns
        'weather': 'medium',        # Atmospheric mixing scales
        'soil': 'tight',           # Highly localized
    }
    
    if domain_knowledge and domain_knowledge in domain_defaults:
        return domain_defaults[domain_knowledge]
    
    # Data-driven suggestion
    if not resolution_ok:
        # Mesh too coarse - use wider priors
        return 'wide'
    
    if obs_density > 10:  # Dense observations
        return 'tight'
    elif obs_density > 1:
        return 'medium'
    else:
        return 'wide'


def create_prior_report(stan_data: Dict, mesh: SPDEMesh) -> str:
    """
    Generate human-readable report of prior choices.
    
    :param stan_data: Prepared Stan data with prior parameters
    :param mesh: Mesh object with diagnostics
    
    :returns: Formatted report of prior specifications
    """
    unit_to_km = stan_data['coordinate_units_to_km']
    units = "km" if unit_to_km == 1.0 else "m" if unit_to_km == 0.001 else "units"
    
    # Compute implied prior medians (for PC priors)
    range_median = stan_data['estimated_range']
    range_km = range_median * unit_to_km
    
    mode_names = {0: "automatic", 1: "tight", 2: "medium", 3: "wide"}
    mode = mode_names[stan_data['prior_mode']]
    
    report = f"""
SPDE Prior Configuration Report
================================
Prior Mode: {mode}
Spatial Fraction: {stan_data['spatial_fraction']:.1%}

Spatial Scale Estimates:
  - Estimated range: {range_median:.1f} {units} ({range_km:.1f} km)
  - Min observation distance: {stan_data['min_distance']:.2f} {units}
  - Median observation distance: {stan_data['median_distance']:.1f} {units}
  - Study region extent: {stan_data['mesh_extent']:.1f} {units}

Implied Priors:
  - Median spatial range: ~{range_median:.1f} {units}
  - Expected spatial SD: ~{stan_data['data_sd'] * np.sqrt(stan_data['spatial_fraction']):.3f}
  - Observation noise SD: determined by data

Mesh Configuration:
  - Vertices: {mesh.vertices.shape[0]:,}
  - Triangles: {mesh.triangles.shape[0]:,}"""
    
    if hasattr(mesh, 'diagnostics'):
        report += f"\n  - Mesh/obs ratio: {mesh.diagnostics['mesh_to_obs_ratio']:.1f}"
    
    if hasattr(mesh, 'scale_diagnostics'):
        if not mesh.scale_diagnostics['validation']['resolution_ok']:
            report += "\n\nWARNING: Mesh resolution may be too coarse for estimated spatial scale"
    
    return report


def compute_log_det_Q_base(Q_matrix_csr: Tuple) -> float:
    """
    Compute log determinant of sparse precision matrix.
    
    :param Q_matrix_csr: CSR format (row_ptr, col_idx, values)
    
    :returns: Log determinant of Q
    """
    # Reconstruct sparse matrix
    row_ptr, col_idx, values = Q_matrix_csr
    n = len(row_ptr) - 1
    Q = csr_matrix((values, col_idx, row_ptr), shape=(n, n))
    
    # Use sparse LU decomposition
    lu = splu(Q.tocsc())
    diagL = lu.L.diagonal()
    diagU = lu.U.diagonal()
    
    # Log det = sum of logs of diagonal elements
    log_det = np.sum(np.log(np.abs(diagL))) + np.sum(np.log(np.abs(diagU)))
    
    return log_det


def sparse_to_stan_csr(matrix: csr_matrix) -> Tuple:
    """
    Convert scipy sparse matrix to Stan CSR format.
    
    :param matrix: Scipy sparse matrix
    
    :returns: (row_ptr, col_idx, values) with 1-based indexing for Stan
    """
    # Stan uses 1-based indexing
    row_ptr = matrix.indptr + 1
    col_idx = matrix.indices + 1
    values = matrix.data
    
    return (row_ptr, col_idx, values)


class StanSPDE:
    """
    SPDE interface for Stan with automatic mesh and prior configuration.
    
    This class provides a streamlined API for Stan SPDE modeling that:
    - Always uses adaptive mesh generation
    - Automatically configures PC priors based on data
    - Provides clear diagnostic reports
    - Outputs Stan-ready data
    
    Examples
    --------
    >>> # Basic usage
    >>> spde = StanSPDE(coords_raw, y_obs)
    >>> stan_data = spde.prepare_stan_data()
    >>> print(spde.get_prior_report())
    
    >>> # With domain knowledge
    >>> spde = StanSPDE(coords_raw, y_obs, domain_knowledge="environmental")
    >>> stan_data = spde.prepare_stan_data(prior_mode="tight")
    """
    
    def __init__(
        self,
        coords_raw: np.ndarray,
        y_obs: np.ndarray,
        domain_knowledge: Optional[DomainKnowledge] = None,
        extension_factor: float = 0.2,
        alpha: int = 2,
        verbose: bool = True
    ):
        """
        Initialize SPDE model with automatic configuration.
        
        :param coords_raw: Raw coordinates (lon/lat or projected)
        :param y_obs: Observations at coordinates
        :param domain_knowledge: Domain hint for prior selection
        :param extension_factor: Boundary extension factor (default 0.2)
        :param alpha: Smoothness (1 for Matérn ν=1/2, 2 for ν=3/2)
        :param verbose: Print progress
        """
        self.coords_raw = coords_raw
        self.y_obs = y_obs
        self.domain_knowledge = domain_knowledge
        self.extension_factor = extension_factor
        self.alpha = alpha
        self.verbose = verbose
        
        # Process coordinates
        self.coords_clean, self.kept_idx, self.proj_info = preprocess_coords(
            coords_raw
        )
        self.y_clean = y_obs[self.kept_idx]
        
        # Create adaptive mesh
        self.mesh = SPDEMesh(self.coords_clean, self.proj_info)
        self.vertices, self.triangles = self.mesh.create_adaptive_mesh(
            extension_factor=extension_factor,
            verbose=verbose
        )
        
        # Compute FEM matrices
        self.C, self.G, self.A, self.Q_base = compute_fem_matrices(
            self.vertices,
            self.triangles,
            self.coords_clean,
            kappa=1.0,  # Reference value
            alpha=alpha,
            verbose=verbose
        )
        
        # Convert to CSR format
        self.A_csr = sparse_to_stan_csr(self.A)
        self.Q_csr = sparse_to_stan_csr(self.Q_base)
        
        # Suggest prior mode if not specified
        self.suggested_prior = suggest_prior_mode(
            self.mesh, self.y_clean, domain_knowledge
        )
        
        self.stan_data = None
    
    def prepare_stan_data(
        self,
        prior_mode: Optional[PriorMode] = None,
        spatial_fraction: float = 0.5,
        user_range_km: Optional[float] = None
    ) -> Dict:
        """
        Prepare Stan data with automatic prior configuration.
        
        :param prior_mode: Override suggested prior mode
        :param spatial_fraction: Expected spatial variance proportion
        :param user_range_km: User-specified range in km
        
        :returns: Stan-ready data
        """
        if prior_mode is None:
            prior_mode = self.suggested_prior
            if self.verbose:
                print(f"Using suggested prior mode: {prior_mode}")
        
        self.stan_data = prepare_stan_data_with_priors(
            y=self.y_clean,
            coords=self.coords_clean,
            mesh=self.mesh,
            A_matrix_csr=self.A_csr,
            Q_matrix_csr=self.Q_csr,
            prior_mode=prior_mode,
            spatial_fraction=spatial_fraction,
            user_range_km=user_range_km,
            alpha=self.alpha
        )
        
        return self.stan_data
    
    def get_prior_report(self) -> str:
        """
        Get human-readable prior configuration report.
        
        :returns: Formatted report
        """
        if self.stan_data is None:
            self.prepare_stan_data()
        
        return create_prior_report(self.stan_data, self.mesh)
    
    def get_mesh_diagnostics(self) -> Dict:
        """
        Get mesh quality diagnostics.
        
        :returns: Mesh diagnostics
        """
        return self.mesh.diagnostics
    
    def get_scale_diagnostics(self) -> Dict:
        """
        Get spatial scale diagnostics.
        
        :returns: Scale diagnostics
        """
        if not hasattr(self.mesh, 'scale_diagnostics'):
            return self.mesh.compute_scale_diagnostics(verbose=False)
        return self.mesh.scale_diagnostics
