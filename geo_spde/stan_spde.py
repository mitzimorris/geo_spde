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

from .coords import preprocess_coords
from .mesh import SPDEMesh
from .matrices import compute_fem_matrices
from .exceptions import ParameterScaleError
from .utils import sparse_to_stan_csr
from .pc_priors import compute_pc_prior_params

PriorMode = Literal["auto", "tight", "medium", "wide", "custom"]
DomainKnowledge = Literal["environmental", "disease", "economic", "weather", "soil"]


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
    ) -> None:
        """Initialize SPDE model with automatic configuration.
        
        :param coords_raw: Raw coordinates (lon/lat or projected)
        :param y_obs: Observations at coordinates
        :param domain_knowledge: Domain hint for prior selection
        :param extension_factor: Boundary extension factor (default 0.2)
        :param alpha: Smoothness (1 for Matern nu=1/2, 2 for nu=3/2)
        :param verbose: Print progress
        """
        self.coords_raw = coords_raw
        self.y_obs = y_obs
        self.domain_knowledge = domain_knowledge
        self.extension_factor = extension_factor
        self.alpha = alpha
        self.verbose = verbose
        
        self.coords_clean, self.kept_idx, self.proj_info = preprocess_coords(
            coords_raw
        )
        self.y_clean = y_obs[self.kept_idx]
        
        self.mesh = SPDEMesh(self.coords_clean, self.proj_info)
        self.vertices, self.triangles = self.mesh.create_adaptive_mesh(
            extension_factor=extension_factor,
            verbose=verbose
        )
        
        self.C, self.G, self.A, self.Q_base = compute_fem_matrices(
            self.vertices,
            self.triangles,
            self.coords_clean,
            kappa=1.0,  # Reference value
            alpha=alpha,
            verbose=verbose
        )
        
        self.A_csr = sparse_to_stan_csr(self.A)
        self.Q_csr = sparse_to_stan_csr(self.Q_base)
        
        self.suggested_prior = self._suggest_prior_mode(
            self.mesh, self.y_clean, domain_knowledge
        )
        
        self.stan_data = None
    
    def prepare_stan_data(
        self,
        prior_mode: Optional[PriorMode] = None,
        spatial_fraction: float = 0.5,
        user_range_km: Optional[float] = None
    ) -> Dict:
        """Prepare Stan data with automatic prior configuration.
        
        :param prior_mode: Override suggested prior mode
        :param spatial_fraction: Expected spatial variance proportion
        :param user_range_km: User-specified range in km
        :return: Stan-ready data
        """
        if prior_mode is None:
            prior_mode = self.suggested_prior
            if self.verbose:
                print(f"Using suggested prior mode: {prior_mode}")
        
        self.stan_data = self._prepare_stan_data_with_priors(
            y=self.y_clean,
            coords=self.coords_clean,
            mesh=self.mesh,
            A_matrix_csr=self.A_csr,
            Q_matrix_csr=self.Q_csr,
            Q_base=self.Q_base,
            prior_mode=prior_mode,
            spatial_fraction=spatial_fraction,
            user_range_km=user_range_km,
            alpha=self.alpha
        )
        
        return self.stan_data
    
    def get_prior_report(self) -> str:
        """Get human-readable prior configuration report.
        
        :return: Formatted report
        """
        if self.stan_data is None:
            self.prepare_stan_data()
        
        return self._create_prior_report(self.stan_data, self.mesh)
    
    def get_mesh_diagnostics(self) -> Dict:
        """Get mesh quality diagnostics.
        
        :return: Mesh diagnostics
        """
        return self.mesh.diagnostics
    
    def get_scale_diagnostics(self) -> Dict:
        """Get spatial scale diagnostics.
        
        :return: Scale diagnostics
        """
        if not hasattr(self.mesh, 'scale_diagnostics'):
            return self.mesh.compute_scale_diagnostics(verbose=False)
        return self.mesh.scale_diagnostics
    
    def _compute_log_det_Q_base(self, Q_matrix: csr_matrix) -> float:
        """Compute log determinant of sparse precision matrix.
        
        :param Q_matrix: Scipy sparse CSR matrix (0-based indexing)
        :return: Log determinant of Q
        """
        Q = Q_matrix
        
        # Use sparse LU for matrix that may be singular
        try:
            lu = splu(Q.tocsc())
            diagL = lu.L.diagonal()
            diagU = lu.U.diagonal()
            
            log_det = np.sum(np.log(np.abs(diagL))) + np.sum(np.log(np.abs(diagU)))
        except RuntimeError as e:
            if "exactly singular" in str(e):
                warnings.warn("Precision matrix is singular, returning -inf for log determinant")
                log_det = -np.inf
            else:
                raise
        
        return log_det
    
    def _create_prior_report(self, stan_data: Dict, mesh: SPDEMesh) -> str:
        """Generate human-readable report of prior choices.
        
        :param stan_data: Prepared Stan data with prior parameters
        :param mesh: Mesh object with diagnostics
        :return: Formatted report of prior specifications
        """
        unit_to_km = stan_data['coordinate_units_to_km']
        units = "km" if unit_to_km == 1.0 else "m" if unit_to_km == 0.001 else "units"
        
        range_median = stan_data['estimated_range']
        range_km = range_median * unit_to_km
        
        mode_names = {0: "auto", 1: "tight", 2: "medium", 3: "wide"}
        mode_name = mode_names.get(stan_data['prior_mode'], "custom")
        
        report_parts = [
            f"SPDE Prior Configuration Report",
            f"=====================================",
            f"",
            f"Prior Mode: {mode_name}",
            f"Spatial Range (median): {range_km:.2f} {units}",
            f"Spatial Variance (median): {stan_data['estimated_sigma']:.3f}",
            f"Alpha (smoothness): {stan_data['alpha']} (Matern nu={(stan_data['alpha']-1)/2:.1f})",
            f"",
            f"PC Prior Hyperparameters:",
            f"  Range: P(rho < {stan_data['rho_0'] * unit_to_km:.2f} {units}) = {stan_data['alpha_rho']:.3f}",
            f"  Variance: P(sigma < {stan_data['sigma_0']:.3f}) = {stan_data['alpha_sigma']:.3f}",
            f"",
            f"Mesh Information:",
            f"  Vertices: {stan_data['N_mesh']:,}",
            f"  Observation points: {stan_data['N_obs']:,}",
            f"  Mesh/obs ratio: {stan_data['N_mesh']/stan_data['N_obs']:.1f}",
        ]
        
        return "\n".join(report_parts)
    
    def _suggest_prior_mode(
        self,
        mesh: SPDEMesh, 
        y: np.ndarray, 
        domain_knowledge: Optional[DomainKnowledge] = None
    ) -> PriorMode:
        """Suggest appropriate prior mode based on mesh and data characteristics.
        
        :param mesh: Configured mesh object
        :param y: Observation values
        :param domain_knowledge: Domain hint ("environmental", "disease", "economic", etc.)
        :return: Suggested prior_mode
        """
        if not hasattr(mesh, 'scale_diagnostics'):
            scale_diag = mesh.compute_scale_diagnostics(verbose=False)
        else:
            scale_diag = mesh.scale_diagnostics
        
        n_obs = len(mesh.coords)
        mesh_extent = scale_diag['suggestions']['mesh_extent']
        mesh_area = mesh_extent ** 2
        obs_density = n_obs / mesh_area if mesh_area > 0 else 0
        
        validation = scale_diag['validation']
        resolution_ok = validation['resolution_ok']
        
        domain_defaults = {
            'environmental': 'medium',  # Moderate spatial correlation expected
            'disease': 'tight',         # Strong local clustering
            'economic': 'wide',         # Broader regional patterns
            'weather': 'medium',        # Atmospheric mixing scales
            'soil': 'tight',           # Highly localized
        }
        
        if domain_knowledge and domain_knowledge in domain_defaults:
            return domain_defaults[domain_knowledge]
        
        y_cv = np.std(y) / np.abs(np.mean(y)) if np.mean(y) != 0 else np.std(y)
        
        if obs_density > 0.01:
            return "tight" if y_cv > 0.5 else "medium" 
        elif obs_density < 0.001:
            return "wide"
        else:
            return "medium" if resolution_ok else "wide"
    
    def _prepare_stan_data_with_priors(
        self,
        y: np.ndarray,
        coords: np.ndarray,
        mesh: SPDEMesh,
        A_matrix_csr: Tuple,
        Q_matrix_csr: Tuple,
        Q_base: csr_matrix,
        prior_mode: PriorMode = "auto",
        spatial_fraction: float = 0.5,
        user_range_km: Optional[float] = None,
        alpha: int = 2
    ) -> Dict:
        """Prepare Stan data with automatic prior configuration using mesh diagnostics.
        
        :param y: Observations at coordinate locations
        :param coords: Cleaned coordinates from preprocess_coords()
        :param mesh: Mesh object after create_adaptive_mesh()
        :param A_matrix_csr: CSR format projection matrix from FEM computation
        :param Q_matrix_csr: CSR format precision matrix from FEM computation
        :param Q_base: Original 0-based precision matrix for log determinant
        :param prior_mode: Prior configuration mode
        :param spatial_fraction: Expected proportion of variance from spatial field (0-1)
        :param user_range_km: User-specified range in km (for custom mode)
        :param alpha: Smoothness parameter (1 for Matern nu=1/2, 2 for nu=3/2)
        :return: Stan data with automatic prior configuration
        """
        if not hasattr(mesh, 'scale_diagnostics'):
            scale_diag = mesh.compute_scale_diagnostics(verbose=False)
        else:
            scale_diag = mesh.scale_diagnostics
        
        spatial_scale = scale_diag['spatial_scale']
        suggestions = scale_diag['suggestions']
        
        unit_to_km = mesh.projection_info.get('unit_to_km', 1.0)
        
        if prior_mode == "custom" and user_range_km is not None:
            user_range_units = user_range_km / unit_to_km
            estimated_range = user_range_units
        else:
            estimated_range = spatial_scale['estimated_range']
        
        mode_map = {"auto": 0, "tight": 1, "medium": 2, "wide": 3, "custom": 0}
        
        # Compute PC prior parameters
        pc_params = compute_pc_prior_params(
            estimated_range=estimated_range,
            data_sd=np.std(y),
            prior_mode=prior_mode,
            spatial_fraction=spatial_fraction,
            mesh_diagnostics=scale_diag,
            alpha=alpha
        )
        
        log_det_Q_base = self._compute_log_det_Q_base(Q_base)
        
        stan_data = {
            # Observation data
            'N_obs': len(y),
            'N_mesh': mesh.vertices.shape[0],
            'y': y.tolist(),
            
            # Sparse matrices (1-based indexing for Stan)
            'A_nnz': len(A_matrix_csr[2]),
            'A_w': A_matrix_csr[2].tolist(),
            'A_v': A_matrix_csr[1].tolist(),
            'A_u': A_matrix_csr[0].tolist(),
            
            'Q_nnz': len(Q_matrix_csr[2]),
            'Q_w': Q_matrix_csr[2].tolist(),
            'Q_v': Q_matrix_csr[1].tolist(),
            'Q_u': Q_matrix_csr[0].tolist(),
            'log_det_Q_base': log_det_Q_base,
            
            # Spatial scale information from mesh diagnostics
            'estimated_range': estimated_range,
            'min_distance': spatial_scale['min_distance'],
            'median_distance': spatial_scale['median_distance'],
            'mesh_extent': suggestions['mesh_extent'],
            'data_sd': np.std(y),
            'coordinate_units_to_km': unit_to_km,
            
            # Prior configuration
            'prior_mode': mode_map[prior_mode],
            'spatial_fraction': spatial_fraction,
            'alpha': alpha,
            
            # PC prior parameters
            'rho_0': pc_params['rho_0'],
            'alpha_rho': pc_params['alpha_rho'],
            'lambda_rho': pc_params['lambda_rho'],
            'kappa_0': pc_params['kappa_0'],
            'sigma_0': pc_params['sigma_0'],
            'alpha_sigma': pc_params['alpha_sigma'],
            'lambda_sigma': pc_params['lambda_sigma'],
            'estimated_sigma': pc_params['expected_sigma'],
        }
        
        return stan_data
