"""
GEO_SPDE: Simplified SPDE modeling with automatic PC priors and adaptive mesh
"""

__version__ = "0.4.0"
__author__ = "Mitzi Morris"

# Main simplified API
from .stan_spde import (
    StanSPDE,
    PriorMode,
    DomainKnowledge,
)

# PC Prior utilities
from .pc_priors import (
    compute_pc_prior_params,
    validate_pc_priors,
    sample_from_pc_prior,
)

# Core components (for advanced users)
from .coords import preprocess_coords
from .mesh import SPDEMesh
from .matrices import compute_fem_matrices
from .utils import sparse_to_stan_csr

# Exceptions
from .exceptions import (
    GeoSpdeError,
    CoordsError,
    MeshError,
    MatrixError,
)

__all__ = [
    # Main API
    'StanSPDE',
    'PriorMode', 
    'DomainKnowledge',
    
    # PC Priors
    'compute_pc_prior_params',
    'validate_pc_priors',
    'sample_from_pc_prior',
    
    # Core components
    'preprocess_coords',
    'SPDEMesh',
    'compute_fem_matrices',
    'sparse_to_stan_csr',
    
    # Exceptions
    'GeoSpdeError',
    'CoordsError',
    'MeshError',
    'MatrixError',
]
