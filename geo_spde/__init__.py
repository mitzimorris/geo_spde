"""
GEO_SPDE: Simplified SPDE modeling with automatic PC priors and adaptive mesh
"""

__version__ = "0.3.0"
__author__ = "Mitzi Morris"

# Main simplified API
from .stan_spde import (
    StanSPDE,
    sparse_to_stan_csr,
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
    'sparse_to_stan_csr',
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
    
    # Exceptions
    'GeoSpdeError',
    'CoordsError',
    'MeshError',
    'MatrixError',
]
