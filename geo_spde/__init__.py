"""
GEO_SPDE: transform geospatial coordinates to spatial adjacency matrices
"""

__version__ = "0.2.0"
__author__ = "Mitzi Morris"

# Core coordinate processing
from .coords import (
    preprocess_coords,
    estimate_characteristic_scale,
)

# Mesh generation
from .mesh import SPDEMesh

# Matrix computation
from .matrices import (
    compute_fem_matrices,
    compute_fem_matrices_scaled,
    select_reference_parameters,
    check_precision_conditioning,
)

# Stan data preparation
from .stan_data_prep import (
    prepare_stan_data,
    validate_prior_compatibility,
    generate_synthetic_data,
    create_stan_init
)

# Exceptions
from .exceptions import (
    GeoSpdeError,
    CoordsError,
    MeshError,
    MatrixError,
    ParameterScaleError,
    ConditioningError
)

__all__ = [
    # Coordinate processing
    'preprocess_coords',
    'estimate_characteristic_scale',
    
    # Mesh generation
    'SPDEMesh',
    
    # Matrix computation
    'compute_fem_matrices',
    'compute_fem_matrices_scaled',
    'select_reference_parameters',
    'check_precision_conditioning',
    
    # Stan integration
    'prepare_stan_data',
    'validate_prior_compatibility',
    'generate_synthetic_data',
    'create_stan_init',
    
    # Exceptions
    'GeoSpdeError',
    'CoordsError',
    'MeshError',
    'MatrixError',
    'ParameterScaleError',
    'ConditioningError'
]
