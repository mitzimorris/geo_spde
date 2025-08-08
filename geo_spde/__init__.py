"""
GEO_SPDE: transform geospatial coordinates to spatial adjacency matrices
"""

__version__ = "0.1.0"
__author__ = "Mitzi Morris"

# Main public API
from .coords import preprocess_coords
from .mesh import SPDEMesh
# from .matrices import compute_projector_matrix, construct_precision_matrix
from .exceptions import GeoSpdeError, CoordsError, MeshError

__all__ = [
    'preprocess_coords',
    'SPDEMesh',
    # 'compute_projector_matrix',
    # 'construct_precision_matrix',
    'GeoSpdeError',
    'CoordsError', 
    'MeshError'
]
