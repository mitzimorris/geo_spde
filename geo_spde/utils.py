"""
Utility functions for working with SPDE matrices in Stan
"""

from typing import Tuple
from scipy.sparse import csr_matrix

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
