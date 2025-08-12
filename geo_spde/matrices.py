"""
FEM matrix computation for SPDE approximations.

This module computes the sparse matrices (C, G, A, Q) needed for Stan's
embedded Laplace approximation of SPDE models using vectorized operations
for maximum computational efficiency.
"""

import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay, KDTree
from typing import Tuple, Optional
import warnings

from geo_spde.exceptions import MatrixError


def compute_fem_matrices(
    vertices: np.ndarray,
    triangles: np.ndarray,
    obs_coords: np.ndarray,
    kappa: Optional[float] = None,
    alpha: int = 1,
    verbose: bool = True
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, Optional[sparse.csr_matrix]]:
    """
    Compute all FEM matrices needed for SPDE approximation.
    
    Parameters
    ----------
    vertices : np.ndarray
        Mesh vertex coordinates of shape (n_mesh, 2)
    triangles : np.ndarray
        Triangle connectivity of shape (n_tri, 3)
    obs_coords : np.ndarray
        Observation coordinates of shape (n_obs, 2)
    kappa : float, optional
        SPDE parameter controlling spatial range. If None, Q matrix not computed
    alpha : int
        SPDE smoothness parameter:
        - alpha=1 for Matern nu=1/2 (exponential covariance)
        - alpha=2 for Matern nu=3/2 (once differentiable)
    verbose : bool
        Print computation progress
        
    Returns
    -------
    C : sparse.csr_matrix
        Mass matrix (n_mesh x n_mesh)
    G : sparse.csr_matrix
        Stiffness matrix (n_mesh x n_mesh)
    A : sparse.csr_matrix
        Projector matrix (n_obs x n_mesh)
    Q : sparse.csr_matrix or None
        Precision matrix (n_mesh x n_mesh), None if kappa not provided
    """
    n_mesh = len(vertices)
    n_obs = len(obs_coords)
    n_tri = len(triangles)
    
    if verbose:
        print(f"Computing FEM matrices (vectorized):")
        print(f"  Mesh: {n_mesh} vertices, {n_tri} triangles")
        print(f"  Observations: {n_obs} points")
    
    # Compute C matrix (Mass matrix)
    if verbose:
        print("  Computing C matrix (Mass)...")
    C = _compute_mass_matrix_vectorized(vertices, triangles)
    
    # Compute G matrix (Stiffness matrix) 
    if verbose:
        print("  Computing G matrix (Stiffness)...")
    G = _compute_stiffness_matrix_vectorized(vertices, triangles)
    
    # Compute A matrix (Projector matrix)
    if verbose:
        print("  Computing A matrix (Projector)...")
    A = _compute_projector_matrix_vectorized(vertices, triangles, obs_coords)
    
    # Compute Q matrix (Precision matrix) if kappa provided
    Q = None
    if kappa is not None:
        if verbose:
            matern_nu = "1/2" if alpha == 1 else "3/2"
            print(f"  Computing Q matrix (Matern nu={matern_nu} precision)...")
        
        if alpha == 1:
            Q = compute_matern_precision_nu_half(C, G, kappa)
        elif alpha == 2:
            Q = compute_matern_precision_nu_three_halves(C, G, kappa)
        else:
            raise ValueError(
                f"Alpha={alpha} not supported. "
                "Use alpha=1 for Matern nu=1/2 or alpha=2 for Matern nu=3/2"
            )
    
    if verbose:
        print("  FEM matrix computation complete")
        _print_matrix_diagnostics(C, G, A, Q)
    
    return C, G, A, Q


def compute_matern_precision_nu_half(
    C: sparse.csr_matrix, 
    G: sparse.csr_matrix, 
    kappa: float
) -> sparse.csr_matrix:
    """
    Compute precision matrix for Matern covariance with nu = 1/2.
    
    The Matern nu = 1/2 field (exponential covariance) is the solution to:
        (kappa^2 - delta)u(s) = W(s)
    
    where delta is the Laplacian operator.
    
    The discretized precision matrix is:
        Q = kappa^2 * C + G
    
    This gives an exponentially decaying spatial correlation with:
        - Range parameter: rho = 1/kappa
        - Smoothness: nu = 1/2 (rough, non-differentiable field)
    
    Parameters
    ----------
    C : sparse.csr_matrix
        Mass matrix from FEM discretization
    G : sparse.csr_matrix
        Stiffness matrix from FEM discretization
    kappa : float
        SPDE range parameter (kappa = 1/rho where rho is correlation range)
        
    Returns
    -------
    sparse.csr_matrix
        Sparse precision matrix Q for Matern nu = 1/2
    """
    return kappa**2 * C + G


def compute_matern_precision_nu_three_halves(
    C: sparse.csr_matrix, 
    G: sparse.csr_matrix, 
    kappa: float
) -> sparse.csr_matrix:
    """
    Compute precision matrix for Matern covariance with nu = 3/2.
    
    The Matern nu = 3/2 field (smoother than nu = 1/2) is the solution to:
        (kappa^2 - delta)^2 u(s) = W(s)
    
    where delta is the Laplacian operator.
    
    The discretized precision matrix is:
        Q = (kappa^2 * C + G) * C^(-1) * (kappa^2 * C + G)
    
    This gives a once-differentiable field with:
        - Range parameter: rho = sqrt(3)/kappa  
        - Smoothness: nu = 3/2 (once differentiable)
    
    NOTE: This computation is more expensive as it requires solving
    linear systems with the mass matrix C.
    
    Parameters
    ----------
    C : sparse.csr_matrix
        Mass matrix from FEM discretization
    G : sparse.csr_matrix
        Stiffness matrix from FEM discretization
    kappa : float
        SPDE range parameter (kappa = sqrt(3)/rho where rho is correlation range)
        
    Returns
    -------
    sparse.csr_matrix
        Sparse precision matrix Q for Matern nu = 3/2
    """
    # First compute kappa^2 * C + G
    B = kappa**2 * C + G
    
    # We need to compute B * C^(-1) * B
    # Rather than inverting C, we solve C * X = B^T for X, then compute B * X
    # This maintains sparsity better than explicit inversion
    
    from scipy.sparse.linalg import spsolve
    
    # Solve C * X = B^T column by column (expensive but necessary)
    n = B.shape[0]
    X = sparse.lil_matrix((n, n))
    
    # Convert to CSC for efficient column access
    B_csc = B.tocsc()
    
    for j in range(n):
        # Solve C * x = B[:, j]
        b_col = B_csc[:, j].toarray().ravel()
        if np.any(b_col != 0):
            x = spsolve(C, b_col)
            X[:, j] = x.reshape(-1, 1)
    
    # Convert back to CSR and compute Q = B * X
    X = X.tocsr()
    Q = B @ X
    
    return Q


def _compute_mass_matrix_vectorized(vertices: np.ndarray, triangles: np.ndarray) -> sparse.csr_matrix:
    """
    Compute mass matrix C where C[i,j] = integral(psi_i(s) * psi_j(s) ds) using vectorized operations.
    
    Here psi_i are the linear basis functions on the triangular mesh.
    
    Parameters
    ----------
    vertices : np.ndarray
        Mesh vertex coordinates
    triangles : np.ndarray
        Triangle connectivity
        
    Returns
    -------
    sparse.csr_matrix
        Mass matrix
    """
    n_vertices = len(vertices)
    n_triangles = len(triangles)
    
    # Get all triangle vertices at once: (n_tri, 3, 2)
    tri_coords = vertices[triangles]
    
    # Vectorized area computation for all triangles
    areas = _triangle_areas_vectorized(tri_coords)
    
    # Check for degenerate triangles
    degenerate_mask = areas <= 1e-12
    if np.any(degenerate_mask):
        n_degenerate = np.sum(degenerate_mask)
        warnings.warn(f"Found {n_degenerate} degenerate triangles, skipping")
        # Filter out degenerate triangles
        valid_mask = ~degenerate_mask
        triangles = triangles[valid_mask]
        areas = areas[valid_mask]
        n_triangles = len(triangles)
    
    # Mass matrix entries for linear elements:
    # Diagonal terms: area/6, Off-diagonal terms: area/12
    diagonal_vals = areas / 6.0
    off_diagonal_vals = areas / 12.0
    
    # Pre-allocate sparse matrix arrays
    # Each triangle contributes 9 entries (3x3)
    total_entries = n_triangles * 9
    row_indices = np.empty(total_entries, dtype=np.int32)
    col_indices = np.empty(total_entries, dtype=np.int32)
    data = np.empty(total_entries, dtype=np.float64)
    
    # Fill entries for all triangles
    entry_idx = 0
    for local_i in range(3):
        for local_j in range(3):
            # Global indices for this (i,j) pair across all triangles
            start_idx = entry_idx
            end_idx = entry_idx + n_triangles
            
            row_indices[start_idx:end_idx] = triangles[:, local_i]
            col_indices[start_idx:end_idx] = triangles[:, local_j]
            
            if local_i == local_j:
                # Diagonal entry
                data[start_idx:end_idx] = diagonal_vals
            else:
                # Off-diagonal entry
                data[start_idx:end_idx] = off_diagonal_vals
            
            entry_idx += n_triangles
    
    # Assemble sparse matrix and sum duplicates
    C = sparse.coo_matrix((data, (row_indices, col_indices)), 
                          shape=(n_vertices, n_vertices))
    C = C.tocsr()
    C.eliminate_zeros()
    
    return C


def _compute_stiffness_matrix_vectorized(vertices: np.ndarray, triangles: np.ndarray) -> sparse.csr_matrix:
    """
    Compute stiffness matrix G where G[i,j] = integral(grad(psi_i) . grad(psi_j) ds) using vectorized operations.
    
    Here psi_i are the linear basis functions and grad denotes the gradient.
    
    Parameters
    ----------
    vertices : np.ndarray
        Mesh vertex coordinates
    triangles : np.ndarray
        Triangle connectivity
        
    Returns
    -------
    sparse.csr_matrix
        Stiffness matrix
    """
    n_vertices = len(vertices)
    n_triangles = len(triangles)
    
    # Get all triangle vertices at once: (n_tri, 3, 2)
    tri_coords = vertices[triangles]
    
    # Vectorized area computation
    areas = _triangle_areas_vectorized(tri_coords)
    
    # Check for degenerate triangles
    degenerate_mask = areas <= 1e-12
    if np.any(degenerate_mask):
        n_degenerate = np.sum(degenerate_mask)
        warnings.warn(f"Found {n_degenerate} degenerate triangles, skipping")
        valid_mask = ~degenerate_mask
        triangles = triangles[valid_mask]
        tri_coords = tri_coords[valid_mask]
        areas = areas[valid_mask]
        n_triangles = len(triangles)
    
    # Compute gradients for all triangles at once: (n_tri, 3, 2)
    gradients = _compute_basis_gradients_vectorized(tri_coords)
    
    # Pre-allocate sparse matrix arrays
    total_entries = n_triangles * 9
    row_indices = np.empty(total_entries, dtype=np.int32)
    col_indices = np.empty(total_entries, dtype=np.int32)
    data = np.empty(total_entries, dtype=np.float64)
    
    # Fill entries for all triangles
    entry_idx = 0
    for local_i in range(3):
        for local_j in range(3):
            start_idx = entry_idx
            end_idx = entry_idx + n_triangles
            
            row_indices[start_idx:end_idx] = triangles[:, local_i]
            col_indices[start_idx:end_idx] = triangles[:, local_j]
            
            # Vectorized dot product: (n_tri,) = sum((n_tri, 2) * (n_tri, 2), axis=1)
            dot_products = np.sum(gradients[:, local_i, :] * gradients[:, local_j, :], axis=1)
            stiffness_vals = areas * dot_products
            
            data[start_idx:end_idx] = stiffness_vals
            entry_idx += n_triangles
    
    # Assemble sparse matrix
    G = sparse.coo_matrix((data, (row_indices, col_indices)), 
                          shape=(n_vertices, n_vertices))
    G = G.tocsr()
    G.eliminate_zeros()
    
    return G


def _compute_projector_matrix_vectorized(
    vertices: np.ndarray, 
    triangles: np.ndarray, 
    obs_coords: np.ndarray
) -> sparse.csr_matrix:
    """
    Compute projector matrix A where A[i,k] = psi_k(s_i) using vectorized operations.
    
    Here psi_k is the k-th basis function evaluated at observation location s_i.
    
    Parameters
    ----------
    vertices : np.ndarray
        Mesh vertex coordinates
    triangles : np.ndarray
        Triangle connectivity
    obs_coords : np.ndarray
        Observation coordinates
        
    Returns
    -------
    sparse.csr_matrix
        Projector matrix
    """
    n_obs = len(obs_coords)
    n_vertices = len(vertices)
    
    # Point location using Delaunay triangulation
    tri_finder = Delaunay(vertices)
    simplex_indices = tri_finder.find_simplex(obs_coords)
    
    # Handle points outside convex hull using KDTree
    outside_mask = simplex_indices == -1
    n_outside = np.sum(outside_mask)
    
    if n_outside > 0:
        if n_outside > 0.1 * n_obs:  # Warn if many points outside
            warnings.warn(f"{n_outside} observation points outside mesh convex hull")
        
        # Use KDTree to find nearest triangle centroids
        centroids = np.mean(vertices[triangles], axis=1)
        tree = KDTree(centroids)
        _, nearest_tri_indices = tree.query(obs_coords[outside_mask])
        simplex_indices[outside_mask] = nearest_tri_indices
    
    # Vectorized barycentric coordinate computation
    bary_coords = _compute_barycentric_vectorized(
        obs_coords, vertices, triangles, simplex_indices
    )
    
    # Build sparse matrix efficiently
    # Each observation has exactly 3 non-zero entries
    obs_indices = np.repeat(np.arange(n_obs), 3)
    vertex_indices = triangles[simplex_indices].ravel()
    values = bary_coords.ravel()
    
    # Filter out near-zero values
    nonzero_mask = np.abs(values) > 1e-12
    obs_indices = obs_indices[nonzero_mask]
    vertex_indices = vertex_indices[nonzero_mask]
    values = values[nonzero_mask]
    
    A = sparse.csr_matrix((values, (obs_indices, vertex_indices)), 
                         shape=(n_obs, n_vertices))
    
    return A


def _triangle_areas_vectorized(tri_coords: np.ndarray) -> np.ndarray:
    """
    Compute areas of all triangles using vectorized operations.
    
    Parameters
    ----------
    tri_coords : np.ndarray
        Triangle coordinates of shape (n_tri, 3, 2)
        
    Returns
    -------
    np.ndarray
        Triangle areas of shape (n_tri,)
    """
    v1, v2, v3 = tri_coords[:, 0], tri_coords[:, 1], tri_coords[:, 2]
    
    # Cross product formula: 0.5 * |det([[v2-v1], [v3-v1]])|
    areas = 0.5 * np.abs(
        (v2[:, 0] - v1[:, 0]) * (v3[:, 1] - v1[:, 1]) - 
        (v3[:, 0] - v1[:, 0]) * (v2[:, 1] - v1[:, 1])
    )
    
    return areas


def _compute_basis_gradients_vectorized(tri_coords: np.ndarray) -> np.ndarray:
    """
    Compute gradients of linear basis functions for all triangles.
    
    For linear basis functions on triangles:
    psi_1 = 1 - xi - eta, psi_2 = xi, psi_3 = eta
    where (xi, eta) are local coordinates.
    
    Parameters
    ----------
    tri_coords : np.ndarray
        Triangle coordinates of shape (n_tri, 3, 2)
        
    Returns
    -------
    np.ndarray
        Basis function gradients of shape (n_tri, 3, 2)
    """
    n_tri = tri_coords.shape[0]
    v1, v2, v3 = tri_coords[:, 0], tri_coords[:, 1], tri_coords[:, 2]
    
    # Create transformation matrices B = [v2-v1, v3-v1] for all triangles
    # B shape: (n_tri, 2, 2)
    B = np.stack([v2 - v1, v3 - v1], axis=2)
    
    # Compute determinants
    det_B = B[:, 0, 0] * B[:, 1, 1] - B[:, 0, 1] * B[:, 1, 0]
    
    # Check for degenerate triangles
    degenerate_mask = np.abs(det_B) < 1e-12
    if np.any(degenerate_mask):
        raise MatrixError(f"Found {np.sum(degenerate_mask)} degenerate triangles in gradient computation")
    
    # Compute inverse matrices B_inv: (n_tri, 2, 2)
    B_inv = np.empty_like(B)
    B_inv[:, 0, 0] = B[:, 1, 1] / det_B
    B_inv[:, 0, 1] = -B[:, 0, 1] / det_B
    B_inv[:, 1, 0] = -B[:, 1, 0] / det_B
    B_inv[:, 1, 1] = B[:, 0, 0] / det_B
    
    # Compute gradients of basis functions
    gradients = np.empty((n_tri, 3, 2))
    
    # grad(psi_1) = -grad(xi) - grad(eta) = -(B_inv[0,:] + B_inv[1,:])
    gradients[:, 0, :] = -(B_inv[:, 0, :] + B_inv[:, 1, :])
    
    # grad(psi_2) = grad(xi) = B_inv[0,:]
    gradients[:, 1, :] = B_inv[:, 0, :]
    
    # grad(psi_3) = grad(eta) = B_inv[1,:]
    gradients[:, 2, :] = B_inv[:, 1, :]
    
    return gradients


def _compute_barycentric_vectorized(
    points: np.ndarray,
    vertices: np.ndarray, 
    triangles: np.ndarray,
    triangle_indices: np.ndarray
) -> np.ndarray:
    """
    Compute barycentric coordinates for all points using vectorized operations.
    
    Barycentric coordinates (lambda_1, lambda_2, lambda_3) satisfy:
    point = lambda_1 * v1 + lambda_2 * v2 + lambda_3 * v3
    with lambda_1 + lambda_2 + lambda_3 = 1
    
    Parameters
    ----------
    points : np.ndarray
        Point coordinates of shape (n_points, 2)
    vertices : np.ndarray
        Mesh vertex coordinates
    triangles : np.ndarray
        Triangle connectivity
    triangle_indices : np.ndarray
        Triangle index for each point
        
    Returns
    -------
    np.ndarray
        Barycentric coordinates of shape (n_points, 3)
    """
    n_points = len(points)
    
    # Get triangle vertices for all points: (n_points, 3, 2)
    tri_verts = vertices[triangles[triangle_indices]]
    
    # Extract vertices
    v1, v2, v3 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
    
    # Vectorized area computation using cross product
    def areas_vectorized(p1, p2, p3):
        return 0.5 * np.abs(
            (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - 
            (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
        )
    
    # Total triangle areas
    total_areas = areas_vectorized(v1, v2, v3)
    
    # Sub-triangle areas
    area1 = areas_vectorized(points, v2, v3)  # lambda_1 (opposite v1)
    area2 = areas_vectorized(v1, points, v3)  # lambda_2 (opposite v2)  
    area3 = areas_vectorized(v1, v2, points)  # lambda_3 (opposite v3)
    
    # Check for degenerate triangles
    degenerate_mask = total_areas <= 1e-12
    if np.any(degenerate_mask):
        raise MatrixError(f"Degenerate triangles in barycentric coordinate computation")
    
    # Barycentric coordinates: (n_points, 3)
    bary_coords = np.column_stack([
        area1 / total_areas,
        area2 / total_areas,
        area3 / total_areas
    ])
    
    return bary_coords


def _print_matrix_diagnostics(
    C: sparse.csr_matrix, 
    G: sparse.csr_matrix, 
    A: sparse.csr_matrix, 
    Q: Optional[sparse.csr_matrix]
) -> None:
    """Print matrix diagnostics for user information."""
    print("\nMatrix Diagnostics:")
    
    # C matrix
    c_nnz = C.nnz
    c_density = c_nnz / (C.shape[0] * C.shape[1]) * 100
    print(f"  C matrix: {C.shape[0]}x{C.shape[1]}, {c_nnz:,} non-zeros ({c_density:.2f}% dense)")
    
    # G matrix  
    g_nnz = G.nnz
    g_density = g_nnz / (G.shape[0] * G.shape[1]) * 100
    print(f"  G matrix: {G.shape[0]}x{G.shape[1]}, {g_nnz:,} non-zeros ({g_density:.2f}% dense)")
    
    # A matrix
    a_nnz = A.nnz
    a_density = a_nnz / (A.shape[0] * A.shape[1]) * 100
    avg_entries_per_obs = a_nnz / A.shape[0]
    print(f"  A matrix: {A.shape[0]}x{A.shape[1]}, {a_nnz:,} non-zeros ({a_density:.3f}% dense)")
    print(f"    Average entries per observation: {avg_entries_per_obs:.1f}")
    
    # Q matrix
    if Q is not None:
        q_nnz = Q.nnz
        q_density = q_nnz / (Q.shape[0] * Q.shape[1]) * 100
        print(f"  Q matrix: {Q.shape[0]}x{Q.shape[1]}, {q_nnz:,} non-zeros ({q_density:.2f}% dense)")
    
    # Memory estimates
    memory_mb = (c_nnz + g_nnz + a_nnz) * 8 / (1024 * 1024)
    if Q is not None:
        memory_mb += Q.nnz * 8 / (1024 * 1024)
    print(f"  Estimated memory: {memory_mb:.1f} MB")


# Convenience functions for integration with SPDEMesh
def compute_spde_matrices_from_mesh(
    mesh,
    kappa: Optional[float] = None,
    alpha: int = 1,
    verbose: bool = True
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, Optional[sparse.csr_matrix]]:
    """
    Compute SPDE matrices directly from SPDEMesh instance.
    
    Parameters
    ----------
    mesh : SPDEMesh
        Mesh instance with generated vertices and triangles
    kappa : float, optional
        SPDE range parameter
    alpha : int
        SPDE smoothness parameter:
        - alpha=1 for Matern nu=1/2 (exponential covariance)
        - alpha=2 for Matern nu=3/2 (once differentiable field)
    verbose : bool
        Print progress
        
    Returns
    -------
    Tuple of sparse matrices (C, G, A, Q)
    """
    if mesh.vertices is None or mesh.triangles is None:
        raise MatrixError("Mesh must be generated before computing matrices")
    
    return compute_fem_matrices(
        vertices=mesh.vertices,
        triangles=mesh.triangles,
        obs_coords=mesh.coords,
        kappa=kappa,
        alpha=alpha,
        verbose=verbose
    )
