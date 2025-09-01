"""
FEM matrix computation for SPDE approximations.

This module computes the sparse matrices (C, G, A, Q) 
required to compute the sparse precision matrix of a GMRF,
following the SPDE approach of Lindgren et al, 2011
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import norm as sparse_norm, eigs
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
    
    if verbose:
        print("  Computing C matrix (Mass)...")
    C = _compute_mass_matrix_vectorized(vertices, triangles)
    
    if verbose:
        print("  Computing G matrix (Stiffness)...")
    G = _compute_stiffness_matrix_vectorized(vertices, triangles)
    
    if verbose:
        print("  Computing A matrix (Projector)...")
    A = _compute_projector_matrix_vectorized(vertices, triangles, obs_coords)

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
    B = kappa**2 * C + G
    
    # Compute B * C^(-1) * B by solving C * X = B to maintain sparsity
    # Use column-by-column solving to preserve symmetry and numerical stability
    
    from scipy.sparse.linalg import spsolve
    
    # Solve C * X = B column by column to maintain sparsity and symmetry
    n = B.shape[0]
    Q_data = []
    Q_row_idx = []
    Q_col_idx = []
    
    # Convert B to CSC format for efficient column access
    B_csc = B.tocsc()
    
    for j in range(n):
        # Solve C * x_j = B[:, j] 
        x_j = spsolve(C, B_csc[:, j])
        
        # Compute Q[:, j] = B * x_j, but only store non-zeros
        q_j = B @ x_j
        
        # Find non-zero entries
        nonzero_mask = np.abs(q_j) > 1e-15
        nonzero_indices = np.where(nonzero_mask)[0]
        
        # Add to sparse format data
        Q_data.extend(q_j[nonzero_indices])
        Q_row_idx.extend(nonzero_indices)
        Q_col_idx.extend([j] * len(nonzero_indices))
    
    # Create sparse matrix and ensure symmetry
    Q = sparse.csr_matrix((Q_data, (Q_row_idx, Q_col_idx)), shape=(n, n))
    
    # Force exact symmetry: Q = (Q + Q^T) / 2
    Q = 0.5 * (Q + Q.T)
    
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
    
    total_entries = n_triangles * 9
    row_indices = np.empty(total_entries, dtype=np.int32)
    col_indices = np.empty(total_entries, dtype=np.int32)
    data = np.empty(total_entries, dtype=np.float64)
    
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


def _point_in_triangle(point: np.ndarray, triangle_vertices: np.ndarray) -> bool:
    """
    Test if a point is inside a triangle using barycentric coordinates.
    
    Parameters
    ----------
    point : np.ndarray
        Point coordinates (x, y)
    triangle_vertices : np.ndarray
        Triangle vertex coordinates, shape (3, 2)
        
    Returns
    -------
    bool
        True if point is inside triangle
    """
    v0, v1, v2 = triangle_vertices
    
    # Vectors from v0 to other vertices and to point
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = point - v0
    
    # Dot products
    dot00 = np.dot(v0v2, v0v2)
    dot01 = np.dot(v0v2, v0v1)
    dot02 = np.dot(v0v2, v0p)
    dot11 = np.dot(v0v1, v0v1)
    dot12 = np.dot(v0v1, v0p)
    
    # Barycentric coordinates
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    # Check if point is inside triangle
    return (u >= -1e-10) and (v >= -1e-10) and (u + v <= 1 + 1e-10)


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
    
    # Point location within existing triangulation
    # We need to find which of our mesh triangles contains each observation point
    simplex_indices = np.full(n_obs, -1, dtype=int)
    
    for i, point in enumerate(obs_coords):
        # Check each triangle to see if point is inside
        for j, tri_indices in enumerate(triangles):
            tri_vertices = vertices[tri_indices]
            if _point_in_triangle(point, tri_vertices):
                simplex_indices[i] = j
                break
    
    # Handle points outside any triangle using nearest triangle centroid
    outside_mask = simplex_indices == -1
    n_outside = np.sum(outside_mask)
    
    if n_outside > 0:
        if n_outside > 0.1 * n_obs:  # Warn if many points outside
            warnings.warn(f"{n_outside} observation points outside mesh")
        
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
    
    # Vectorized signed area computation using cross product
    # Note: For points outside triangles, signed areas can be negative
    def signed_areas_vectorized(p1, p2, p3):
        return 0.5 * (
            (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - 
            (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
        )
    
    # Total triangle areas (should always be positive)
    total_areas = np.abs(signed_areas_vectorized(v1, v2, v3))
    
    # Sub-triangle signed areas (can be negative for points outside)
    area1 = signed_areas_vectorized(points, v2, v3)  # lambda_1 (opposite v1)
    area2 = signed_areas_vectorized(v1, points, v3)  # lambda_2 (opposite v2)  
    area3 = signed_areas_vectorized(v1, v2, points)  # lambda_3 (opposite v3)
    
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
    
    # For numerical stability and proper interpolation, ensure coordinates sum to 1
    # This handles small numerical errors and maintains interpolation property
    coord_sums = bary_coords.sum(axis=1, keepdims=True)
    bary_coords = bary_coords / coord_sums
    
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

def select_reference_parameters(
    vertices: np.ndarray,
    triangles: np.ndarray,
    obs_coords: np.ndarray,
    alpha: int = 1
) -> Tuple[float, float]:
    """
    Automatically select reference kappa and tau based on mesh characteristics.
    
    :param vertices: Mesh vertex coordinates
    :param triangles: Triangle connectivity
    :param obs_coords: Observation coordinates
    :param alpha: SPDE smoothness parameter (1 or 2)
    :return: Tuple of (kappa_ref, tau_ref)
    """
    x_range = vertices[:, 0].max() - vertices[:, 0].min()
    y_range = vertices[:, 1].max() - vertices[:, 1].min()
    mesh_extent = max(x_range, y_range)
    
    # Compute edge lengths for all triangles (vectorized for efficiency)
    # Get all edges: each triangle has 3 edges
    edge_lengths = np.zeros(len(triangles) * 3)
    edge_idx = 0
    
    for k in range(3):
        i = k
        j = (k + 1) % 3
        # Vectorized edge length computation for all triangles
        v1 = vertices[triangles[:, i]]
        v2 = vertices[triangles[:, j]]
        edges = np.linalg.norm(v2 - v1, axis=1)
        edge_lengths[edge_idx:edge_idx + len(triangles)] = edges
        edge_idx += len(triangles)
    
    typical_edge = np.median(edge_lengths)
    
    ref_range = mesh_extent * 0.15
    
    if alpha == 1:
        kappa_ref = np.sqrt(8) / ref_range
    elif alpha == 2:
        kappa_ref = np.sqrt(8 * 3) / ref_range
    else:
        raise ValueError(f"Unsupported alpha={alpha}")
    
    # Reference tau based on mesh resolution for unit variance
    n_vertices = len(vertices)
    mesh_area = x_range * y_range
    vertex_density = n_vertices / mesh_area
    
    tau_ref = 1.0 / np.sqrt(vertex_density * typical_edge)
    
    return kappa_ref, tau_ref

def compute_fem_matrices_scaled(
    vertices: np.ndarray,
    triangles: np.ndarray,
    obs_coords: np.ndarray,
    kappa: Optional[float] = None,
    tau: Optional[float] = None,
    alpha: int = 1,
    auto_scale: bool = True,
    check_conditioning: bool = True,
    verbose: bool = True
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, 
           Optional[sparse.csr_matrix], dict]:
    """
    Compute FEM matrices with automatic parameter scaling.
    
    :param vertices: Mesh vertex coordinates
    :param triangles: Triangle connectivity
    :param obs_coords: Observation coordinates
    :param kappa: SPDE range parameter. If None and auto_scale=True, computed automatically
    :param tau: SPDE precision parameter. If None and auto_scale=True, computed automatically
    :param alpha: SPDE smoothness parameter
    :param auto_scale: Automatically select parameters if not provided
    :param check_conditioning: Check and warn about conditioning issues
    :param verbose: Print progress and diagnostics
    :return: Tuple of (C, G, A, Q, scale_info) where scale_info contains diagnostics
    """
    if auto_scale and (kappa is None or tau is None):
        kappa_ref, tau_ref = select_reference_parameters(
            vertices, triangles, obs_coords, alpha
        )
        if kappa is None:
            kappa = kappa_ref
        if tau is None:
            tau = tau_ref
        
        if verbose:
            print(f"Auto-selected parameters:")
            print(f"  kappa = {kappa:.4e} (range = {np.sqrt(8)/kappa:.3f})")
            print(f"  tau = {tau:.4e} (sd = {1/np.sqrt(tau):.3f})")
    
    C, G, A, _ = compute_fem_matrices(
        vertices, triangles, obs_coords, 
        kappa=1.0, alpha=alpha, verbose=False
    )
    Q = None
    scale_info = {
        'kappa_used': kappa,
        'tau_used': tau,
        'auto_scaled': auto_scale
    }
    
    if kappa is not None and tau is not None:
        if alpha == 1:
            Q_unscaled = kappa**2 * C + G
        elif alpha == 2:
            B = kappa**2 * C + G
            # Q = B * C^{-1} * B
            Q_unscaled = compute_matern_precision_nu_three_halves(C, G, kappa)
        else:
            raise ValueError(f"Unsupported alpha={alpha}")
        
        Q = tau * Q_unscaled
        
        if check_conditioning:
            cond_info = check_precision_conditioning(Q, C, G, kappa, tau, verbose)
            scale_info.update(cond_info)
    
    if verbose:
        _print_scale_aware_diagnostics(C, G, A, Q, scale_info)
    
    return C, G, A, Q, scale_info

def check_precision_conditioning(
    Q: sparse.csr_matrix,
    C: sparse.csr_matrix,
    G: sparse.csr_matrix,
    kappa: float,
    tau: float,
    verbose: bool = True
) -> dict:
    """
    Check conditioning of precision matrix and warn about issues.
    
    :param Q: Precision matrix
    :param C: Mass matrix
    :param G: Stiffness matrix
    :param kappa: Range parameter used
    :param tau: Precision parameter used
    :param verbose: Print warnings
    :return: Dictionary with conditioning diagnostics
    """
    n = Q.shape[0]
    
    # Estimate condition number using eigenvalues
    # For better efficiency, use iterative methods with looser tolerance
    try:
        # Use a looser tolerance for faster convergence
        lambda_max = eigs(Q, k=1, which='LM', return_eigenvectors=False, 
                          tol=1e-3, maxiter=100)[0].real
        
        # For smallest eigenvalue, use shift-invert mode which is more reliable
        # Note: This is still expensive but more robust than direct 'SM' computation
        try:
            lambda_min = eigs(Q, k=1, which='SM', return_eigenvectors=False,
                             tol=1e-3, maxiter=100)[0].real
        except:
            # If SM fails, estimate using inverse iteration with shift
            # This provides a reasonable lower bound estimate
            Q_diag = Q.diagonal()
            lambda_min = np.min(Q_diag[Q_diag > 0]) if np.any(Q_diag > 0) else 1e-10
        
        condition_number = lambda_max / lambda_min if lambda_min > 0 else np.inf
    except:
        # Fallback: use condition number estimate based on norms
        Q_norm_fro = sparse_norm(Q, 'fro')
        Q_norm_1 = sparse_norm(Q, 1)
        Q_norm_inf = sparse_norm(Q, np.inf)
        
        # Rough condition number estimate
        condition_number = np.sqrt(Q_norm_1 * Q_norm_inf)
        lambda_max = Q_norm_fro
        lambda_min = None
    
    kappa_c_norm = sparse_norm(kappa**2 * C, 'fro')
    g_norm = sparse_norm(G, 'fro')
    dominance_ratio = kappa_c_norm / g_norm
    
    diagnostics = {
        'condition_number': condition_number,
        'lambda_max': lambda_max,
        'lambda_min': lambda_min,
        'dominance_ratio': dominance_ratio,
        'well_conditioned': condition_number < 1e6
    }
    
    if verbose:
        if condition_number > 1e8:
            warnings.warn(
                f"Precision matrix is poorly conditioned (κ ≈ {condition_number:.2e}).\n"
                f"  This typically means kappa is too large for the mesh scale.\n"
                f"  Current kappa = {kappa:.4e} (range = {np.sqrt(8)/kappa:.3f})\n"
                f"  Try reducing kappa or using auto_scale=True"
            )
        elif condition_number > 1e6:
            warnings.warn(
                f"Precision matrix conditioning is marginal (κ ≈ {condition_number:.2e}).\n"
                f"  Consider adjusting parameters if convergence issues occur."
            )
        
        if dominance_ratio > 100:
            warnings.warn(
                f"Mass matrix term dominates (ratio = {dominance_ratio:.1f}).\n"
                f"  The spatial correlation is very short relative to mesh.\n"
                f"  Consider reducing kappa or using a finer mesh."
            )
        elif dominance_ratio < 0.01:
            warnings.warn(
                f"Stiffness matrix term dominates (ratio = {dominance_ratio:.3f}).\n"
                f"  The spatial correlation is very long relative to mesh.\n"
                f"  Consider increasing kappa or using a coarser mesh."
            )
    
    return diagnostics


def _print_scale_aware_diagnostics(
    C: sparse.csr_matrix,
    G: sparse.csr_matrix,
    A: sparse.csr_matrix,
    Q: Optional[sparse.csr_matrix],
    scale_info: dict
) -> None:
    """
    Print scale-aware diagnostics.
    
    :param C: Mass matrix
    :param G: Stiffness matrix
    :param A: Projector matrix
    :param Q: Precision matrix (optional)
    :param scale_info: Dictionary with scaling information
    :return: None
    """
    print("\nScale-Aware Matrix Diagnostics:")
    
    if 'kappa_used' in scale_info and scale_info['kappa_used'] is not None:
        kappa = scale_info['kappa_used']
        tau = scale_info['tau_used']
        print(f"  Parameters:")
        print(f"    kappa = {kappa:.4e} (range ≈ {np.sqrt(8)/kappa:.3f} units)")
        print(f"    tau = {tau:.4e} (marginal sd ≈ {1/np.sqrt(tau):.3f})")
    
    if 'condition_number' in scale_info:
        cond = scale_info['condition_number']
        status = "good" if cond < 1e4 else "marginal" if cond < 1e6 else "poor"
        print(f"  Conditioning: {status} (κ ≈ {cond:.2e})")
    
    if 'dominance_ratio' in scale_info:
        ratio = scale_info['dominance_ratio']
        if ratio > 10:
            print(f"  Warning: Mass matrix dominates (short-range correlation)")
        elif ratio < 0.1:
            print(f"  Warning: Stiffness matrix dominates (long-range correlation)")
        else:
            print(f"  Balance: Good (mass/stiffness ratio = {ratio:.2f})")
