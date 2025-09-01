"""
Coordinate preprocessing utilities using libpysal where available.
This version leverages libpysal's computational geometry module for improved performance.
"""

import numpy as np
import pyproj
import warnings

try:
    import libpysal
    HAS_LIBPYSAL = True
except ImportError:
    HAS_LIBPYSAL = False
    warnings.warn("libpysal not installed. Using scipy implementations instead.")

from scipy.spatial import ConvexHull, distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from geo_spde.exceptions import CoordsError
from typing import Tuple, Dict, Union, List, Optional


def estimate_characteristic_scale(coords: np.ndarray, method: str = 'mst') -> Dict[str, float]:
    """
    Estimate characteristic spatial scale from point pattern.
    
    :param coords: Coordinate array (n_obs, 2)
    :param method: 'mst' for minimum spanning tree (fast), 'nn' for nearest neighbor distances
    :return: Dict with scale estimates: characteristic_scale, min_distance, median_distance, mesh_recommended_edge
    """
    n_obs = len(coords)
    
    if HAS_LIBPYSAL and method == 'nn':
        kd = libpysal.cg.KDTree(coords)
        nn_distances = []
        for i in range(min(n_obs, 500)):
            dists, _ = kd.query(coords[i], k=2)
            nn_distances.append(dists[1])  # Second closest is first neighbor
        characteristic_scale = np.median(nn_distances) * 5  # Scale up from NN
        
        dists = distance_matrix(coords, coords)
        dists_flat = dists[np.triu_indices(n_obs, k=1)]
        min_dist = np.min(dists_flat[dists_flat > 0]) if np.any(dists_flat > 0) else 1.0
        median_dist = np.median(dists_flat)
    else:
        dists = distance_matrix(coords, coords)
        dists_flat = dists[np.triu_indices(n_obs, k=1)]
        min_dist = np.min(dists_flat[dists_flat > 0]) if np.any(dists_flat > 0) else 1.0
        median_dist = np.median(dists_flat)
        
        if method == 'mst':
            coords_sub = coords
            dist_matrix = distance_matrix(coords_sub, coords_sub)
            mst = minimum_spanning_tree(dist_matrix)
            mst_edges = mst.tocoo().data
            characteristic_scale = np.percentile(mst_edges, 75)
        else:
            nn_distances = []
            for i in range(min(n_obs, 500)):
                dists_i = dists[i] if n_obs < 500 else distance_matrix(coords[i:i+1], coords)[0]
                dists_i[i] = np.inf  # Exclude self
                nn_distances.append(np.min(dists_i))
            characteristic_scale = np.median(nn_distances) * 5  # Scale up from NN
    
    return {
        'characteristic_scale': characteristic_scale,
        'min_distance': min_dist,
        'median_distance': median_dist,
        'mesh_recommended_edge': characteristic_scale * 0.3
    }


def compute_convex_hull_diameter(coords: np.ndarray) -> float:
    """
    Compute maximum distance between convex hull vertices.
    
    :param coords: Coordinate array of shape (n_obs, 2)
    :return: Maximum distance between hull vertices
    """
    if len(coords) < 3:
        if len(coords) == 2:
            return np.linalg.norm(coords[1] - coords[0])
        return 0.0
    
    try:
        if HAS_LIBPYSAL:
            # Use libpysal's convex hull
            hull_points = libpysal.cg.convex_hull(coords)
            # Convert to array if needed
            if hasattr(hull_points, 'vertices'):
                hull_points = np.array(hull_points.vertices)
            elif not isinstance(hull_points, np.ndarray):
                hull_points = np.array(list(hull_points))
        else:
            # Use scipy's ConvexHull
            hull = ConvexHull(coords)
            hull_points = coords[hull.vertices]
        
        # Compute all pairwise distances between hull vertices
        distances = pdist(hull_points)
        return np.max(distances)
    except Exception:
        # Fallback for degenerate cases (collinear points)
        distances = pdist(coords)
        return np.max(distances) if len(distances) > 0 else 0.0


def remove_duplicate_coords_libpysal(coords: np.ndarray, tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Remove duplicate coordinates using libpysal's KDTree for efficient nearest neighbor search.
    
    :param coords: Coordinate array of shape (n_obs, 2)
    :param tolerance: Distance tolerance for considering points duplicates
    :return: Tuple of (unique_coords, unique_indices, n_duplicates)
    """
    if not HAS_LIBPYSAL or len(coords) <= 1:
        # Fall back to original implementation
        return remove_duplicate_coords(coords, tolerance)
    
    # Use KDTree for efficient duplicate detection
    kd = libpysal.cg.KDTree(coords)
    
    # Find all points within tolerance of each point
    indices_to_remove = set()
    for i in range(len(coords)):
        if i in indices_to_remove:
            continue
        # Query for neighbors within tolerance
        neighbors = kd.query_ball_point(coords[i], tolerance)
        # Remove self and mark higher-indexed duplicates for removal
        for j in neighbors:
            if j > i:
                indices_to_remove.add(j)
    
    # Keep non-duplicate indices
    unique_indices = np.array([i for i in range(len(coords)) if i not in indices_to_remove])
    unique_coords = coords[unique_indices]
    n_duplicates = len(coords) - len(unique_coords)
    
    return unique_coords, unique_indices, n_duplicates


# Keep original implementation for compatibility
def remove_duplicate_coords(coords: np.ndarray, tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Remove duplicate coordinates within tolerance (scipy implementation).
    
    :param coords: Coordinate array of shape (n_obs, 2)
    :param tolerance: Distance tolerance for considering points duplicates
    :return: Tuple of (unique_coords, unique_indices, n_duplicates)
    """
    if len(coords) <= 1:
        return coords, np.arange(len(coords)), 0
    
    # Compute pairwise distances
    distances = squareform(pdist(coords))
    
    # Find duplicates
    i_indices, j_indices = np.where(distances < tolerance)
    off_diagonal_mask = i_indices != j_indices
    duplicate_pairs = (i_indices[off_diagonal_mask], j_indices[off_diagonal_mask])
    
    # Build set of indices to remove (keep first occurrence)
    indices_to_remove = set()
    for i, j in zip(duplicate_pairs[0], duplicate_pairs[1]):
        if i < j:  # Only consider upper triangle
            indices_to_remove.add(j)
    
    # Keep non-duplicate indices
    unique_indices = np.array([i for i in range(len(coords)) if i not in indices_to_remove])
    unique_coords = coords[unique_indices]
    n_duplicates = len(coords) - len(unique_coords)
    
    return unique_coords, unique_indices, n_duplicates


# Additional libpysal-specific utilities
def compute_alpha_shape(coords: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
    """
    Compute alpha shape (concave hull) using libpysal.
    
    :param coords: Coordinate array of shape (n_obs, 2)
    :param alpha: Alpha parameter (None for auto-detection)
    :return: Array of points forming the alpha shape boundary
    """
    if not HAS_LIBPYSAL:
        # Fall back to convex hull
        warnings.warn("libpysal not available, falling back to convex hull")
        hull = ConvexHull(coords)
        return coords[hull.vertices]
    
    if alpha is None:
        # Use auto alpha shape
        from libpysal.cg import alpha_shape_auto
        alpha_shape = alpha_shape_auto(coords)
    else:
        from libpysal.cg import alpha_shape
        alpha_shape = alpha_shape(coords, alpha)
    
    # Extract boundary points
    if hasattr(alpha_shape, 'boundary'):
        return np.array(alpha_shape.boundary.coords)
    else:
        return np.array(list(alpha_shape.exterior.coords))


# Export the improved version if libpysal is available
if HAS_LIBPYSAL:
    print("Using libpysal-enhanced coordinate functions")
    # Override the original function with the libpysal version
    remove_duplicate_coords_original = remove_duplicate_coords
    remove_duplicate_coords = remove_duplicate_coords_libpysal


# Import remaining functions from original module
from geo_spde.coords import (
    is_geographic,
    detect_antimeridian_crossing,
    unwrap_antimeridian,
    degrees_to_km_estimate,
    determine_projection_scale,
    determine_utm_zone,
    compute_albers_standard_parallels,
    create_projection_string,
    project_coordinates,
    preprocess_coords
)