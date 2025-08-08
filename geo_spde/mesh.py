"""
SPDE mesh generation for Stan integration.

This module creates triangular meshes from spatial coordinates for use in
SPDE approximations following Lindgren et al. (2011).
"""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from typing import Tuple, Dict, Optional
import meshpy.triangle as triangle

from geo_spde.exceptions import MeshError


class SPDEMesh:
    """
    SPDE mesh generation with data-adaptive parameters.
    
    This class creates triangular meshes optimized for SPDE approximations,
    with automatic parameter selection based on observation density.
    
    Attributes
    ----------
    coords : np.ndarray
        Projected coordinates from preprocessing step
    projection_info : Dict
        Metadata from coordinate preprocessing
    mesh_params : Dict
        Computed mesh generation parameters
    vertices : np.ndarray
        Mesh vertex coordinates
    triangles : np.ndarray
        Triangle connectivity matrix
    diagnostics : Dict
        Mesh quality and performance metrics
    """
    
    def __init__(
        self, 
        coords: np.ndarray,
        projection_info: Optional[Dict] = None
    ):
        """
        Initialize mesh generator with preprocessed coordinates.
        
        Parameters
        ----------
        coords : np.ndarray
            Clean projected coordinates from preprocess_coords()
        projection_info : Dict, optional
            Projection metadata from preprocess_coords()
        """
        if not isinstance(coords, np.ndarray):
            coords = np.asarray(coords)
        
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise MeshError(f"Expected coords shape (n_obs, 2), got {coords.shape}")
        
        if len(coords) < 3:
            raise MeshError(
                f"Need at least 3 coordinates for mesh generation, got {len(coords)}"
            )
        
        self.coords = coords
        self.projection_info = projection_info or {}
        self.mesh_params = None
        self.vertices = None
        self.triangles = None
        self.diagnostics = None
        
    def create_mesh(
        self,
        boundary: Optional[np.ndarray] = None,
        extension_factor: float = 0.2,
        target_edge_factor: float = 0.5,
        min_angle: float = 20.0,
        max_area: Optional[float] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate triangular mesh with data-adaptive parameters.
        
        Parameters
        ----------
        boundary : np.ndarray, optional
            Custom boundary polygon. If None, creates extended convex hull
        extension_factor : float
            Factor to extend boundary beyond convex hull (default: 0.2 = 20%)
        target_edge_factor : float
            Target edge length as fraction of median observation distance
        min_angle : float
            Minimum angle constraint in degrees (default: 20)
        max_area : float, optional
            Maximum triangle area. If None, computed automatically
        verbose : bool
            Print mesh generation progress and diagnostics
            
        Returns
        -------
        vertices : np.ndarray
            Mesh vertex coordinates of shape (n_mesh, 2)
        triangles : np.ndarray
            Triangle connectivity of shape (n_tri, 3)
        """
        if verbose:
            print(f"Creating mesh for {len(self.coords)} observation points...")
        
        # Step 1: Compute adaptive parameters
        self.mesh_params = self._compute_mesh_parameters(
            target_edge_factor, max_area
        )
        
        if verbose:
            print(f"  Target edge length: {self.mesh_params['max_edge']:.1f} units")
            print(f"  Cutoff distance: {self.mesh_params['cutoff']:.3f} units")
        
        # Step 2: Create boundary
        if boundary is None:
            boundary = self._create_extended_boundary(extension_factor)
            if verbose:
                print(f"  Created extended boundary ({extension_factor*100:.0f}% buffer)")
        
        # Step 3: Generate mesh
        self.vertices, self.triangles = self._generate_basic_mesh(
            boundary, min_angle, self.mesh_params['max_area']
        )
        
        # Step 4: Compute diagnostics
        self.diagnostics = self._compute_mesh_diagnostics()
        
        if verbose:
            self._print_diagnostics()
        
        return self.vertices, self.triangles
    
    def _compute_mesh_parameters(
        self,
        target_edge_factor: float,
        max_area: Optional[float]
    ) -> Dict[str, float]:
        """
        Compute data-adaptive mesh parameters based on observation density.
        
        Parameters
        ----------
        target_edge_factor : float
            Target edge length as fraction of median observation distance
        max_area : float, optional
            Override automatic area calculation
            
        Returns
        -------
        Dict with keys:
            - max_edge: Maximum edge length
            - max_area: Maximum triangle area
            - cutoff: Minimum distance threshold
            - min_distance: Minimum observation distance
            - median_distance: Median observation distance
        """
        # Compute pairwise distances
        distances = pdist(self.coords)
        distances_nonzero = distances[distances > 0]
        
        if len(distances_nonzero) == 0:
            raise MeshError("All coordinates are identical")
        
        min_distance = np.min(distances_nonzero)
        median_distance = np.median(distances_nonzero)
        
        # Data-adaptive parameters
        max_edge = median_distance * target_edge_factor
        
        if max_area is None:
            max_area = (max_edge ** 2) / 2.0
        
        # Cutoff should be smaller than minimum observation distance
        cutoff = min_distance * 0.1
        
        return {
            'max_edge': max_edge,
            'max_area': max_area,
            'cutoff': cutoff,
            'min_distance': min_distance,
            'median_distance': median_distance
        }
    
    def _create_extended_boundary(self, extension_factor: float) -> np.ndarray:
        """
        Create extended boundary polygon from convex hull.
        
        Parameters
        ----------
        extension_factor : float
            Factor to extend boundary beyond convex hull
            
        Returns
        -------
        np.ndarray
            Extended boundary points
        """
        # Compute convex hull
        hull = ConvexHull(self.coords)
        hull_points = self.coords[hull.vertices]
        
        # Compute centroid
        centroid = np.mean(hull_points, axis=0)
        
        # Extend points outward from centroid
        extended_points = centroid + (hull_points - centroid) * (1 + extension_factor)
        
        return extended_points
    
    def _generate_basic_mesh(
        self,
        boundary: np.ndarray,
        min_angle: float,
        max_area: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate triangular mesh using MeshPy.
        
        Parameters
        ----------
        boundary : np.ndarray
            Boundary polygon points
        min_angle : float
            Minimum angle constraint in degrees
        max_area : float
            Maximum triangle area
            
        Returns
        -------
        vertices : np.ndarray
            Mesh vertices
        triangles : np.ndarray
            Triangle connectivity
        """
        # Set up MeshPy mesh info
        mesh_info = triangle.MeshInfo()
        
        # Combine observation points and boundary points
        points_list = self.coords.tolist()
        boundary_list = boundary.tolist()
        all_points = points_list + boundary_list
        
        # Set all points at once
        mesh_info.set_points(all_points)
        
        # Create boundary facets (segments)
        n_boundary = len(boundary)
        offset = len(points_list)  # Boundary points start after observation points
        facets = []
        for i in range(n_boundary):
            facets.append([offset + i, offset + ((i + 1) % n_boundary)])
        mesh_info.set_facets(facets)
        
        # Build mesh with quality constraints
        mesh = triangle.build(
            mesh_info,
            min_angle=min_angle,
            max_volume=max_area,
            attributes=False,
            generate_faces=False
        )
        
        # Extract vertices and triangles
        vertices = np.array(mesh.points)
        triangles = np.array(mesh.elements, dtype=np.int32)
        
        return vertices, triangles
    
    def _compute_mesh_diagnostics(self) -> Dict:
        """
        Compute user-interpretable mesh diagnostics.
        
        Returns
        -------
        Dict with mesh quality and performance metrics
        """
        n_vertices = len(self.vertices)
        n_triangles = len(self.triangles)
        n_obs = len(self.coords)
        
        # Compute triangle areas
        areas = []
        for tri in self.triangles:
            v0, v1, v2 = self.vertices[tri]
            area = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - 
                           (v2[0] - v0[0]) * (v1[1] - v0[1]))
            areas.append(area)
        
        areas = np.array(areas)
        
        # Compute mesh density
        total_area = np.sum(areas)
        mesh_density = n_vertices / total_area if total_area > 0 else 0
        
        # Memory estimates (rough)
        # A matrix: n_obs × n_mesh, sparse ~3 entries per row
        # C, G matrices: n_mesh × n_mesh, sparse ~7 entries per row
        sparse_entries_A = n_obs * 3
        sparse_entries_CG = n_vertices * 7 * 2
        memory_mb = (sparse_entries_A + sparse_entries_CG) * 8 / (1024 * 1024)
        
        return {
            'n_vertices': n_vertices,
            'n_triangles': n_triangles,
            'n_observations': n_obs,
            'mesh_to_obs_ratio': n_vertices / n_obs,
            'mean_triangle_area': np.mean(areas),
            'total_area': total_area,
            'mesh_density': mesh_density,
            'memory_estimate_mb': memory_mb
        }
    
    def _print_diagnostics(self):
        """Print user-friendly diagnostics."""
        d = self.diagnostics
        
        print("\nMesh Generation Complete:")
        print(f"  {d['n_vertices']:,} mesh vertices")
        print(f"  {d['n_triangles']:,} triangles")
        print(f"  Mesh/observation ratio: {d['mesh_to_obs_ratio']:.1f}")
        
        # Convert area to km² if projection info available
        area_unit = "units^2"
        if self.projection_info.get('system', '').startswith('UTM'):
            area_unit = "km^2"
            area_scale = 1e-6
        else:
            area_scale = 1
        
        print(f"  Total area: {d['total_area'] * area_scale:.1f} {area_unit}")
        print(f"  Estimated memory for Stan: {d['memory_estimate_mb']:.1f} MB")
        
        # Performance warnings
        if d['n_vertices'] > 5000:
            print("  WARNING: Large mesh - expect slower Stan sampling")
        if d['mesh_to_obs_ratio'] > 10:
            print("  WARNING: High mesh/observation ratio - consider coarser mesh")
    
    def get_mesh_info(self) -> Dict:
        """
        Get comprehensive mesh information.
        
        Returns
        -------
        Dict containing mesh parameters, diagnostics, and metadata
        """
        if self.vertices is None:
            raise MeshError("Mesh not yet generated. Call create_mesh() first.")
        
        return {
            'mesh_params': self.mesh_params,
            'diagnostics': self.diagnostics,
            'projection_info': self.projection_info,
            'n_vertices': len(self.vertices),
            'n_triangles': len(self.triangles),
            'n_observations': len(self.coords)
        }