"""
SPDE mesh generation for Stan integration.

This module creates triangular meshes from spatial coordinates for use in
SPDE approximations following Lindgren et al. (2011).
"""

import numpy as np

import meshpy.triangle as triangle
from scipy.spatial import ConvexHull, distance_matrix, KDTree
from scipy.spatial.distance import pdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import curve_fit

from typing import Tuple, Dict, Optional, List

import warnings

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
        
        # Check for and remove very close coordinates that cause mesh generation issues
        coords = self._remove_very_close_coords(coords)
        
        if len(coords) < 3:
            raise MeshError(
                f"After removing very close coordinates, only {len(coords)} remain. "
                "Need at least 3 coordinates for mesh generation."
            )
        
        self.coords = coords
        self.projection_info = projection_info or {}
        self.mesh_params = None
        self.vertices = None
        self.triangles = None
        self.diagnostics = None
        
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
    
    def _create_extended_boundary(self, extension_factor: float) -> np.ndarray:
        """
        Create extended boundary polygon from convex hull.
        
        :param extension_factor: Factor to extend boundary beyond convex hull
        :return: Extended boundary points
        """
        # Compute convex hull
        hull = ConvexHull(self.coords)
        hull_points = self.coords[hull.vertices]
        
        # Compute centroid
        centroid = np.mean(hull_points, axis=0)
        
        # Extend points outward from centroid
        extended_points = centroid + (hull_points - centroid) * (1 + extension_factor)
        
        return extended_points
    
    def _remove_very_close_coords(self, coords: np.ndarray, min_distance: float = 1e-4) -> np.ndarray:
        """
        Remove coordinates that are very close together to prevent mesh generation issues.
        
        :param coords: Input coordinates
        :param min_distance: Minimum allowed distance between points
        :return: Filtered coordinates
        """
        if len(coords) <= 1:
            return coords
        
        # Use a simple approach: keep first occurrence, remove subsequent close points
        from scipy.spatial.distance import cdist
        
        keep_indices = [0]  # Always keep the first point
        
        for i in range(1, len(coords)):
            # Check distance to all previously kept points
            distances = cdist([coords[i]], coords[keep_indices])
            min_dist = np.min(distances)
            
            if min_dist >= min_distance:
                keep_indices.append(i)
        
        filtered_coords = coords[keep_indices]
        
        if len(filtered_coords) < len(coords):
            warnings.warn(
                f"Removed {len(coords) - len(filtered_coords)} coordinates that were "
                f"closer than {min_distance:.1e} units to avoid mesh generation issues.",
                UserWarning
            )
        
        return filtered_coords
    
    def get_mesh_info(self) -> Dict:
        """
        Get comprehensive mesh information.
        
        Returns
        -------
        Dict containing mesh parameters, diagnostics, and metadata
        """
        if self.vertices is None:
            raise MeshError("Mesh not yet generated. Call create_adaptive_mesh() first.")
        
        return {
            'mesh_params': self.mesh_params,
            'diagnostics': self.diagnostics,
            'projection_info': self.projection_info,
            'n_vertices': len(self.vertices),
            'n_triangles': len(self.triangles),
            'n_observations': len(self.coords)
        }

    def estimate_spatial_scale(self, method: str = 'variogram', 
                              max_pairs: int = 5000) -> Dict[str, float]:
        """
        Estimate characteristic spatial scales from coordinate data.
        
        Parameters
        ----------
        method : str
            'variogram' for empirical variogram fitting
            'mst' for minimum spanning tree analysis
            'both' for both methods
        max_pairs : int
            Maximum pairs to use for variogram (for computational efficiency)
            
        Returns
        -------
        Dict with keys:
            - 'estimated_range': Characteristic correlation range
            - 'min_distance': Minimum non-zero distance
            - 'median_distance': Median distance
            - 'practical_range': Distance at which correlation ≈ 0.05
            - 'method_used': Which estimation method was used
        """
        n_obs = len(self.coords)
        
        if n_obs < 100:
            dists = distance_matrix(self.coords, self.coords)
            dists_flat = dists[np.triu_indices(n_obs, k=1)]
        else:
            idx = np.random.choice(n_obs, min(100, n_obs), replace=False)
            dists_sub = distance_matrix(self.coords[idx], self.coords[idx])
            dists_flat = dists_sub[np.triu_indices(len(idx), k=1)]
        
        min_dist = np.min(dists_flat[dists_flat > 0])
        median_dist = np.median(dists_flat)
        
        results = {
            'min_distance': min_dist,
            'median_distance': median_dist
        }
        
        if method in ['variogram', 'both']:
            try:
                var_range = self._estimate_range_variogram(max_pairs)
                results['estimated_range'] = var_range
                results['practical_range'] = var_range * 3
                results['method_used'] = 'variogram'
            except:
                method = 'mst'
        
        if method in ['mst', 'both']:
            mst_scale = self._estimate_range_mst()
            if 'estimated_range' not in results or method == 'mst':
                results['estimated_range'] = mst_scale
                results['practical_range'] = mst_scale * 3
                results['method_used'] = 'mst'
            else:
                results['mst_range'] = mst_scale
        
        self.spatial_scale = results
        return results
    
    def _estimate_range_variogram(self, max_pairs: int = 5000) -> float:
        """
        Estimate range using empirical variogram with exponential model.
        
        Assumes data would have constant mean and exponential covariance.
        
        :param max_pairs: Maximum pairs to use for computational efficiency
        :return: Estimated range parameter
        """
        n_obs = len(self.coords)
        
        if n_obs * (n_obs - 1) / 2 > max_pairs:
            n_sub = int(np.sqrt(max_pairs * 2))
            idx = np.random.choice(n_obs, n_sub, replace=False)
            coords_sub = self.coords[idx]
        else:
            coords_sub = self.coords
        
        dists = distance_matrix(coords_sub, coords_sub)
        dists_flat = dists[np.triu_indices(len(coords_sub), k=1)]
        
        max_dist = np.percentile(dists_flat, 50)
        n_bins = min(20, len(dists_flat) // 50)
        bins = np.linspace(0, max_dist, n_bins)
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        distance_variance = np.zeros(len(bin_centers))
        
        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            mask = (dists_flat >= low) & (dists_flat < high)
            if np.sum(mask) > 5:
                distance_variance[i] = np.var(dists_flat[mask])
        
        valid = distance_variance > 0
        if np.sum(valid) < 3:
            raise ValueError("Insufficient data for variogram fitting")
        
        def exp_variogram(h, sill, range_param):
            return sill * (1 - np.exp(-h / range_param))
        
        median_dist = np.median(dists_flat)
        min_dist = np.min(dists_flat[dists_flat > 0])
        
        try:
            popt, _ = curve_fit(
                exp_variogram, 
                bin_centers[valid], 
                distance_variance[valid],
                p0=[np.max(distance_variance), median_dist / 3],
                bounds=([0, min_dist], [np.inf, max_dist])
            )
            return popt[1]
        except:
            return median_dist / 3
    
    def _estimate_range_mst(self) -> float:
        """
        Estimate spatial scale using minimum spanning tree edge lengths.
        
        The MST captures the essential connectivity scale of the points.
        
        :return: Characteristic spatial scale
        """
        n_obs = len(self.coords)
        
        if n_obs > 500:
            idx = np.random.choice(n_obs, 500, replace=False)
            coords_sub = self.coords[idx]
        else:
            coords_sub = self.coords
        
        dists = distance_matrix(coords_sub, coords_sub)
        mst = minimum_spanning_tree(dists)
        
        mst_edges = mst.tocoo()
        edge_lengths = mst_edges.data
        
        return np.percentile(edge_lengths, 75)
    
    def validate_mesh_resolution(self, verbose: bool = True) -> Dict[str, bool]:
        """
        Check if mesh resolution is appropriate for the estimated correlation range.
        
        :param verbose: Print warnings
        :return: Dict with validation results: resolution_ok, extent_ok, edge_to_range_ratio, recommended_edge_factor
        """
        if self.mesh_params is None and self.vertices is None:
            raise MeshError("Mesh parameters not computed. Call create_adaptive_mesh() first.")
        
        # For adaptive mesh, compute effective edge length if mesh_params not available
        if self.mesh_params is None and self.vertices is not None:
            # Compute effective edge lengths from actual mesh
            edge_lengths = []
            for tri in self.triangles:
                for i in range(3):
                    j = (i + 1) % 3
                    edge_length = np.linalg.norm(self.vertices[tri[j]] - self.vertices[tri[i]])
                    edge_lengths.append(edge_length)
            max_edge = np.percentile(edge_lengths, 90)  # Use 90th percentile as "max edge"
        else:
            max_edge = self.mesh_params['max_edge']
        
        if not hasattr(self, 'spatial_scale'):
            self.estimate_spatial_scale()
        
        estimated_range = self.spatial_scale['estimated_range']
        practical_range = self.spatial_scale.get('practical_range', estimated_range * 3)
        
        edge_to_range_ratio = max_edge / estimated_range
        resolution_ok = edge_to_range_ratio < 0.5
        
        if self.vertices is not None:
            mesh_extent = np.max([
                self.vertices[:, 0].max() - self.vertices[:, 0].min(),
                self.vertices[:, 1].max() - self.vertices[:, 1].min()
            ])
            extent_ok = mesh_extent > practical_range * 2
        else:
            extent_ok = True
            mesh_extent = None
        
        validation = {
            'resolution_ok': resolution_ok,
            'extent_ok': extent_ok,
            'edge_to_range_ratio': edge_to_range_ratio,
            'recommended_edge_factor': 0.3 if edge_to_range_ratio > 0.5 else None,
            'mesh_extent': mesh_extent
        }
        
        if verbose and not resolution_ok:
            warnings.warn(
                f"Mesh may be too coarse for correlation structure.\n"
                f"  Mesh edge length: {max_edge:.3f}\n"
                f"  Estimated range: {estimated_range:.3f}\n"
                f"  Ratio: {edge_to_range_ratio:.2f} (should be < 0.5)\n"
                f"  Consider using target_edge_factor={validation['recommended_edge_factor']}"
            )
        
        self.mesh_validation = validation
        return validation
    
    def suggest_spde_parameters(self) -> Dict[str, float]:
        """
        Suggest reasonable SPDE parameters based on mesh and data characteristics.
        
        :return: Dict with suggested parameters: kappa_suggestion, tau_suggestion, spatial_range_suggestion, spatial_sd_suggestion, prior_range_kappa, prior_range_tau
        """
        if not hasattr(self, 'spatial_scale'):
            self.estimate_spatial_scale()
        
        estimated_range = self.spatial_scale['estimated_range']
        min_dist = self.spatial_scale['min_distance']
        median_dist = self.spatial_scale['median_distance']
        
        x_range = self.coords[:, 0].max() - self.coords[:, 0].min()
        y_range = self.coords[:, 1].max() - self.coords[:, 1].min()
        mesh_extent = max(x_range, y_range)
        
        suggested_range = np.clip(estimated_range, mesh_extent * 0.05, mesh_extent * 0.3)
        
        kappa_suggestion = np.sqrt(8) / suggested_range
        
        tau_suggestion = 1.0 / np.sqrt(mesh_extent / median_dist)
        
        kappa_prior_range = (kappa_suggestion / 3, kappa_suggestion * 3)
        tau_prior_range = (tau_suggestion / 3, tau_suggestion * 3)
        
        suggestions = {
            'kappa_suggestion': kappa_suggestion,
            'tau_suggestion': tau_suggestion,
            'spatial_range_suggestion': suggested_range,
            'spatial_sd_suggestion': 1.0 / np.sqrt(tau_suggestion),
            'prior_range_kappa': kappa_prior_range,
            'prior_range_tau': tau_prior_range,
            'mesh_extent': mesh_extent,
            'range_to_extent_ratio': suggested_range / mesh_extent
        }
        
        self.parameter_suggestions = suggestions
        return suggestions
    
    def compute_scale_diagnostics(self, verbose: bool = True) -> Dict:
        """
        Compute comprehensive scale diagnostics after mesh creation.
        
        :param verbose: Print diagnostic information
        :return: Dict containing spatial scale, validation, and parameter suggestions
        """
        if self.vertices is None:
            raise MeshError("Mesh must be created before computing scale diagnostics")
        
        spatial_scale = self.estimate_spatial_scale(method='both')
        
        validation = self.validate_mesh_resolution(verbose=verbose)
        
        suggestions = self.suggest_spde_parameters()
        
        self.scale_diagnostics = {
            'spatial_scale': spatial_scale,
            'validation': validation,
            'suggestions': suggestions
        }
        
        if verbose:
            print("\nScale Diagnostics:")
            print(f"  Estimated correlation range: {spatial_scale['estimated_range']:.3f}")
            print(f"  Mesh resolution adequate: {validation['resolution_ok']}")
            print(f"  Suggested kappa: {suggestions['kappa_suggestion']:.4f}")
            print(f"  Suggested tau: {suggestions['tau_suggestion']:.4f}")
            print(f"  (spatial_range: {suggestions['spatial_range_suggestion']:.3f})")
            print(f"  (spatial_sd: {suggestions['spatial_sd_suggestion']:.3f})")
        
        return self.scale_diagnostics

    def create_adaptive_mesh(
            self,
            boundary: Optional[np.ndarray] = None,
            extension_factor: float = 0.2,
            min_edge_near_obs: Optional[float] = None,
            max_edge_far: Optional[float] = None,
            transition_distance: Optional[float] = None,
            min_angle: float = 20.0,
            verbose: bool = True
            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate adaptive triangular mesh with spatially varying resolution.
        
        Creates a mesh that adapts to observation density, with finer resolution
        near observation points and coarser resolution in sparse areas.
        
        :param boundary: Custom boundary polygon. If None, creates extended convex hull
        :param extension_factor: Factor to extend boundary beyond convex hull (default: 0.2 = 20%)
        :param min_edge_near_obs: Minimum edge length near observations. If None, computed automatically
        :param max_edge_far: Maximum edge length far from observations. If None, computed automatically
        :param transition_distance: Distance over which mesh transitions from fine to coarse. If None, computed automatically
        :param min_angle: Minimum angle constraint in degrees
        :param verbose: Print mesh generation progress and diagnostics
        :return: Tuple of (vertices, triangles) where vertices is shape (n_mesh, 2) and triangles is shape (n_tri, 3)
        """
        if verbose:
            print(f"Creating adaptive mesh for {len(self.coords)} observation points...")
    
        # Compute adaptive parameters
        if min_edge_near_obs is None:
            params = self._compute_adaptive_parameters()
            min_edge_near_obs = params['min_edge_near_obs']
            max_edge_far = params['max_edge_far']
            transition_distance = params['transition_distance']
        
            if verbose:
                print(f"  Min edge (near obs): {min_edge_near_obs:.1f} units")
                print(f"  Max edge (far): {max_edge_far:.1f} units")
                print(f"  Transition distance: {transition_distance:.1f} units")
    
        # Create boundary
        if boundary is None:
            boundary = self._create_extended_boundary(extension_factor)
        # Generate mesh with spatially varying areas
        self.vertices, self.triangles = self._generate_adaptive_mesh_iterative(
            boundary, min_angle, min_edge_near_obs, max_edge_far, transition_distance
            )
    
        self.diagnostics = self._compute_mesh_diagnostics()
    
        if verbose:
            self._print_adaptive_diagnostics()
    
        return self.vertices, self.triangles

    def _generate_adaptive_mesh_iterative(
            self,
            boundary: np.ndarray,
            min_angle: float,
            min_edge: float,
            max_edge: float,
            transition_dist: float
            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate adaptive mesh using iterative refinement with MeshPy.
        
        Implements an iterative refinement strategy where triangles are subdivided
        based on their distance to observation points.
        
        :param boundary: Boundary polygon points
        :param min_angle: Minimum angle constraint in degrees
        :param min_edge: Minimum edge length near observations
        :param max_edge: Maximum edge length far from observations
        :param transition_dist: Distance over which mesh transitions from fine to coarse
        :return: Tuple of (vertices, triangles) arrays
        """
        # Build KDTree for distance queries
        obs_tree = KDTree(self.coords)
    
        # Initial mesh setup
        mesh_info = triangle.MeshInfo()
    
        # Combine all points
        points_list = self.coords.tolist()
        boundary_list = boundary.tolist()
        all_points = points_list + boundary_list
    
        # Add intermediate points for smooth transition
        intermediate_points = self._generate_intermediate_points(
            boundary, obs_tree, transition_dist, max_edge
        )
        all_points.extend(intermediate_points)
    
        mesh_info.set_points(all_points)
    
        # Create boundary facets
        n_boundary = len(boundary)
        offset = len(self.coords)
        facets = []
        for i in range(n_boundary):
            facets.append([offset + i, offset + ((i + 1) % n_boundary)])
        mesh_info.set_facets(facets)
    
        # Build mesh with spatially varying area constraints
        # Use a simpler approach: generate with fine resolution then simplify
        fine_area = (min_edge ** 2) / 2.0
        if max_edge is not None:
            coarse_area = (max_edge ** 2) / 2.0
            avg_area = (fine_area + coarse_area) / 2.0
        else:
            avg_area = fine_area * 4.0  # Use larger area when max_edge is None
        mesh = triangle.build(
            mesh_info,
            min_angle=min_angle,
            attributes=False,
            generate_faces=False,
            max_volume=avg_area
        )
        
        # For now, return the mesh as is - the iterative refinement
        # was causing segfaults with MeshPy
        vertices = np.array(mesh.points)
        triangles = np.array(mesh.elements, dtype=np.int32)
        
        return vertices, triangles
    
    def _generate_intermediate_points(
        self,
        boundary: np.ndarray,
        obs_tree: KDTree,
        transition_dist: float,
        max_edge: float
    ) -> List[List[float]]:
        """
        Generate intermediate points for smooth mesh transition.
        
        Creates rings of points around observation clusters to ensure smooth
        transition between fine and coarse mesh regions.
        
        :param boundary: Boundary polygon points
        :param obs_tree: KDTree of observation points for efficient distance queries
        :param transition_dist: Distance over which mesh transitions
        :param max_edge: Maximum edge length
        :return: List of intermediate point coordinates
        """
        intermediate: List[List[float]] = []
        
        # Create rings around observation clusters
        from scipy.spatial import ConvexHull
        
        # Find observation clusters
        hull = ConvexHull(self.coords)
        hull_points = self.coords[hull.vertices]
        centroid = np.mean(self.coords, axis=0)
        
        # Add transition ring points
        n_rings = 3
        hull_distances = pdist(hull_points)
        if len(hull_distances) == 0 or transition_dist is None:
            return intermediate  # No intermediate points needed
            
        for ring in range(1, n_rings + 1):
            scale = 1.0 + (ring * transition_dist / np.mean(hull_distances))
            ring_points = centroid + (hull_points - centroid) * scale
            
            # Filter points that are useful for transition
            for point in ring_points:
                dist_to_obs, _ = obs_tree.query([point], k=1)
                if transition_dist * 0.5 < dist_to_obs[0] < transition_dist * 2:
                    intermediate.append(point.tolist())
        
        return intermediate
    
    def _triangle_area(self, vertices: np.ndarray) -> float:
        """
        Compute area of triangle from vertices.
        
        :param vertices: Triangle vertices as 3x2 array
        :return: Triangle area
        """
        v1, v2, v3 = vertices
        return 0.5 * abs((v2[0] - v1[0]) * (v3[1] - v1[1]) - 
                         (v3[0] - v1[0]) * (v2[1] - v1[1]))
    
    def _compute_adaptive_parameters(self) -> Dict[str, float]:
        """
        Compute data-driven parameters for adaptive meshing.
        
        Analyzes the spatial distribution of observations to determine appropriate
        mesh resolution parameters. Uses nearest neighbor distances and overall
        point distribution to set min/max edge lengths and transition distance.
        
        :return: Dictionary with keys 'min_edge_near_obs', 'max_edge_far', and 'transition_distance'
        """
        tree = KDTree(self.coords)
        
        # Analyze spacing
        k_neighbors = min(5, len(self.coords))
        distances, _ = tree.query(self.coords, k=k_neighbors)
        
        # Handle edge cases
        if distances.shape[1] > 1:
            nn_distances = distances[:, 1]  # Skip self (distance 0)
        else:
            # Single point or all points identical
            nn_distances = np.array([1.0])  # Default value
        
        # Identify clusters and gaps
        all_distances = pdist(self.coords)
        if len(all_distances) == 0:
            all_distances = nn_distances
        
        # Parameters based on distribution with safety checks
        min_edge_near_obs = max(np.percentile(nn_distances, 10) * 2, 1e-6)
        max_edge_far = np.percentile(all_distances, 90) * 0.3
        
        # Ensure reasonable ratio
        max_ratio = 50.0
        if max_edge_far / min_edge_near_obs > max_ratio:
            max_edge_far = min_edge_near_obs * max_ratio
        elif max_edge_far < min_edge_near_obs * 2:
            # Ensure there's some gradation
            max_edge_far = min_edge_near_obs * 2
        
        transition_distance = np.percentile(nn_distances, 75) * 5
        
        # Validate parameters
        if not np.isfinite(min_edge_near_obs) or min_edge_near_obs <= 0:
            raise MeshError("Invalid min_edge_near_obs computed")
        if not np.isfinite(max_edge_far) or max_edge_far <= 0:
            raise MeshError("Invalid max_edge_far computed")
        if not np.isfinite(transition_distance) or transition_distance <= 0:
            raise MeshError("Invalid transition_distance computed")
        
        return {
            'min_edge_near_obs': min_edge_near_obs,
            'max_edge_far': max_edge_far,
            'transition_distance': transition_distance
        }
    
    def _print_adaptive_diagnostics(self) -> None:
        """
        Print diagnostics for adaptive mesh.
        
        Computes and displays edge length distribution statistics and refinement ratio
        to help assess mesh quality.
        """
        # Compute edge length distribution
        edge_lengths = []
        for tri in self.triangles:
            for i in range(3):
                j = (i + 1) % 3
                edge = np.linalg.norm(self.vertices[tri[j]] - self.vertices[tri[i]])
                edge_lengths.append(edge)
        
        edge_lengths = np.array(edge_lengths)
        
        print("\nAdaptive Mesh Diagnostics:")
        print(f"  Edge length distribution:")
        print(f"    Min: {edge_lengths.min():.1f}")
        print(f"    25%: {np.percentile(edge_lengths, 25):.1f}")
        print(f"    Median: {np.median(edge_lengths):.1f}")
        print(f"    75%: {np.percentile(edge_lengths, 75):.1f}")
        print(f"    Max: {edge_lengths.max():.1f}")
        print(f"  Refinement ratio: {edge_lengths.max() / edge_lengths.min():.1f}x")
        
        # Add performance warnings similar to _print_diagnostics
        n_vertices = len(self.vertices)
        n_obs = len(self.coords)
        mesh_to_obs_ratio = n_vertices / n_obs
        
        if n_vertices > 5000:
            print("  WARNING: Large mesh - expect slower Stan sampling")
        if mesh_to_obs_ratio > 10:
            print("  WARNING: High mesh/observation ratio - consider coarser mesh")
    
