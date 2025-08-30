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

from typing import Tuple, Dict, Optional

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
        
        # Basic distance statistics
        if n_obs < 100:
            dists = distance_matrix(self.coords, self.coords)
            dists_flat = dists[np.triu_indices(n_obs, k=1)]
        else:
            # Subsample for efficiency
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
                results['practical_range'] = var_range * 3  # For exponential
                results['method_used'] = 'variogram'
            except:
                method = 'mst'  # Fallback
        
        if method in ['mst', 'both']:
            mst_scale = self._estimate_range_mst()
            if 'estimated_range' not in results or method == 'mst':
                results['estimated_range'] = mst_scale
                results['practical_range'] = mst_scale * 3
                results['method_used'] = 'mst'
            else:
                results['mst_range'] = mst_scale
        
        # Store for later use
        self.spatial_scale = results
        return results
    
    def _estimate_range_variogram(self, max_pairs: int = 5000) -> float:
        """
        Estimate range using empirical variogram with exponential model.
        Assumes data would have constant mean and exponential covariance.
        """
        n_obs = len(self.coords)
        
        # Create subset if needed
        if n_obs * (n_obs - 1) / 2 > max_pairs:
            n_sub = int(np.sqrt(max_pairs * 2))
            idx = np.random.choice(n_obs, n_sub, replace=False)
            coords_sub = self.coords[idx]
        else:
            coords_sub = self.coords
        
        # Compute pairwise distances
        dists = distance_matrix(coords_sub, coords_sub)
        dists_flat = dists[np.triu_indices(len(coords_sub), k=1)]
        
        # Bin distances
        max_dist = np.percentile(dists_flat, 50)
        n_bins = min(20, len(dists_flat) // 50)
        bins = np.linspace(0, max_dist, n_bins)
        
        # Distance-based variance estimation
        bin_centers = (bins[:-1] + bins[1:]) / 2
        distance_variance = np.zeros(len(bin_centers))
        
        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            mask = (dists_flat >= low) & (dists_flat < high)
            if np.sum(mask) > 5:
                distance_variance[i] = np.var(dists_flat[mask])
        
        # Fit exponential decay
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
        """
        n_obs = len(self.coords)
        
        if n_obs > 500:
            idx = np.random.choice(n_obs, 500, replace=False)
            coords_sub = self.coords[idx]
        else:
            coords_sub = self.coords
        
        # Build MST
        dists = distance_matrix(coords_sub, coords_sub)
        mst = minimum_spanning_tree(dists)
        
        # Get MST edge lengths
        mst_edges = mst.tocoo()
        edge_lengths = mst_edges.data
        
        # Use 75th percentile of MST edges as characteristic scale
        return np.percentile(edge_lengths, 75)
    
    def validate_mesh_resolution(self, verbose: bool = True) -> Dict[str, bool]:
        """
        Check if mesh resolution is appropriate for the estimated correlation range.
        
        Parameters
        ----------
        verbose : bool
            Print warnings
            
        Returns
        -------
        Dict with validation results:
            - 'resolution_ok': Is mesh fine enough for correlation range?
            - 'extent_ok': Is mesh extent large enough?
            - 'edge_to_range_ratio': Ratio of edge length to correlation range
            - 'recommended_edge_factor': Suggested target_edge_factor
        """
        if self.mesh_params is None:
            raise MeshError("Mesh parameters not computed. Call create_mesh() first.")
        
        # Get spatial scale if not already computed
        if not hasattr(self, 'spatial_scale'):
            self.estimate_spatial_scale()
        
        max_edge = self.mesh_params['max_edge']
        estimated_range = self.spatial_scale['estimated_range']
        practical_range = self.spatial_scale.get('practical_range', estimated_range * 3)
        
        # Rule of thumb: need at least 3-5 mesh elements per correlation range
        edge_to_range_ratio = max_edge / estimated_range
        resolution_ok = edge_to_range_ratio < 0.5
        
        # Check mesh extent
        if self.vertices is not None:
            mesh_extent = np.max([
                self.vertices[:, 0].max() - self.vertices[:, 0].min(),
                self.vertices[:, 1].max() - self.vertices[:, 1].min()
            ])
            extent_ok = mesh_extent > practical_range * 2
        else:
            extent_ok = True  # Can't check without mesh
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
        
        Returns
        -------
        Dict with suggested parameters:
            - 'kappa_suggestion': Suggested kappa value
            - 'tau_suggestion': Suggested tau value
            - 'spatial_range_suggestion': In mesh units
            - 'spatial_sd_suggestion': Marginal standard deviation
            - 'prior_range_kappa': (lower, upper) for kappa prior
            - 'prior_range_tau': (lower, upper) for tau prior
        """
        # Ensure we have spatial scale estimates
        if not hasattr(self, 'spatial_scale'):
            self.estimate_spatial_scale()
        
        estimated_range = self.spatial_scale['estimated_range']
        min_dist = self.spatial_scale['min_distance']
        median_dist = self.spatial_scale['median_distance']
        
        # Compute mesh extent
        x_range = self.coords[:, 0].max() - self.coords[:, 0].min()
        y_range = self.coords[:, 1].max() - self.coords[:, 1].min()
        mesh_extent = max(x_range, y_range)
        
        # Suggest range as fraction of domain
        suggested_range = np.clip(estimated_range, mesh_extent * 0.05, mesh_extent * 0.3)
        
        # Convert to kappa (for Matérn nu=1/2)
        kappa_suggestion = np.sqrt(8) / suggested_range
        
        # Tau suggestion based on domain size
        # Smaller tau for larger domains to maintain reasonable variance
        tau_suggestion = 1.0 / np.sqrt(mesh_extent / median_dist)
        
        # Prior ranges: within order of magnitude
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
        
        Parameters
        ----------
        verbose : bool
            Print diagnostic information
            
        Returns
        -------
        Dict containing spatial scale, validation, and parameter suggestions
        """
        if self.vertices is None:
            raise MeshError("Mesh must be created before computing scale diagnostics")
        
        # Estimate spatial scales
        spatial_scale = self.estimate_spatial_scale(method='both')
        
        # Validate mesh
        validation = self.validate_mesh_resolution(verbose=verbose)
        
        # Get parameter suggestions
        suggestions = self.suggest_spde_parameters()
        
        # Store everything
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
