"""
Unit tests for geo_spde.matrices module.

Tests FEM matrix computation for SPDE approximations including:
- Mass matrix (C)
- Stiffness matrix (G)
- Projector matrix (A)
- Precision matrices (Q) for different Matern covariances
"""

import pytest
import numpy as np
from scipy import sparse
from geo_spde.matrices import (
    compute_fem_matrices,
    compute_matern_precision_nu_half,
    compute_matern_precision_nu_three_halves,
    _triangle_areas_vectorized,
    _compute_basis_gradients_vectorized,
    _compute_barycentric_vectorized,
    _compute_mass_matrix_vectorized,
    _compute_stiffness_matrix_vectorized,
    _compute_projector_matrix_vectorized,
    select_reference_parameters,
    compute_fem_matrices_scaled,
    check_precision_conditioning
)


class TestFEMMatrices:
    """Test suite for FEM matrix computations."""
    
    def setup_method(self):
        """Create simple test meshes."""
        # Unit square mesh with 4 vertices
        self.vertices_simple = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1]
        ])
        self.triangles_simple = np.array([
            [0, 1, 2], [0, 2, 3]
        ])
        self.obs_coords_simple = np.array([
            [0.5, 0.5], [0.25, 0.25]
        ])
        
        # Regular grid mesh
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        xx, yy = np.meshgrid(x, y)
        self.vertices_grid = np.column_stack([xx.ravel(), yy.ravel()])
        
    def test_triangle_areas(self):
        """Test triangle area computation."""
        tri_coords = self.vertices_simple[self.triangles_simple]
        areas = _triangle_areas_vectorized(tri_coords)
        
        # Each triangle should have area 0.5
        np.testing.assert_allclose(areas, [0.5, 0.5])
        
    def test_degenerate_triangle_detection(self):
        """Test handling of degenerate triangles."""
        vertices = np.array([[0, 0], [1, 0], [0.5, 0]])  # Collinear points
        triangles = np.array([[0, 1, 2]])
        tri_coords = vertices[triangles]
        areas = _triangle_areas_vectorized(tri_coords)
        assert areas[0] < 1e-12
        
    def test_mass_matrix_properties(self):
        """Test mass matrix computation and properties."""
        C = _compute_mass_matrix_vectorized(
            self.vertices_simple, self.triangles_simple
        )
        
        # Check symmetry
        assert np.allclose(C.toarray(), C.T.toarray())
        
        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(C.toarray())
        assert np.all(eigenvalues > -1e-10)
        
        # Check row sums (should be proportional to vertex areas)
        row_sums = np.array(C.sum(axis=1)).flatten()
        assert np.all(row_sums > 0)
        
    def test_stiffness_matrix_properties(self):
        """Test stiffness matrix computation and properties."""
        G = _compute_stiffness_matrix_vectorized(
            self.vertices_simple, self.triangles_simple
        )
        
        # Check symmetry
        assert np.allclose(G.toarray(), G.T.toarray())
        
        # Check positive semi-definiteness
        eigenvalues = np.linalg.eigvalsh(G.toarray())
        assert np.all(eigenvalues > -1e-10)
        
        # Check that constant functions are in nullspace
        ones = np.ones(len(self.vertices_simple))
        assert np.allclose(G @ ones, 0, atol=1e-10)
        
    def test_projector_matrix_properties(self):
        """Test projector matrix computation."""
        A = _compute_projector_matrix_vectorized(
            self.vertices_simple, self.triangles_simple, self.obs_coords_simple
        )
        
        # Check shape
        assert A.shape == (len(self.obs_coords_simple), len(self.vertices_simple))
        
        # Check row sums (should be 1 for interpolation)
        row_sums = np.array(A.sum(axis=1)).flatten()
        np.testing.assert_allclose(row_sums, 1.0)
        
        # Check non-negativity (for points inside mesh)
        assert np.all(A.toarray() >= -1e-10)
        
    def test_barycentric_coordinates(self):
        """Test barycentric coordinate computation."""
        points = np.array([[0.5, 0.5]])
        triangle_indices = np.array([0])
        
        bary = _compute_barycentric_vectorized(
            points, self.vertices_simple, self.triangles_simple, triangle_indices
        )
        
        # Check that coordinates sum to 1
        np.testing.assert_allclose(bary.sum(axis=1), 1.0)
        
        # Check non-negativity for interior points
        assert np.all(bary >= -1e-10)
        
    def test_matern_precision_nu_half(self):
        """Test Matern nu=1/2 precision matrix."""
        C = _compute_mass_matrix_vectorized(
            self.vertices_simple, self.triangles_simple
        )
        G = _compute_stiffness_matrix_vectorized(
            self.vertices_simple, self.triangles_simple
        )
        
        kappa = 2.0
        Q = compute_matern_precision_nu_half(C, G, kappa)
        
        # Check symmetry
        assert np.allclose(Q.toarray(), Q.T.toarray())
        
        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(Q.toarray())
        assert np.all(eigenvalues > -1e-10)
        
    def test_matern_precision_nu_three_halves(self):
        """Test Matern nu=3/2 precision matrix."""
        C = _compute_mass_matrix_vectorized(
            self.vertices_simple, self.triangles_simple
        )
        G = _compute_stiffness_matrix_vectorized(
            self.vertices_simple, self.triangles_simple
        )
        
        kappa = 2.0
        Q = compute_matern_precision_nu_three_halves(C, G, kappa)
        
        # Check symmetry
        assert np.allclose(Q.toarray(), Q.T.toarray())
        
        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(Q.toarray())
        assert np.all(eigenvalues > -1e-10)
        
    def test_compute_fem_matrices_integration(self):
        """Test full FEM matrix computation."""
        C, G, A, Q = compute_fem_matrices(
            self.vertices_simple, 
            self.triangles_simple,
            self.obs_coords_simple,
            kappa=2.0,
            alpha=1,
            verbose=False
        )
        
        assert C.shape == (4, 4)
        assert G.shape == (4, 4)
        assert A.shape == (2, 4)
        assert Q.shape == (4, 4)
        
        # Test without Q matrix
        C2, G2, A2, Q2 = compute_fem_matrices(
            self.vertices_simple,
            self.triangles_simple,
            self.obs_coords_simple,
            kappa=None,
            verbose=False
        )
        assert Q2 is None
        
    def test_reference_parameter_selection(self):
        """Test automatic parameter selection."""
        kappa_ref, tau_ref = select_reference_parameters(
            self.vertices_simple,
            self.triangles_simple,
            self.obs_coords_simple,
            alpha=1
        )
        
        assert kappa_ref > 0
        assert tau_ref > 0
        
        # Test for alpha=2
        kappa_ref2, tau_ref2 = select_reference_parameters(
            self.vertices_simple,
            self.triangles_simple,
            self.obs_coords_simple,
            alpha=2
        )
        
        assert kappa_ref2 > kappa_ref  # Should be larger for alpha=2
        
    def test_scaled_matrices(self):
        """Test scaled FEM matrix computation."""
        C, G, A, Q, scale_info = compute_fem_matrices_scaled(
            self.vertices_simple,
            self.triangles_simple,
            self.obs_coords_simple,
            auto_scale=True,
            verbose=False
        )
        
        assert 'kappa_used' in scale_info
        assert 'tau_used' in scale_info
        assert 'condition_number' in scale_info
        
    def test_conditioning_check(self):
        """Test precision matrix conditioning check."""
        C = _compute_mass_matrix_vectorized(
            self.vertices_simple, self.triangles_simple
        )
        G = _compute_stiffness_matrix_vectorized(
            self.vertices_simple, self.triangles_simple
        )
        Q = compute_matern_precision_nu_half(C, G, kappa=2.0)
        
        diagnostics = check_precision_conditioning(
            Q, C, G, kappa=2.0, tau=1.0, verbose=False
        )
        
        assert 'condition_number' in diagnostics
        assert 'dominance_ratio' in diagnostics
        assert 'well_conditioned' in diagnostics
        
    def test_points_outside_mesh(self):
        """Test handling of observation points outside mesh."""
        outside_coords = np.array([[-1, -1], [2, 2]])
        
        with pytest.warns(UserWarning, match="observation points outside"):
            A = _compute_projector_matrix_vectorized(
                self.vertices_simple,
                self.triangles_simple,
                outside_coords
            )
        
        # Should still produce valid matrix
        assert A.shape == (2, 4)
        row_sums = np.array(A.sum(axis=1)).flatten()
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)
        
    def test_basis_gradient_computation(self):
        """Test basis function gradient computation."""
        tri_coords = self.vertices_simple[self.triangles_simple]
        gradients = _compute_basis_gradients_vectorized(tri_coords)
        
        # Check shape
        assert gradients.shape == (2, 3, 2)
        
        # Sum of gradients should be zero (partition of unity)
        grad_sum = gradients.sum(axis=1)
        np.testing.assert_allclose(grad_sum, 0, atol=1e-10)
        
    def test_invalid_alpha_values(self):
        """Test error handling for invalid alpha values."""
        with pytest.raises(ValueError, match="Alpha=3 not supported"):
            compute_fem_matrices(
                self.vertices_simple,
                self.triangles_simple,
                self.obs_coords_simple,
                kappa=2.0,
                alpha=3
            )
            
    def test_sparse_matrix_efficiency(self):
        """Test that matrices maintain sparsity."""
        # Create larger mesh
        n = 20
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        xx, yy = np.meshgrid(x, y)
        vertices = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Simple triangulation
        triangles = []
        for i in range(n-1):
            for j in range(n-1):
                idx = i * n + j
                triangles.append([idx, idx+1, idx+n])
                triangles.append([idx+1, idx+n+1, idx+n])
        triangles = np.array(triangles)
        
        C = _compute_mass_matrix_vectorized(vertices, triangles)
        G = _compute_stiffness_matrix_vectorized(vertices, triangles)
        
        # Check sparsity (should be < 5% dense for this mesh)
        c_density = C.nnz / (C.shape[0] * C.shape[1])
        g_density = G.nnz / (G.shape[0] * G.shape[1])
        
        assert c_density < 0.05
        assert g_density < 0.05
        
    def test_interpolation_accuracy(self):
        """Test that projector matrix correctly interpolates vertex values."""
        # Define a linear function on vertices
        f_vertices = self.vertices_simple[:, 0] + 2 * self.vertices_simple[:, 1]
        
        # Compute projector matrix
        A = _compute_projector_matrix_vectorized(
            self.vertices_simple, self.triangles_simple, self.obs_coords_simple
        )
        
        # Interpolate to observation points
        f_obs = A @ f_vertices
        
        # True values at observation points
        f_true = self.obs_coords_simple[:, 0] + 2 * self.obs_coords_simple[:, 1]
        
        # Should match exactly for linear functions on triangular elements
        np.testing.assert_allclose(f_obs, f_true, rtol=1e-10)
        
    def test_matrix_consistency(self):
        """Test consistency between different matrix computations."""
        # Test that compute_fem_matrices produces same results as individual functions
        C1, G1, A1, Q1 = compute_fem_matrices(
            self.vertices_simple,
            self.triangles_simple,
            self.obs_coords_simple,
            kappa=2.0,
            alpha=1,
            verbose=False
        )
        
        C2 = _compute_mass_matrix_vectorized(
            self.vertices_simple, self.triangles_simple
        )
        G2 = _compute_stiffness_matrix_vectorized(
            self.vertices_simple, self.triangles_simple
        )
        A2 = _compute_projector_matrix_vectorized(
            self.vertices_simple, self.triangles_simple, self.obs_coords_simple
        )
        Q2 = compute_matern_precision_nu_half(C2, G2, kappa=2.0)
        
        assert np.allclose(C1.toarray(), C2.toarray())
        assert np.allclose(G1.toarray(), G2.toarray())
        assert np.allclose(A1.toarray(), A2.toarray())
        assert np.allclose(Q1.toarray(), Q2.toarray())
        
    def test_edge_length_computation(self):
        """Test that edge length computation uses all triangles."""
        # Create a mesh with varying edge lengths
        vertices = np.array([
            [0, 0], [1, 0], [2, 0],  # Bottom row
            [0, 1], [1, 1], [2, 1]   # Top row
        ])
        triangles = np.array([
            [0, 1, 3], [1, 4, 3],  # Left square
            [1, 2, 4], [2, 5, 4]   # Right square
        ])
        obs_coords = np.array([[0.5, 0.5], [1.5, 0.5]])
        
        kappa_ref, tau_ref = select_reference_parameters(
            vertices, triangles, obs_coords, alpha=1
        )
        
        # Should compute edge lengths for all triangles (4 triangles * 3 edges each)
        # Even though some edges are shared, the computation considers all
        assert kappa_ref > 0
        assert tau_ref > 0
        
    def test_batch_solving_efficiency(self):
        """Test that batch solving in Matern nu=3/2 works correctly."""
        # Use a slightly larger mesh for meaningful test
        n = 10
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        xx, yy = np.meshgrid(x, y)
        vertices = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Simple triangulation
        triangles = []
        for i in range(n-1):
            for j in range(n-1):
                idx = i * n + j
                triangles.append([idx, idx+1, idx+n])
                triangles.append([idx+1, idx+n+1, idx+n])
        triangles = np.array(triangles)
        
        C = _compute_mass_matrix_vectorized(vertices, triangles)
        G = _compute_stiffness_matrix_vectorized(vertices, triangles)
        
        # Compute Matern nu=3/2 precision matrix
        Q = compute_matern_precision_nu_three_halves(C, G, kappa=2.0)
        
        # Check that result is valid
        assert Q.shape == (n*n, n*n)
        assert np.allclose(Q.toarray(), Q.T.toarray())  # Symmetry
        
        # Check positive definiteness (with tolerance for numerical errors)
        eigenvalues = np.linalg.eigvalsh(Q.toarray())
        assert np.all(eigenvalues > -1e-8)