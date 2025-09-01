data {
  int<lower=1> N_obs;
  vector[N_obs] y;
  int<lower=1> N_mesh;
  
  // Sparse matrix A (projector) in CSR format
  int<lower=0> A_nnz;
  vector[A_nnz] A_w;
  array[A_nnz] int A_v;
  array[N_obs + 1] int A_u;
  
  // Sparse matrix Q (precision) in CSR format  
  int<lower=0> Q_nnz;
  vector[Q_nnz] Q_w;
  array[Q_nnz] int Q_v;
  array[N_mesh + 1] int Q_u;
}

parameters {
  sum_to_zero_vector[N_mesh] u_field;
  real<lower=0> sigma_obs;
  real<lower=0> rho;
  real<lower=0> sigma_sp;
}

transformed parameters {
  real<lower=0> kappa = sqrt(8) / rho;
  real<lower=0> tau = 1 / square(sigma_sp);
  vector[N_obs] spatial_effect = csr_matrix_times_vector(N_obs, N_mesh, A_w, A_v, A_u, u_field);
}

model {
  // Scale Q matrix by kappa^2 * tau
  vector[Q_nnz] Q_w_scaled = Q_w * square(kappa) * tau;
  
  // Compute quadratic form: u' * Q * u
  real quadratic_form = csr_matrix_times_vector(N_mesh, N_mesh, Q_w_scaled, Q_v, Q_u, u_field)' * u_field;
  
  // GMRF log probability (improper, rank-deficient by 1)
  target += -0.5 * quadratic_form;
  
  // Log determinant contribution (accounting for rank deficiency)
  target += (N_mesh - 1) * log(kappa) + 0.5 * (N_mesh - 1) * log(tau);
  
  // The sum-to-zero constraint implicitly handles the scaling
  // No explicit prior on u_field needed due to GMRF structure
  
  // Priors on hyperparameters
  rho ~ normal(300, 200);  // Weakly informative
  sigma_sp ~ normal(1, 1);  // Weakly informative
  sigma_obs ~ normal(0.5, 0.5);  // Weakly informative
  
  // Likelihood
  y ~ normal(spatial_effect, sigma_obs);
}