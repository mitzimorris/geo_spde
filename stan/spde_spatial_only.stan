
functions {
  // Log density of GMRF with sparse precision matrix
  real sparse_gmrf_lpdf(vector u_field, 
                        array[] int Q_u, array[] int Q_v, vector Q_w,
                        real log_det_Q) {
    int N = num_elements(u_field);
    real quadratic_form = csr_matrix_times_vector(N, N, Q_w, Q_v, Q_u, u_field)' * u_field;
    return -0.5 * (quadratic_form - log_det_Q);
  }
}

data {
  int<lower=1> N_obs;                 // number of observations
  int<lower=1> N_mesh;                // number of mesh vertices
  
  // Sparse matrix A (projector) in CSR format
  int<lower=0> A_nnz;                 // number of non-zeros in A
  vector[A_nnz] A_w;                  // values
  array[A_nnz] int A_v;               // column indices (1-indexed)
  array[N_obs + 1] int A_u;           // row pointers (1-indexed)
  
  // Sparse matrix Q (precision) in CSR format  
  int<lower=0> Q_nnz;                 // number of non-zeros in Q
  vector[Q_nnz] Q_w;                  // values (for kappa=1, tau=1)
  array[Q_nnz] int Q_v;               // column indices (1-indexed)
  array[N_mesh + 1] int Q_u;          // row pointers (1-indexed)
  
  real log_det_Q_base;                // log determinant for base Q (kappa=1, tau=1)
  
  // Observations
  vector[N_obs] y;                    // observed values
}

parameters {
  vector[N_mesh] u_field;              // spatial field at mesh vertices
  real<lower=0> sigma;                 // observation noise SD
  real<lower=0> kappa;
  real<lower=0> tau;
  //  real<lower=0> spatial_sd;     // Standard deviation of spatial field
  //  real<lower=0> spatial_range;  // Practical correlation range
}
transformed parameters {
  vector[N_obs] spatial_effect = csr_matrix_times_vector(N_obs, N_mesh, A_w, A_v, A_u, u_field);
  //  real tau = 1 / square(spatial_sd);
  //  real kappa = sqrt(8) / spatial_range;  // For Matérn ν=1/2
}
model {
  // Scale precision matrix Q by kappa^2 and tau
  vector[Q_nnz] Q_w_scaled = Q_w * square(kappa) * tau;
  real log_det_Q = log_det_Q_base + 2 * N_mesh * log(kappa) + N_mesh * log(tau);
  
  // GMRF prior on mesh vertices
  target += sparse_gmrf_lpdf(u_field | Q_u, Q_v, Q_w_scaled, log_det_Q);
  
  // Likelihood
  y ~ normal(spatial_effect, sigma);
  
  // Priors
  tau ~ std_normal();
  kappa ~ normal(4, 1);          // Moderate range (roughly 1/3 of region)
  sigma ~ std_normal();
  //  spatial_sd ~ exponential(1);           // P(sd > 1) ≈ 0.37
  //  spatial_range ~ inv_gamma(2, 1);       // P(range > 0.5) ≈ 0.37
  //  sigma ~ exponential(1);              // Prior for observation noise
}

// generated quantities {
//   array[N_obs] real y_rep;             // posterior predictive samples
//   for (i in 1:N_obs) {
//     y_rep[i] = normal_rng(spatial_effect[i], sigma);
//   }
// }
