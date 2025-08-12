// Spatial Regression for PM2.5 Data using SPDE
// Uses CSR format for efficient sparse matrix operations

functions {
  // Compute quadratic form: x' * Q * x for CSR matrix Q
  real sparse_quad_form_csr(vector x, vector Q_w, array[] int Q_v, array[] int Q_u) {
    int N = num_elements(x);
    real result = 0;
    
    for (i in 1:N) {
      // For row i, entries are from Q_u[i] to Q_u[i+1]-1
      for (j in Q_u[i]:(Q_u[i+1]-1)) {
        result += x[i] * Q_w[j] * x[Q_v[j]];
      }
    }
    return result;
  }
}
data {
  int<lower=1> N_obs;
  int<lower=1> N_countries;
  
  vector[N_obs] y;
  array[N_obs] int<lower=1,upper=N_countries> country;
  array[N_obs] int<lower=0,upper=1> is_urban;

  int<lower=1> N_vertices;  // mesh
  
  // Sparse matrix A (projector) in CSR format
  int<lower=0> A_nnz;
  vector[A_nnz] A_w;          // values
  array[A_nnz] int A_v;       // column indices (1-indexed)
  array[N_obs + 1] int A_u;   // row pointers (1-indexed)
  
  // Sparse matrix Q (precision) in CSR format
  int<lower=0> Q_nnz;
  vector[Q_nnz] Q_w;              // values
  array[Q_nnz] int Q_v;           // column indices (1-indexed)
  array[N_vertices + 1] int Q_u;  // row pointers (1-indexed)
}
parameters {
  real alpha;
  real beta_urban;
  vector[N_countries] beta_country;
  real<lower=0> sigma;

  vector[N_vertices] w;              // Latent spatial process
  real<lower=0> tau;                 // Field amplitude
}
model {
  alpha ~ normal(0, 5);
  sigma ~ normal(0, 5);
  beta_country ~ normal(0, 2.5);
  beta_urban ~ normal(0, 2.5);
  tau ~ normal(0, 5);      // Field amplitude
  
  // SPDE prior: w ~ N(0, tau^(-2) * Q^{-1})
  // Equivalently: tau * sqrt(Q) * w ~ N(0, I)
  // Implementation: -0.5 * tau^2 * w' * Q * w
  target += -0.5 * tau * tau * sparse_quad_form_csr(w, Q_w, Q_v, Q_u);
  
  // Log determinant adjustment for proper density
  // For precision Q: log|Q|^{1/2} = 0.5 * log|Q|
  // Approximation: use N_vertices * log(tau) for the tau scaling
  target += N_vertices * log(tau);
  
  // Linear predictor: mu + A*w
  vector[N_obs] eta = alpha + beta_country[country] + beta_urban * to_vector(is_urban)
                      + csr_matrix_times_vector(N_obs, N_vertices, A_w, A_v, A_u, w);
  y ~ normal(eta, sigma);
}

generated quantities {
  vector[N_obs] y_pred = alpha + beta_country[country] + beta_urban * to_vector(is_urban)
                         + csr_matrix_times_vector(N_obs, N_vertices, A_w, A_v, A_u, w);
  array[N_obs] real y_rep = normal_rng(y_pred, rep_vector(sigma, N_obs));
}
