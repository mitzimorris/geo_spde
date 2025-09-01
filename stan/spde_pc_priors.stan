functions {
  // Log density of GMRF with sparse precision matrix
  real sparse_gmrf_lpdf(vector u_field, 
                        array[] int Q_u, array[] int Q_v, vector Q_w,
                        real log_det_Q) {
    int N = num_elements(u_field);
    real quadratic_form = csr_matrix_times_vector(N, N, Q_w, Q_v, Q_u, u_field)' * u_field;
    return -0.5 * (quadratic_form - log_det_Q);
  }
  
  // PC prior for log(kappa) - spatial range parameter
  real pc_prior_log_kappa_lpdf(real log_kappa, real lambda) {
    real kappa = exp(log_kappa);
    // PC prior on kappa with Jacobian adjustment for log transform
    return log(lambda) - lambda * kappa + log_kappa;
  }
  
  // PC prior for log(tau) - spatial variance parameter
  real pc_prior_log_tau_lpdf(real log_tau, real lambda) {
    real tau = exp(log_tau);
    // PC prior on tau with Jacobian adjustment for log transform
    return log(lambda) - lambda * tau + log_tau;
  }
}

data {
  // Dimensions
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
  
  // Spatial scale information from mesh diagnostics
  real<lower=0> estimated_range;      // From spatial_scale['estimated_range']
  real<lower=0> min_distance;         // From spatial_scale['min_distance']
  real<lower=0> median_distance;      // From spatial_scale['median_distance']
  real<lower=0> mesh_extent;          // From suggestions['mesh_extent']
  
  // Data characteristics
  real<lower=0> data_sd;              // Standard deviation of observations
  real<lower=0> coordinate_units_to_km; // From projection_info['unit_to_km']
  
  // User control
  int<lower=0,upper=3> prior_mode;    // 0=auto, 1=tight, 2=medium, 3=wide
  real<lower=0,upper=1> spatial_fraction; // Expected spatial variance contribution
  
  // SPDE smoothness
  int<lower=1,upper=2> alpha;         // 1 for Matérn ν=1/2, 2 for ν=3/2
}

transformed data {
  // Compute PC prior hyperparameters based on mesh diagnostics
  real rho_0;          // Reference range
  real alpha_rho;      // P(range < rho_0)
  real sigma_0;        // Reference spatial SD
  real alpha_sigma;    // P(sigma > sigma_0)
  
  // Scale factor for range-kappa relationship
  real scale_factor;
  if (alpha == 1) {
    scale_factor = 2.0;  // sqrt(8 * 0.5)
  } else {
    scale_factor = sqrt(12.0);  // sqrt(8 * 1.5)
  }
  
  if (prior_mode == 0) {
    // Automatic: Use mesh diagnostics
    rho_0 = estimated_range;
    alpha_rho = 0.5;
    sigma_0 = data_sd * sqrt(spatial_fraction);
    alpha_sigma = 0.05;
    
  } else if (prior_mode == 1) {
    // Tight priors (strong spatial structure expected)
    rho_0 = min_distance * 10;
    alpha_rho = 0.9;
    sigma_0 = data_sd * 0.3;
    alpha_sigma = 0.01;
    
  } else if (prior_mode == 2) {
    // Medium priors (default)
    rho_0 = median_distance * 3;
    alpha_rho = 0.5;
    sigma_0 = data_sd * 0.5;
    alpha_sigma = 0.05;
    
  } else {
    // Wide priors (weak spatial structure)
    rho_0 = mesh_extent * 0.3;
    alpha_rho = 0.1;
    sigma_0 = data_sd * 0.7;
    alpha_sigma = 0.1;
  }
  
  // Compute PC prior rate parameters
  real lambda_kappa = -log(alpha_rho) * scale_factor / rho_0;
  real lambda_tau = -log(alpha_sigma) / sigma_0;
  
  // Print prior configuration
  print("PC Prior Configuration:");
  print("  Reference range (rho_0): ", rho_0, " units");
  print("  P(range < rho_0): ", alpha_rho);
  print("  Reference spatial SD (sigma_0): ", sigma_0);
  print("  P(sigma > sigma_0): ", alpha_sigma);
  print("  Lambda for kappa: ", lambda_kappa);
  print("  Lambda for tau: ", lambda_tau);
}

parameters {
  vector[N_mesh] u_field;              // spatial field at mesh vertices
  real<lower=0> sigma_obs;             // observation noise SD
  real log_kappa;                      // log of spatial range parameter
  real log_tau;                        // log of spatial variance parameter
}

transformed parameters {
  real<lower=0> kappa = exp(log_kappa);
  real<lower=0> tau = exp(log_tau);
  
  // Spatial range in original units
  real<lower=0> range = scale_factor / kappa;
  
  // Spatial standard deviation
  real<lower=0> sigma_spatial = 1 / sqrt(tau);
  
  // Project field to observation locations
  vector[N_obs] spatial_effect = csr_matrix_times_vector(N_obs, N_mesh, A_w, A_v, A_u, u_field);
}

model {
  // Scale precision matrix Q by kappa^2 and tau
  vector[Q_nnz] Q_w_scaled = Q_w * square(kappa) * tau;
  real log_det_Q = log_det_Q_base + 2 * N_mesh * log(kappa) + N_mesh * log(tau);
  
  // GMRF prior on mesh vertices with PC priors
  target += sparse_gmrf_lpdf(u_field | Q_u, Q_v, Q_w_scaled, log_det_Q);
  
  // PC priors on hyperparameters
  target += pc_prior_log_kappa_lpdf(log_kappa | lambda_kappa);
  target += pc_prior_log_tau_lpdf(log_tau | lambda_tau);
  
  // Prior on observation noise
  sigma_obs ~ exponential(1 / data_sd);  // Weakly informative
  
  // Likelihood
  y ~ normal(spatial_effect, sigma_obs);
}

generated quantities {
  // Posterior predictive checks
  vector[N_obs] y_rep;
  for (i in 1:N_obs) {
    y_rep[i] = normal_rng(spatial_effect[i], sigma_obs);
  }
  
  // Compute variance partition
  real var_spatial = variance(spatial_effect);
  real var_obs = square(sigma_obs);
  real var_total = var_spatial + var_obs;
  real spatial_fraction_posterior = var_spatial / var_total;
  
  // Range in km for reporting
  real range_km = range * coordinate_units_to_km;
}