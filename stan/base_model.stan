data {
  int<lower=1> N_obs;
  int<lower=1> N_countries;
  vector[N_obs] y;
  array[N_obs] int<lower=1,upper=N_countries> country;
  array[N_obs] int<lower=0,upper=1> is_urban;
}
parameters {
  real alpha;
  real beta_urban;
  vector[N_countries] beta_country;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 5);
  sigma ~ normal(0, 5);
  beta_country ~ normal(0, 2.5);
  beta_urban ~ normal(0, 2.5);

  vector[N_obs] eta = alpha + beta_country[country] + beta_urban * to_vector(is_urban);
  y ~ normal(eta, sigma);
}
generated quantities {
  vector[N_obs] y_pred = alpha + beta_country[country] + beta_urban * to_vector(is_urban);
  array[N_obs] real y_rep = normal_rng(y_pred, rep_vector(sigma, N_obs));
}
