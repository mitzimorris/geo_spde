"""
Penalized Complexity (PC) priors for SPDE models.

This module implements PC priors following Simpson et al. (2017) and 
Fuglstad et al. (2019), providing automatic prior configuration based 
on mesh and data characteristics.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.special import gamma as gamma_func
from scipy.stats import gamma, expon


def pc_prior_range(
    rho_0: float,
    alpha_rho: float,
    alpha: int = 2
) -> Dict[str, float]:
    """
    Compute PC prior parameters for spatial range.
    
    The PC prior for range rho has the form:
    pi(rho) proportional to rho^(-1) exp(-lambda_rho * rho)
    
    where lambda_rho is chosen such that P(rho < rho_0) = alpha_rho
    
    :param rho_0: Reference range value
    :param alpha_rho: Probability that range < rho_0 (typically 0.5)
    :param alpha: Smoothness parameter (1 for Matern nu=1/2, 2 for nu=3/2)
    
    :returns: Dict with lambda_rho (rate parameter), kappa_0 (reference kappa), prior_params
    """
    # For Matern, relationship between range and kappa
    # range = sqrt(8*nu) / kappa
    # For nu=1/2 (alpha=1): sqrt(8*0.5) = 2
    # For nu=3/2 (alpha=2): sqrt(8*1.5) = sqrt(12)
    
    if alpha == 1:
        scale_factor = 2.0
    elif alpha == 2:
        scale_factor = np.sqrt(12)
    else:
        raise ValueError(f"Unsupported alpha value: {alpha}")
    
    kappa_0 = scale_factor / rho_0
    lambda_rho = -np.log(alpha_rho) / rho_0

    return {
        'lambda_rho': lambda_rho,
        'kappa_0': kappa_0,
        'rho_0': rho_0,
        'alpha_rho': alpha_rho,
        'scale_factor': scale_factor
    }


def pc_prior_variance(
    sigma_0: float,
    alpha_sigma: float
) -> Dict[str, float]:
    """
    Compute PC prior parameters for spatial standard deviation.
    
    The PC prior for standard deviation sigma has the form:
    pi(sigma) proportional to exp(-lambda_sigma * sigma)
    
    where lambda_sigma is chosen such that P(sigma > sigma_0) = alpha_sigma
    
    :param sigma_0: Reference standard deviation value
    :param alpha_sigma: Probability that sigma > sigma_0 (typically 0.05)
    
    :returns: Dict with lambda_sigma (rate parameter), sigma_0 (reference value), prior_params
    """
    lambda_sigma = -np.log(alpha_sigma) / sigma_0

    return {
        'lambda_sigma': lambda_sigma,
        'sigma_0': sigma_0,
        'alpha_sigma': alpha_sigma
    }


def compute_pc_prior_params(
    estimated_range: float,
    data_sd: float,
    prior_mode: str,
    spatial_fraction: float,
    mesh_diagnostics: Dict,
    alpha: int = 2
) -> Dict[str, float]:
    """
    Compute complete PC prior parameters based on mode and diagnostics.
    
    :param estimated_range: Estimated spatial range from mesh diagnostics
    :param data_sd: Standard deviation of observations
    :param prior_mode: Prior configuration mode ("auto", "tight", "medium", or "wide")
    :param spatial_fraction: Expected proportion of variance from spatial field
    :param mesh_diagnostics: Mesh scale diagnostics
    :param alpha: Smoothness parameter
    
    :returns: Complete PC prior parameters for Stan
    """
    min_distance = mesh_diagnostics['spatial_scale']['min_distance']
    median_distance = mesh_diagnostics['spatial_scale']['median_distance']
    mesh_extent = mesh_diagnostics['suggestions']['mesh_extent']
    
    if prior_mode == "auto":
        rho_0 = estimated_range
        alpha_rho = 0.5
        sigma_0 = data_sd * np.sqrt(spatial_fraction)
        alpha_sigma = 0.05
        
    elif prior_mode == "tight":
        rho_0 = min_distance * 10
        alpha_rho = 0.9
        sigma_0 = data_sd * 0.3
        alpha_sigma = 0.01
        
    elif prior_mode == "medium":
        rho_0 = median_distance * 3
        alpha_rho = 0.5
        sigma_0 = data_sd * 0.5
        alpha_sigma = 0.05
        
    else:  # wide
        rho_0 = mesh_extent * 0.3
        alpha_rho = 0.1
        sigma_0 = data_sd * 0.7
        alpha_sigma = 0.1
    
    range_prior = pc_prior_range(rho_0, alpha_rho, alpha)
    variance_prior = pc_prior_variance(sigma_0, alpha_sigma)
    
    return {
        'rho_0': rho_0,
        'alpha_rho': alpha_rho,
        'lambda_rho': range_prior['lambda_rho'],
        'kappa_0': range_prior['kappa_0'],
        'sigma_0': sigma_0,
        'alpha_sigma': alpha_sigma,
        'lambda_sigma': variance_prior['lambda_sigma'],
        'expected_range': rho_0 / (1 - alpha_rho),  # Approximate median
        'expected_sigma': sigma_0 * (-np.log(0.5) / -np.log(alpha_sigma))  # Median
    }


def validate_pc_priors(
    pc_params: Dict[str, float],
    mesh_diagnostics: Dict,
    verbose: bool = True
) -> Dict[str, bool]:
    """
    Validate PC prior choices against mesh resolution.
    
    :param pc_params: PC prior parameters
    :param mesh_diagnostics: Mesh diagnostics
    :param verbose: Print warnings
    
    :returns: Validation results
    """
    validation = {
        'range_compatible': True,
        'variance_reasonable': True,
        'mesh_adequate': True
    }
    
    min_edge = mesh_diagnostics['edge_lengths']['min']
    max_edge = mesh_diagnostics['edge_lengths']['max']
    
    if pc_params['rho_0'] < min_edge * 3:
        validation['range_compatible'] = False
        if verbose:
            print(f"WARNING: Prior range ({pc_params['rho_0']:.2f}) may be too small "
                  f"for mesh resolution (min edge: {min_edge:.2f})")
    
    if pc_params['rho_0'] > max_edge * 100:
        validation['mesh_adequate'] = False
        if verbose:
            print(f"WARNING: Prior range ({pc_params['rho_0']:.2f}) much larger than "
                  f"mesh extent - consider wider mesh")
    
    if pc_params['sigma_0'] < 1e-6:
        validation['variance_reasonable'] = False
        if verbose:
            print(f"WARNING: Prior variance very small ({pc_params['sigma_0']:.6f})")
    
    return validation


def generate_pc_prior_code() -> str:
    """
    Generate Stan code for PC priors.
    
    :returns: Stan functions for PC priors
    """
    return """
functions {
  // PC prior for log(kappa) 
  real pc_prior_log_kappa_lpdf(real log_kappa, real lambda_kappa, real log_kappa_0) {
    real kappa = exp(log_kappa);
    real kappa_0 = exp(log_kappa_0);
    
    // PC prior: pi(kappa) proportional to exp(-lambda * (kappa - kappa_0))
    // With change of variables to log(kappa)
    return log(lambda_kappa) - lambda_kappa * (kappa - kappa_0) + log_kappa;
  }
  
  // PC prior for log(tau)
  real pc_prior_log_tau_lpdf(real log_tau, real lambda_tau) {
    real tau = exp(log_tau);
    
    // PC prior: pi(tau) proportional to exp(-lambda * tau)
    // With change of variables to log(tau)
    return log(lambda_tau) - lambda_tau * tau + log_tau;
  }
  
  // Alternative: PC prior for range directly
  real pc_prior_range_lpdf(real range, real lambda_rho, real d) {
    // d is dimension (2 for 2D spatial)
    return log(d) + log(lambda_rho) + (d - 1) * log(range) - lambda_rho * range;
  }
  
  // PC prior for standard deviation
  real pc_prior_sigma_lpdf(real sigma, real lambda_sigma) {
    return log(lambda_sigma) - lambda_sigma * sigma;
  }
}
"""


def sample_from_pc_prior(
    pc_params: Dict[str, float],
    n_samples: int = 1000,
    alpha: int = 2
) -> Dict[str, np.ndarray]:
    """
    Sample from PC priors for visualization.
    
    :param pc_params: PC prior parameters
    :param n_samples: Number of samples to draw
    :param alpha: Smoothness parameter
    
    :returns: Samples from prior distributions
    """
    lambda_rho = pc_params['lambda_rho']
    range_samples = expon.rvs(scale=1/lambda_rho, size=n_samples)
    
    lambda_sigma = pc_params['lambda_sigma']
    sigma_samples = expon.rvs(scale=1/lambda_sigma, size=n_samples)
    
    if alpha == 1:
        scale_factor = 2.0  # sqrt(8 * 0.5) for Matern nu=1/2
    else:
        scale_factor = np.sqrt(12)  # sqrt(8 * 1.5) for Matern nu=3/2
    
    kappa_samples = scale_factor / range_samples
    tau_samples = gamma.rvs(a=1, scale=1/lambda_sigma, size=n_samples)
    
    return {
        'range': range_samples,
        'sigma': sigma_samples,
        'kappa': kappa_samples,
        'tau': tau_samples
    }
