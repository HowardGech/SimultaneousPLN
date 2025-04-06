#ifndef ADMM_H
#define ADMM_H

void update_mu(
    double* mu_M,
    double* y,
    double* sigma,
    double* ridge_inv,
    double* log_diff,
    int p,
    double rho,
    int max_iter,
    int min_iter,
    double alpha_init,
    double mu_lowbound,
    double mu_highbound,
    double eps
);

void update_sigma(
    double* Sigma,
    double* Omega,
    double* mu,
    int p,
    int max_iter,
    double sigma_lowbound,
    double sigma_highbound,
    double eps
);

#endif