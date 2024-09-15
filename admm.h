#ifndef ADMM_H
#define ADMM_H

void update_mu(
    double* mu_M,
    double* mu_N,
    double* y,
    double* sigma,
    double** ridge_inv,
    double* log_diff,
    int p,
    double* alpha,
    double rho,
    int max_iter,
    int min_iter,
    double alpha_init,
    double eps
);

void update_sigma(
    double* Sigma,
    double** Omega,
    double* mu,
    int p
);

#endif