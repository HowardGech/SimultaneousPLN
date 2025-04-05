#include <math.h>
#include <stdlib.h>
#include "admm.h"

// Helper function to compute L2 derivative
static double L2_der(double mu, double y, double sigma, double alpha, double rho, double mu_N) {
    return -y + exp(mu + sigma / 2.0) + alpha + rho * (mu - mu_N);
}

// Helper function to compute the derivative of L2_der
static double L2_der_prime(double mu, double sigma, double rho) {
    return exp(mu + sigma / 2.0) + rho;
}

// Helper function to find optimal mu using bisection method
static double L2_opt(double y, double sigma, double alpha, double rho, double mu_N, int max_iter, double low_bound, double high_bound, double tol, double init) {
    double x = init, diff = 10.0, temp;
    int count = 0;
    
    // if (L2_der(low_bound, y, sigma, alpha, rho, mu_N) >= 0.0) {
    //     return low_bound;
    // }
    // if (L2_der(high_bound, y, sigma, alpha, rho, mu_N) <= 0.0) {
    //     return high_bound;
    // }

    // while (diff > tol && count < max_iter) {
    //     count++;
    //     temp = L2_der((a + b) / 2.0, y, sigma, alpha, rho, mu_N);
    //     if (temp > 0.0) {
    //         b = (a + b) / 2.0;
    //     } else {
    //         a = (a + b) / 2.0;
    //     }
    //     diff = fabs(temp);
    // }
    // return (a + b) / 2.0;

    // Newton's method for finding the root
    while (diff > tol && count < max_iter) {
        count++;
        x -= L2_der(x, y, sigma, alpha, rho, mu_N) / L2_der_prime(x, sigma, rho);
        if (x < low_bound) {
            return low_bound; // enforce lower bound
        } else if (x > high_bound) {
            return high_bound; // enforce upper bound
        }
        diff = fabs(L2_der(x, y, sigma, alpha, rho, mu_N));
    }
    return x; // return the optimal mu

}

// L1 optimization for ADMM
static void L1_opt(double* mu_N, double* alpha, double rho, double* mu_M, double* log_diff, double* ridge_inv, int p) {
    int i, j;
    for (i = 0; i < p; i++) {
        mu_N[i] = 0;
        for (j = 0; j < p; j++) {
            mu_N[i] += (rho * mu_M[j] + alpha[j] + log_diff[j]) * ridge_inv[j*p+i];
        }
    }
}

// Function to update mu in the ADMM algorithm
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
) {
    int i, count = 0;
    double delta = 1e6;
    double* alpha = (double*)malloc(p * sizeof(double));
    double* mu_N = (double*)malloc(p * sizeof(double));

    for (i = 0; i < p; i++) {
        alpha[i] = alpha_init;
        mu_N[i] = mu_M[i];
    }

    while (count <= max_iter) {
        count++;

        // Update mu_M using L2 optimization for each element
        for (i = 0; i < p; i++) {
            mu_M[i] = L2_opt(y[i], sigma[i], alpha[i], rho, mu_N[i], max_iter, mu_lowbound, mu_highbound, eps, mu_M[i]);
        }

        // Update mu_N using L1 optimization
        L1_opt(mu_N, alpha, rho, mu_M, log_diff, ridge_inv, p);

        // Update dual variable alpha
        for (i = 0; i < p; i++) {
            alpha[i] += rho * (mu_M[i] - mu_N[i]);
        }

        // Check convergence
        delta = 0;
        for (i = 0; i < p; i++) {
            double diff = (mu_M[i] - mu_N[i]) * (mu_M[i] - mu_N[i]) / (fabs(mu_M[i]) + 10.0 * eps) / (fabs(mu_N[i]) + 10.0 * eps);
            if (diff > delta) {
                delta = diff;
            }
        }

        if (delta < eps && count >= min_iter) {
            break;
        }
    }
    free(alpha);
    free(mu_N);
}


static double eq_sigma(double sigma2, double omega, double mu) {
    return 1.0 - (omega * sigma2) - exp(mu + sigma2 / 2.0) * sigma2;
}
// Function to solve for sigma
static double solve_sigma(double omega, double mu, int max_iter, double low_bound, double high_bound, double tol) {
    double diff = 10.0, a = low_bound, b = high_bound, temp;
    int count = 0;

    if (eq_sigma(low_bound, omega, mu) < 0.0) {
        return low_bound;
    }
    if (eq_sigma(high_bound, omega, mu) > 0.0) {
        return high_bound;
    }

    while (diff > tol && count < max_iter) {
        count++;
        temp = eq_sigma((a + b) / 2.0, omega, mu);
        if (temp > 0.0) {
            a = (a + b) / 2.0;
        } else {
            b = (a + b) / 2.0;
        }
        diff = fabs(temp);
    }

    return (a + b) / 2.0;
}

// Function to update sigma in the ADMM algorithm
void update_sigma(double* Sigma, double* Omega, double* mu, int p, int max_iter, double sigma_lowbound, double sigma_highbound, double eps) {
    int i;
    for (i = 0; i < p; i++) {
        Sigma[i] = solve_sigma(Omega[i*p+i], mu[i], max_iter, sigma_lowbound, sigma_highbound, eps); // initial guess is current value of Sigma[i]
    }
}
