#include <math.h>
#include <stdlib.h>
#include "admm.h"

// Helper function to compute L2 derivative
double L2_der(double mu, double y, double sigma, double alpha, double rho, double mu_N) {
    return -y + exp(mu + sigma / 2.0) + alpha + rho * (mu - mu_N);
}

// Helper function to find optimal mu using bisection method
double L2_opt(double y, double sigma, double alpha, double rho, double mu_N, int max_iter) {
    double a = -30.0, b = 30.0, diff = 10.0, temp;
    int count = 0;

    if (L2_der(-30.0, y, sigma, alpha, rho, mu_N) >= 0.0) {
        return -30.0;
    }
    if (L2_der(30.0, y, sigma, alpha, rho, mu_N) <= 0.0) {
        return 30.0;
    }

    while (diff > 1e-6 && count < max_iter) {
        count++;
        temp = L2_der((a + b) / 2.0, y, sigma, alpha, rho, mu_N);
        if (temp > 0.0) {
            b = (a + b) / 2.0;
        } else {
            a = (a + b) / 2.0;
        }
        diff = fabs(temp);
    }

    return (a + b) / 2.0;
}

// L1 optimization for ADMM
void L1_opt(double* mu_N, double* alpha, double rho, double* mu_M, double* log_diff, double** ridge_inv, int p) {
    int i, j;
    for (i = 0; i < p; i++) {
        mu_N[i] = 0;
        for (j = 0; j < p; j++) {
            mu_N[i] += (rho * mu_M[j] + alpha[j] + log_diff[j]) * ridge_inv[j][i];
        }
    }
}

// Function to update mu in the ADMM algorithm
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
) {
    int i, count = 0;
    double delta = 1e6;

    for (i = 0; i < p; i++) {
        alpha[i] = alpha_init;
        mu_N[i] = mu_M[i];
    }

    while (count <= max_iter) {
        count++;

        // Update mu_M using L2 optimization for each element
        for (i = 0; i < p; i++) {
            mu_M[i] = L2_opt(y[i], sigma[i], alpha[i], rho, mu_N[i], max_iter);
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
            double diff = (mu_M[i] - mu_N[i]) * (mu_M[i] - mu_N[i]) / (fabs(mu_M[i]) + 1e-4) / (fabs(mu_N[i]) + 1e-4);
            if (diff > delta) {
                delta = diff;
            }
        }

        if (delta < eps && count >= min_iter) {
            break;
        }
    }
}

// Equation used to update sigma
double eq_sigma(double sigma2, double omega, double mu) {
    return 1 - sigma2 * omega - sigma2 * exp(mu + sigma2 / 2.0);
}

// Function to solve for sigma using a bisection method
double solve_sigma(double omega, double mu, int max_iter) {
    double a = 1e-4, b = 1e2, diff = 10.0, temp;
    int count = 0;

    if (eq_sigma(1e-4, omega, mu) < 0.0) {
        return 1e-4;
    }
    if (eq_sigma(2e2, omega, mu) > 0.0) {
        return 2e2;
    }

    while (diff > 1e-6 && count < max_iter) {
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
void update_sigma(double* Sigma, double** Omega, double* mu, int p) {
    int i;
    for (i = 0; i < p; i++) {
        Sigma[i] = solve_sigma(Omega[i][i], mu[i], 100);
    }
}