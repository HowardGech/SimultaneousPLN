cimport numpy as cnp
import numpy as np
from libc.stdlib cimport malloc, free
cdef extern from "admm.h":
    void update_mu(double* mu_M, double* mu_N, double* y, double* sigma, double** ridge_inv, double* log_diff, int p, double* alpha, double rho, int max_iter, int min_iter, double alpha_init, double eps)
    void update_sigma(double* Sigma, double** Omega, double* mu, int p)

# Cython wrapper for update_mu
def update_mu_py(cnp.ndarray[double, ndim=1] mu, cnp.ndarray[double, ndim=1] y, cnp.ndarray[double, ndim=1] sigma, cnp.ndarray[double, ndim=2] ridge_inv, cnp.ndarray[double, ndim=1] log_diff, double rho, int max_iter=100, int min_iter=10, double alpha_init=0, double eps=1e-4):
    cdef int p = y.shape[0]
    cdef double* mu_M = <double*>malloc(p * sizeof(double))
    cdef double* mu_N = <double*>malloc(p * sizeof(double))
    cdef double* alpha = <double*>malloc(p * sizeof(double))
    
    # Convert ridge_inv to C array
    cdef double** ridge_inv_c = <double**>malloc(p * sizeof(double*))
    for i in range(p):
        ridge_inv_c[i] = &ridge_inv[i, 0]
    
    # Initialize mu_M and mu_N
    for i in range(p):
        mu_M[i] = mu[i]
        mu_N[i] = mu[i]
    # Call the C function
    update_mu(mu_M, mu_N, &y[0], &sigma[0], ridge_inv_c, &log_diff[0], p, alpha, rho, max_iter, min_iter, alpha_init, eps)

    # Convert result back to NumPy array
    result = np.array([mu_M[i] for i in range(p)])

    # Free allocated memory
    free(mu_M)
    free(mu_N)
    free(alpha)
    free(ridge_inv_c)

    return result

# Cython wrapper for update_sigma
def update_sigma_py(cnp.ndarray[double, ndim=2] Omega, cnp.ndarray[double, ndim=1] mu):
    cdef int p = mu.shape[0]
    cdef double* Sigma = <double*>malloc(p * sizeof(double))

    # Convert Omega to C array
    cdef double** Omega_c = <double**>malloc(p * sizeof(double*))
    for i in range(p):
        Omega_c[i] = &Omega[i, 0]

    # Call the C function
    update_sigma(Sigma, Omega_c, &mu[0], p)

    # Convert result back to NumPy array
    result = np.array([Sigma[i] for i in range(p)])

    # Free allocated memory
    free(Sigma)
    free(Omega_c)

    return result