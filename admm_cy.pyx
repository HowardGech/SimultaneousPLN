cimport numpy as cnp
cimport cython
import numpy as np
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange

cdef extern from "admm.h":
    void update_mu(double* mu_M, double* y, double* sigma, double* ridge_inv, double* log_diff, int p, double rho, int max_iter, int min_iter, double alpha_init, double mu_lowbound, double mu_highbound, double eps ) nogil
    void update_sigma(double* Sigma, double* Omega, double* mu, int p, int max_iter, double sigma_lowbound, double sigma_highbound, double eps) nogil

def update_py(double[:, :] mu, double[:, :] y, double[:, :] sigma, double[:, :] Omega, double[:, :] ridge_inv, double[:, :] log_diff, double rho, int max_iter, int min_iter, double alpha_init, double eps, double mu_lowbound, double mu_highbound, double sigma_lowbound, double sigma_highbound):
    cdef int p = y.shape[1]
    cdef int n = y.shape[0]
    cdef Py_ssize_t i
    
    for i in range(n):
        update_mu(&mu[i,0], &y[i,0], &sigma[i,0], &ridge_inv[0,0], &log_diff[i,0], p, rho, max_iter, min_iter, alpha_init, mu_lowbound, mu_highbound, eps)
        
    for i in range(n):
        update_sigma(&sigma[i,0], &Omega[0,0], &mu[i,0], p, max_iter, sigma_lowbound, sigma_highbound, eps)

    return



@cython.boundscheck(False)
@cython.wraparound(False)
def update_py_par(double[:, :] mu, double[:, :] y, double[:, :] sigma, double[:, :] Omega, double[:, :] ridge_inv, double[:, :] log_diff, double rho, int max_iter, int min_iter, double alpha_init, double eps, double mu_lowbound, double mu_highbound, double sigma_lowbound, double sigma_highbound, int threads=0):
    cdef int p = y.shape[1]
    cdef int n = y.shape[0]
    cdef Py_ssize_t i

    if threads == 0:           
        for i in prange(n, nogil=True):
            update_mu(&mu[i,0], &y[i,0], &sigma[i,0], &ridge_inv[0,0], &log_diff[i,0], p, rho, max_iter, min_iter, alpha_init, mu_lowbound, mu_highbound, eps)
        for i in prange(n, nogil=True):
            update_sigma(&sigma[i,0], &Omega[0,0], &mu[i,0], p, max_iter, sigma_lowbound, sigma_highbound, eps)
    else:
        for i in prange(n, nogil=True, num_threads=threads):
            update_mu(&mu[i,0], &y[i,0], &sigma[i,0], &ridge_inv[0,0], &log_diff[i,0], p, rho, max_iter, min_iter, alpha_init, mu_lowbound, mu_highbound, eps)
        for i in prange(n, nogil=True, num_threads=threads):
            update_sigma(&sigma[i,0], &Omega[0,0], &mu[i,0], p, max_iter, sigma_lowbound, sigma_highbound, eps)


    return