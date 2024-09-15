cimport numpy as cnp
from libc.stdlib cimport malloc, free
from admm_cy import update_mu_py, update_sigma_py
from scipy.special import gammaln
import numpy as np
import math
import logging
from inverse_covariance import QuicGraphicalLasso
import multiprocessing as mp
logging.basicConfig(level=logging.INFO)

cdef class SimultaneousPLN:
    """
    Class to perform Simultaneous Poisson Log-Normal (PLN) model fitting
    using ADMM optimization and graphical lasso for precision matrix estimation.
    """
    cdef list Omega  # Precision matrix
    cdef list y  # Observed count data
    cdef list mu  # Mean of the variational distribution
    cdef list sigma  # Variance of the variational distribution
    cdef object Offset  # Optional offset matrix
    cdef object z  # Optional covariate matrix
    cdef list EBIC_sample  # EBIC for each sample
    cdef double EBIC  # Overall EBIC

    def __init__(self, list y, list Omega_init = [], list mu_init=[], list sigma_init=[], Offset=None, z=None):
        """
        Initialize the Simultaneous PLN model parameters.

        Parameters:
        y          : Observed count data (list of np.ndarray for each sample group).
        Omega_init : Initial estimate of the precision matrix (list of np.ndarray for each sample group).
        mu_init    : Initial estimate of the mean for the variational distribution (list of np.ndarray for each sample group).
        sigma_init : Initial estimate of the variance for the variational distribution (list of np.ndarray for each sample group).
        Offset     : Optional offset matrix (default is None).
        z          : Optional covariate matrix (default is None).
        """
        self.y = y
        self.Omega = [np.copy(Omega_init[i]) for i in range(len(Omega_init))]
        self.mu = [np.copy(mu_init[i]) for i in range(len(mu_init))]
        self.sigma = [np.copy(sigma_init[i]) for i in range(len(sigma_init))]
        self.Offset = Offset
        self.z = z
        self.EBIC_sample = []
        self.EBIC = 0.0

    def initialize(self):
        """
        Initialize the model parameters if not provided.
        """
        # Initialize Omega, mu, and sigma if they are None
        cdef int p = self.y[0].shape[1]
        cdef int n = self.y[0].shape[0]
        cdef int I = len(self.y)
        if not self.mu:
            self.mu = [np.log(self.y[i]+0.5) for i in range(I)]
        if not self.sigma:
            self.sigma = [np.repeat([np.repeat(1.1,p)],n,axis=0) for _ in range(I)]
        if not self.Omega:
            self.Omega = [np.linalg.inv(np.cov(self.mu[i].T))  for i in range(I)]
        
    def compute(self, list Omega, list y, list mu_init, list sigma_init, Offset=None, z=None, 
                double pi=0.5, double rho=0.5, double v1=0.5, double v0=0.05, double tau=10, 
                bint pen_diag=True, int min_iter=15, int max_iter=100, double eps=1e-5, bint parallel=True, int process=0, bint info = True):
        """
        Utility function that performs iterative computation of mean, variance, 
        and precision matrix using a variational approach combined with graphical lasso.
        """
        cdef int p = y[0].shape[1]
        cdef int I = len(y)
        cdef int d = z[0].shape[1] if z is not None else 0
        cdef int count = 0
        cdef list n_i = [y[i].shape[0] for i in range(I)]
        
        cdef list mu = mu_init.copy()
        cdef list sigma = sigma_init.copy()
        cdef list l = [None] * I
        cdef list beta = [None] * I
        cdef list err = [None] * I
        cdef list log_diff = [None] * I
        cdef list inv_XTX = [None] * I
        cdef list Xinv_XTX = [None] * I
        
        if Offset is None:
            Offset = [np.zeros((y[i].shape[0], p)) for i in range(I)]
        if z is None:
            z = [None] * I

        # Precompute X.T @ X inverse for each group
        for j in range(I):
            inv_XTX[j] = np.linalg.inv(z[j].T @ z[j]) if z[j] is not None else None

        # Compute beta estimates for each group
        for j in range(I):
            Xinv_XTX[j] = inv_XTX[j] @ z[j].T if z[j] is not None else None
            beta[j] = Xinv_XTX[j] @ (mu[j] - Offset[j]) if z[j] is not None else None

        # Compute linear predictor l for each group
        for j in range(I):
            l[j] = Offset[j] + z[j] @ beta[j] if beta[j] is not None else Offset[j]

        Omega = self.Omega
        ridge_inv = [None] * I

        # Iterative updates for mu, sigma, and Omega
        while count <= max_iter:
            Omega_old = np.copy(Omega)
            count += 1
            # Calculate norm of Omega for penalization
            Omega_norm = np.sum([np.abs(Omega[j]) for j in range(I)], axis=0)
            Gamma = 1 / (1 + (1 - pi) / pi * (v1 / v0) ** I * np.exp(-(1 / v0 - 1 / v1) * Omega_norm))
            np.fill_diagonal(Gamma, 1)  # Ensure diagonal remains 1
            Pen = Gamma / v1 + (1 - Gamma) / v0
            np.fill_diagonal(Pen, 1 / tau)

            if not pen_diag:
                np.fill_diagonal(Pen, 0)

            # Multiprocessing support for updating mu and sigma
            if parallel:
                if process == 0:
                    num_workers = mp.cpu_count()
                else:
                    num_workers = process
                with mp.Pool(num_workers) as pool:
                    if info and count ==1:
                        logging.info(f"{num_workers} processes requested, {pool._processes} are working")
                    for j in range(I):
                        n = y[j].shape[0]
                        ridge_inv[j] = np.linalg.inv(Omega[j] + rho * np.eye(Omega[j].shape[0]))
                        log_diff[j] = l[j] @ Omega[j]

                        # Prepare data for multiprocessing (mu update)
                        zipped_mu = [(mu[j][i, :], y[j][i, :], sigma[j][i, :], ridge_inv[j], log_diff[j][i, :], rho) for i in range(n)]
                        results_mu = pool.starmap(update_mu_py, zipped_mu)

                        # Prepare data for multiprocessing (sigma update)
                        zipped_sigma = [(Omega[j], results_mu[i]) for i in range(n)]
                        results_sigma = pool.starmap(update_sigma_py, zipped_sigma)

                        # Collect the results
                        for i in range(n):
                            mu[j][i, :] = results_mu[i]
                            sigma[j][i, :] = results_sigma[i]
            else:
                # Sequential update when multiprocessing is not used
                for j in range(I):
                    n = y[j].shape[0]
                    ridge_inv[j] = np.linalg.inv(Omega[j] + rho * np.eye(Omega[j].shape[0]))
                    log_diff[j] = l[j] @ Omega[j]

                    for i in range(n):
                        mu[j][i, :] = update_mu_py(mu[j][i, :], y[j][i, :], sigma[j][i, :], ridge_inv[j], log_diff[j][i, :], rho)
                        sigma[j][i, :] = update_sigma_py(Omega[j], mu[j][i, :])

            # Update beta and linear predictor l
            for j in range(I):
                beta[j] = Xinv_XTX[j] @ (mu[j] - Offset[j]) if z[j] is not None else None
                l[j] = Offset[j] + z[j] @ beta[j] if beta[j] is not None else Offset[j]

                # Update covariance matrix s and penalization term Penn_indi
                s = (mu[j] - l[j]).T @ (mu[j] - l[j]) / n + np.diag(np.mean(sigma[j], axis=0))
                Penn_indi = Pen / n

                # Update Omega using QuicGraphicalLasso
                model = QuicGraphicalLasso(lam=Penn_indi, init_method=self.iden)
                Omega[j] = model.fit(s).precision_
                err[j] = np.mean((Omega[j] - Omega_old[j]) ** 2)
            
            if info:
                print(f'Iteration {count}, Error: {np.max(err):.2e}')
            # Check convergence
            if count > min_iter and np.max(err) < eps:
                break

        if count >= max_iter:
            logging.warning('Algorithm did not converge. Consider increasing max_iter or adjusting other parameters.')

        # Compute EBIC
        EBIC_sample = []
        for i in range(I):
            vilm = y[i] * mu[i] - np.exp(mu[i] + sigma[i] / 2) + np.log(sigma[i]) / 2
            vil = np.sum(vilm)
            logdet_temp = np.linalg.slogdet(Omega[i])
            vil += n_i[i] * logdet_temp[0] * logdet_temp[1] / 2 - n_i[i] * (((mu[i] - l[i]).T @ (mu[i] - l[i]) / n_i[i] + np.diag(np.mean(sigma[i], axis=0))) @ Omega[i]).trace()
            edge_size = (np.sum(Omega[i] != 0) - p) / 2
            EBIC_sample.append(-2 * vil + np.log(n_i[i]) * (edge_size + d * p) + 0.5 * (gammaln((p + 1) * p / 2 + 1) - gammaln((p + 1) * p / 2 - edge_size + 1) - gammaln(edge_size + 1)))


        return Omega, mu, sigma, EBIC_sample

    def iden(self, x):
        """
        Identity function used for initializing the QuicGraphicalLasso model.
        """
        return x, 1.0

    def fit(self, pi=0.5, rho=0.5, v1=0.5, v0=0.05, tau=10, pen_diag=True, min_iter=15, max_iter=100, eps=1e-5, parallel=True, process=0, info=True):
        """
        Fit the Simultaneous PLN model to the data using variational inference and graphical lasso.

        Parameters:
        pi        : Prior probability of a non-zero entry in the precision matrix.
        rho       : Regularization parameter for the ADMM.
        v1, v0    : Prior variances for non-zero and zero entries in the precision matrix, respectively.
        tau       : Penalty term for diagonal entries in the precision matrix.
        pen_diag  : Whether to penalize diagonal entries of the precision matrix (default True).
        min_iter  : Minimum number of iterations before checking for convergence.
        max_iter  : Maximum number of iterations allowed for convergence.
        eps       : Tolerance for convergence based on the precision matrix difference.
        parallel  : Whether to use explicit parallel computing (default true).
        process   : Number of processes to use for parallel computation. If 0, all available cpu cores are used (default 0).
        info      : Whether to display logging information (default True).
        """


        self.Omega, self.mu, self.sigma, self.EBIC_sample = self.compute(self.Omega, self.y, self.mu, self.sigma, 
                                                                        self.Offset, self.z, pi, rho, v1, v0, tau, pen_diag, 
                                                                        min_iter, max_iter, eps, parallel, process, info)
        self.EBIC = np.sum(self.EBIC_sample)
