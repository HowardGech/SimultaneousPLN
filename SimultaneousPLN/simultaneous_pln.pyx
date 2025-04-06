# cython: language_level=3, boundscheck=False, wraparound=False
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from scipy.special import gammaln
import numpy as np
import logging
from SimultaneousPLN.admm_cy import update_py_par, update_py
from SimultaneousPLN.pyquic import quic, quic_par
from collections import defaultdict
logging.basicConfig(level=logging.INFO)
class SPLN:
    """
    Class to perform Simultaneous Poisson Log-Normal (PLN) model fitting
    using ADMM optimization and graphical lasso for precision matrix estimation.
    """
    def __init__(self, list y, Offset=None, z=None, list Omega_init = [], list mu_init=[], list sigma_init=[]):
        """
        Initialize the Simultaneous PLN model parameters.

        Parameters:
        y          : Observed count data (list of np.ndarray for each sample group).
        Offset     : Optional offset matrix (default is None).
        z          : Optional covariate matrix (default is None).
        Omega_init : Initial precision matrices (list of np.ndarray, default is empty list).
        mu_init    : Initial mean estimates (list of np.ndarray, default is empty list).
        sigma_init : Initial variance estimates (list of np.ndarray, default is empty list).
        Note: If mu_init, sigma_init, or Omega_init are not provided, default values are computed.
        """
        self.y = y
        self.Offset = Offset
        self.z = z
        self.criterion = dict()
        self.Omega = []
        self.mu = []
        self.sigma = []
        self.beta = []
        self.param = dict()
        cdef int p = self.y[0].shape[1]
        cdef int I = len(self.y)

        # check if all ys have the same number of columns
        for i in range(1, I):
            if self.y[i].shape[1] != p:
                raise ValueError(f"All y matrices must have the same number of columns. Expected {p}, but got {self.y[i].shape[1]} in group {i+1}.")
        # check if all given parameters have the same length
        if Omega_init and len(Omega_init) != I:
            raise ValueError(f"Omega_init must be a list of length {I}.")
        if mu_init and len(mu_init) != I:
            raise ValueError(f"mu_init must be a list of length {I}.")
        if sigma_init and len(sigma_init) != I:
            raise ValueError(f"sigma_init must be a list of length {I}.")
        if Offset is not None and len(Offset) != I:
            raise ValueError(f"Offset must be a list of length {I}.")
        if z is not None and len(z) != I:
            raise ValueError(f"z must be a list of length {I}.")

        if not mu_init:
            self.mu = [np.log(self.y[i]+0.5) for i in range(I)]
        else:
            self.mu = [np.copy(mu_init[i]) for i in range(I)]
        if not sigma_init:
            self.sigma = [np.repeat([np.repeat(1.1,p)],self.y[i].shape[0],axis=0) for i in range(I)]
        else:
            self.sigma = [np.copy(sigma_init[i]) for i in range(I)]
        if not Omega_init:
            self.Omega = [np.linalg.inv(np.cov(self.mu[i].T))  for i in range(I)]
        else:
            self.Omega = [np.copy(Omega_init[i]) for i in range(I)]

        # make sure arrays are contiguous
        for i in range(I):
            self.mu[i] = np.ascontiguousarray(self.mu[i])
            self.sigma[i] = np.ascontiguousarray(self.sigma[i])
            self.Omega[i] = np.ascontiguousarray(self.Omega[i])
            self.y[i] = np.ascontiguousarray(self.y[i])
            if self.Offset is not None:
                self.Offset[i] = np.ascontiguousarray(self.Offset[i])
            if self.z is not None:
                self.z[i] = np.ascontiguousarray(self.z[i])
        
    def compute(self, list Omega, list y, list mu, list sigma, Offset=None, z=None, criterion = None,
                double pi=0.5, double rho=0.5, double v1=0.5, double v0=0.05, double tau=10, double ebic_gamma=0.5,
                bint pen_diag=False, bint keep_Omega=True, int min_iter=10, int max_iter=100, int min_inner = 10, int max_inner = 100,int max_quic = 100,
                double alpha_init=0, double tol=1e-5, double tol_quic = 1e-6, double tol_inner = 1e-6, double mu_lowbound = -30, double mu_highbound = 30, double sigma_lowbound = 1e-4, double sigma_highbound = 25,
                 bint parallel=False, int threads=0, bint verbose = True):
        """
        Utility function that performs iterative computation of mean, variance, 
        and precision matrix using a variational approach combined with graphical lasso.
        """
        cdef int p = y[0].shape[1]
        cdef int I = len(y)
        cdef int d = z[0].shape[1] if z is not None else 0
        cdef int count = 0
        cdef list n_i = [y[i].shape[0] for i in range(I)]
        cdef list l = [None] * I
        cdef list beta = [None] * I
        cdef list err = [None] * I
        cdef list log_diff = [None] * I
        cdef list inv_XTX = [None] * I
        cdef list Xinv_XTX = [None] * I
        cdef list Omega_old = [np.copy(Omega[i]) for i in range(I)]
        cdef list ridge_inv = [np.zeros_like(Omega[i]) for i in range(I)]
        cdef dict crit_value = dict()
        cdef dict param_value = dict()
        cdef list Theta = [np.eye(p) for _ in range(I)]
        cdef list sample_cov = [np.eye(p) for _ in range(I)]
        cdef list Pen_list = [np.zeros((p, p)) for _ in range(I)]
        param_value['pi'], param_value['rho'], param_value['v1'], param_value['v0'], param_value['tau'], param_value['ebic_gamma'] = pi, rho, v1, v0, tau, ebic_gamma
        param_value['pen_diag'], param_value['keep_Omega'] = pen_diag, keep_Omega
        param_value['min_iter'], param_value['max_iter'], param_value['min_inner'], param_value['max_inner'], param_value['max_quic'] = min_iter, max_iter, min_inner, max_inner, max_quic
        param_value['tol'], param_value['tol_quic'], param_value['tol_inner'], param_value['alpha_init'] = tol, tol_quic, tol_inner, alpha_init
        param_value['mu_lowbound'], param_value['mu_highbound'], param_value['sigma_lowbound'], param_value['sigma_highbound'] = mu_lowbound, mu_highbound, sigma_lowbound, sigma_highbound
        param_value['parallel'], param_value['threads'] = parallel, threads

    
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

        # 

        # Iterative updates for mu, sigma, and Omega
        Omega_norm = np.zeros((p, p))
        Gamma = np.zeros((p, p))
        Pen = np.zeros((p, p))
        err_prev = .0
        while count <= max_iter:
            Omega_old = [np.copy(Omega[i]) for i in range(I)]

            # Calculate norm of Omega for penalization
            Omega_norm = np.sum([np.abs(Omega[j]) for j in range(I)], axis=0)
            Gamma = 1 / (1 + (1 - pi) / pi * (v1 / v0) ** I * np.exp(-(1 / v0 - 1 / v1) * Omega_norm))
            np.fill_diagonal(Gamma, 1)  # Ensure diagonal remains 1
            Pen = Gamma / v1 + (1 - Gamma) / v0
            np.fill_diagonal(Pen, 1 / tau)
            
            if not pen_diag:
                np.fill_diagonal(Pen, 0)

            for j in range(I):
                if verbose:
                    print(' ' * 100, end="\r")
                    print(f'Iteration {count}: calculating VEM on group {j+1}/{I}; error {err_prev:.2e}', end="\r")
                ridge_inv[j] = np.linalg.inv(Omega[j] + rho * np.eye(Omega[j].shape[0]))
                log_diff[j] = l[j] @ Omega[j]
                if not parallel:
                    update_py(mu[j], y[j], sigma[j], Omega[j], ridge_inv[j], log_diff[j], rho, max_inner, min_inner, alpha_init, tol_inner, mu_lowbound, mu_highbound, sigma_lowbound, sigma_highbound)
                else:
                    update_py_par(mu[j], y[j], sigma[j], Omega[j], ridge_inv[j], log_diff[j], rho, max_inner, min_inner, alpha_init, tol_inner, mu_lowbound, mu_highbound, sigma_lowbound, sigma_highbound, threads)

            # Update beta and linear predictor l
            for j in range(I):
                beta[j] = Xinv_XTX[j] @ (mu[j] - Offset[j]) if z[j] is not None else None
                l[j] = Offset[j] + z[j] @ beta[j] if beta[j] is not None else Offset[j]

                # Update covariance matrix s and penalization term Penn_indi
                sample_cov[j] = (mu[j] - l[j]).T @ (mu[j] - l[j]) / n_i[j] + np.diag(np.mean(sigma[j], axis=0))
                Pen_list[j] = Pen/n_i[j]
            print(' ' * 100, end="\r")
            print(f'Iteration {count}: solving graphical Lasso; error {err_prev:.2e}', end="\r")
            if count == 0 or not keep_Omega:
                for j in range(I):
                    Omega[j][:,:] = np.eye(p)
                    Theta[j][:,:] = np.eye(p)
            if not parallel: 
                quic(sample_cov, Pen_list, Omega, Theta, tol_quic, max_quic)
            else:
                quic_par(sample_cov, Pen_list, Omega, Theta, tol_quic, max_quic, threads)
            for j in range(I):
                err[j] = np.mean((Omega[j] - Omega_old[j]) ** 2)
            err_prev = np.max(err)
            # Check convergence
            if count > min_iter and np.max(err) < tol:
                break
            count += 1

        if count > max_iter:
            logging.warning('Algorithm did not converge. Consider increasing max_iter or adjusting other parameters.')

        if criterion is not None:
            crit_set = set(criterion) if isinstance(criterion, list) else {criterion}
            if not crit_set.issubset({'loglik', 'AIC', 'BIC', 'EBIC'}):
                raise ValueError("Only 'loglik', 'AIC', 'BIC' and 'EBIC' are supported as criteria.")
            BIC_value = []
            EBIC_value = []
            AIC_value = []
            loglik_value = []
            for i in range(I):
                vilm = y[i] * mu[i] - np.exp(mu[i] + sigma[i] / 2) + np.log(sigma[i]) / 2
                vil = np.sum(vilm)
                logdet_temp = np.linalg.slogdet(Omega[i])
                vil += n_i[i] * logdet_temp[0] * logdet_temp[1] / 2 - n_i[i] * (((mu[i] - l[i]).T @ (mu[i] - l[i]) / n_i[i] + np.diag(np.mean(sigma[i], axis=0))) @ Omega[i]).trace()
                edge_size = (np.sum(Omega[i] != 0) - p) / 2
                if 'EBIC' in crit_set:
                    EBIC_value.append(-2 * vil + np.log(n_i[i]) * (edge_size + d * p) + ebic_gamma * (gammaln((p + 1) * p / 2 + 1) - gammaln((p + 1) * p / 2 - edge_size + 1) - gammaln(edge_size + 1)))
                if 'BIC' in crit_set:
                    BIC_value.append(-2 * vil + np.log(n_i[i]) * (edge_size + d * p))
                if 'AIC' in crit_set:
                    AIC_value.append(-2 * vil + 2 * (edge_size + d * p))
                if 'loglik' in crit_set:
                    loglik_value.append(-2 * vil)
            if 'BIC' in crit_set:
                crit_value['BIC'] = BIC_value
            if 'EBIC' in crit_set:
                crit_value['EBIC'] = EBIC_value
            if 'AIC' in crit_set:
                crit_value['AIC'] = AIC_value
            if 'loglik' in crit_set:
                crit_value['loglik'] = loglik_value
        return Omega, mu, sigma, crit_value, beta, param_value




    def fit(self, **kwargs):
        """
        Fit the Simultaneous PLN model to the data using variational inference and graphical lasso.

        Parameters:
        kwargs : Additional parameters for the compute method.
        """


        _, _, _, self.criterion, self.beta, self.param = self.compute(self.Omega, self.y, self.mu, self.sigma, 
                                                                        self.Offset, self.z, **kwargs)
    
    def ModelSelect(self, pi_list = [0.5], rho_list = [0.5], ratio_list = [10], v0_list = [0.05], tau_list = [10], criterion = 'EBIC', param_verbose = True, param_choice = 'split', **kwargs):
        """
        Fit the Simultaneous PLN model to the data over a parameter grid and select the best model based on the provided criterion.

        Parameters:
        pi_list          : List of values for the pi parameter (default is [0.5]).
        rho_list         : List of values for the rho parameter (default is [0.5]).
        ratio_list       : List of values for the v1/v0 parameter (default is [10]).
        v0_list          : List of values for the v0 parameter (default is [0.05]).
        tau_list         : List of values for the tau parameter (default is [10]).
        criterion        : Criterion for model selection ('BIC' or 'EBIC', default is 'EBIC').
        param_verbose    : Whether to print parameter choices (default is True).
        param_choice     : Choice for best parameter ('split' or 'union', default is 'split').
        kwargs           : Additional parameters for the compute method.
        """


        if param_choice not in ['split', 'union']:
            raise ValueError("param_choice must be either 'split' or 'union'.")
        if criterion not in ['loglik', 'AIC', 'BIC', 'EBIC']:
            raise ValueError("criterion must be one of 'loglik', 'AIC', 'BIC' or 'EBIC'.")

        ## Initialize parameters
        I = len(self.y)
        self.__param_grid = [(pi, rho, ratio, v0, tau) for pi in pi_list for rho in rho_list for ratio in ratio_list for v0 in v0_list for tau in tau_list]
        total_params = len(self.__param_grid)
        self.__crit_value, self.__mus, self.__sigmas, self.__omegas = [None] * total_params, [None] * total_params, [None] * total_params, [None] * total_params
        self.__betas, self.__params = [None] * total_params, [None] * total_params
        self.beta = [None] * I
        

        ## Fit model
        for index, (pi, rho, ratio, v0, tau) in enumerate(self.__param_grid):
            v1 = ratio * v0
            if param_verbose:
                print(f'Param set [{index+1}/{total_params}]. Current setting: pi={pi}, rho={rho}, v1={v1:.2e}, v0={v0:.2e}, tau={tau}')
            self.__mus[index] = [np.copy(self.mu[i]) for i in range(I)]
            self.__sigmas[index] = [np.copy(self.sigma[i]) for i in range(I)]
            self.__omegas[index] = [np.copy(self.Omega[i]) for i in range(I)]
            _, _, _, crit_dict, self.__betas[index], self.__params[index] = self.compute(self.__omegas[index], self.y, self.__mus[index], self.__sigmas[index], 
                                                        self.Offset, self.z, criterion = criterion, pi=pi, rho=rho, v1=v1, v0=v0, tau=tau,
                                                        **kwargs)
            self.__crit_value[index] = crit_dict[criterion]
            
        ## Select best parameters
        self.criterion[criterion] = [None] * I
        if param_choice == 'split':
            self.param = defaultdict(list)
            for i in range(I):
                index = np.argmin([self.__crit_value[j][i] for j in range(total_params)])
                self.criterion[criterion][i] = self.__crit_value[index][i]
                self.mu[i] = self.__mus[index][i]
                self.sigma[i] = self.__sigmas[index][i]
                self.Omega[i] = self.__omegas[index][i]
                self.beta[i] = self.__betas[index][i]
                for key in self.__params[index]:
                    self.param[key].append(self.__params[index][key])
        elif param_choice == 'union':
            index = np.argmin([np.mean([self.__crit_value[j][i] for i in range(I)]) for j in range(total_params)])
            self.criterion[criterion] = self.__crit_value[index]
            self.mu = self.__mus[index]
            self.sigma = self.__sigmas[index]
            self.Omega = self.__omegas[index]
            self.beta = self.__betas[index]
            self.param = self.__params[index]

