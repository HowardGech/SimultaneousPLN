import numpy as np
import os
from simultaneous_pln import SimultaneousPLN  # Import the SimultaneousPLN class
import time  # Import time module for measuring runtime

# Set up the dimensions for the problem
I = 30  # Number of groups/samples
n = 500  # Number of data points per sample
p = 40  # Number of features/variables per data point

# Initialize a random matrix (p x p) for sparsity structure
indi_temp = np.random.rand(p, p)

# Set the upper triangular part (including the diagonal) to zero
for k in range(p):
    indi_temp[k, k:] = 0

# Make the matrix symmetric
indi_temp = indi_temp + indi_temp.T

# Define sparsity levels
sparsity = 0.25  # Sparsity level for the individual covariance matrix
sparsity_group = 0.8  # Sparsity level for group structure

# Create an indicator matrix (True where there is structure, False elsewhere)
indi = (indi_temp < sparsity)
np.fill_diagonal(indi, True)  # Ensure the diagonal is filled with True

# Initialize placeholders for Omega, data, Offset, and covariates (X)
Omega = [None] * I  # Precision matrices for each group
y = [None] * I  # Data points for each group
Offset = [None] * I  # Offsets for each group
X = [None] * I  # Covariates for each group

# Loop to generate precision matrices (Omega) for each group
for i in range(I):
    # Generate a random symmetric group structure for covariance matrix
    sg = np.random.uniform(0.2, 0.5, (p, p)) * (1 - 2 * np.random.binomial(1, 0.5, (p, p)))
    for k in range(p):
        sg[k, k:] = 0  # Ensure upper triangle is zero
    sg = sg + sg.T  # Make the matrix symmetric
    for k in range(p):
        sg[k, k] = np.random.uniform(0.8, 1.2)  # Ensure diagonal entries are positive
    
    # Generate a group structure with sparsity
    indi_group_temp = np.random.rand(p, p)
    for k in range(p):
        indi_group_temp[k, k:] = 0  # Set upper triangular part to zero
    indi_group_temp = indi_group_temp + indi_group_temp.T
    indi_group = (indi_group_temp < sparsity_group)
    np.fill_diagonal(indi_group, True)  # Fill diagonal with True
    indi_group = indi_group * indi  # Apply overall sparsity structure
    
    # Create the precision matrix for group `i`
    Omega[i] = indi_group * sg
    eps = np.min(np.linalg.eig(Omega[i])[0])  # Check the smallest eigenvalue to ensure positive definiteness
    if eps < 0:
        eps = -eps + 0.01  # Adjust to make the matrix positive definite
    else:
        eps = 0.01
    Omega[i] = Omega[i] + eps * np.eye(p)  # Add epsilon to the diagonal to ensure positive definiteness

# Loop to generate synthetic data for each group
for i in range(I):
    # Generate multivariate normal samples for each group `i`
    x = np.random.multivariate_normal(np.random.normal(0, 1, p), np.linalg.inv(Omega[i]), n)
    x_exp = np.exp(x)  # Take the exponential of the samples
    y[i] = np.zeros((n, p))  # Initialize y for group `i`
    
    # Generate Poisson-distributed samples based on the exponentiated normal samples
    for j in range(n):
        for k in range(p):
            y[i][j, k] = np.random.poisson(lam=x_exp[j, k], size=1)[0]  # Poisson-distributed data

# Initialize Offset and X (covariates) for each group
for i in range(I):
    Offset[i] = np.zeros((n, p))  # No offset initially
    X[i] = np.ones((n, 1))  # Covariate matrix with all ones (e.g., intercept)

# Main block for model fitting
if __name__ == '__main__':
    scale0 = 1  # Scaling factor for the small variance
    scale1 = 10  # Scaling factor for the large variance

    # Initialize the Simultaneous PLN model
    model = SimultaneousPLN(y, Offset=None, z=X)

    # Initialize the model parameters
    model.initialize()

    # Set prior variance values based on the problem size
    v1 = scale1 * scale0 / np.sqrt(n * np.log(p))  # Prior variance for the non-sparse entries
    v0 = scale0 / np.sqrt(n * np.log(p))  # Prior variance for the sparse entries

    # Measure the runtime of model fitting
    start = time.time()
    model.fit(v1=v1, v0=v0, max_iter=30, parallel=True, process=0, eps=1e-4, info=True)  # Fit the model
    end = time.time()

    # Print the runtime of the model fitting process
    print("The runtime of model fitting is: ", (end - start), "s")
