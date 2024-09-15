import numpy as np
import os
from simultaneous_pln import SimultaneousPLN
import time
I = 30
n = 500
p = 40

indi_temp = np.random.rand(p,p)
for k in range(p):
    indi_temp[k,k:]=0
indi_temp = indi_temp + indi_temp.T
sparsity = 0.25
sparsity_group = 0.8
indi = (indi_temp<sparsity)
np.fill_diagonal(indi, True)

Omega = [None]*I
Omega_init = [None]*I
y = [None]*I
Sigma_init = [None]*I
Offset = [None]*I
X = [None]*I

for i in range(I):
    sg = np.random.uniform(0.2,0.5,(p,p))*(1-2*np.random.binomial(1, 0.5,(p,p)))
    for k in range(p):
        sg[k,k:] = 0
    sg = sg + sg.T
    for k in range(p):
        sg[k,k] = np.random.uniform(0.8,1.2)
    indi_group_temp = np.random.rand(p,p)
    for k in range(p):
        indi_group_temp[k,k:]=0
    indi_group_temp = indi_group_temp + indi_group_temp.T
    indi_group = (indi_group_temp<sparsity_group)
    np.fill_diagonal(indi_group, True)
    indi_group = indi_group * indi
    Omega[i] = indi_group * sg
    eps = np.min(np.linalg.eig(Omega[i])[0])
    if eps<0:
        eps = -eps + 0.01
    else:
        eps = 0.01
    Omega[i] = Omega[i] + eps*np.eye(p)
    
    
for i in range(I):
    x = np.random.multivariate_normal(np.random.normal(0,1,p), np.linalg.inv(Omega[i]),n)
    x_exp = np.exp(x)
    y[i] = np.zeros((n,p))
    for j in range(n):
        for k in range(p):
            y[i][j,k] = np.random.poisson(lam=x_exp[j,k],size=1)[0]



for i in range(I):
    Sigma_init[i] = np.repeat([np.repeat(1.1,p)],n,axis=0)
    Offset[i] = np.zeros((n,p))
    X[i] = np.ones((n,1))
        
   

mu_init = [None]*I

for i in range(I):
    mu_init[i] = np.log(y[i]+0.5)
    Omega_init[i] = np.linalg.inv(np.cov(mu_init[i].T))
    

if __name__ == '__main__':

    scale0 = 1
    scale1 = 10


    model = SimultaneousPLN(Omega_init, y, mu_init, Sigma_init, Offset = None, z=X)
    v1 = scale1*scale0/np.sqrt(n*np.log(p))
    v0 = scale0/np.sqrt(n*np.log(p))

    start = time.time()
    model.fit(v1 = v1, v0 = v0, max_iter=30, parallel=True, process=0, eps=1e-4, info=True)
    end = time.time()

    print("The runtime of model fitting is: ", (end-start), "s")