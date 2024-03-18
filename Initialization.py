
import pandas as pd
from statsmodels.tsa.vector_ar import vecm
import numpy as np 
import scipy as sp
from scipy.linalg import sqrtm as sqrt
from comm_functions import sigma_estimate, rnd
import random



def initialize_step_1(df,lags,regimes, exog= None, beta =None):
    model = vecm.VECM(endog=df,
                      exog = exog,
                    k_ar_diff=lags,
                    coint_rank=1,
                    dates=df.index,
                    deterministic="colo",
                    freq='d')
    vecm_result = model.fit()
    print(vecm_result.summary())
    print(f'this is base model llf:{vecm_result.llf}')
    residuals =vecm_result.resid.T
    if beta == None:
        beta = vecm_result.beta
    data_mat = vecm._endog_matrices(model.y, model.endog, model.exog, lags, "colo")
    delta_yt = data_mat[1]
    k_vars=model.endog.shape[1]
    beta_trn_y_t_1 = beta.T@data_mat[2][:k_vars,:]
    exog_var = data_mat[2][k_vars:,:]
    if model.deterministic == 'colo':
        insert_at=2
    else: 
        print('not implemented')
        insert_at = None 
    mat2 = np.roll(data_mat[3][:-k_vars,:], -(model.k_ar_diff*k_vars), axis=0)
    zt = np.insert(mat2,insert_at,beta_trn_y_t_1, 0)
    if model.exog is not None:  
        zt =np.insert(zt,insert_at,exog_var, 0)  
    b = sqrt(vecm_result.sigma_u)
    start_prob = np.ones([regimes,1])/regimes     
    transition_prob = np.ones([regimes, regimes]) / regimes 

    return b, start_prob, transition_prob, delta_yt, zt, residuals, beta


def initialize_step_2(b,regimes):
    k_vars= b.shape[0]
    b_rand = np.array([rnd() for _ in range(k_vars*k_vars)])
    b = b + b_rand.reshape(k_vars,k_vars)
    length_x0 = k_vars*k_vars+(regimes-1)*k_vars
    x0 = np.zeros(length_x0).reshape(-1,1)
    x0[:(k_vars*k_vars),[0]] =  b.T.reshape(-1,1) 
    x0=x0.ravel()
    lam_m = np.zeros([k_vars,k_vars,regimes-1])

    for regime in range(regimes-1):
        lam_m[:,:, regime] = np.identity(k_vars)*np.array([np.random.rand() for _ in range(k_vars)])*3 

    sigmas = sigma_estimate(b, lam_m)    
    return sigmas , x0