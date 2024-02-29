import warnings
from statsmodels.tools.sm_exceptions import EstimationWarning
import numpy as np
from scipy.optimize import minimize
from autograd import elementwise_grad as egrad
from autograd import jacobian
import autograd.numpy as np
from comm_functions import sigma_estimate, draw_from_Rademacher, convert_wlsparam, svar_ma_rep
from expectation import expectation_run
from scipy.stats import norm


def bootstrap_optimization_run(smoothed_prob, residuals,lam_m,
                     x0,  zt, delta_yt):

        b_matrix = numerical_opt_b_lambda(x0, residuals, smoothed_prob,lam_m)
        sigmas = sigma_estimate(b_matrix, lam_m)
        wls_params = wls_estimate(sigmas, zt, delta_yt, smoothed_prob)
        
        return b_matrix, wls_params


def sigma_likelihood(x, residuals, smoothed_prob, lam_m_):
    """
    :param x: must be a column vector of guesses taht is b.T.flatten()
    :param residuals:
    :param smoothed_prob:
    :param lam_m: 3 dimesion lam_m this will be converted to a list withing the function
    :return:
    """

    k_vars, obs = residuals.shape
    regimes = smoothed_prob.shape[0]
    lam_m = []
    for i in range(regimes-1 ):
        lam_m.append(lam_m_[:,:,i])
    b_matrix = x.reshape(k_vars,k_vars).T
    weighted_sum_res = np.zeros([k_vars, k_vars, regimes])
    for regime in range(regimes):
        temp_wt_sum = 0
        for t in range(obs):
            temp_wt_sum = temp_wt_sum + \
                          smoothed_prob[regime, t] * \
                          residuals[:, [t]] @ residuals[:, [t]].T
        weighted_sum_res[:, :, regime] = temp_wt_sum

    # likelihood terms
    b_matrix_trans_inv = np.linalg.pinv(b_matrix.T)
    on = True
    while on:
        try:
            b_matrix_inv = np.linalg.pinv(b_matrix)
        except:
            print('cannot invert b metrix')
        on = False

    term_1 = obs * np.log(abs(np.linalg.det(b_matrix))) #/ 2  # TODO change back to original sum(smoothed_prob[0, :])
    term_2 = np.trace(b_matrix_trans_inv @ b_matrix_inv @ weighted_sum_res[:, :, 0]) / 2
    term_3 = 0
    term_4 = 0
    for regime in range(regimes-1):
        lam_inv = np.linalg.pinv(lam_m[regime])
        term_3 += np.sum(smoothed_prob[regime+1, :]) * np.log(np.linalg.det(lam_m[regime ])) / 2
        term_4 += np.trace(b_matrix_trans_inv @ lam_inv @ b_matrix_inv @ weighted_sum_res[:, :, regime+1]) / 2
    negative_likelihood = term_1 + term_2 + term_3 + term_4

    return negative_likelihood  




def fprime(x, residuals, smoothed_prob, lam_m):
    func = lambda x : sigma_likelihood(x, residuals, smoothed_prob, lam_m)
    g= egrad(func)
    return g(x)


def hessian(x,  residuals, smoothed_prob, lam_m):
  func = lambda x : sigma_likelihood(x, residuals, smoothed_prob, lam_m)
  H_f = jacobian(egrad(func))
  return H_f(x)

def wls_estimate(sigma, zt, delta_yt, smoothed_prob):
    regimes, obs = smoothed_prob.shape

    sigma_inv = np.zeros(sigma.shape)

    for regime in range(regimes):
        sigma_inv[:, :, regime] = np.linalg.pinv(sigma[:, :, regime])

    tsum_denom = 0
    m_sum_denom = 0
    tsum_numo = 0
    m_sum_numo = 0
    for t in range(obs):
        for regime in range(regimes):
            m_sum_denom += np.kron(smoothed_prob[regime, t] * zt[:, [t]] @ zt[:, [t]].T, sigma_inv[:, :, regime])
            m_sum_numo += np.kron(smoothed_prob[regime, t] * zt[:, [t]], sigma_inv[:, :, regime]) @ delta_yt[:, [t]]
        tsum_denom += m_sum_denom
        tsum_numo += m_sum_numo

    wls_params =  np.linalg.pinv(tsum_denom) @ tsum_numo #sp.linalg.solve(tsum_denom,tsum_numo)
    return  wls_params 



def numerical_opt_b_lambda(x0, residuals, smoothed_prob, lam_m):
    k_vars = residuals.shape[0]
    func = lambda x : sigma_likelihood(x, residuals, smoothed_prob, lam_m)
    grad1 = lambda x : fprime(x, residuals, smoothed_prob, lam_m) # gradient based on Autograd 
    hess1 = lambda x :  hessian(x,  residuals, smoothed_prob, lam_m)
    initial_x0 = x0
    method_ = 'trust-krylov' # starting optimization with gradient based optimizer

    for  j  in range(5): 
        #print(f'\n this is the current optimization method: \n {method_}')
        try:
            result = minimize(func, x0, 
                                method = method_, 
                                tol= 1e-6, 
                                jac=grad1,hess=hess1,
                                options={'maxiter': 30000})
            
            x0 = result.x
            if j< 3:     
                num_grad = result.jac
            else: 
                try:
                    num_grad = grad1(x0)
                except: 
                    print('autograd could not find the gradient')
                    num_grad = np.array([1,1]) 
            message = result.success   
        except:
            print('optimization resulted in an error')
            message = False 

        if message == True:
            break 
        
        length_x0 = k_vars*k_vars
        if j==0:
            x0 = np.zeros(length_x0).reshape(-1,1)
            x0[:(k_vars*k_vars),[0]] =  np.identity(k_vars).reshape(-1,1) #
            x0=x0.ravel()
        elif j==1:
            x0 = np.ones(length_x0).reshape(-1,1)*(3)
            x0[:(k_vars*k_vars),[0]] =  np.identity(k_vars).reshape(-1,1) #
            x0=x0.ravel()         
        elif j==2:
            x0 = np.ones(length_x0).reshape(-1,1)*(-3)
            x0[:(k_vars*k_vars),[0]] =  np.identity(k_vars).reshape(-1,1) #
            x0=x0.ravel()   
        else:
            x0 = initial_x0 
            method_ = 'COBYLA' # if gradient based method doesn't work try gradient free method

    b  = np.array(result.x).reshape(k_vars,k_vars).T

    return b




def bootstrap_ma(beta,wls_params,zt,x0, residuals,lam_m,smoothed_prob,  lags,k_vars, steps):
  k_vars, obs = residuals.shape
  sample_res = np.zeros(residuals.shape)
  sample_delta_yt = np.zeros(residuals.shape)

  # changes to the expectation run 
  #expectation_run(sigma,residuals,start_prob,transition_prob)

  for t in range(obs):
    sample_res[:,[t]] = residuals[:,[t]]*draw_from_Rademacher(1).reshape(-1,1)
    sample_delta_yt[:,[t]] = np.kron(zt[:, [t]].T, np.eye(k_vars)) @ wls_params + sample_res[:,[t]]

  sample_b_matrix, sample_wls_params = bootstrap_optimization_run(smoothed_prob, sample_res, lam_m,x0,  zt, sample_delta_yt)
  alpha, gamma = convert_wlsparam(sample_wls_params, k_vars)
  neqs = k_vars
  k_ar = lags +1
  return svar_ma_rep(sample_b_matrix, beta, alpha, gamma, neqs, k_ar, maxn= steps)

def get_quantiles(observations):
  """
  This function estimates the normal distribution and returns the 16th and 84th quantiles values.

  Args:
    observations: A list of observations.

  Returns:
    A list of the 16th and 84th quantiles values.
  """

  # Estimate the normal distribution.
  loc, scale = norm.fit(observations)

  # Return the 16th and 84th quantiles values.
  return [norm.ppf(0.16, loc, scale), norm.ppf(0.84, loc, scale)]




def bootstrap_run( result_dict,lags,k_vars, steps, draws=1000):
    smoothed_prob = np.array(result_dict['smoothed_prob'])
    lam_m = np.array(result_dict['lam_m'])
    residuals = np.array(result_dict['residuals'])
    x0 = np.array(result_dict['b_matrix']).T.flatten()
    zt = np.array(result_dict['zt'])
    wls_params = np.array(result_dict['wls_params'])
    beta = np.array(result_dict['beta']).reshape(-1,1)
    upper_bound = np.zeros([steps,k_vars,k_vars]) 
    lower_bound = np.zeros([steps,k_vars,k_vars]) 
    result = []

    for _ in range(draws):
        result.append(bootstrap_ma(beta,wls_params,zt,x0, residuals,lam_m,smoothed_prob,  lags,k_vars, steps)) 
    
    for k in range(steps):
        for i in range(k_vars):
            for j in range(k_vars):
                param_dst = np.zeros(draws)
                for d in range(draws):
                    param_dst[d] = result[d][k,i,j]
                bounds = get_quantiles(param_dst)     
                lower_bound[k,i,j] = bounds[0]
                upper_bound[k,i,j]= bounds[1]
    return lower_bound, upper_bound

    



