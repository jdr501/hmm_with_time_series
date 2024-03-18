'''
This functions are used across different files. 
Best practice is to save these functions in a seperate file so there are no circular import issues.
'''

from statsmodels.tsa.vector_ar.var_model import ma_rep
import numpy as np 
import random

def sigma_estimate(b_matrix, lam_m):
    regimes = 1 + lam_m.shape[2]
    k_vars = b_matrix.shape[0]
    sigma = np.zeros([k_vars, k_vars, regimes])
    for regime in range(regimes):
        if regime == 0:
            sigma[:, :, 0] = b_matrix @ b_matrix.T
        else:
            sigma[:, :, regime] = b_matrix @ lam_m[:, :, regime - 1] @ b_matrix.T
    return sigma


def rnd():
    significand =   random.random()
    return significand * 10e-6



def zscore(s, window, thresh=3, return_all=False):
  roll = s.rolling(window=window, min_periods=1, center=True)
  avg = roll.mean()
  std = roll.std(ddof=0)
  z = s.sub(avg).div(std)
  m = z.between(-thresh, thresh)

  if return_all:
      return z, avg, std, m
  return s.where(m, avg)


def check_parameter_converge(old_dict, new_dict, xtol):
  each_converge= []
  for key in old_dict:
    each_converge.append(all([new_dict[key][i] - old_dict[key][i] < xtol for i in range(len(new_dict[key]))]) )
  return all(each_converge)


def draw_from_Rademacher(n):
  """Draws n samples from the Rademacher distribution.

  Args:
    n: The number of samples to draw.

  Returns:
    A numpy array of size n containing the samples.
  """

  return np.random.choice([-1, 1], size=n)




def convert_wlsparam(wls_params, k_vars):
  regressors = int(len(wls_params)/ k_vars)
  param_mat = np.zeros([k_vars,regressors])
  start = 0
  end = k_vars
  for i in range(regressors):
    param_mat[:,[i]]=wls_params[start:end]
    start= end
    end = start+k_vars

  alpha = param_mat[:,[2]]
  gamma = param_mat[:,3:]
  return alpha, gamma 




def var_rep(beta, alpha, gamma, neqs, k_ar):
  pi = alpha.dot(beta.T)
  gamma = gamma
  K = neqs
  A = np.zeros((k_ar, K, K))
  A[0] = pi + np.identity(K)
  if gamma.size > 0:
      A[0] += gamma[:, :K]
      A[k_ar - 1] = -gamma[:, K * (k_ar - 2) :]
      for i in range(1, k_ar - 1):
          A[i] = (
              gamma[:, K * i : K * (i + 1)]
              - gamma[:, K * (i - 1) : K * i]
          )
  return A




def svar_ma_rep(B, beta, alpha, gamma, neqs, k_ar, maxn=10):
  P = B
  coefs = var_rep(beta, alpha, gamma, neqs, k_ar)
  ma_mats = ma_rep(coefs, maxn)
  return np.array([np.dot(coefs, P) for coefs in ma_mats])

def bound_gen(res_mat,x0):
  restriction_array = res_mat.T.flatten().reshape(-1,1)
  bounds = [(-np.Inf, np.inf)] * len(x0)
  for i in range(len(restriction_array)):
    if restriction_array[i] ==0:
      bounds[i] = (0, 0)
    else:
      pass 
  return bounds
