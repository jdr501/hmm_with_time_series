import warnings
from statsmodels.tools.sm_exceptions import EstimationWarning
import numpy as np
from scipy.optimize import minimize
from autograd import elementwise_grad as egrad
from autograd import jacobian
import autograd.numpy as np

from comm_functions import sigma_estimate

def optimization_run(smoothed_prob, joint_smoothed_prob,sigmas,
                     x0,  zt, delta_yt, regimes):
    
    transition_prob, start_prob = estimate_transition_prob(smoothed_prob, joint_smoothed_prob, regimes)
    wls_params = wls_estimate(sigmas, zt, delta_yt, smoothed_prob)
    residuals = residuals_estimate(delta_yt, zt, wls_params)
    x0, \
        b_matrix, \
        lam_m, \
        sigmas, \
            num_grad = numerical_opt_b_lambda(x0, residuals, smoothed_prob)
    
    return transition_prob, start_prob, x0, b_matrix, lam_m, sigmas, wls_params, num_grad, residuals



def estimate_transition_prob(smoothed_prob, joint_smoothed_prob, regimes):
    regime_transition = em_regime_transition(smoothed_prob, joint_smoothed_prob,regimes)
    trans_prob_temp = regime_transition_matrix(regime_transition, regimes)
    start_prob = initial_probabilities(trans_prob_temp).reshape(-1, 1)
    transition_prob = trans_prob_temp[:, :, 0]
    return transition_prob, start_prob



def em_regime_transition(smoothed_marginal_probabilities, smoothed_joint_probabilities, k_regimes):
    """
    TOOK FROM THE STATS MODEL TSA REGIME SWITCHING MODEL!!!
    EM step for regime transition probabilities
    """
    # Marginalize the smoothed joint probabilities to just S_t, S_{t-1} | T
    tmp = smoothed_joint_probabilities
    for i in range(tmp.ndim - 3):
        tmp = np.sum(tmp, -2)
    smoothed_joint_probabilities = tmp
    # Transition parameters (recall we're not yet supporting TVTP here)

    regime_transition = np.zeros((k_regimes, (k_regimes-1)))

    for i in range(k_regimes):  # S_{t_1}
        for j in range(k_regimes - 1):  # S_t
            regime_transition[i, j] = (
                    np.sum(smoothed_joint_probabilities[j, i]) /
                    np.sum(smoothed_marginal_probabilities[i]))

        # It may be the case that due to rounding error this estimates
        # transition probabilities that sum to greater than one. If so,
        # re-scale the probabilities and warn the user that something
        # is not quite right
        delta = np.sum(regime_transition[i,:]) - 1
        print(f'this is delta{delta}')
        print(f'this is sum: {np.sum(regime_transition[i,:])}')
        if delta > 0:
            print('delta is greater than 0')
            warnings.warn('Invalid regime transition probabilities'
                          ' estimated in EM iteration; probabilities have'
                          ' been re-scaled to continue estimation.',
                          EstimationWarning)
            regime_transition[i] /= 1 + delta + 1e-6

    return regime_transition




def regime_transition_matrix(regime_transition, k_regimes):
    """
    Construct the left-stochastic transition matrix
    TOOK FROM THE STATS MODEL TSA REGIME SWITCHING MODEL!!!
    Notes
    -----
    This matrix will either be shaped (k_regimes, k_regimes, 1) or if there
    are time-varying transition probabilities, it will be shaped
    (k_regimes, k_regimes, nobs).

    The (i,j)th element of this matrix is the probability of transitioning
    from regime j to regime i; thus the previous regime is represented in a
    column and the next regime is represented by a row.

    It is left-stochastic, meaning that each column sums to one (because
    it is certain that from one regime (j) you will transition to *some
    other regime*).
    """
    # transition_matrix = regime_transition.reshape(k_regimes,k_regimes,1)
    if True:
        transition_matrix = np.zeros((k_regimes, k_regimes, 1), dtype=np.float64)
        transition_matrix[:-1, :, 0] = regime_transition.T #np.reshape(regime_transition,
                                                  #(k_regimes - 1, k_regimes))
        transition_matrix[-1, :, 0] = (
                1 - np.sum(transition_matrix[:-1, :, 0], axis=0))

    return transition_matrix


def initial_probabilities(regime_transition):
    """
    TOOK FROM THE STATS MODEL TSA REGIME SWITCHING MODEL!!!
    Retrieve initial probabilities
    """
    if regime_transition.ndim == 3:
        regime_transition = regime_transition[..., 0]
    m = regime_transition.shape[0]
    A = np.c_[(np.eye(m) - regime_transition).T, np.ones(m)].T
    try:
        probabilities = np.linalg.pinv(A)[:, -1]
    except np.linalg.LinAlgError:
        raise RuntimeError('Steady-state probabilities could not be'
                           ' constructed.')

    # Slightly bound probabilities away from zero (for filters in log
    # space)
    probabilities = np.maximum(probabilities, 1e-20)

    return probabilities


def sigma_likelihood(x, residuals, smoothed_prob):
    """
    :param x: must be a column vector of guesses
    :param residuals:
    :param smoothed_prob:
    :return:
    """

    k_vars, obs = residuals.shape
    regimes = smoothed_prob.shape[0]
    b_matrix, lam_m = reconstitute_b_lambda(x, k_vars, regimes)
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


def reconstitute_b_lambda(x, k_vars, regimes):
    x = x.reshape(-1, 1)
    lam_m = [] 
    b_matrix = x[:k_vars * k_vars, [0]].reshape(k_vars, k_vars).T
    identity_mat = np.eye(k_vars)
    for regime in range(regimes - 1):
        if regime == 0:
            start = k_vars * k_vars
            end = start + k_vars
        else:
            start = end
            end = start + k_vars
        lam_m.append(identity_mat * np.exp(x[start:end, [0]]))

    return b_matrix, lam_m



def fprime(x, residuals, smoothed_prob):
    func = lambda x : sigma_likelihood(x, residuals, smoothed_prob)
    g= egrad(func)
    return g(x)


def hessian(x,  residuals, smoothed_prob):
  func = lambda x : sigma_likelihood(x, residuals, smoothed_prob)
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

def residuals_estimate(delta_yt, zt, wls_params):
    k_vars, obs = delta_yt.shape
    residuals = np.zeros(delta_yt.shape)

    for t in range(obs):
        residuals[:, [t]] = delta_yt[:, [t]] - np.kron(zt[:, [t]].T, np.eye(k_vars)) @ wls_params
    return residuals



def numerical_opt_b_lambda(x0, residuals, smoothed_prob):
    k_vars = residuals.shape[0]
    regimes = smoothed_prob.shape[0]
    func = lambda x : sigma_likelihood(x, residuals, smoothed_prob)
    grad1 = lambda x : fprime(x, residuals, smoothed_prob) # gradient based on Autograd 
    hess1 = lambda x :  hessian(x,  residuals, smoothed_prob)
    initial_x0 = x0
    method_ = 'trust-krylov' # starting optimization with gradient based optimizer

    for  j  in range(5): 
        print(f'\n this is the current optimization method: \n {method_}')
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
            print(result)
        except:
            print('optimization resulted in an error')
            message = False 

        if message == True:
            break 
        
        length_x0 = k_vars*k_vars+(regimes-1)*k_vars
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

    b, lam_m_list  = reconstitute_b_lambda(result.x, k_vars, regimes)
    # above lambda is a list convert into a 3d array 
    lam_m = np.zeros([k_vars,k_vars,regimes-1])
    for regime in range(regimes-1):
        lam_m[:,:,regime] = lam_m_list[regime]

    sigmas = sigma_estimate(b, lam_m)

    return result.x, b, lam_m, sigmas, num_grad

