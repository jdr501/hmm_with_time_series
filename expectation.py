import numpy as np
from scipy.stats import multivariate_normal
from statsmodels.tsa.regime_switching.markov_switching import cy_kim_smoother_log, cy_hamilton_filter_log

'''
below are the wrapper functions to  convert my estimated parameters to pass to statsmodels tsa regime switching hamilton filter
and kim smoother functions. 
Their smoother functions more sofisticated than general user written functions.These are written in C languge for efficiency.
They use diferent methods to  reduce numerical overflow and underflow issues, hence their methods are more numerically stable. 
For more details visit:
https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/regime_switching/markov_switching.py

'''
def expectation_run(sigma, residuals, start_prob, transition_prob):

    conditional_density = normal_cond_dens(sigma, residuals)
    
    joint_smoothed_prob, \
        smoothed_prob, \
        loglikelihood,\
        loglikelihoods_obs  = smoothed(start_prob, transition_prob, conditional_density)
  

    return loglikelihood, smoothed_prob, joint_smoothed_prob, loglikelihoods_obs


def normal_cond_dens(sigma, residuals):
    regimes = sigma.shape[2]
    obs = residuals.shape[1]
    conditional_density = np.zeros([regimes, obs])  # y_t|s_t = j conditional density of Y for a given state
    for r in range(regimes):
        conditional_density[r, :] = multivariate_normal(mean=None,
                                                        cov=sigma[:, :, r], allow_singular = True).logpdf(residuals.T).T
        for i in range(len(conditional_density[r, :])):
            if conditional_density[r, i] < -1E16:
                conditional_density[r, i] = -1E16
                
    return conditional_density


def smoothed(initial_prob, transition_prob, conditional_density):
    initial_prob = np.squeeze(initial_prob)
    regimes, obs = conditional_density.shape
    trans_prob = transition_prob.reshape(regimes, regimes, 1)
    cond_dens = np.zeros([regimes, regimes, obs])
    for regime in range(regimes):
        for t in range(obs):
            cond_dens[:, regime, [t]] = conditional_density[:, [t]]
            #cond_dens[:, 1, [t]] = conditional_density[:, [t]]
    filtered_marginal_probabilities, \
        predicted_joint_probabilities, \
        joint_loglikelihoods, \
        filtered_joint_probabilities, \
        predicted_joint_probabilities_log, \
        filtered_joint_probabilities_log = cy_hamilton_filter_log(initial_prob, trans_prob, cond_dens,
                                                                  0)  # model order doesn't matter in this case
    smoothed_joint_probabilities, \
        smoothed_marginal_probabilities = cy_kim_smoother_log(trans_prob, predicted_joint_probabilities_log,
                                                              filtered_joint_probabilities_log)



    return smoothed_joint_probabilities, smoothed_marginal_probabilities, sum(joint_loglikelihoods), joint_loglikelihoods
