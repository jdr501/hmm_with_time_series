from  Initialization import *
from expectation import *
from optimization import *
from comm_functions import *
import time

from functools import partial
import multiprocessing
import os
import random




def em_algorithm(sigmas, residuals, start_prob, transition_prob, beta,  zt, delta_yt,x0):
    regimes = start_prob.shape[0]
    llf = []
    for i in range(1000): 
        if i>1 and  abs(llf[-1] - llf[-2])/ abs(llf[-2])< 1e-6  and parameter_converged: # / abs(llf[-2])
            print('em converged')
            break         
        loglikelihood, \
        smoothed_prob,\
        joint_smoothed_prob,\
        loglikelihoods_obs = expectation_run(sigmas, residuals, start_prob, transition_prob)

    
        transition_prob, \
        start_prob, x0, \
        b_matrix, \
        lam_m, sigmas,\
        wls_params, num_grad, \
        residuals = optimization_run(smoothed_prob, joint_smoothed_prob,sigmas,x0,  zt, delta_yt,regimes)
        

        print(f'this is new log likelihood {loglikelihood}')
        llf.append(loglikelihood)

        param_convergence_dict = {'numerical_opt_x': x0.tolist(),
                                'start_prob': start_prob.reshape(-1,1).ravel().tolist(),
                                'transition_prob': transition_prob.reshape(-1,1).ravel().tolist()}

        if i >0:
            parameter_converged = check_parameter_converge(previous_param_convergence_dict, param_convergence_dict, 1e-6)
        previous_param_convergence_dict = param_convergence_dict
    
    print(f'================================ \n it took {i} runs to converge \n ================================')


    em_result = {'likelihood': loglikelihood.tolist(),
                'start_prob': start_prob.tolist(),
                'smoothed_prob': smoothed_prob.tolist(),
                'joint_smoothed_prob': joint_smoothed_prob.tolist(),
                'transition_prob': transition_prob.tolist(),
                'b_matrix': b_matrix.tolist(),
                'grad': num_grad.tolist(),
                'lam_m': lam_m.tolist(),
                'sigma': sigmas.tolist(),
                'wls_params': wls_params.tolist(),
                'beta': beta.tolist(), 
                'residuals': residuals.tolist(),
                'llf': llf,
                'zt': zt.tolist()}
    return em_result



def em_run(df,lags,regimes, runs, parrelel_run = False,  exog= None):
    b, start_prob, \
    transition_prob, \
    delta_yt, zt,\
    residuals, beta = initialize_step_1(df,lags,regimes, exog)

    if parrelel_run:
        result_dic =parallel_run(b, residuals, start_prob,
                                transition_prob, beta,  zt, 
                                delta_yt, regimes,runs)


    else:
        result_dic = {}
        for i in range(runs):
            sigmas, x0 = initialize_step_2(b,regimes)
            em_result = em_algorithm(sigmas, residuals, start_prob, transition_prob, beta,  zt, delta_yt,x0)
            result_dic.update({f'{i}': em_result})
    runs_llf = np.zeros(runs)


    for i in range(runs):
        runs_llf[i]= result_dic[f'{i}']['likelihood']
    index = np.argmax(runs_llf)
    best_result = result_dic[f'{index}']
    return best_result





def parallel_run(b, residuals, start_prob,
                  transition_prob, beta,  zt, 
                  delta_yt, regimes,runs):  
    sigmas_list = []
    result_dic = {}
    d = 0
    for _ in range(runs):
        sigmas = initialize_step_2(b,regimes)
        sigmas_list.append(sigmas)
   
    partial_em = partial(em_algorithm, residuals, start_prob, transition_prob, beta,  zt, delta_yt)

    if __name__ == "em_parallel":
        ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
        print(f'this is number of CPUs per node {ncpus*2}')
        with multiprocessing.Pool(processes=ncpus) as pool:
            output = [pool.apply_async(partial_em, args=(x,)) for x in sigmas_list]

            for p in output: 
                print(f'this is {d}')
                result_dic.update({f'{d}': p.get()})
                d += 1


    return result_dic 
 