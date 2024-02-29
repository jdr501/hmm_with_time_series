
from  comm_functions import zscore 
import pandas as pd 
from expectation_maximization import *
from impulse_response import irf_plot
from bootstrap_std_err import bootstrap_run
import numpy as np
import json
df = pd.read_csv('tether_data.csv')
df.set_index('date', inplace =True)

df['BTC_price'] = (df['BTC_price']/100 )  
df['USDT_price'] =  (df['USDT_price']*100) 
df['USDT_supply'] = (df['USDT_supply']/100000000) 
df =df[df.index > '2018-04-01'] # 2019 april is the migration date to etherium 
#df =df[df.index < '2022-01-01']
df2 = pd.DataFrame()
for column in df.columns:
  z, avg, std, m = zscore(df[column], window=45, return_all=True)
  df2[column] = df[column]
  df2.loc[~m, column] = avg[~m]

df = df2 # 


k_vars = 3
regimes = 3
lags = 3
runs = 1

# estimation of parameters 
result_dict = em_run(df,lags,regimes, runs, parrelel_run = False,  exog= None)
file_name =  '/globalhome/jdr501/HPC/estimated_results/results_3_regime_parallel_1.json'
with open(file_name, 'w') as fp:
    json.dump(result_dict, fp)

B_mat = np.array(result_dict['b_matrix'])
wls_params = np.array(result_dict['wls_params'])
beta = np.array(result_dict['beta']).reshape(-1,1)
alpha, gamma = convert_wlsparam(wls_params, k_vars)
neqs = k_vars
k_ar = lags +1
steps = 40
phi = svar_ma_rep(B_mat, beta, alpha, gamma, neqs, k_ar, maxn= steps)

print('standard errors step')
stderr = bootstrap_run( result_dict,lags,k_vars, steps+1, draws=1000)

plot = irf_plot(phi, stderr, names = ['USDT_price',	'USDT_supply',	'BTC_price'])
plot.savefig('irf_plot.pdf', dpi=300)

