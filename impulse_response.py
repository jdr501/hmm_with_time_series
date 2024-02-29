import statsmodels.tsa.vector_ar.plotting as plotting

def irf_plot(phi, stderr, names = ['USDT_price',	'USDT_supply',	'BTC_price']):
    
    return plotting.irf_grid_plot(phi, stderr, None, None, names, None,
                  signif=0.05, hlines=None, subplot_params=None,
                  plot_params=None, figsize=(12,12), stderr_type='mc')

