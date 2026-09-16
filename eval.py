import numpy as np
from sklearn.utils import resample
import statsmodels.api as sm
import scipy.stats as st

def block_bootstrap_metric(diff_series, block_size=60, n_boot=2000, seed=0):
    """
    diff_series: (T,) series of loss differences (modelA - modelB) aligned in time order
    Returns: bootstrap means array and 95% CI tuple
    """
    np.random.seed(seed)
    T = len(diff_series)
    n_blocks = int(np.ceil(T / block_size))
    boots = []
    for _ in range(n_boot):
        starts = np.random.randint(0, T - block_size + 1, size=n_blocks)
        samp = []
        for s in starts:
            samp.append(diff_series[s:s+block_size])
        samp = np.concatenate(samp)[:T]
        boots.append(samp.mean())
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return boots, (lo, hi)

def diebold_mariano(loss_a, loss_b, h=1):
    """
    loss_a, loss_b: arrays (T,) of per-forecast loss (aligned)
    Returns DM statistic and two-sided p-value using Newey-West HAC (lag=h-1)
    """
    d = loss_a - loss_b
    T = len(d)
    mean_d = d.mean()
    lag = max(0, h-1)
    X = np.ones((T,1))
    model = sm.OLS(d, X).fit(cov_type='HAC', cov_kwds={'maxlags':lag})
    var = model.cov_params()[0,0]
    dm_stat = mean_d / np.sqrt(var / T)
    pval = 2 * (1 - st.norm.cdf(abs(dm_stat)))
    return dm_stat, pval