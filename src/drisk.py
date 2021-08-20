# Purpose: Estimate drought risk illustrative probability
# Record of revisions:
# Date 		Programmer 		Description of change
# ======== 	============= 	=====================
# 04/2021 	A. Elkouk     	Original code


import numpy as np
import xarray as xr
import os
from scipy import stats


def drisk_pr(exp_norm, hdi_norm, yi, outpath, outfname):
    nssp, nth, ngrd = exp_norm.shape
    diri = exp_norm.transpose(1,0,2) * hdi_norm[yi].reshape(nssp, ngrd)
    dsout = xr.Dataset(data_vars={'Diri':(('nth', 'nssp', 'ngrd'), diri.transpose(1,0,2))})
    dsout.to_netcdf(os.path.join(outpath, outfname))
    return diri


def lognorm_exposure(exp_grd, exp_grd_ref, outpath, outfname):

    nssp, nth, ngrd = exp_grd.shape
    exp_normalized = np.full((nssp, nth, ngrd), np.nan)
    for thi in range(nth):
        yy = exp_grd[:, thi].ravel()
        xx = exp_grd_ref[:, thi].ravel()
        exp_normalized[:, thi] = lognormal(yy, xx).reshape(nssp, ngrd)
    dsout = xr.Dataset(data_vars={'Dexp_normalized': (('nssp', 'nth', 'ngrd'), exp_normalized),})
    dsout.to_netcdf(os.path.join(outpath, outfname))

    return exp_normalized

def lognormal(exp, expref):
    ''' Normalize population using log10-normal cdf'''
    logdata = np.log(expref[expref > 0])
    params = (np.nanmean(logdata), np.nanstd(logdata))
    expcdf = stats.norm.cdf(np.log(exp), *params)
    return expcdf