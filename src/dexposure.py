# Purpose: Calculate people exposed to drought
# Record of revisions:
# Date 		Programmer 		Description of change
# ======== 	============= 	=====================
# 04/2021 	A. Elkouk     	Original code


import numpy as np
import xarray as xr
from scipy import stats
import os

def calc_exp_dths(dpr, pop, countries_ma, ncoun, tp, yi, outpath, outfname):

    nth, ngrd = dpr.shape
    ntp, ny, nssp, nlat, nlon = pop.shape
    exp_grd = np.full((nssp, nth, ngrd), np.nan)
    exp_country = np.full((nssp, nth, ncoun), np.nan)
    for thi in range(nth):
        exp_grd[:, thi] = dpr[thi] * pop[tp,yi].reshape(nssp, ngrd)
        for ci in range(ncoun):
            ci_ma = (countries_ma == ci)
            exp_country[:, thi, ci] = np.nansum(exp_grd[:, thi][:, ci_ma], axis=-1)

    dsout = xr.Dataset(data_vars={'Dexp_grd':(('nssp', 'nth', 'ngrd'), exp_grd),
                                  'Dexp_country':(('nssp', 'nth', 'ncountry'), exp_country)})
    dsout.to_netcdf(os.path.join(outpath, outfname))
    print(outfname, 'DONE')

    return exp_grd, exp_country

def calc_exp_diri(diri, pop, countries_ma, ncoun, rclass, tp, yi, outpath, outfname):

    nssp, nth, ngrd = diri.shape
    ntp, ny, nssp, nlat, nlon = pop.shape
    nrcl = rclass.size
    exp_grd = np.full((nrcl, nssp, nth, ngrd), np.nan)
    exp_country = np.full((nrcl, nssp, nth, ncoun), np.nan)
    for thi in range(nth):
        for rci in range(nrcl):
            exp_grd[rci, :, thi] = (diri[thi]>=rclass[rci]) * pop[tp,yi].reshape(nssp, ngrd)
            for ci in range(ncoun):
                ci_ma = (countries_ma == ci)
                exp_country[rci, :, thi, ci] = np.nansum(exp_grd[rci, :, thi][:, ci_ma], axis=-1)

    dsout = xr.Dataset(data_vars={'Dexp_grd':(('niri', 'nssp', 'nth', 'ngrd'), exp_grd),
                                  'Dexp_country':(('niri', 'nssp', 'nth', 'ncountry'), exp_country)})
    dsout.to_netcdf(os.path.join(outpath, outfname))
    print(outfname, 'DONE')

    return exp_grd, exp_country