# Purpose:
# Record of revisions:
# Date 		Programmer 		Description of change
# ======== 	============= 	=====================
# 05/2021 	A. Elkouk     	Original code


import numpy as np
import xarray as xr
import os

from dindices import calc_dindex
from dfreq import calc_dpr_dths
from dexposure import calc_exp_dths, calc_exp_diri
from drisk import lognorm_exposure, drisk_pr

# Configuration
basepath = '/storage/elkoukah/empirical/'
os.chdir(basepath)

hist_fpath = '0_data/ISIMIP2b_soilmoist/clm45_gfdl-esm2m_ewembi_historical_2005soc_co2_soilmoist_global_monthly_1861_2005.nc4'
prj_fpath = '0_data/ISIMIP2b_soilmoist/clm45_gfdl-esm2m_ewembi_rcp85_2005soc_co2_soilmoist_global_monthly_2006_2099.nc'
outpath = '2_pipeline/drisk/tmp/'
index_flag = 'SMI'
hm, gcm, _, clim = os.path.basename(prj_fpath).split('_')[:4]
Thist = slice(-360, None)

ths = np.array([0.3, 0.2, 0.1, 0.05, 0.01])
outfname_dpr = '_'.join((hm,gcm,clim,index_flag+'.nc'))
outfname_dpr_ref = '_'.join((hm,gcm,'historical',index_flag+'.nc'))
Ti = slice(-360, None)
Ti_ref = slice(-360, None)

ds_pop = xr.load_dataarray('0_data/POP/pop_total_rural.nc')
pop = ds_pop.values
ds_countries = xr.load_dataarray('0_data/COUNTRIES/countries.nc')
countries_ma = ds_countries.values.ravel()
ncoun = len(ds_countries.ISO)
tp = 1
yi = 6
yi_ref = 0
outfname_exp = '_'.join((hm,gcm,clim,index_flag,'exposure.nc'))
outfname_exp_ref = '_'.join((hm,gcm,'historical',index_flag,'exposure.nc'))
outfname_exp_cstclim = '_'.join((hm,gcm,'historical',index_flag,'exposure_cstclim.nc'))
outfname_exp_cstpop = '_'.join((hm,gcm,clim,index_flag,'exposure_cstpop.nc'))

outfname_expnorm = '_'.join((hm,gcm,clim,index_flag,'exposure_norm.nc'))
outfname_expnorm_cstclim = '_'.join((hm,gcm,'historical',index_flag,'exposure_norm_cstclim.nc'))
outfname_expnorm_cstpop = '_'.join((hm,gcm,clim,index_flag,'exposure_norm_cstpop.nc'))

ds_hdi = xr.load_dataarray('0_data/POP/vi_norm.nc')
hdi_norm = ds_hdi.values[:, :-1]
hdi_norm_ref = ds_hdi.values[:, [-1]*5] # replicate hdiref 5 times for consistency with SSPs1-5
yi_hdi = 12
yi_hdi_ref = 0
outfname_diri = '_'.join((hm,gcm,clim,index_flag,'diri.nc'))
outfname_diri_cstclim = '_'.join((hm,gcm,'historical',index_flag,'diri_cstclim.nc'))
outfname_diri_cstpop = '_'.join((hm,gcm,clim,index_flag,'diri_cstpop.nc'))

rclass = np.arange(0., 0.91, 0.1)
outfname_expdiri = '_'.join((hm,gcm,clim,index_flag,'expdiri.nc'))
outfname_expdiri_cstclim = '_'.join((hm,gcm,'historical',index_flag,'expdiri_cstclim.nc'))
outfname_expdiri_cstpop = '_'.join((hm,gcm,clim,index_flag,'expdiri_cstpop.nc'))

# Calc drought indices
di = calc_dindex(index_flag, hist_fpath, prj_fpath, usehist_flag=False, outpath=outpath, Thist=Thist)
di_ref = calc_dindex(index_flag, hist_fpath, _, usehist_flag=True, outpath=outpath, Thist=Thist)

# Calc drought frequency
dpr = calc_dpr_dths(di, ths, Ti, outpath, outfname_dpr)
dpr_ref = calc_dpr_dths(di_ref, ths, Ti_ref, outpath, outfname_dpr_ref)

# Calc exposure to drought
exp_grd, _ = calc_exp_dths(dpr, pop, countries_ma, ncoun, tp, yi, outpath, outfname_exp)
exp_grd_ref, _ = calc_exp_dths(dpr_ref, pop, countries_ma, ncoun, tp, yi_ref, outpath, outfname_exp_ref)
exp_grd_cstclim, _ = calc_exp_dths(dpr_ref, pop, countries_ma, ncoun, tp, yi, outpath, outfname_exp_cstclim)
exp_grd_cstpop, _ = calc_exp_dths(dpr, pop, countries_ma, ncoun, tp, yi_ref, outpath, outfname_exp_cstpop)

# Normalize exposure to drought
exp_normalized = lognorm_exposure(exp_grd, exp_grd_ref, outpath, outfname_expnorm)
exp_normalized_cstclim = lognorm_exposure(exp_grd_cstclim, exp_grd_ref, outpath, outfname_expnorm_cstclim)
exp_normalized_cstpop = lognorm_exposure(exp_grd_cstpop, exp_grd_ref, outpath, outfname_expnorm_cstpop)

# Calc risk probability
diri = drisk_pr(exp_normalized, hdi_norm, yi_hdi, outpath, outfname_diri)
diri_cstclim = drisk_pr(exp_normalized_cstclim, hdi_norm, yi_hdi, outpath, outfname_diri_cstclim)
diri_cstpop = drisk_pr(exp_normalized_cstpop, hdi_norm_ref, yi_hdi_ref, outpath, outfname_diri_cstclim)

# Calc population at risk
diri_exp_grd, diri_exp_country = calc_exp_diri(diri, pop, countries_ma, ncoun, rclass, tp, yi, outpath, outfname_expdiri)
diri_exp_grd_cstclim, diri_exp_country_cstclim = calc_exp_diri(diri_cstclim, pop, countries_ma, ncoun, rclass, tp, yi, outpath, outfname_expdiri_cstclim)
diri_exp_grd_cstpop, diri_exp_country_cstpop = calc_exp_diri(diri_cstpop, pop, countries_ma, ncoun, rclass, tp, yi_ref, outpath, outfname_expdiri_cstpop)