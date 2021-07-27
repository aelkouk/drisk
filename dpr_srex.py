# Purpose:
# Record of revisions:
# Date 		Programmer 		Description of change
# ======== 	============= 	=====================
# 11/09/19 	A. Elkouk     	Original code

import numpy as np
import xarray as xr
from scipy import stats
import os
from mpi4py import MPI

gcms = ['GFDL-ESM2M', 'HADGEM2-ES', 'IPSL-CM5A-LR', 'MIROC5']
hms = ['clm45', 'jules-w1', 'orchidee', 'matsiro', 'lpjml', 'h08', 'pcr-globwb', 'watergap2']
rcps = ['historical', 'rcp26', 'rcp60', 'rcp85']
basepath = '/storage/elkoukah/empirical/'
varnames = ['SMI', 'RI']
inpath = basepath + '2_pipeline/drisk/store/dpr/'
ncfs = os.listdir(inpath)
ths = np.array([0.3, 0.2, 0.1, 0.05, 0.01])
nth = ths.size
ngrd = (360*180)*4
nhm = len(hms)
ngcm = len(gcms)
nrcp = len(rcps)
nT = 2 # 2031-60, 2071-100
nvar = len(varnames)

ens_dpr = np.full((nvar, nrcp, ngcm, nhm, nth, nT, ngrd), np.nan)
for ncf in ncfs:
    hmi, gcmi, rcpi, vari = ncf[:-3].split('_')
    hmidx, gcmidx, rcpidx, varidx = hms.index(hmi), gcms.index(gcmi.upper()), rcps.index(rcpi), varnames.index(vari)
    ds = xr.load_dataset(inpath+ncf)
    dpr = ds['DPr'].values
    if rcps[0] in ncf:
        dpr = np.stack([dpr]*nT).transpose(1,0,2)
    ens_dpr[varidx, rcpidx, gcmidx, hmidx] = dpr
    print(ncf)

mafile = basepath+'0_data/IPCC/SREX_regions_mask_0.5x0.5.nc'
ds_ma = xr.load_dataarray(mafile)
subregions_ma = ds_ma.values.ravel()
nreg = 26+1 # +Global

ens_dpr_srex = np.full((nvar, nrcp, ngcm, nhm, nth, nT, nreg), np.nan)
for regi in range(nreg):
    if regi==(nreg-1):
        ma = ~np.isnan(subregions_ma)
    else:
        ma = (subregions_ma==regi+1)
    ens_dpr_srex[:,:,:,:,:,:, regi] = np.nanmean(ens_dpr[:,:,:,:,:,:, ma], axis=-1)

dsout = xr.Dataset(data_vars={'DPr_ens_srex': (('nvar', 'nrcp', 'ngcm', 'nhm', 'nth', 'nT', 'nreg'), ens_dpr_srex),
                              })
dsout.to_netcdf(basepath + '2_pipeline/drisk/store/dpr_ensstats/dpr_ens_srex.nc')
print('DONE')