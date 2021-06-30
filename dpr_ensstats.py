# Purpose:
# Record of revisions:
# Date 		Programmer 		Description of change
# ======== 	============= 	=====================
# 11/09/19 	A. Elkouk     	Original code

import numpy as np
import xarray as xr
from scipy import stats
import os

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

ens_mean = np.nanmean(ens_dpr, axis=(2,3))
# ens_mean_ch = ens_mean[:,1:]-ens_mean[:,0][:, None]
# ens_mean_snr =

ens_ch = ens_dpr[:,1:]-ens_dpr[:,0][:, None]
ens_ch_mean = np.nanmean(ens_ch, axis=(2,3))
ens_ch_snr = ens_ch_mean/np.nanstd(ens_ch, axis=(2,3))
# ens_ch_r = np.where(np.isnan(ens_ch), 0, ens_ch)
ens_ch_agg = (np.sign(ens_ch.transpose(2,3,0,1,4,5,6)) == np.sign(ens_ch_mean)).sum(axis=(0,1))
ens_size = np.array([[32,32,24], [28,28,24]])
agg = ens_ch_agg.T/ens_size.T*100
agg = agg.T

dsout = xr.Dataset(data_vars={'DPr_ch_ensmean': (('nvar', 'nrcp', 'nth', 'nT', 'ngrd'), ens_ch_mean),
                              'DPr_ch_snr': (('nvar', 'nrcp', 'nth', 'nT', 'ngrd'), ens_ch_snr),
                              'DPr_ch_agg': (('nvar', 'nrcp', 'nth', 'nT', 'ngrd'), agg),
                               'DPr_ensmean': (('nvar', 'nclim', 'nth', 'nT', 'ngrd'), ens_mean),
                              })
dsout.to_netcdf(basepath + '2_pipeline/drisk/store/dpr_ensstats/dpr_ensstats.nc')
print('DONE')
