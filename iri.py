# Purpose:
# Record of revisions:
# Date 		Programmer 		Description of change
# ======== 	============= 	=====================
# 11/09/19 	A. Elkouk     	Original code

import numpy as np
import xarray as xr
from scipy import stats
import os
from utilities import lognormal_population

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
nssp = 5

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

# Ensemble mean pr
ens_mean = np.nanmean(ens_dpr, axis=(2,3))
## Uncertainty
ens_std = np.nanstd(ens_dpr, axis=(2,3))

# Future exposure
ds_pop = xr.load_dataarray(basepath + '0_data/POP/pop_total_rural.nc')
pop_tt_rur = ds_pop.values.reshape(2,10,nssp,ngrd)
pop_tt_rur = pop_tt_rur[[1,0]] # change to match SMI-rur/RI-tot
pop_nT = np.stack([pop_tt_rur[:, 2], pop_tt_rur[:, 6]]) # Total pop 2030, 2070
## Reshape to the same dims
tmp_pop = np.stack([pop_nT]*nth*(nrcp-1)).reshape((nrcp-1),nth,nT,nvar,nssp,ngrd).transpose(3,0,4,1,2,5)
tmp_dpr = np.stack([ens_mean[:, 1:]]*nssp).transpose(1,2,0,3,4,5)
exp_rcpssp = tmp_pop*tmp_dpr
## Uncertainty
tmp_dpr = np.stack([ens_std[:, 1:]]*nssp).transpose(1,2,0,3,4,5)
exp_rcpssp_unc = tmp_pop*tmp_dpr

# Ref exposure
# ds_pop_ref = xr.load_dataarray(basepath + '0_data/POP/pop_total_UN-Adjusted.nc')
# pop_ref = ds_pop_ref.values[3] # 2015 population counts
# exp_ref = ens_mean[:, 0]*pop_ref.ravel()
tmp_ref = np.stack([pop_tt_rur[:, 0, 1]]*nth*nT).reshape(nth,nT,nvar,ngrd).transpose(2,0,1,3)
exp_ref = ens_mean[:, 0]*tmp_ref

# Log-normalize exposure
popnorm = np.full((nvar, (nrcp-1), nssp, nth, nT, ngrd), np.nan)
popnorm_unc = np.full((nvar, (nrcp-1), nssp, nth, nT, ngrd), np.nan)
for vari in range(nvar):
    for thi in range(nth):
        ref = exp_ref[vari, thi].ravel()
        est = exp_rcpssp[vari, :, :, thi].ravel()
        lognorm = lognormal_population(est, ref)
        popnorm[vari, :, :, thi] = np.reshape(lognorm, ((nrcp-1), nssp, nT, ngrd))

        est = exp_rcpssp_unc[vari, :, :, thi].ravel()
        lognorm = lognormal_population(est, ref)
        popnorm_unc[vari, :, :, thi] = np.reshape(lognorm, ((nrcp-1), nssp, nT, ngrd))
# IRI index
## VI in 2030 and 2070
dsvi = xr.load_dataarray(basepath+'0_data/POP/vi_norm.nc')
vi = dsvi.values[:, :-1]
tmp_vi = np.stack([vi[4], vi[12]])
tmp_vi = np.stack([tmp_vi]*nvar*nth*(nrcp-1)).reshape(nvar,(nrcp-1),nth,nT,nssp,ngrd).transpose(0,1,4,2,3,5)
iri = tmp_vi*popnorm
## Uncertainty
iri_unc = tmp_vi*popnorm_unc
# Non Normalized IRI
dsvi = xr.load_dataarray(basepath+'0_data/POP/vi_nonnorm.nc')
vi = dsvi.values[:, :-1]
tmp_vi = np.stack([vi[4], vi[12]])
tmp_vi = np.stack([tmp_vi]*nvar*nth*(nrcp-1)).reshape(nvar,(nrcp-1),nth,nT,nssp,ngrd).transpose(0,1,4,2,3,5)
iri_nonnorm = tmp_vi*exp_rcpssp

dsout = xr.Dataset(data_vars={'IRI': (('nvar', 'nrcp', 'nssp', 'nth', 'nT', 'ngrd'), iri),})
                              #'nonIRI': (('nvar', 'nrcp', 'nssp', 'nth', 'nT', 'ngrd'), iri_nonnorm),})
                              #'IRIunc': (('nvar', 'nrcp', 'nssp', 'nth', 'nT', 'ngrd'), iri_unc)})
dsout.to_netcdf(basepath + '2_pipeline/drisk/store/IRI/iri.nc')
print('DONE')

dsout = xr.Dataset(data_vars={'expnorm': (('nvar', 'nrcp', 'nssp', 'nth', 'nT', 'ngrd'), popnorm),
                              'expnonnorm': (('nvar', 'nrcp', 'nssp', 'nth', 'nT', 'ngrd'), exp_rcpssp),})
dsout.to_netcdf(basepath + '2_pipeline/drisk/store/IRI/exp_grd.nc')
print('DONE')


# IRI under fixed climate and socio-eco change
## Fix socio-eco exposure
tmp_ref = np.stack([pop_tt_rur[:, 0, 1]]*(nrcp-1)*nth*nT).reshape((nrcp-1),nth,nT,nvar,ngrd).transpose(3,0,1,2,4)
exp_clim = ens_mean[:, 1:]*tmp_ref
## Fix climate exposure
tmp_dpr = np.stack([ens_mean[:, 0]]*nssp).transpose(1,0,2,3,4)
tmp_pop = np.stack([pop_nT]*nth).reshape(nth,nT,nvar,nssp,ngrd).transpose(2,3,0,1,4)
exp_ssp = tmp_pop*tmp_dpr

## Exposure
exp_clim_norm = np.full((nvar, (nrcp-1), nth, nT, ngrd), np.nan)
exp_ssp_norm = np.full((nvar, nssp, nth, nT, ngrd), np.nan)
for vari in range(nvar):
    for thi in range(nth):
        ### Reference 2015pop, Histclim
        ref = exp_ref[vari, thi].ravel()

        est_clim = exp_clim[vari, :, thi].ravel()
        est_ssp = exp_ssp[vari, :, thi].ravel()
        lognorm_clim = lognormal_population(est_clim, ref)
        lognorm_ssp = lognormal_population(est_ssp, ref)
        exp_clim_norm[vari, :, thi] = np.reshape(lognorm_clim, ((nrcp-1), nT, ngrd))
        exp_ssp_norm[vari, :, thi] = np.reshape(lognorm_ssp, (nssp, nT, ngrd))

## IRI fixed socio-eco
vi_cst = dsvi.values[0, -1]
vi_cst = np.stack([vi_cst]*nvar*nth*nT*(nrcp-1)).reshape(nvar,nth,nT,(nrcp-1),ngrd).transpose(0,3,1,2,4)
iri_clim = exp_clim_norm*vi_cst

## IRI fixed climate, 2030 and 2070
tmp_vi = np.stack([vi[4], vi[12]])
tmp_vi = np.stack([tmp_vi]*nvar*nth).reshape(nvar,nth,nT,nssp,ngrd).transpose(0,3,1,2,4)
iri_ssp = exp_ssp_norm*tmp_vi

dsout = xr.Dataset(data_vars={'IRIclim': (('nvar', 'nrcp', 'nth', 'nT', 'ngrd'), iri_clim),
                              'IRIssp': (('nvar', 'nssp', 'nth', 'nT', 'ngrd'), iri_ssp),})
dsout.to_netcdf(basepath + '2_pipeline/drisk/store/IRI/iri_climssp_ct.nc')
print('DONE')

