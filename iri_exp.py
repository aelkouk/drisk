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
di = 2 # 10th percentile

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
tmp_dpr = np.stack([ens_dpr[:, 1:]]*nssp).transpose(3,4,1,2,0,5,6,7)
exp_rcpssp_ensunc = tmp_pop*tmp_dpr
exp_rcpssp_unc = np.nanstd(exp_rcpssp_ensunc, axis=(0,1))


# IRI classes (nvar, nrcp, nssp, nth, nT, ngrd)
ds_iri = xr.load_dataset(basepath+'2_pipeline/drisk/store/IRI/iri.nc')
iri = ds_iri['IRI'].values#[:,:,:, di]
iri_unc = ds_iri['IRIunc'].values#[:,:,:, di]
rclass = np.arange(0., 0.91, 0.1)
nrcl = rclass.size

ds_countries = xr.load_dataarray(basepath + '0_data/COUNTRIES/countries.nc')
countries = ds_countries.values.ravel()
ncoun = len(ds_countries.ISO)

iri_exp = np.zeros((nrcl, nvar, (nrcp-1), nssp, nth, nT, ncoun))
iri_exp_glob = np.zeros((nrcl, nvar, (nrcp-1), nssp, nth, nT))
iri_exp_unc = np.zeros((nrcl, nvar, (nrcp-1), nssp, nth, nT, ncoun))
iri_exp_glob_unc = np.zeros((nrcl, nvar, (nrcp-1), nssp, nth, nT))
for rci in range(nrcl):
    exprci = (iri>=rclass[rci]) * exp_rcpssp
    exprci_unc = (iri >= rclass[rci]) * exp_rcpssp_unc
    for ci in range(ncoun):
        ci_ma = (countries == ci)
        iri_exp[rci, :, :, :, :, :, ci] = np.nansum(exprci[:, :, :, :, :, ci_ma], axis=-1)
        iri_exp_unc[rci, :, :, :, :, :, ci] = np.nansum(exprci_unc[:, :, :, :, :, ci_ma], axis=-1)
    iri_exp_glob[rci] = np.nansum(exprci, axis=-1)
    iri_exp_glob_unc[rci] = np.nansum(exprci_unc, axis=-1)


# Exposure clim and socioeco change
## Fix socio-eco exposure
tmp_ref = np.stack([pop_tt_rur[:, 0, 1]]*(nrcp-1)*nth*nT).reshape((nrcp-1),nth,nT,nvar,ngrd).transpose(3,0,1,2,4)
exp_clim = ens_mean[:, 1:]*tmp_ref
## Fix climate exposure
tmp_dpr = np.stack([ens_mean[:, 0]]*nssp).transpose(1,0,2,3,4)
tmp_pop = np.stack([pop_nT]*nth).reshape(nth,nT,nvar,nssp,ngrd).transpose(2,3,0,1,4)
exp_ssp = tmp_pop*tmp_dpr

# IRI
ds_iri = xr.load_dataset(basepath+'2_pipeline/drisk/store/IRI/iri_climssp_ct.nc')
iri_clim = ds_iri['IRIclim'].values#[:,:, di]
iri_ssp = ds_iri['IRIssp'].values#[:,:, di]

iri_exp_clim = np.zeros((nrcl, nvar, (nrcp-1), nth, nT, ncoun))
iri_exp_glob_clim = np.zeros((nrcl, nvar, (nrcp-1), nth, nT))

iri_exp_ssp = np.zeros((nrcl, nvar, nssp, nth, nT, ncoun))
iri_exp_glob_ssp = np.zeros((nrcl, nvar, nssp, nth, nT))

for rci in range(nrcl):
    exprci_clim = (iri_clim>=rclass[rci]) * exp_clim
    exprci_ssp = (iri_ssp>=rclass[rci]) * exp_ssp
    for ci in range(ncoun):
        ci_ma = (countries == ci)
        iri_exp_clim[rci, :,:, :, :, ci] = np.nansum(exprci_clim[:, :, :, :, ci_ma], axis=-1)
        iri_exp_ssp[rci, :,:, :, :, ci] = np.nansum(exprci_ssp[:, :, :, :, ci_ma], axis=-1)
    iri_exp_glob_clim[rci] = np.nansum(exprci_clim, axis=-1)
    iri_exp_glob_ssp[rci] = np.nansum(exprci_ssp, axis=-1)

dsout = xr.Dataset(data_vars={'IRIexp': (('nrcl', 'nvar', 'nrcp', 'nssp', 'nth', 'nT', 'ncoun'), iri_exp),
                              'IRIexp_glob': (('nrcl', 'nvar', 'nrcp', 'nssp', 'nth', 'nT'), iri_exp_glob),
                              'IRIexp_clim': (('nrcl', 'nvar', 'nrcp', 'nth', 'nT', 'ncoun'), iri_exp_clim),
                              'IRIexp_clim_glob': (('nrcl', 'nvar', 'nrcp', 'nth', 'nT'), iri_exp_glob_clim),
                              'IRIexp_ssp': (('nrcl', 'nvar', 'nssp', 'nth', 'nT', 'ncoun'), iri_exp_ssp),
                              'IRIexp_ssp_glob': (('nrcl', 'nvar', 'nssp', 'nth', 'nT'), iri_exp_glob_ssp),
                              'IRIexp_unc': (('nrcl', 'nvar', 'nrcp', 'nssp', 'nth', 'nT', 'ncoun'), iri_exp_unc),
                              'IRIexp_glob_unc': (('nrcl', 'nvar', 'nrcp', 'nssp', 'nth', 'nT'), iri_exp_glob_unc)
                              })
dsout.to_netcdf(basepath + '2_pipeline/drisk/store/IRI/iri_exp.nc')
print('DONE')
