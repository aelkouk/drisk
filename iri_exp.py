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

ds_countries = xr.load_dataarray(basepath + '0_data/COUNTRIES/countries.nc')
countries = ds_countries.values.ravel()
ncoun = len(ds_countries.ISO)

ds_pop_ref = xr.load_dataarray(basepath + '0_data/POP/pop_total_UN-Adjusted.nc')
pop_ref = ds_pop_ref.values[3] # 2015 population counts

ds_pop = xr.load_dataarray(basepath + '0_data/POP/pop_total_rural.nc')
pop_tt_rur = ds_pop.values.reshape(2,10,nssp,ngrd)
pop_nT = np.stack([pop_tt_rur[0, 2], pop_tt_rur[0, 6]]) # Total pop 2030, 2070

ds_iri = xr.load_dataset(basepath+'2_pipeline/drisk/store/IRI/iri.nc')
iri = ds_iri['IRI'].values#[:,:,:, di]

rclass = np.arange(0.1, 1.1, 0.1)
nrcl = rclass.size

tmp_pop = np.stack([pop_nT]*nvar*nth*(nrcp-1)).reshape(nvar,(nrcp-1),nth,nT,nssp,ngrd).transpose(0,1,4,2,3,5)
iri_exp = np.zeros((nrcl, nvar, (nrcp-1), nssp, nth, nT, ncoun))
iri_exp_glob = np.zeros((nrcl, nvar, (nrcp-1), nssp, nth, nT))
for rci in range(nrcl):
    exprci = (iri<rclass[rci]) * tmp_pop
    for ci in range(ncoun):
        ci_ma = (countries == ci)
        iri_exp[rci, :,:,:,:,:, ci] = np.nansum(exprci[:,:,:,:,:, ci_ma], axis=-1)
    iri_exp_glob[rci] = np.nansum(exprci, axis=-1)

# Clim and Socioeconomic
ds_iri = xr.load_dataset(basepath+'2_pipeline/drisk/store/IRI/iri_climssp_ct.nc')
iri_clim = ds_iri['IRIclim'].values#[:,:, di]
iri_ssp = ds_iri['IRIssp'].values#[:,:, di]

iri_exp_clim = np.zeros((nrcl, nvar, (nrcp-1), nth, nT, ncoun))
iri_exp_glob_clim = np.zeros((nrcl, nvar, (nrcp-1), nth, nT))

tmp_pop = np.stack([pop_nT]*nvar*nth).reshape(nvar,nth,nT,nssp,ngrd).transpose(0,3,1,2,4)
iri_exp_ssp = np.zeros((nrcl, nvar, nssp, nth, nT, ncoun))
iri_exp_glob_ssp = np.zeros((nrcl, nvar, nssp, nth, nT))

for rci in range(nrcl):
    exprci_clim = (iri_clim<rclass[rci]) * pop_ref.ravel()
    exprci_ssp = (iri_ssp<rclass[rci]) * tmp_pop
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
                              })
dsout.to_netcdf(basepath + '2_pipeline/drisk/store/IRI/iri_exp.nc')
print('DONE')
