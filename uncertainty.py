# Purpose:
# Record of revisions:
# Date 		Programmer 		Description of change
# ======== 	============= 	=====================
# 11/09/19 	A. Elkouk     	Original code

import numpy as np
import xarray as xr
from scipy import stats
import os
from utilities import lognormal_population, calc_exp_uncer_glob

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
ny = 94

# IRI Uncertainty
## Future exposure
basepath = '/storage/elkoukah/empirical/'
ds_countries = xr.load_dataarray(basepath + '0_data/COUNTRIES/countries.nc')
countries = ds_countries.values.ravel()
ncoun = len(ds_countries.ISO)
ds_pop = xr.load_dataarray(basepath + '0_data/POP/pop_total_rural.nc')
pop_tt_rur = ds_pop.values
pop_ny = np.vstack([[pop_tt_rur[:, i]]*10 for i in range(10)])

inpath = basepath + '2_pipeline/drisk/store/RI/'
varname = 'RI'
ths = [0.3, 0.2, 0.1, 0.05, 0.01]
ncfiles = os.listdir(inpath)

ens_exp = np.full((nvar, nrcp, ngcm, nhm, nssp, nth, ny, ngrd), np.nan)
for ncf in ncfiles:
    parts = ncf[:-3].split('_')
    hmi, gcmi, rcpi, vari = parts[0], parts[1], parts[3], varname
    hmidx, gcmidx, rcpidx, varidx = hms.index(hmi), gcms.index(gcmi.upper()), rcps.index(rcpi), varnames.index(vari)
    infile = inpath + ncf
    if 'historical' in ncf:
        pop_10 = np.stack([pop_tt_rur[:, 0]]*30)
        expi = calc_exp_uncer_glob(infile, varname, pop_10, ths)
        ens_exp[varidx, rcpidx, gcmidx, hmidx,:,:,:30] = expi[0]
    else:
        expi = calc_exp_uncer_glob(infile, varname, pop_ny, ths)
        ens_exp[varidx, rcpidx, gcmidx, hmidx] = expi[0] # total population

inpath = basepath + '2_pipeline/drisk/store/SMI/'
varname = 'SMI'
ncfiles = os.listdir(inpath)
for ncf in ncfiles:
    parts = ncf[:-3].split('_')
    hmi, gcmi, rcpi, vari = parts[0], parts[1], parts[3], varname
    hmidx, gcmidx, rcpidx, varidx = hms.index(hmi), gcms.index(gcmi.upper()), rcps.index(rcpi), varnames.index(vari)
    infile = inpath + ncf
    if 'historical' in ncf:
        pop_10 = np.stack([pop_tt_rur[:, 0]]*30)
        expi = calc_exp_uncer_glob(infile, varname, pop_10, ths)
        ens_exp[varidx, rcpidx, gcmidx, hmidx,:,:,:30] = expi[0]
    else:
        expi = calc_exp_uncer_glob(infile, varname, pop_ny, ths)
        ens_exp[varidx, rcpidx, gcmidx, hmidx] = expi[0] # rural population

## Log-normalize exposure
popnorm = np.full((nvar, (nrcp-1), ngcm, nhm, nssp, nth, ny, ngrd), np.nan)
for vari in range(nvar):
    for gcmi in range(ngcm):
        for hmi in range(nhm):
            for thi in range(nth):
                ref = ens_exp[vari, 0, gcmi, hmi, 1, thi].ravel()
                est = ens_exp[vari, 1:, gcmi, hmi, :, thi].ravel()
                if np.isnan(est).all():
                    continue
                lognorm = lognormal_population(est, ref)
                popnorm[vari, :, gcmi, hmi, :, thi] = np.reshape(lognorm, ((nrcp-1), nssp, ny, ngrd))

## IRI index
dsvi = xr.load_dataarray(basepath+'0_data/POP/vi_norm.nc')
vi = dsvi.values[:, :-1]
vi_ny = np.vstack([[vi[i]]*5 for i in range(14)])
vi_ny = np.vstack([vi_ny, np.stack([vi_ny[-1]]*24)])
tmp_vi = np.stack([vi_ny]*nvar*(nrcp-1)*nth).reshape(nvar, (nrcp-1), nth, ny, nssp, ngrd).transpose(0,1,4,2,3,5)
iri = tmp_vi*popnorm.transpose(2,3,0,1,4,5,6,7)

dsout = xr.Dataset(data_vars={'IRI_ens': (('nhm', 'ngcm', 'nvar', 'nrcp', 'nssp', 'nth', 'ny', 'ngrd'), iri)})
dsout.to_netcdf(basepath + '2_pipeline/drisk/store/IRI/iri_ens.nc')
print('DONE')
