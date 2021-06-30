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

from utilities import calc_exp_dths, calc_exp_dths_cstclim, calc_exp_dths_cstpop

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()

basepath = '/storage/elkoukah/empirical/'
ds_countries = xr.load_dataarray(basepath + '0_data/COUNTRIES/countries.nc')
countries = ds_countries.values.ravel()
ncoun = len(ds_countries.ISO)

ds_pop = xr.load_dataarray(basepath + '0_data/POP/pop_total_rural.nc')
pop_tt_rur = ds_pop.values
pop_ny = np.vstack([[pop_tt_rur[:, i]]*10 for i in range(10)])
# pop_ny = pop_ny[6:]

ds_pop_ref = xr.load_dataarray(basepath + '0_data/POP/pop_total_UN-Adjusted.nc')
pop_ref = ds_pop_ref.values[0] # 2000 population counts
pop_ref_ny = np.stack([pop_ref]*100)

inpath = basepath + '2_pipeline/drisk/store/SMI/'
# outpath = basepath + '2_pipeline/drisk/store/SMIthexp/'
varname = 'SMI'
# inpath = basepath + '2_pipeline/drisk/store/RI/'
# varname = 'RI'
ths = [0.3, 0.2, 0.1, 0.05, 0.01]
ncfiles = os.listdir(inpath)

# Total future change
# outpath = basepath + '2_pipeline/drisk/store/RIthexp/'
# ncfiles = [ncf for ncf in ncfiles if ('matsiro' in ncf) or ('jules' in ncf)]
# print(len(ncfiles))
# infile = inpath+ncfiles[my_rank]
# calc_exp_dths(infile, varname, pop_ny, countries, ncoun, ths, outpath)

# Constant population
ncfiles = [ncf for ncf in ncfiles if 'rcp' in ncf]
outpath = basepath + '2_pipeline/drisk/store/%sthexp_cstpop/' % varname
infile = inpath+ncfiles[my_rank+44]
# print(len(ncfiles))
calc_exp_dths_cstpop(infile, varname, pop_ref_ny, countries, ncoun, ths, outpath)

# Constant climate
# ncfiles = [ncf for ncf in ncfiles if 'historical' in ncf]
# outpath = basepath + '2_pipeline/drisk/store/%sthexp_cstclim/' % varname
# infile = inpath+ncfiles[my_rank]
# print(len(ncfiles))
# calc_exp_dths_cstclim(infile, varname, pop_ny[6:], countries, ncoun, ths, outpath)
