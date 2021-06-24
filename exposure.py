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

from utilities import calc_exp_dths

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

# inpath = basepath + '2_pipeline/drisk/store/SMI/'
# outpath = basepath + '2_pipeline/drisk/store/SMIthexp/'
# varname = 'SMI'
inpath = basepath + '2_pipeline/drisk/store/RI/'
outpath = basepath + '2_pipeline/drisk/store/RIthexp/'
varname = 'RI'
ths = [0.3, 0.2, 0.1, 0.05, 0.01]

ncfiles = os.listdir(inpath)
infile = inpath+ncfiles[my_rank+40]
# for my_rank in range(len(ncfiles)):
    # infile = inpath + ncfiles[my_rank]
calc_exp_dths(infile, varname, pop_ny, countries, ncoun, ths, outpath)
