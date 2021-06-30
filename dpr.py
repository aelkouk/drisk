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

from utilities import calc_dpr_dths


comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()

basepath = '/storage/elkoukah/empirical/'
# varname = 'SMI'
varname = 'RI'
inpath = basepath + '2_pipeline/drisk/store/%s/' % varname
outpath = basepath + '2_pipeline/drisk/store/dpr/'

ths = np.array([0.3, 0.2, 0.1, 0.05, 0.01])
ncfiles = os.listdir(inpath)
# for my_rank in range(len(ncfiles)):
infile = inpath+ncfiles[my_rank+80]
# assert os.path.isfile(infile)
calc_dpr_dths(infile, varname, ths, outpath)