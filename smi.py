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

from utilities import dict_inputs, mpi_calc_smi

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()

basedir = '/storage/elkoukah/empirical/'
inpath = basedir + '0_data/ISIMIP2b_soilmoist/'
outpath = basedir + '2_pipeline/drisk/store/SMI/'
infiles = dict_inputs(inpath)
fn_rank = []
for key, vals in infiles.items():
    key = inpath + key
    fn_rank.append((key, key, True))
    # for val in vals:
    #     val = inpath + val
    #     fn_rank.append((key, val, False))
hist_fpath, prj_fpath, usehist_flag = fn_rank[my_rank]
if usehist_flag:
    mpi_calc_smi(hist_fpath, prj_fpath, usehist_flag=usehist_flag, outpath=outpath)
else:
    print('SKIPED, ', os.path.basename(prj_fpath))
