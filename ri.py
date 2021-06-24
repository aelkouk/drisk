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

from utilities import ro_inputs, mpi_calc_ri

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()

basedir = '/storage/elkoukah/empirical/'
inpath = basedir + '0_data/QTOT/'
outpath = basedir + '2_pipeline/drisk/store/RI/'
fn_rank = ro_inputs(inpath)
hist_fpath, prj_fpath, usehist_flag = fn_rank[my_rank+60]
print(len(fn_rank))
if not usehist_flag:
    mpi_calc_ri(hist_fpath, prj_fpath, usehist_flag=usehist_flag, outpath=outpath)
else:
    print('SKIPED, ', os.path.basename(prj_fpath[0]))
# print(len(fn_rank))
# for i in fn_rank:
#     print(i[1])
