# Purpose: Estimate drought frequency
# Record of revisions:
# Date 		Programmer 		Description of change
# ======== 	============= 	=====================
# 11/09/19 	A. Elkouk     	Original code

import numpy as np
import xarray as xr


def calc_dpr_dths(di, ths, Ti, outpath, outfname):

    _, ngrd = di.shape
    nth = ths.size
    di_Ti = di[Ti]
    di_ths = np.stack([di_Ti] * nth).transpose(1, 2, 0)
    dpr = (di_ths <= ths).sum(axis=0) / di_Ti.shape[0]
    dsout = xr.Dataset(data_vars={'DPr': (('nth', 'ngrd'), dpr.T)})
    dsout.to_netcdf(os.path.join(outpath, outfname))
    print(outfname, 'DONE')

    return dpr