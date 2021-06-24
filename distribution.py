# Purpose:
# Record of revisions:
# Date 		Programmer 		Description of change
# ======== 	============= 	=====================
# 11/09/19 	A. Elkouk     	Original code

import numpy as np
import xarray as xr
from scipy import stats


def cal_smi(sm_ref, sm_dw, nodata, nt, nt_rcp, ngrd):
    smi_dw = np.full((nt_rcp, ngrd), nodata)
    for igrd in range(ngrd):
        smgrd = sm_ref[:, igrd]
        if np.isnan(smgrd).all():
            continue
        for cmonth in range(12):
            xs = smgrd[cmonth::12]
            if (xs == 0.0).all():
                continue
            bw = 1.05922384104881 / (xs.size ** 0.2) * np.std(xs)
            if bw == 0:
                continue
            xs_dw = sm_dw[:, igrd][cmonth::12]
            if np.isnan(xs_dw).all():
                continue
            ys_dwcdf = kde_cdf(xs, xs_dw, bw)
            smi_dw[:, igrd][cmonth::12] = ys_dwcdf
    return smi_dw


def kde_cdf(xs, xsout, bw):
    outer = (xsout[:, None] - xs) / bw
    cdf = stats.norm.cdf(outer, *(0, 1)).mean(axis=1)
    cdf[np.isnan(cdf)] = 0.0
    return cdf
