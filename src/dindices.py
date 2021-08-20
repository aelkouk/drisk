# Purpose: Transform into percentiles-based index
# Record of revisions:
# Date 		Programmer 		Description of change
# ======== 	============= 	=====================
# 04/2021 	A. Elkouk     	Original code


import numpy as np
import xarray as xr
from scipy import stats
import os

def calc_dindex(index, hist_fpath, prj_fpath, usehist_flag=False, outpath='', Thist=slice(-360, None)):

    if index == 'RI':
        outIndex = calc_ri(hist_fpath, prj_fpath, usehist_flag, outpath, Thist)
    elif index == 'SMI':
        outIndex = calc_smi(hist_fpath, prj_fpath, usehist_flag, outpath, Thist)

    return outIndex

def calc_smi(hist_fpath, prj_fpath, usehist_flag=False, outpath='', Thist=slice(-360, None)):
    ds = xr.open_dataset(hist_fpath, decode_times=False)
    ds_hist = ds.isel(time=Thist)
    hm_nm = os.path.basename(hist_fpath).split('_')[0]
    sm_hist = sum_sm_depth(ds_hist, hm_nm)
    nt_hist, nlat, nlon = sm_hist.shape
    ngrd = nlat * nlon
    sm_histix = np.reshape(sm_hist, (nt_hist, ngrd))
    if usehist_flag:
        ds = ds_hist
        nt_prj = nt_hist
        sm_prjix = sm_histix
        outfname = os.path.basename(hist_fpath)
    else:
        ds = xr.open_dataset(prj_fpath, decode_times=False)
        sm_prj = sum_sm_depth(ds, hm_nm)
        nt_prj, _, _ = sm_prj.shape
        sm_prjix = np.reshape(sm_prj, (nt_prj, ngrd))
        outfname = os.path.basename(prj_fpath)
    smi_out = cal_smi(sm_histix, sm_prjix, np.nan, None, nt_prj, ngrd)
    dsout = xr.Dataset(data_vars={'SMI':(('time', 'ngrd'), smi_out)},
                       coords={'time': ds.time})
    dsout.to_netcdf(os.path.join(outpath, outfname), encoding={'SMI': {"dtype": "f4"}})
    print(outfname, 'DONE')

    return smi_out


def calc_ri(hist_fpath, prj_fpath, usehist_flag=False, outpath='', Thist=slice(-360, None)):
    ds_qtot = calc_qtot(hist_fpath)
    qtot_hist = ds_qtot.isel(time=Thist)
    nt_hist, nlat, nlon = qtot_hist.shape
    ngrd = nlat * nlon
    qtot_histix = np.reshape(qtot_hist.values, (nt_hist, ngrd))
    if usehist_flag:
        ds_qtot = qtot_hist
        nt_prj = nt_hist
        qtot_prjix = qtot_histix
        outfname = os.path.basename(hist_fpath[0].replace('qs', 'qtot'))
    else:
        ds_qtot = calc_qtot(prj_fpath)
        nt_prj, _, _ = ds_qtot.shape
        qtot_prjix = np.reshape(ds_qtot.values, (nt_prj, ngrd))
        outfname = os.path.basename(prj_fpath[0].replace('qs', 'qtot'))
    ri_out = cal_smi(qtot_histix, qtot_prjix, np.nan, None, nt_prj, ngrd)
    dsout = xr.Dataset(data_vars={'RI':(('time', 'ngrd'), ri_out)},
                       coords={'time': ds_qtot.time})
    dsout.to_netcdf(os.path.join(outpath, outfname), encoding={'RI': {"dtype": "f4"}})
    print(outfname, 'DONE')

def sum_sm_depth(ds, hm):
    depths = {'clm45':[0.0175, 0.0276, 0.0455, 0.0750, 0.1236, 0.2038, 0.3360],
            'lpjml':[0.2, 0.3, 0.5],
            'orchidee':[0.001, 0.003, 0.006, 0.012, 0.023, 0.047, 0.094, 0.188, 0.375],
            'matsiro':[0.05, 0.2, 0.75],
            'jules-w1':[0.1, 0.25, 0.65]}
    if hm in depths.keys():
        if hm == 'orchidee':
            dsa = ds.sel(solay=slice(1, 9))
        elif hm == 'clm45':
            dsa = ds.sel(depth=slice(0, 7))
        elif hm == 'jules-w1':
            dsa = ds.sel(soil=slice(0.1, 0.65))
        elif hm == 'lpjml':
            dsa = ds.sel(depth=slice(0.1, 0.75))
        elif hm == 'matsiro':
            dsa = ds.sel(lev=slice(1, 3))
        z = np.array(depths[hm])
        sm_weighted = dsa['soilmoist'].values.transpose(0, 2, 3, 1) * (z / z.sum())
        sm = np.sum(sm_weighted, axis=-1)
    else:
        sm = ds['soilmoist'].values
    return sm

def calc_qtot(fpaths):
    ds_qs = xr.open_dataset(fpaths[0], decode_times=False)
    ds_qsb = xr.open_dataset(fpaths[1], decode_times=False)
    ds_qtot = ds_qs['qs']+ds_qsb['qsb']
    return ds_qtot

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
